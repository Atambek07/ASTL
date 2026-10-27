"""
main.py — главный цикл «Умного светофора» (MVP v1.0)

Связывает: чтение кадров -> инференс нейросети -> принятие решения (FSM) ->
отправка сигнала на GPIO.

Запуск:
    python3 main.py --config config.yaml

Параметры:
    --config PATH   путь к конфигурационному файлу (по умолчанию config.yaml)
    --video PATH    переопределение видеофайла (удобно для быстрых тестов)
    --no-window     отключить отрисовку окна (headless-режим)
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import time

import cv2
import yaml

from detector import MockDetector, TFLiteDetector, VideoSource, draw_frame
from decision_engine import TrafficFSM
from gpio_controller import RelayController

logger = logging.getLogger("main")


# ---------------------------------------------------------------- configuration
def load_config(path: str) -> dict:
    """Загрузка YAML-конфигурации с проверкой ключевых полей."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("Конфигурация пустая или некорректная")
    # режим
    mode = config.get("mode", "test")
    if mode not in ("test", "live"):
        raise ValueError(f"Недопустимый mode в конфиге: {mode}")
    config["mode"] = mode
    return config


def setup_logging(config: dict):
    """Настройка логирования: консоль + файл (ротация)."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    log_file = log_cfg.get("file", "./logs/app.log")
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            handlers.append(fh)
        except Exception as exc:
            logger.error("Не удалось создать файл лога %s: %s", log_file, exc)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        handlers=handlers,
    )


def create_detector(config: dict):
    """Создание детектора: TFLite, при неудаче — MockDetector (fallback)."""
    try:
        detector = TFLiteDetector(config)
        logger.info("Детектор: TFLite (YOLOv8n)")
        return detector
    except FileNotFoundError as exc:
        logger.warning("Модель не найдена: %s", exc)
    except ImportError as exc:
        logger.warning("TFLite не установлен: %s", exc)
    except Exception as exc:
        logger.error("Ошибка инициализации TFLite-детектора: %s", exc)

    if config.get("allow_mock_fallback", True):
        logger.warning("Запускаю синтетический детектор (MockDetector, demo без ИИ)")
        return MockDetector(config)
    logger.error("Запуск невозможен: модель не загружена и fallback запрещён.")
    sys.exit(1)


def option_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Умный светофор — MVP v1.0")
    parser.add_argument("--config", default="config.yaml", help="путь к config.yaml")
    parser.add_argument("--video", default=None, help="переопределить видеофайл")
    parser.add_argument("--no-window", action="store_true", help="отключить окно отрисовки")
    return parser.parse_args()


# ---------------------------------------------------------------------- main
def main():
    args = option_parse()
    config = load_config(args.config)

    if args.video:
        config["mode"] = "test"
        config["video_path"] = args.video

    setup_logging(config)
    logger.info("====== «Умный светофор» MVP v1.0 | режим: %s ======", config["mode"])

    # --- 1. Источник кадров ---
    try:
        source = VideoSource(config)
    except Exception as exc:
        logger.error("Не удалось открыть источник видео: %s", exc)
        sys.exit(1)

    # --- 2. Детектор ---
    detector = create_detector(config)

    # --- 3. FSM принятия решений ---
    fsm = TrafficFSM(config)

    # --- 4. GPIO / реле ---
    gpio = RelayController(config)

    split_fraction = float(config.get("phase_split_fraction", 0.5))
    show_video = bool(config.get("show_video", True)) and not args.no_window
    window_scale = float(config.get("window_scale", 1.0))
    window_name = "Smart Traffic Light — MVP v1.0"

    if show_video:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    start_ts = time.monotonic()

    try:
        while True:
            now_ms = int(time.monotonic() * 1000)
            ok, frame = source.read()
            if not ok:
                logger.warning("Конец источника видео — завершение сеанса")
                break

            # --- детекция ---
            batch = detector.infer(frame)

            # --- разделение на фазы A/B и расчёт очередей ---
            counter_a, counter_b = batch.get_phase_counters(split_fraction)

            # --- решение ---
            result = fsm.update(counter_a, counter_b, now_ms)
            for ev in result["events"]:
                logger.info("EVENT: %s | PT-A=%.1f PT-B=%.1f | state=%s",
                            ev, result["demand_a"], result["demand_b"], fsm.state)
            # редкое логирование (не захламлять каждый кадр)
            frame_idx += 1

            # --- управление реле/светофором ---
            gpio.apply_outputs(fsm.outputs)

            # --- отрисовка ---
            if show_video:
                frame = draw_frame(
                    frame, batch,
                    counter_a, counter_b,
                    result["demand_a"], result["demand_b"],
                    fsm.state, split_fraction, fsm.current_phase,
                )
                if window_scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        (int(frame.shape[1] * window_scale),
                         int(frame.shape[0] * window_scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC — выход
                    logger.info("Получен ESC — завершение")
                    break
            else:
                # в headless-режиме период = 50 мс, чтобы не грузить CPU без нужды
                time.sleep(0.05)

            # статус каждые ~5 секунд
            if frame_idx % 100 == 0:
                fps = (frame_idx) / max(time.monotonic() - start_ts, 1e-6)
                logger.info("Кадров: %d, fps=%.1f, состояние=%s, выходы=%s",
                            frame_idx, fps, fsm.state, fsm.outputs)

    except KeyboardInterrupt:
        logger.info("Прерывание пользователем (Ctrl+C)")
    except Exception as exc:
        logger.exception("Критическая ошибка главного цикла: %s", exc)
    finally:
        # --- безопасное завершение: гашение всех сигналов ---
        logger.info("Завершение работы: гашу все сигналы и освобождаю ресурсы")
        gpio.cleanup()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
pipeline.py — поток обработки кадров для веб-интерфейса «Умного светофора»

Запускает источник кадров -> детектор -> FSM -> GPIO в отдельном потоке и
публикует два «общих» ресурса:
  * last_jpeg — последний обработанный кадр в формате JPEG (для MJPEG-стрима);
  * last_info — снимок состояния (FSM, счётчики, спрос, FPS, выходы GPIO);

Веб-сервер (webapp.py) только читает эти ресурсы, поэтому между потоком
обработки и HTTP-запросами нет гонок.

Дополнительно здесь реализованы «живые» настройки калибровки камеры:
  * rotation_angle         — поворот кадра на угол (компенсация наклонной камеры);
  * phase_split_fraction   — вертикальная граница А/В (0..1 от ширины кадра);
которые можно менять прямо из интерфейса без перезапуска.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter

import cv2

from detector import URLVideoSource, draw_frame

logger = logging.getLogger("pipeline")

COLORS_DOMAIN = {
    "pedestrian": "Пешеход",
    "passenger_car": "Легковая",
    "bus": "Автобус",
    "truck": "Грузовик",
    "motorcycle": "Мотоцикл",
    "bicycle": "Велосипед",
    "emergency_vehicle": "Спецтранспорт",
}


class LiveState:
    """Разделяемые между потоком обработки и веб-сервером данные."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_jpeg: bytes | None = None       # последний кадр (JPEG)
        self.last_info: dict = {}                  # снимок состояния
        self.source_ok: bool = False               # открыт ли источник
        self.error: str | None = None


class Pipeline:
    """Поток обработки кадров. Стартует в фоне при запуске веб-сервера."""

    def __init__(self, config, source, detector, fsm, gpio):
        self.config = config
        self.source = source
        self.local_source = source          # исходный источник (камера/файл)
        self.detector = detector
        self.fsm = fsm
        self.gpio = gpio
        self.state = LiveState()

        self.window_scale = float(config.get("window_scale", 1.0))

        # «Живые» настройки (меняются из веб-интерфейса)
        self.cfg_lock = threading.Lock()
        self.split_fraction = float(config.get("phase_split_fraction", 0.5))
        self.rotation_angle = float(config.get("camera_rotation_angle", 0.0))
        self.show_boxes = bool(config.get("show_boxes", True))
        self.current_url: str | None = None   # активный интернет-стрим

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._loop_started = time.monotonic()

    def set_stream(self, url: str) -> bool:
        """Переключает источник на интернет-стрим (url) или обратно на локальный.

        Возвращает True при успехе (в т.ч. при возврате к локальному источнику).
        Смена и освобождение источника выполняются под cfg_lock, чтобы поток
        обработки (_run) никогда не взял ссылку на уже освобождённый источник.
        """
        url = (url or "").strip()

        with self.cfg_lock:
            # старая ссылка (не локальная) — отсоединяем
            try:
                if self.source is not self.local_source:
                    self.source.release()
            except Exception:
                pass

            if not url:
                # возврат к локальной камере / файлу
                self.source = self.local_source
                self.current_url = None
                return True

            try:
                new_src = URLVideoSource(url, self.config.get("target_fps", 0))
            except Exception as exc:
                logging.getLogger("pipeline").error("Не удалось открыть стрим %s: %s", url, exc)
                self.source = self.local_source
                self.current_url = None
                return False
            self.source = new_src
            self.current_url = url
            logging.getLogger("pipeline").info("Переключено на интернет-стрим: %s", url)
            return True

    def get_stream_info(self) -> dict:
        return {"url": self.current_url or ""}

    # ------------------------------------------------------------ настройки
    def set_setting(self, name: str, value):
        """Изменение «живой» настройки из интерфейса (под валидацией)."""
        with self.cfg_lock:
            if name == "split_fraction":
                self.split_fraction = max(0.05, min(0.95, float(value)))
                return True
            if name == "rotation":
                self.rotation_angle = max(-180.0, min(180.0, float(value)))
                return True
            if name == "show_boxes":
                self.show_boxes = bool(value)
                return True
        return False

    def get_settings(self) -> dict:
        with self.cfg_lock:
            return {
                "split_fraction": round(self.split_fraction, 3),
                "rotation": round(self.rotation_angle, 1),
                "show_boxes": self.show_boxes,
            }

    # ------------------------------------------------------------ поворот
    @staticmethod
    def rotate_frame(frame, angle: float):
        """Поворот кадра на угол (град., против часовой) с сохранением размера.

        Для компенсации наклонной камеры. При 0 — возвращает кадр без изменений.
        """
        if not angle:
            return frame
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            frame, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    # ------------------------------------------------------------ поток
    def start(self):
        self._thread = threading.Thread(target=self._run, name="pipeline",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self):
        logger.info("Поток обработки запущен")
        self.state.source_ok = True
        while not self._stop.is_set():
            now_ms = int(time.monotonic() * 1000)
            try:
                # берём ссылку на источник под блокировкой — set_stream() может
                # заменить/освободить источник в любой момент из HTTP-потока
                with self.cfg_lock:
                    source = self.source
                ok, frame = source.read()
                if not ok:
                    self.state.error = "Источник видео недоступен"
                    time.sleep(0.5)
                    continue
                self.state.error = None

                with self.cfg_lock:
                    split_frac = self.split_fraction
                    angle = self.rotation_angle
                    show_boxes = self.show_boxes

                # компенсация наклона камеры (если задан поворот)
                if angle:
                    frame = self.rotate_frame(frame, angle)

                # детекция
                batch = self.detector.infer(frame)

                # разделение на фазы А/В и спрос
                counter_a, counter_b = batch.get_phase_counters(split_frac)
                result = self.fsm.update(counter_a, counter_b, now_ms)

                # управление светофором (GPIO)
                self.gpio.apply_outputs(self.fsm.outputs)

                # отрисовка кадра со всеми пояснениями
                canvas = draw_frame(
                    frame, batch,
                    counter_a, counter_b,
                    result["demand_a"], result["demand_b"],
                    self.fsm.state, split_frac, self.fsm.current_phase,
                    show_boxes=show_boxes,
                )

                # масштаб кадра под видео
                if self.window_scale != 1.0:
                    canvas = cv2.resize(
                        canvas,
                        (int(canvas.shape[1] * self.window_scale),
                         int(canvas.shape[0] * self.window_scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                # сжатие в JPEG
                ok_jpg, buf = cv2.imencode(
                    ".jpg", canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                self._frame_count += 1

                # снимок состояния для JSON API и схемы перекрёстка
                info = {
                    "mode": self.config.get("mode", "test"),
                    "state": self.fsm.state,
                    "phase": self.fsm.current_phase,
                    "outputs": dict(self.fsm.outputs),
                    "demand_a": round(result["demand_a"], 1),
                    "demand_b": round(result["demand_b"], 1),
                    "queue_a": sum(counter_a.values()),
                    "queue_b": sum(counter_b.values()),
                    "counters": {
                        "a": {k: v for k, v in counter_a.items()},
                        "b": {k: v for k, v in counter_b.items()},
                    },
                    "detections": len(batch.class_ids),
                    "latency_ms": batch.t_ms,
                    "fps": round(
                        self._frame_count / max(time.monotonic() - self._loop_started, 1e-6), 1
                    ),
                    "settings": self.get_settings(),
                }
                with self.state.lock:
                    if ok_jpg:
                        self.state.last_jpeg = buf.tobytes()
                    self.state.last_info = info
                    self.state.source_ok = True

            except KeyboardInterrupt:
                break
            except Exception as exc:  # не роняем поток — логируем и продолжаем
                logger.exception("Ошибка в потоке обработки: %s", exc)
                self.state.error = str(exc)
                time.sleep(0.5)

        logger.info("Поток обработки остановлен")

"""
webapp.py — веб-интерфейс «Умного светофора»

Запускает в фоне поток обработки (pipeline.Pipeline) и отдаёт удобный
браузерный интерфейс:

  /              — главная страница (видео + схема перекрёстка + настройки)
  /video_feed    — MJPEG-стрим обработанного видео
  /api/state     — JSON-снимок состояния (для схемы и статусов)
  /api/settings  — GET/POST «живых» настроек (граница фаз, поворот камеры)
  /api/log       — последние события/статус из логов (для журнала)

Запуск:
    python3 webapp.py [--config config.yaml] [--port 5000]
    Затем откройте http://localhost:5000 в браузере.
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

import pipeline as pipeline_mod
from webapp_log import MemoryLogHandler

app = Flask(__name__)
CORS(app)

# Глобальные объекты, инициализируются в main()
PIPELINE: pipeline_mod.Pipeline | None = None
CONFIG: dict | None = None


# ------------------------------------------------------------------ страница
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------ стрим
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if PIPELINE is not None and PIPELINE.state.last_jpeg is not None:
                jpeg = PIPELINE.state.last_jpeg
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            import time
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------------ API
@app.route("/api/state")
def api_state():
    payload = {"connected": False}
    if PIPELINE is not None:
        with PIPELINE.state.lock:
            payload = dict(PIPELINE.state.last_info)
            payload["connected"] = True
            payload["source_ok"] = PIPELINE.state.source_ok
            payload["error"] = PIPELINE.state.error
    return jsonify(payload)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if PIPELINE is not None:
            for key, value in data.items():
                PIPELINE.set_setting(key, value)
    if PIPELINE is not None:
        return jsonify(PIPELINE.get_settings())
    return jsonify({})


@app.route("/api/log")
def api_log():
    handler = MemoryLogHandler.get_instance()
    lines = list(handler.buffer)
    return jsonify({"log": lines})


@app.route("/api/stream", methods=["POST"])
def api_stream():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    ok = False
    if PIPELINE is not None:
        ok = PIPELINE.set_stream(url)
    return jsonify({"ok": ok, "url": url})


@app.route("/api/stream", methods=["GET"])
def api_stream_get():
    if PIPELINE is not None:
        return jsonify(PIPELINE.get_stream_info())
    return jsonify({"url": ""})


# ------------------------------------------------------------------ main
def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("Конфигурация пустая или некорректная")
    mode = config.get("mode", "test")
    if mode not in ("test", "live"):
        raise ValueError(f"Недопустимый mode в конфиге: {mode}")
    config["mode"] = mode
    return config


def build_components(config: dict):
    """Создаёт источник, детектор, FSM и GPIO (как в main.py)."""
    from detector import TFLiteDetector, MockDetector, VideoSource
    from decision_engine import TrafficFSM
    from gpio_controller import RelayController

    source = VideoSource(config)

    # детектор с fallback
    detector = None
    try:
        detector = TFLiteDetector(config)
    except Exception as exc:
        logging.getLogger("webapp").warning("TFLite недоступен (%s) — запускаю демо-детектор", exc)
        detector = MockDetector(config)

    fsm = TrafficFSM(config)
    gpio = RelayController(config)
    return source, detector, fsm, gpio


def option_parse():
    parser = argparse.ArgumentParser(description="Умный светофор — веб-интерфейс")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    global PIPELINE, CONFIG
    args = option_parse()

    if not os.path.exists(args.config):
        print(f"Конфигурационный файл не найден: {args.config}", file=sys.stderr)
        sys.exit(1)

    CONFIG = load_config(args.config)

    # логирование: файл + память (для журнала в интерфейсе)
    handlers = [MemoryLogHandler.get_instance()]
    log_file = CONFIG.get("logging", {}).get("file", "./logs/app.log")
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
                )
            )
        except Exception as exc:
            logging.getLogger("webapp").error("Не удалось создать файл лога: %s", exc)
    logging.basicConfig(
        level=getattr(logging, str(CONFIG.get("logging", {}).get("level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        handlers=handlers,
    )

    # компоненты
    source, detector, fsm, gpio = build_components(CONFIG)
    PIPELINE = pipeline_mod.Pipeline(CONFIG, source, detector, fsm, gpio)
    PIPELINE.start()
    logging.getLogger("webapp").info(
        "Веб-интерфейс запущен: http://%s:%d", args.host, args.port
    )

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    finally:
        PIPELINE.stop()
        gpio.cleanup()
        source.release()


if __name__ == "__main__":
    main()

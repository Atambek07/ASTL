#!/usr/bin/env python3
"""
get_yolov8n_tflite.py — загрузка и конвертация YOLOv8n в TFLite int8.

Два пути:
  A) Быстрый (рекомендуется для старта): скачать готовый .tflite с GitHub
     (см. URL ниже). Работает без установки torch/ultralytics.
  B) Собственный: сконвертировать модель ultralytics в формат TFLite int8
     (нужны torch, ultralytics; требует доступного датасета для калибровки).
"""
import argparse
import os
import sys
import urllib.request

DEFAULT_URL = (
    "https://github.com/shubham0204/Android-YOLOv8-Object-Detection-TFLite/"
    "raw/main/model/yolov8n_int8.tflite"
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def download(url: str, dest: str):
    """Скачивание файла модели по URL."""
    print(f"Скачивание модели: {url}")
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — публичная CDN
        os.replace(tmp, dest)
        print(f"Модель сохранена: {dest}")
    except Exception as exc:
        print(f"Ошибка скачивания: {exc}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(1)


def convert_from_ultralytics(weights: str, dest: str, calib_video: str | None):
    """Конвертация из формата ultralytics (опционально, требует GPU/CPU+torch)."""
    try:
        from ultralytics import YOLO
        import tflite
    except ImportError as exc:
        print("Нужны пакеты: pip install ultralytics tensorflow", file=sys.stderr)
        sys.exit(1)

    print(f"Конвертация из: {weights} -> {dest}")
    model = YOLO(weights)

    # Экспорт в TFLite int8 (квантование требует калибровочных данных)
    kwargs = dict(format="tflite", imgsz=640, half=False)
    if calib_video and os.path.exists(calib_video):
        kwargs["int8"] = True
        kwargs["data"] = calib_video
    elif not calib_video:
        print("Предупреждение: int8-квантование требует калибровки; пропускаю")
        kwargs["int8"] = False

    exported = model.export(**kwargs)
    print(f"Экспортирована модель: {exported}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(exported, dest)
    print(f"Модель перемещена: {dest}")


def main():
    ap = argparse.ArgumentParser(description="Загрузка YOLOv8n TFLite-модели")
    ap.add_argument("--url", default=DEFAULT_URL, help="URL готовой TFLite-модели")
    ap.add_argument("--weights", default=None, help="путь к .pt для конвертации (опция)")
    ap.add_argument("--calib-video", default=None, help="видео для калибровки int8")
    ap.add_argument("--dest", default=None, help="путь сохранения модели")
    args = ap.parse_args()

    dest = args.dest or os.path.join(MODEL_DIR, "yolov8n-int8.tflite")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if args.weights:
        convert_from_ultralytics(args.weights, dest, args.calib_video)
    else:
        download(args.url, dest)


if __name__ == "__main__":
    main()

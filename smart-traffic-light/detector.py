"""
detector.py — модуль компьютерного зрения «Умного светофора» (MVP v1.0)

Задачи:
  * Загрузка TFLite-модели (YOLOv8n, int8) и инференс по кадру.
  * Препроцессинг кадра (letterbox), постпроцессинг (NMS, масштабирование).
  * Подсчёт объектов по доменным классам (пешеходы, легковые, автобус, …).
  * Разбиение детекций на две «фазы» (A/B) по вертикальной границе кадра —
    упрощение для стендового MVP; в боевой конфигурации камера ↔ подход 1:1.
  * Класс VideoSource: чтение из видеофайла с зацикливанием или с камеры.
  * Класс MockDetector: синтетический детектор (демо-режим без модели ИИ).
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections import Counter

import cv2
import numpy as np

logger = logging.getLogger("detector")

# Классы доменной модели продукта
DOMAIN_CLASSES = [
    "pedestrian",
    "passenger_car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "emergency_vehicle",
]

# Сопоставление индексов COCO -> доменные классы
# (для стандартной YOLOv8n, обученной на COCO)
COCO_TO_DOMAIN = {
    0: "pedestrian",      # person
    1: "bicycle",
    2: "passenger_car",   # car
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Цвета отрисовки по классам (BGR)
CLASS_COLORS = {
    "pedestrian": (0, 255, 0),
    "passenger_car": (0, 255, 255),
    "bus": (0, 0, 255),
    "truck": (0, 128, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (0, 200, 200),
    "emergency_vehicle": (255, 0, 0),
}


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    """Размер кадра с сохранением пропорций + дополнение (letterbox).

    Возвращает (img, ratio, pad), где pad = (pad_left, pad_top).
    Выравнивание до кратного stride согласуется с letterbox YOLOv8.
    """
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Примечание: не ограничиваем 1.0 — модели нужен ТОЧНЫЙ размер входа
    # (например 640x640), поэтому маленькие кадры тоже растягиваются.

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[1] - new_unpad[0])
    dh = (new_shape[0] - new_unpad[1])

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))
    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (left, top)


def nms(boxes, scores, iou_threshold=0.45):
    """Простейший Non-Maximum Suppression. boxes: (N, 4) в формате xyxy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def postprocess_yolov8(pred, conf_thres=0.35, iou_thres=0.45):
    """Разбор выхода YOLOv8 (TFLite).

    Выход модели: (1, 84, N) или (1, N, 84); 84 = 4 координаты + 80 классов COCO.
    Координаты ожидаются в пикселях входа модели (box-space после letterbox).
    Возвращает (boxes_xyxy, scores, class_indices).
    """
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] != 84:           # если пришло (N, 84) — транспонируем
        pred = pred.T
    cx = pred[0]
    cy = pred[1]
    bw = pred[2]
    bh = pred[3]
    class_scores = pred[4:]                    # (80, N)
    class_ids = class_scores.argmax(axis=0)
    conf = class_scores.max(axis=0)
    mask = conf >= conf_thres
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    conf, class_ids = conf[mask], class_ids[mask]
    if conf.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = nms(boxes, conf, iou_thres)
    return boxes[keep], conf[keep], class_ids[keep]


class DetectionBatch:
    """Контейнер с детекциями одного кадра."""

    __slots__ = ("frame_shape", "boxes", "scores", "class_ids", "t_ms")

    def __init__(self, frame_shape, boxes, scores, class_ids, t_ms):
        self.frame_shape = frame_shape      # (h, w)
        self.boxes = boxes                  # (N, 4) xyxy
        self.scores = scores                # (N,)
        self.class_ids = list(class_ids)    # доменные имена классов
        self.t_ms = t_ms

    def count_by_class(self):
        """Счётчик объектов по классам во всём кадре."""
        return Counter(self.class_ids)

    def get_phase_counters(self, split_fraction=0.5):
        """Разделяет скопления на фазу A (лево) и фазу B (право).

        Упрощение для стендового MVP: вертикальная граница делит кадр
        на две зоны подходов. В боевой конфигурации каждая камера
        соответствует своему подходу (данные приходят раздельно).
        Возвращает (counter_a, counter_b) — Counter по доменным классам.
        """
        h, w = self.frame_shape
        mid_x = int(w * split_fraction)
        counter_a = Counter()
        counter_b = Counter()
        for box, cls in zip(self.boxes, self.class_ids):
            center_x = (box[0] + box[2]) / 2.0
            if center_x <= mid_x:
                counter_a[cls] += 1
            else:
                counter_b[cls] += 1
        return counter_a, counter_b


class TFLiteDetector:
    """Детектор на базе TensorFlow Lite (YOLOv8n int8, 640x640)."""

    def __init__(self, config):
        self.config = config
        self.size = int(config.get("model_input_size", 640))
        self.conf_thres = float(config.get("conf_threshold", 0.35))
        self.iou_thres = float(config.get("iou_threshold", 0.45))
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._nchw = False          # True, если вход модели вида [1,3,H,W]
        self._input_as_float = False  # True, если вход float32 (нужна /255)
        self._load_model()
        logger.info(
            "TFLite-дететкор готов (вход %dx%d, layout=%s, dtype=%s)",
            self.size, self.size,
            "NCHW" if self._nchw else "NHWC",
            self.input_details[0]["dtype"],
        )

    def _load_model(self):
        """Загрузка интерпретатора TFLite и тензоров модели."""
        path = os.path.abspath(self.config.get("model_path", "./models/yolov8n_int8.tflite"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        try:
            # Приоритет — лёгкий пакет tflite-runtime (Raspberry Pi ARM)
            import tflite_runtime.interpreter as tflite
            logger.info("Используется tflite-runtime")
        except ImportError:
            try:
                from tensorflow import lite as tflite
                logger.info("Используется tensorflow.lite")
            except ImportError as exc:
                raise ImportError(
                    "TensorFlow Lite не установлен. Выполните: "
                    "pip install tflite-runtime"
                ) from exc

        try:
            self.interpreter = tflite.Interpreter(model_path=path, num_threads=4)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as exc:
            raise RuntimeError(f"Ошибка загрузки модели TFLite: {exc}") from exc

        if not self.input_details or not self.output_details:
            raise RuntimeError("Модель не содержит входов/выходов (невалидный .tflite)")

        # --- определение формата входа: [1,3,H,W] (NCHW) или [1,H,W,3] (NHWC)
        in_shape = tuple(self.input_details[0]["shape"])
        logger.info("Размер входа модели: %s (dtype=%s)", in_shape,
                    self.input_details[0]["dtype"])
        self._nchw = False
        self._input_as_float = False
        if len(in_shape) == 4:
            self._nchw = in_shape[1] == 3
            self.size = int(in_shape[-1])  # квадратный вход
        self._input_as_float = self.input_details[0]["dtype"] == np.float32

    def _prepare_tensor(self, img):
        """Letterbox-кадр -> тензор с нужным порядком каналов и нормализацией."""
        data = np.transpose(img, (2, 0, 1)) if self._nchw else img
        if self._input_as_float:
            data = data.astype(np.float32) / 255.0
        else:
            data = data.astype(np.uint8)
        return data[None, ...]  # [1,3,H,W] или [1,H,W,3]

    def infer(self, frame):
        """Инференс по кадру -> DetectionBatch в координатах исходного кадра."""
        t0 = time.monotonic()
        h, w = frame.shape[:2]

        # 1) препроцессинг: letterbox до (size, size)
        img, ratio, (pad_l, pad_t) = letterbox(frame, (self.size, self.size))

        # 2) подготовка тензора: нормализация и порядок каналов (NHWC/NCHW)
        input_data = self._prepare_tensor(img)

        # 3) инференс
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]["index"])

        # 4) постпроцессинг: NMS + перевод координат в исходный кадр
        boxes, scores, ids = postprocess_yolov8(raw, self.conf_thres, self.iou_thres)
        if boxes.size:
            boxes = boxes - np.array([pad_l, pad_t, pad_l, pad_t], dtype=np.float32)
            boxes = boxes / ratio
            np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1], out=boxes)

        # 5) отображение индексов COCO -> доменные классы
        domain_ids = []
        keep_idx = []
        for i, coco_id in enumerate(ids):
            dom = COCO_TO_DOMAIN.get(int(coco_id))
            if dom is not None:
                domain_ids.append(dom)
                keep_idx.append(i)
        if keep_idx:
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

        t_ms = int((time.monotonic() - t0) * 1000)
        return DetectionBatch((h, w), boxes, scores, domain_ids, t_ms)


class MockDetector:
    """Синтетический детектор — демо-режим без модели ИИ (fallback).

    Эмулирует несколько объектов классов со случайным блужданием по кадру,
    чтобы проверить весь конвейер (детекция -> FSM -> GPIO) на «живом» видео
    даже без самой модели. Честно помечается в логах как демо-режим.
    """

    def __init__(self, config):
        random.seed(7)
        self.config = config
        # Набор классов, из которого набираются «объекты»
        self.class_pool = [
            "passenger_car", "passenger_car", "bus", "passenger_car",
            "pedestrian", "pedestrian", "passenger_car",
        ]
        self.objects = []

    def infer(self, frame):
        h, w = frame.shape[:2]

        # первичное размещение объектов по кадру
        if not self.objects:
            for cls in self.class_pool:
                size = 90 if cls == "bus" else (28 if cls == "pedestrian" else 60)
                x = random.uniform(0.08 * w, 0.92 * w)
                y = random.uniform(0.25 * h, 0.80 * h)
                self.objects.append({
                    "cls": cls, "x": x, "y": y, "s": size,
                    "vx": random.uniform(-2.5, 2.5),
                    "vy": random.uniform(-1.2, 1.2),
                })

        boxes, scores, ids = [], [], []
        for obj in self.objects:
            # шаг случайного блуждания + отражение от границ кадра
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]
            if obj["x"] < obj["s"] / 2 or obj["x"] > w - obj["s"] / 2:
                obj["vx"] = -obj["vx"]
            if obj["y"] < obj["s"] / 2 or obj["y"] > h - obj["s"] / 2:
                obj["vy"] = -obj["vy"]
            boxes.append([obj["x"] - obj["s"] / 2, obj["y"] - obj["s"] / 2,
                          obj["x"] + obj["s"] / 2, obj["y"] + obj["s"] / 2])
            scores.append(0.90)
            ids.append(obj["cls"])

        return DetectionBatch(
            (h, w),
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            ids,
            0,
        )


class VideoSource:
    """Источник кадров: видеофайл (с зацикливанием) или камера."""

    def __init__(self, config):
        self.config = config
        self.mode = config["mode"]
        self.loop = bool(config.get("video_loop", True))
        self.target_fps = float(config.get("target_fps", 0) or 0)
        self._n = 0
        self._t0 = time.monotonic()
        self._src = config.get("video_path", "") if self.mode != "live" \
            else config.get("camera_source", 0)
        self.cap = cv2.VideoCapture(self._src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник видео: {self._src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25

    def read(self):
        """Возвращает (True, frame). При окончании видео в test-режиме — зацикливает."""
        if not self.cap.isOpened() and self.mode == "live":
            # переоткрытие "зависшей" камеры (восстановление после сбоя)
            time.sleep(1.0)
            self.cap.open(self._src)
            return (True, self.cap.read()[1]) if self.cap.isOpened() else (False, None)

        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                # конец файла -> перемотка в начало
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if ok:
                    logger.debug("Видео зациклено (начало файла)")
            if not ok:
                return False, None

        # троттлинг под целевой fps (без него демо летит со скоростью декодера)
        self._n += 1
        if self.target_fps > 0:
            expected = self._n / self.target_fps
            dt = expected - (time.monotonic() - self._t0)
            if dt > 0:
                time.sleep(dt)

        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class URLVideoSource:
    """Источник кадров из интернета (live-стрим): HTTP(S)-MJPEG, RTSP, HLS.

    Использует возможности OpenCV с бэкендом FFmpeg/GStreamer.
    Режим без зацикливания: если стрим оборвался — открывает повторно.
    """

    def __init__(self, url: str, target_fps: float = 0.0):
        self.url = url
        self.target_fps = float(target_fps or 0)
        self._n = 0
        self._t0 = time.monotonic()
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть поток: {url}")

    def read(self):
        if not self.cap.isOpened():
            # попытка восстановления оборвавшегося стрима
            time.sleep(1.0)
            self.cap.open(self.url)
            if not self.cap.isOpened():
                return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        self._n += 1
        if self.target_fps > 0:
            expected = self._n / self.target_fps
            dt = expected - (time.monotonic() - self._t0)
            if dt > 0:
                time.sleep(dt)
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


def draw_frame(frame, batch, count_a, count_b, demand_a, demand_b, state,
               split_fraction=0.5, phase=None, show_boxes=True):
    """Отрисовка bounding boxes, счётчиков и служебной панели на кадре.

    show_boxes=False — рамки детекций и подписи не рисуются (остаётся панель
    метрик и граница фаз).
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # вертикальная граница фаз
    mid_x = int(w * split_fraction)
    cv2.line(canvas, (mid_x, 0), (mid_x, h), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PHASE A", (max(mid_x - 80, 6), 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PHASE B", (min(mid_x + 8, w - 120), 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # bounding boxes
    if show_boxes:
        for box, score, cls in zip(batch.boxes, batch.scores, batch.class_ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(y1 - th - 6, 0)), (x1 + tw, y1), color, -1)
            cv2.putText(canvas, label, (x1, max(y1 - 4, th)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # информационная панель
    panel_lines = [
        f"State: {state}",
        f"Phase: {phase if phase else '—'}",
        f"PT-A: {demand_a:.1f} pers | queue: {sum(count_a.values())} obj",
        f"PT-B: {demand_b:.1f} pers | queue: {sum(count_b.values())} obj",
        f"Latency: {batch.t_ms} ms",
        f"Detections: {len(batch.class_ids)}",
    ]
    start_y = 46
    for i, line in enumerate(panel_lines):
        cv2.putText(canvas, line, (6, start_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (6, start_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

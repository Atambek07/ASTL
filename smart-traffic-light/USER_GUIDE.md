---
title: "Умный светофор — Руководство пользователя и техническое описание"
subtitle: "MVP v1.0 (edge, Raspberry Pi 4)"
author: "Инженерный отдел"
lang: ru-RU
---

# Умный светофор: руководство пользователя и техническое описание

## 1. Суть проекта и метрика Person-Throughput

«Умный светофор» — edge-система динамического управления фазами светофорного
объекта на базе Raspberry Pi 4 Model B. Вместо классического подсчёта автомобилей
(vehicle throughput) системой вводится метрика **Person-Throughput (PT)** —
«сколько людей проходит за единицу времени», учитывающая пассажиров автобусов.

**Базовая формула:**

```
PT_фазы = Σ (count_vehicle_class × occupancy_class) + Σ пешеходов
```

Коэффициенты вместимости (настраиваются в `config.yaml`, секция `occupancy`):

| Класс | Коэффициент | Комментарий |
|---|---|---|
| passenger_car | 1.2 | средняя загрузка |
| bus | 35.0 | маршрутный автобус |
| truck | 1.1 | водитель + пассажир |
| motorcycle | 1.0 | водитель |
| bicycle | 1.0 | велосипедист |
| pedestrian | 1.0 | пешеход — точный счёт |
| emergency_vehicle | 0.0 | не считается, только событие (приоритет) |

**Алгоритм:** локальный конечный автомат (FSM) глядит на очереди обеих фаз
и жадным правилом отдаёт зелёный фазе с максимальным приведённым скоплением
людей. Экстренный транспорт прерывает цикл немедленно.

Для стендового MVP кадр делится на две половины (зоны фаз A/B) вертикальной
границей — параметр `phase_split_fraction`. В боевой версии каждая камера
отображается на свой подход 1:1.

---

## 2. Состав пакета

```
smart-traffic-light/
├── config.yaml              # настройки системы
├── detector.py              # CV-модуль (TFLite + OpenCV)
├── decision_engine.py       # FSM и метрика PT
├── gpio_controller.py       # управление реле (pigpio / RPi.GPIO / mock)
├── main.py                  # главный цикл
├── requirements.txt         # зависимости
├── HARDWARE_WIRING.md       # схема подключения
├── USER_GUIDE.md            # этот документ
├── scripts/get_yolov8n_tflite.py  # загрузка/конвертация модели
├── models/                  # сюда кладём .tflite
└── logs/                    # файлы логов (создаются автоматически)
```

---

## 3. Установка зависимостей на Raspberry Pi OS

### 3.1 Базовая система (Raspberry Pi OS Lite 64-bit рекомендуем)

```bash
sudo apt update
sudo apt upgrade -y

# Базовые утилиты и Python
sudo apt install -y python3-pip python3-dev python3-venv git curl

# OpenCV (системный — быстрее, чем pip-сборка на ARM)
sudo apt install -y python3-opencv libatlas-base-dev

# pigpio (демон + обвязка для точного GPIO)
sudo apt install -y pigpio python3-pigpio
sudo systemctl enable --now pigpiod

# Экранный пакет (для окна cv2.imshow) — на сервере без монитора можно пропустить
sudo apt install -y libqt5gui5 2>/dev/null || true
```

### 3.2 Python-зависимости (в виртуальном окружении, рекомендовано)

```bash
cd smart-traffic-light
python3 -m venv .venv
source .venv/bin/activate

# Основные пакеты
pip install --upgrade pip
pip install PyYAML numpy

# OpenCV (если не ставили через apt; для ARM-headless версия)
pip install opencv-python-headless

# TensorFlow Lite Runtime — варианты:
#  A) На Raspberry Pi (ARM):
pip install tflite-runtime
#  B) На ПК для отладки (x86/Windows/macOS) — полный TensorFlow:
# pip install tensorflow

pip install -r requirements.txt   # остальное
```

> **Проверка окружения:**
> ```bash
> python3 -c "import cv2, numpy, yaml; print('OK', cv2.__version__)"
> python3 -c "import tflite_runtime.interpreter as t; print('TFLite OK')"
> ```

### 3.3 Установка модели ИИ

```bash
cd smart-traffic-light
python3 scripts/get_yolov8n_tflite.py                # скачивание готовой YOLOv8n int8
# либо своя: python3 scripts/get_yolov8n_tflite.py --weights yolov8n.pt --calib-video sample.mp4
ls -la models/                                       # должен появиться файл
```

Модель кладём в `models/yolov8n-int8.tflite` (путь настроен в `config.yaml`).
Если модель отсутствует, программа автоматически запустит **MockDetector**
(синтетические объекты) — демо заработает без ИИ.

---

## 4. Запуск в тестовом режиме (по видеофайлу)

1. Положите `.mp4` с записью перекрёстка рядом с проектом, например: `test_video.mp4`.
2. В `config.yaml` убедитесь, что:
   ```yaml
   mode: test
   video_path: "./test_video.mp4"
   video_loop: true          # зацикливание
   show_video: true
   ```
3. Запустите:

   ```bash
   source .venv/bin/activate
   python3 main.py --config config.yaml
   ```
   (или с другой записью: `python3 main.py --video /path/to/clip.mp4`)

4. Откроется окно с кадрами: bounding boxes, счётчики, метрики **PT-A / PT-B**,
   текущее состояние FSM. Наблюдайте, как система переключает фазу A/B в
   зависимости от скопления людей. Выход из программы — `ESC` (в окне) или `Ctrl+C`.

> **Если окно не появляется** (сервер без графики): добавьте в конфиг
> `show_video: false`, а метрики и события увидите в логе:
> `tail -f logs/app.log`.

---

## 5. Переключение в боевой режим (реальная камера)

### 5.1 USB-камера (проще всего)

```yaml
mode: live
camera_source: 0        # 0 — /dev/video0
show_video: true
```

```bash
python3 main.py --config config.yaml
```

### 5.2 Камера CSI (пи коннектор)

```yaml
mode: live
# URL конвейера gstreamer для libcamera (установите libcamera-compat и gst):
camera_source: "libcamerasrc ! camera name=cam0 ! libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! appsink"
```

Перед использованием CSI-камеры на Raspberry Pi OS (Bookworm) выполните:

```bash
sudo apt install -y python3-libcamera python3-picamera2 libcamera-tools
# иначе камера будет доступна через /dev/video0/1 после загрузки драйвера
```

### 5.3 Что изменить в конфиге при «бое»-инсталляции

- `occupancy` — настройте коэффициенты по замерам вашего перекрёстка.
- `phase_split_fraction` — при одной камере на два подхода оставьте 0.5,
  при двух камерах (по подготовке из архитеktурного документа) — данные фаз
  приходят раздельно, и граница не нужна.
- `show_video: false` в продуктовой инсталляции.
- `timings` и `thresholds` согласуйте со светофорным контроллером/инженером.

---

## 6. Отказоустойчивость (краткие гарантии MVP)

| Сбой | Реакция |
|---|---|
| Модель не найдена | Авто-fallback на MockDetector (демо продолжает работать) |
| Видео закончилось | Зацикливание (в тестовом режиме) |
| Демон pigpiod не запущен | Автопереход на RPi.GPIO, затем на MOCK-эмуляцию |
| Камера «зависла» | Переоткрытие источника, лог ошибки |
| Ctrl+C / ESC | Гашение всех сигналов, корректное освобождение ресурсов |
| Исключение в кадре | Логируется, цикл продолжается (система не «падает») |

> **Безопасность BOM**: критичные изменения таймингов применяются только в
> конфиг-файле и только инженером с доступом к устройству.

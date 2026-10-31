# «Умный светофор» — MVP v1.0 (Raspberry Pi 4)

**Динамическое управление фазами светофорного объекта по метрике
Person-Throughput (PT) — «люди/час», а не машины.**

Подробное техописание аппаратной части — `docs/Умный_светофор_техническая_документация_v1.0.md`.

## Что в пакете

```
smart-traffic-light/
├── config.yaml                   # настройки (модель, видео, тайминги, пины GPIO)
├── detector.py                   # CV: TFLite + OpenCV (NCHW/NHWC, letterbox, NMS)
├── decision_engine.py            # FSM: метрика PT + переключение фаз
├── gpio_controller.py            # реле: pigpio → RPi.GPIO → MOCK
├── pipeline.py                   # поток обработки кадров (для веб-интерфейса)
├── webapp.py                     # веб-интерфейс (Flask): видео, схема, настройки
├── webapp_log.py                 # буфер журнала событий для веб-интерфейса
├── templates/index.html          # страница веб-интерфейса
├── main.py                       # главный цикл (консольное окно OpenCV)
├── requirements.txt              # для Raspberry Pi
├── requirements-laptop.txt       # для ноутбука (отладка)
├── HARDWARE_WIRING.md            # схема подключения реле/светодиодов
├── USER_GUIDE.md                 # полное руководство (запуск RPi + ноутбук)
└── scripts/get_yolov8n_tflite.py # загрузка/конвертация модели
```

> **Уже в комплекте:** модель `models/yolov8n_int8.tflite` и тестовое видео
> `test_video.mp4` — можно запускать сразу.

---

## Быстрый старт

### Вариант A — Ноутбук (Linux / macOS / Windows)

```bash
cd smart-traffic-light
python3 -m venv .venv && source .venv/bin/activate
# Windows (PowerShell):  python -m venv .venv; .\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements-laptop.txt

python main.py --config config.yaml
```

Откроется окно с детекциями и метриками PT-A / PT-B (переходы фаз в логе).
GPIO автоматически уйдёт в MOCK-эмуляцию. Выход — `ESC` / `Ctrl+C`.

### Вариант Б — Raspberry Pi (с реле и светодиодами)

```bash
# 1) базовая система
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev \
                   pigpio python3-pigpio
sudo systemctl enable --now pigpiod

# 2) проект
cd ~ && git clone <repo> smart-traffic-light && cd smart-traffic-light

# 3) зависимости
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install tflite-runtime PyYAML numpy      # (или: pip install -r requirements.txt)

# 4) тест на видео
python main.py --config config.yaml

# 5) боевой режим — в config.yaml: mode: live, camera_source: 0
python main.py --config config.yaml
```

Схема подключения — в `HARDWARE_WIRING.md`. Полные пошаговые инструкции
для обеих платформ — в `USER_GUIDE.md`.

---

## Как это работает (3 шага)

1. **detector.py** — детектирует классы (легковой, автобус, пешеход, …)
   через TFLite-модель YOLOv8n (640×640), рисует bounding boxes.
2. **decision_engine.py** — считает `PT = Σ(count × occupancy)` для каждой фазы
   (автобус «весит» 35 человек vs 1.2 у легкового) и конечным автоматом
   решает, какой фазе дать зелёный (минимальный/максимальный зелёный,
   жёлтый, очистка перекрёстка, приоритет спецтранспорта).
3. **gpio_controller.py** — включает нужные светодиоды через реле.

---

## Требования к модели

- Формат: **TFLite** (`.tflite`), вход **640×640**.
- Поддерживаются NCHW `[1,3,640,640]` и NHWC `[1,640,640,3]`,
  float32 и uint8. Код определяет формат автоматически.
- Скачивание готовой модели:
  ```bash
  python scripts/get_yolov8n_tflite.py
  # или конвертация из ultralytics-весов:
  python scripts/get_yolov8n_tflite.py --weights yolov8n.pt --calib-video sample.mp4
  ```
- Нет модели — включится **MockDetector** (синтетические объекты, демо без ИИ).

---

## Веб-интерфейс (редиска в браузере)

Основной способ работы — **веб-интерфейс** (`webapp.py`). Он показывает видео
с камеры, схему перекрёстка и позволяет настраивать систему вживую.

```bash
python webapp.py --config config.yaml --port 5000
# откройте в браузере:  http://localhost:5000
```

Возможности интерфейса:
- 📹 **Видео** — прямой поток с камеры/файла с рамками детекций и границей фаз.
- 🚦 **Схема перекрёстка** — адаптивная схема: фазы A/B с реальным состоянием
  светофоров (красный/жёлтый/зелёный), спросом и очередями.
- 🛠 **Настройка камеры вживую**:
  - *Граница фаз A/B* — сдвиг вертикальной границы между фазами;
  - *Компенсация наклона камеры* — поворот кадра, если камера стоит «криво»
    (светофоры/подходы не попадают в нужные зоны);
  - **Видео из интернета** — вставьте ссылку на онлайн-стрим (HTTP-MJPEG,
    RTSP `rtsp://…`, HLS `…/playlist.m3u8`) и кнопка «Смотреть» покажет его
    в окне видео. Кнопка «Локальная камера» вернёт обычный источник.

Калибровку можно проводить прямо на «живой» картинке без перезапуска.

HTTP API: `GET /api/state` (состояние FSM), `GET/POST /api/settings`
(граница/поворот), `GET /api/log` (журнал событий), `POST /api/stream`
(переключение на интернет-стрим).

## Лицензия/статус
MVP v1.0 для стендовой отладки. Боевое внедрение — только после согласования
с владельцем светофорного объекта и ГИБДД (см. техдокументацию, глава 7).

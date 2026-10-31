---
title: "Умный светофор — Руководство пользователя и техническое описание"
subtitle: "MVP v1.0 (edge, Raspberry Pi 4)"
author: "Инженерный отдел"
lang: ru-RU
---

# Умный светофор: руководство пользователя и техническое описание

## 1. Суть проекта и метрика Person-Throughput

«Умный светофор» — edge-система динамического управления фазами светофорного
объекта на базе Raspberry Pi 4 Model B. Система оптимизирует **пропускную
способность людей**, а не машин — через метрику **Person-Throughput (PT)**.

```
PT_фазы = Σ (count_vehicle_class × occupancy_class) + Σ пешеходов
```

Коэффициенты вместимости (настраиваются в `config.yaml`):

| Класс | Коэффициент | Комментарий |
|---|---|---|
| passenger_car | 1.2 | средняя загрузка |
| bus | 35.0 | маршрутный автобус |
| truck | 1.1 | водитель + пассажир |
| motorcycle | 1.0 | водитель |
| bicycle | 1.0 | велосипедист |
| pedestrian | 1.0 | пешеход |
| emergency_vehicle | 0.0 | спецтранспорт — событие, приоритет |

**Алгоритм:** конечный автомат (FSM) сравнивает скопления обеих фаз и жадным
правилом отдаёт зелёный фазе с максимальным приведённым числом людей.
Экстренный транспорт (класс `emergency_vehicle`) прерывает цикл немедленно.

Для стендового MVP кадр делится двумя зонами фаз A/B вертикальной границей
(`phase_split_fraction`). В боевой версии каждая камера → свой подход 1:1.

---

## 2. Состав пакета

```
smart-traffic-light/
├── README.md                # быстрый старт
├── config.yaml              # настройки
├── detector.py              # CV-модуль (TFLite + OpenCV)
├── decision_engine.py       # FSM и метрика PT
├── gpio_controller.py       # управление реле
├── main.py                  # главный цикл
├── requirements.txt
├── HARDWARE_WIRING.md       # схема подключения
├── USER_GUIDE.md            # этот документ
├── scripts/get_yolov8n_tflite.py
├── models/                  # yolov8n_int8.tflite (уже здесь)
└── test_video.mp4           # тестовое видео (уже здесь, опционально)
```

---

## 3. Часть I. Запуск на НОУТБУКЕ (отладка без железа)

> Здесь нет реле и пинов — GPIO автоматически уйдёт в **MOCK-эмуляцию**,
> но детекция, FSM и отрисовка работают в полном объёме. Это лучший способ
> проверить алгоритм и модель до установки на Pi.

### Шаг 1. Установите Python 3.9+

- **Windows:** установите с python.org (галка «Add to PATH»).
- **Linux (Ubuntu/Debian):** `sudo apt install python3 python3-pip python3-venv`
- **macOS:** `brew install python` или установите с python.org.

### Шаг 2. Откройте терминал и скопируйте проект

```bash
cd /путь/к/вашей/папке
git clone <ваш-репозиторий> smart-traffic-light   # или просто скопируйте папку
cd smart-traffic-light
```

### Шаг 3. Создайте виртуальное окружение и поставьте зависимости

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Установка пакетов:
```bash
pip install --upgrade pip
pip install PyYAML numpy opencv-python-headless
pip install tensorflow          # на ПК проще поставить полный TF (Lite внутри)
# (медленно? см. раздел 5 «Депенденсы по платформам»)
```

Проверка:
```bash
python -c "import cv2, numpy, yaml; print('OK', cv2.__version__)"
```

### Шаг 4. Проверьте наличие модели

В папке уже должна лежать `models/yolov8n_int8.tflite`. Если нет:
```bash
python scripts/get_yolov8n_tflite.py        # скачает готовую модель
# или:  python scripts/get_yolov8n_tflite.py --weights yolov8n.pt
```
> Без модели программа **не падает** — включится синтетический MockDetector
> и пойдёт демо-режим (помечается в логе как «demo без ИИ»).

### Шаг 5. Проверьте/положите тестовое видео

В папке уже есть `test_video.mp4` (4-секундная запись с автобусом и пешеходами).
Можете заменить своим роликом с перекрёстка.

Убедитесь, что в `config.yaml`:
```yaml
mode: test
video_path: "./test_video.mp4"
video_loop: true
show_video: true
```

### Шаг 6. Запуск

```bash
python main.py --config config.yaml
```

**Что увидите:** окно с детекциями (bounding boxes), панель с метриками
`PT-A / PT-B`, текущим состоянием FSM (`GREEN_A` → `AMBER_A` → ...) и
вертикальной линией границы фаз. В консоли — лог событий переключения фаз.

**Выход:** `ESC` в окне или `Ctrl+C` в терминале.

> **Headless (сервер без монитора):** в `config.yaml` поставьте
> `show_video: false` — метрики увидите в логе: `tail -f logs/app.log`.

---

## 4. Часть II. Запуск на RASPBERRY PI (полный цикл с реле)

### Шаг 0. Подготовка SD-карты и ОС

1. Запишите **Raspberry Pi OS (64-bit) Lite** (без рабочего стола) на SD-карту
   (официальный Raspberry Pi Imager).
2. В настройках Imager включите SSH, задайте имя пользователя/пароль, WiFi (если нужно).
3. Вставьте SD, включите Pi, найдите IP и подключитесь:
   ```bash
   ssh pi@<ip-pi>
   ```

### Шаг 1. Обновление системы и базовые пакеты

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl
```

### Шаг 2. Установка опенсорсных библиотек (быстрее через apt, чем pip)

```bash
sudo apt install -y python3-opencv libatlas-base-dev
```

### Шаг 3. Установка pigpio (демон GPIO) и запуск

```bash
sudo apt install -y pigpio python3-pigpio
sudo systemctl enable --now pigpiod
pigs t            # проверка: должен вывести "1000"
```

> Ролик-демонов: если pigpiod не запущен, программа автоматически использует
> RPi.GPIO (без демона), а при отсутствии железа — MOCK. Но для точности
> включения реле рекомендуем pigpio.

### Шаг 4. Скопируйте проект на Pi

```bash
cd ~
git clone <ваш-репозиторий> smart-traffic-light
cd smart-traffic-light
# либо rsync/scp из ноутбука:
# scp -r smart-traffic-light pi@<ip>:/home/pi/
```

### Шаг 5. Виртуальное окружение и зависимости для RPi

```bash
cd ~/smart-traffic-light
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install PyYAML numpy
pip install opencv-python-headless     # если не поставили через apt
pip install tflite-runtime             # лёгкий рантайм для ARM
pip install pigpio
# (requirements.txt тоже подойдёт, но tflite-runtime предпочтительнее полного TF)
```

Проверка:
```bash
python -c "import cv2, numpy, yaml; print('OK', cv2.__version__)"
python -c "import tflite_runtime.interpreter; print('TFLite OK')"
```

### Шаг 6. Проверьте модель и тестовое видео

Как на ноутбуке: `models/yolov8n_int8.tflite` уже в репозитории.
При отсутствии — `python scripts/get_yolov8n_tflite.py`.

### Шаг 7. Тест на видео без камеры

```bash
python main.py --config config.yaml
```
GPIO перейдёт в **REAL-режим** (pigpio или RPi.GPIO), но без подключённых
светодиодов ты увидишь только лог «применяю сигнал». Чтобы проверить реальное
железо, подключите схему (см. `HARDWARE_WIRING.md`) и используйте
быстрый тест из раздела 5 этого руководства.

### Шаг 8. Переключаем в БОЕВОЙ режим (реальная камера)

Отредактируйте `config.yaml`:
```yaml
mode: live
camera_source: 0        # USB-камера /dev/video0
show_video: false       # на реальном объекте окно не нужно
```
Запуск:
```bash
python main.py --config config.yaml
```

**USB-камера** проще всего. **CSI-камера** (пи-кабель) — см. раздел 6.

### Шаг 9. Автозапуск при загрузке (systemd)

Создайте `/etc/systemd/system/smarttraffic.service`:
```ini
[Unit]
Description=Smart Traffic Light MVP
After=network-online.target pigpiod.service
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/smart-traffic-light
ExecStart=/home/pi/smart-traffic-light/.venv/bin/python main.py --config config.yaml --no-window
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now smarttraffic
# логи:
journalctl -u smarttraffic -f
```

---

## 5. Депенденсы по платформам (коротко)

| Платформа | Быстрый вариант | Рекомендуемый |
|---|---|---|
| **Windows ноутбук** | `pip install tensorflow opencv-python-headless PyYAML numpy` | Full TF |
| **Linux/macOS ноут** | то же | Full TF |
| **Raspberry Pi (ARM)** | `pip install tflite-runtime opencv-python-headless PyYAML numpy pigpio` | tflite-runtime |

`requirements.txt` в пакете написан под **RPi** (tflite-runtime). На ноутбуке
можете просто закомментировать строку `tflite-runtime` и добавить `tensorflow`.

---

## 6. Камеры: CSI и USB

- **USB:** `camera_source: 0` — самый простой вариант.
- **CSI (пи-шлейф):** на Raspberry Pi OS (Bookworm) нужен libcamera:
  ```bash
  sudo apt install -y libcamera-tools python3-libcamera python3-picamera2
  ```
  В `config.yaml`:
  ```yaml
  camera_source: "libcamerasrc ! videoconvert ! video/x-raw,width=1280,height=720,framerate=30/1 ! appsink"
  ```

---

## 7. Модель ИИ: что за вероятность и как свою поставить

Пакет работает с **YOLOv8n в формате TFLite (640×640)**. Модель в репозитории
`models/yolov8n_int8.tflite` обучена на COCO (80 классов) — детектор
мапит их в 7 доменных классов (легковой, автобус, грузовик, мото, вело,
пешеход). Для спецтранспорта (`emergency_vehicle`) со стандартной COCO-моделью
класс не появляется — его надо дообучить (пункт «обучение» в техдокументации).

В Types: `scripts/get_yolov8n_tflite.py` поддерживает и конвертацию из
`.pt`-весов ultralytics: `--weights yolov8n.pt --calib-video sample.mp4`.

---

## 8. Быстрый тест реле/светодиодов (отдельно от программы)

```bash
cd smart-traffic-light
python - <<'PY'
import yaml, time
from gpio_controller import RelayController
cfg = yaml.safe_load(open("config.yaml"))
g = RelayController(cfg)
print("Backend:", g.backend)
for pair in [({"A":"GREEN","B":"RED"}, 2),
             ({"A":"YELLOW","B":"RED"}, 1),
             ({"A":"RED","B":"GREEN"}, 2)]:
    g.apply_outputs(pair[0]); time.sleep(pair[1])
g.cleanup()
print("OK — сигналы переключаются")
PY
```
Ожидаемый результат: зелёный A → жёлтый A → зелёный B.

---

## 9. Отказоустойчивость (что делает система)

| Сбой | Реакция |
|---|---|
| Модель не найдена | Mock-детектор (демо продолжает) |
| Видео закончилось | Зацикливание (тест-режим) |
| pigpiod не запущен | Авто: RPi.GPIO → MOCK |
| Камера «зависла» | Переоткрытие источника |
| Ctrl+C / ESC | Гашение сигналов, корректный cleanup |
| Ошибка в кадре | Логирование, цикл продолжается |

---

## 10. Устранение неполадок (FAQ)

| Симптом | Причина/решение |
|---|---|
| «Модуль не найден: cv2» | postavte opencv: `pip install opencv-python-headless` |
| «No module named yaml» | `pip install PyYAML` |
| Модель грузится медленно | На RPi используйте `tflite-runtime`, не полный TF |
| Нет окна на сервере | `show_video: false` |
| Светодиоды не загораются | Проверьте `relay_active_low`, питание модуля 5V, джемпер JD-VCC |
| pip не находит tflite-runtime на ПК | На ПК поставьте `tensorflow` |

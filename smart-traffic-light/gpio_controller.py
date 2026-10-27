"""
gpio_controller.py — модуль управления реле «Умного светофора» (MVP v1.0)

Поддержка трёх бэкендов (выбор автоматический, приоритет сверху вниз):
  1. pigpio          — рекомендуется (ОЗУ пигасы, нет джиттера, pigpiod)
  2. RPi.GPIO        — запасной вариант без демона
  3. MockGPIO        — эмуляция без железа (для ПК/отладки вне Raspberry Pi)

Управление светодиодами светофора через релейный модуль (активный LOW):
  реле срабатывает (замыкает контакты COM-NO) при логическом 0 на входе IN.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("gpio")

COLORS = ("RED", "YELLOW", "GREEN")


class RelayController:
    """Контроллер релейного модуля для двух светофорных групп (A и B)."""

    def __init__(self, config):
        gpio_cfg = config.get("gpio", {})
        self.active_low = bool(gpio_cfg.get("relay_active_low", True))

        # Карта пинов: {'A': {'RED': pin, 'YELLOW': pin, 'GREEN': pin}, ...}
        self.pins = {}
        for phase in ("A", "B"):
            pc = phase.lower()
            self.pins[phase] = {
                "RED": int(gpio_cfg.get(f"pin_red_{pc}", -1)),
                "YELLOW": int(gpio_cfg.get(f"pin_yellow_{pc}", -1)),
                "GREEN": int(gpio_cfg.get(f"pin_green_{pc}", -1)),
            }

        self.backend = "none"
        self._pi = None
        self._GPIO = None
        self._mock_pins = {}
        self._init_backend()

    # ------------------------------------------------------------------ helpers
    def _level_to_write(self, on: bool) -> int:
        """Переводит логическое состояние светодиода в уровень на выводе.

        Для реле активного LOW включение = уровень 0 (GND).
        """
        if on:
            return 0 if self.active_low else 1
        return 1 if self.active_low else 0

    def _init_backend(self):
        """Выбирает доступный бэкенд: pigpio -> RPi.GPIO -> MockGPIO."""
        # 1) pigpio
        try:
            import pigpio
            self._pi = pigpio.pi()
            if self._pi is None or not self._pi.connected:
                logger.warning("pigpiod не запущен — попробую RPi.GPIO")
                raise RuntimeError("pigpiod unavailable")
            for phase_pins in self.pins.values():
                for pin in phase_pins.values():
                    if pin < 0:
                        continue
                    self._pi.set_mode(pin, pigpio.OUTPUT)
                    self._pi.write(pin, self._level_to_write(False))
            self.backend = "pigpio"
            logger.info("GPIO-бэкенд: pigpio (pid=%d)", self._pi.get_pid() if hasattr(self._pi, "get_pid") else -1)
            return
        except ImportError:
            logger.info("pigpio не установлен")
        except Exception as exc:  # demo без pigpiod
            logger.warning("pigpio недоступен: %s", exc)

        # 2) RPi.GPIO
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for phase_pins in self.pins.values():
                for pin in phase_pins.values():
                    if pin < 0:
                        continue
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH if self.active_low else GPIO.LOW)
            self._GPIO = GPIO
            self.backend = "rpi"
            logger.info("GPIO-бэкенд: RPi.GPIO")
            return
        except ImportError:
            logger.info("RPi.GPIO не установлен")
        except Exception as exc:
            logger.warning("RPi.GPIO недоступен: %s", exc)

        # 3) MockGPIO (эмуляция, если запущено не на Raspberry Pi)
        self.backend = "mock"
        self._mock_pins = {pin: False for phase_pins in self.pins.values() for pin in phase_pins.values()}
        logger.warning("GPIO-бэкенд: MOCK (эмуляция, реальное железо не задействовано)")

    # ------------------------------------------------------------------ public
    def set_color(self, phase: str, color: str, on: bool):
        """Включить/выключить один сигнал одной фазы."""
        if phase not in self.pins:
            logger.error("Неизвестная фаза: %s", phase)
            return
        if color not in COLORS:
            logger.error("Неизвестный цвет: %s", color)
            return
        pin = self.pins[phase][color]
        level = self._level_to_write(on)
        try:
            if self.backend == "pigpio":
                self._pi.write(pin, level)
            elif self.backend == "rpi":
                self._GPIO.output(pin, self._GPIO.HIGH if level else self._GPIO.LOW)
            elif self.backend == "mock":
                self._mock_pins[pin] = on
            self._log(phase, color, on)
        except Exception as exc:
            logger.error("Ошибка записи вывода %s/%s: %s", phase, color, exc)

    def apply_outputs(self, outputs: dict):
        """Применяет комбинацию светов: {'A': 'GREEN', 'B': 'RED'}.

        Всегда строго один свет на фазу (остальные погашены) — исключает
        «красный+зелёный» одновременно на одной группе.
        """
        for phase in ("A", "B"):
            state = outputs.get(phase, "RED")  # безопасное значение по умолчанию
            if state not in COLORS:
                logger.error("Некорректное состояние %s для фазы %s — ставлю RED", state, phase)
                state = "RED"
            for color in COLORS:
                self.set_color(phase, color, color == state)

    def all_off(self):
        """Аварийное гашение всех сигналов (безопасное состояние)."""
        for phase in ("A", "B"):
            for color in COLORS:
                self.set_color(phase, color, False)

    def _log(self, phase, color, on):
        logger.debug("GPIO[%s] %s -> %s", phase, color, "ON" if on else "OFF")

    def cleanup(self):
        """Корректное освобождение ресурсов."""
        try:
            self.all_off()
        except Exception as exc:
            logger.warning("Не удалось погасить сигналы: %s", exc)
        try:
            if self.backend == "pigpio":
                self._pi.stop()
            elif self.backend == "rpi":
                self._GPIO.cleanup()
        except Exception as exc:
            logger.warning("Ошибка очистки GPIO: %s", exc)
        self.backend = "none"
        logger.info("GPIO освобождён")

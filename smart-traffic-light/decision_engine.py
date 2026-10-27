"""
decision_engine.py — конечный автомат (FSM) «Умного светофора» (MVP v1.0)

Реализует метрику Person-Throughput (PT) из технической документации:
    PT_фазы = Σ (count_class * occupancy_class) + Σ пешеходов

Двухфазный FSM для стендового MVP (фазы A и B):
    GREEN_A -> AMBER_A -> ALL_RED -> GREEN_B -> AMBER_B -> ALL_RED -> ...

Правила переключения:
  1. Экстренный транспорт (emergency_vehicle) — немедленный приоритет.
  2. Переключение на фазу-конкурента, если её PT > текущей * crowd_delta_ratio
     И уже отработано min_green И собралась минимальная очередь (min_waiting).
  3. Принудительный переход по max_green, если на другой стороне есть спрос.
"""
from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger("decision")

# Состояния конечного автомата
S_GREEN_A = "GREEN_A"
S_AMBER_A = "AMBER_A"
S_GREEN_B = "GREEN_B"
S_AMBER_B = "AMBER_B"
S_ALL_RED = "ALL_RED"


class TrafficFSM:
    """Конечный автомат управления фазами светофорного объекта."""

    def __init__(self, config):
        self.config = config
        timings = config.get("timings", {})
        th = config.get("thresholds", {})
        self.min_green = int(timings.get("min_green", 15000))
        self.max_green = int(timings.get("max_green", 60000))
        self.amber = int(timings.get("amber", 3000))
        self.all_red = int(timings.get("all_red_clearance", 2500))
        self.crowd_delta_ratio = float(th.get("crowd_delta_ratio", 1.15))
        self.min_waiting_to_switch = int(th.get("min_waiting_to_switch", 2))
        self.ev_confirm_frames = int(th.get("ev_confirm_frames", 2))
        self.occupancy = config.get("occupancy", {})

        self.state = S_GREEN_A
        self.state_since_ms = 0      # момент входа в текущее состояние (монотонные мс)
        self.current_phase = "A"     # фаза, которой сейчас дано «право на зелёный»
        self.outputs = {"A": "RED", "B": "RED"}
        self._ev_streak = 0          # счётчик кадров подтверждения спецтранспорта

        logger.info(
            "FSM инициализирован: min_green=%d мс, max_green=%d мс, amber=%d мс, "
            "all_red=%d мс, ratio=%.2f",
            self.min_green, self.max_green, self.amber, self.all_red,
            self.crowd_delta_ratio,
        )

    # ------------------------------------------------------------------ helpers
    def _pt(self, counter: Counter) -> float:
        """Person-Throughput: приведение числа объектов к числу людей."""
        return float(sum(
            count * self.occupancy.get(cls, 1.0)
            for cls, count in counter.items()
        ))

    @staticmethod
    def _queue_size(counter: Counter) -> int:
        """Общее число объектов в очереди (для порога min_waiting_to_switch)."""
        return sum(counter.values())

    def _ev_present(self, counter_a: Counter, counter_b: Counter) -> str | None:
        """Возвращает фазу, где присутствует спецтранспорт, либо None."""
        if counter_a.get("emergency_vehicle", 0) > 0:
            return "A"
        if counter_b.get("emergency_vehicle", 0) > 0:
            return "B"
        return None

    # ------------------------------------------------------------------ core
    def update(self, counter_a: Counter, counter_b: Counter, now_ms: int):
        """Единый шаг FSM.

        Аргументы:
            counter_a, counter_b — Counter'ы объектов по доменным классам
                                    для фаз A и B (из detector).
            now_ms                — монотонные миллисекунды (time.monotonic()*1000).

        Возвращает:
            dict с полями: events (list[str]), demand_a, demand_b (float PT).
        """
        # Сглаживание: на случай пропущенных кадров используем последние данные
        self._last_a = counter_a
        self._last_b = counter_b

        demand_a = self._pt(counter_a)
        demand_b = self._pt(counter_b)
        events: list[str] = []

        # ------------------------------------------------- 1) Экстренный транспорт
        ev_phase = self._ev_present(counter_a, counter_b)
        if ev_phase is not None:
            events.append("EMERGENCY_VEHICLE_DETECTED")
            if ev_phase == self.current_phase:
                # уже зелёный нужной стороне — держим (hold)
                self.state_since_ms = now_ms  # «обновляем» таймер, не уходим
                events.append("EV_HOLD")
            else:
                # прерываем текущую фазу через amber -> нужный зелёный
                if self.state == S_GREEN_A and ev_phase == "B" and now_ms - self.state_since_ms >= self.min_green:
                    self._set_state(S_AMBER_A, now_ms, events)
                elif self.state == S_GREEN_B and ev_phase == "A" and now_ms - self.state_since_ms >= self.min_green:
                    self._set_state(S_AMBER_B, now_ms, events)
                elif self.state not in (S_GREEN_A, S_GREEN_B):
                    # во время переходов — просто продолжаем, скоро дойдём
                    pass
        else:
            self._ev_streak = 0

        # ------------------------------------------------- 2) Основной FSM
        if self.state == S_GREEN_A:
            self._tick_green(counter_a, counter_b, demand_a, demand_b, now_ms, events)
        elif self.state == S_AMBER_A:
            if now_ms - self.state_since_ms >= self.amber:
                self._set_state(S_ALL_RED, now_ms, events)
        elif self.state == S_ALL_RED:
            if now_ms - self.state_since_ms >= self.all_red:
                if self.current_phase == "A":
                    self.state = S_GREEN_B
                    self.current_phase = "B"
                else:
                    self.state = S_GREEN_A
                    self.current_phase = "A"
                self.state_since_ms = now_ms
                events.append(self.state)
                logger.info("Смена фазы -> %s", self.state)
        elif self.state == S_GREEN_B:
            self._tick_green(counter_a, counter_b, demand_a, demand_b, now_ms, events)
        elif self.state == S_AMBER_B:
            if now_ms - self.state_since_ms >= self.amber:
                self._set_state(S_ALL_RED, now_ms, events)

        self._update_outputs()
        return {"events": events, "demand_a": demand_a, "demand_b": demand_b}

    # ------------------------------------------------------------------ internal
    def _tick_green(self, counter_a, counter_b, demand_a, demand_b, now_ms, events):
        """Логика решения во время зелёной фазы: продлить или переключить."""
        current_phase = self.current_phase  # "A" или "B"
        elapsed = now_ms - self.state_since_ms

        if current_phase == "A":
            own_demand, other_demand = demand_a, demand_b
            other_counter = counter_b
            amber_state = S_AMBER_A
        else:
            own_demand, other_demand = demand_b, demand_a
            other_counter = counter_a
            amber_state = S_AMBER_B

        # Минимальная длительность уже отработана — можно оценивать переключение
        if elapsed < self.min_green:
            return

        other_queue = self._queue_size(other_counter)
        switch = False

        if elapsed >= self.max_green:
            # принудительный переход, если на другой стороне есть хоть кто-то
            switch = other_queue >= 1
        else:
            # жадное правило: конкурент должен «перевесить» с запасом
            ratio = self.crowd_delta_ratio
            if (other_queue >= self.min_waiting_to_switch
                    and other_demand > own_demand * ratio):
                switch = True

        if switch:
            self._set_state(amber_state, now_ms, events)

    def _set_state(self, new_state, now_ms, events):
        self.state = new_state
        self.state_since_ms = now_ms
        events.append(new_state)
        logger.info("Переход состояния -> %s", new_state)

    def _update_outputs(self):
        """Отображает состояние FSM на световые комбинации обеих фаз."""
        if self.state == S_GREEN_A:
            self.outputs = {"A": "GREEN", "B": "RED"}
        elif self.state == S_AMBER_A:
            self.outputs = {"A": "YELLOW", "B": "RED"}
        elif self.state == S_GREEN_B:
            self.outputs = {"A": "RED", "B": "GREEN"}
        elif self.state == S_AMBER_B:
            self.outputs = {"A": "RED", "B": "YELLOW"}
        else:  # S_ALL_RED
            self.outputs = {"A": "RED", "B": "RED"}

    def reset(self):
        """Возврат в начальное состояние (перезапуск пилота)."""
        self.state = S_GREEN_A
        self.state_since_ms = 0
        self.current_phase = "A"
        self.outputs = {"A": "RED", "B": "RED"}
        logger.info("FSM сброшен в начальное состояние")

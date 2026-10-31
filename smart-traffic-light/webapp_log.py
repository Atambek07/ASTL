"""
webapp_log.py — буфер логов в памяти для журнала событий веб-интерфейса.

Разделяется между потоком обработки и HTTP-запросами через синглтон и
ограниченную очередь (последние N сообщений).
"""
from __future__ import annotations

import logging
import threading
from collections import deque


class MemoryLogHandler(logging.Handler):
    """Собирает последние сообщения в кольцевой буфер для отображения в интерфейсе."""

    _instance: "MemoryLogHandler | None" = None
    _lock = threading.Lock()

    def __init__(self, capacity: int = 200):
        super().__init__(level=logging.INFO)
        self.buffer: deque[str] = deque(maxlen=capacity)

    @classmethod
    def get_instance(cls, capacity: int = 200) -> "MemoryLogHandler":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(capacity)
            return cls._instance

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        with self._lock:
            self.buffer.append(msg)

    def format(self, record: logging.LogRecord) -> str:
        return f"{self.formatter.format(record) if self.formatter else record.getMessage()}"

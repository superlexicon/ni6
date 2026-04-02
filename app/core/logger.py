import logging
import threading
import json
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque


class Logger:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._setup_logger()
        self._memory_logs = deque(maxlen=1000)
        self._memory_lock = threading.Lock()

    def _setup_logger(self):
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG for signature verification debugging

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def _add_to_memory(
        self, level: str, message: str, context: Optional[Dict[str, Any]] = None
    ):
        with self._memory_lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "context": context or {},
            }
            self._memory_logs.append(log_entry)

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        formatted_msg = self._format_message(message, context)
        self.logger.debug(formatted_msg)
        self._add_to_memory("DEBUG", message, context)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        formatted_msg = self._format_message(message, context)
        self.logger.info(formatted_msg)
        self._add_to_memory("INFO", message, context)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        formatted_msg = self._format_message(message, context)
        self.logger.warning(formatted_msg)
        self._add_to_memory("WARNING", message, context)

    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        formatted_msg = self._format_message(message, context)
        self.logger.error(formatted_msg)
        self._add_to_memory("ERROR", message, context)

    def critical(self, message: str, context: Optional[Dict[str, Any]] = None):

        formatted_msg = self._format_message(message, context)
        self.logger.critical(formatted_msg)
        self._add_to_memory("CRITICAL", message, context)

    def _format_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        if context:
            return f"{message} | Context: {json.dumps(context, default=str)}"
        return message

    def set_level(self, level: str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
            for handler in self.logger.handlers:
                handler.setLevel(level_map[level.upper()])
        else:
            raise ValueError(f"Invalid log level: {level}")

    def add_file_handler(self, filename: str, level: str = "INFO"):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def disable_console(self):
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                self.logger.removeHandler(handler)

    def get_memory_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._memory_lock:
            if level:
                return [
                    log for log in self._memory_logs if log["level"] == level.upper()
                ]
            return list(self._memory_logs)

    def clear_memory_logs(self):
        with self._memory_lock:
            self._memory_logs.clear()

    def export_logs(self) -> str:
        return json.dumps(self.get_memory_logs(), indent=2)

    def set_max_memory_logs(self, max_logs: int):
        with self._memory_lock:
            new_deque = deque(self._memory_logs, maxlen=max_logs)
            self._memory_logs = new_deque


def get_logger() -> Logger:
    return Logger()

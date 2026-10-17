import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

class Logger:
    def __init__(
        self,
        name: str = "hidden_thoughts",
        log_level: int = logging.INFO,
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
        use_rich: bool = True,
        log_format: Optional[str] = None,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []
        self.logger.propagate = (
            False
        )

        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        if log_to_console:
            if use_rich:
                console = Console()
                console_handler = RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    markup=True,
                    show_time=True,
                    show_path=True,
                )
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = logging.Formatter(log_format)
                console_handler.setFormatter(console_formatter)

            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)

        if log_to_file and log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir_path / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Logging to file: {log_file}")

    @staticmethod
    def from_config(config: dict) -> "Logger":
        return Logger(
            name=config.get("name", "hidden_thoughts"),
            log_level=config.get("log_level", logging.INFO),
            log_dir=config.get("log_dir"),
            log_to_file=config.get("log_to_file", True),
            log_to_console=config.get("log_to_console", True),
            use_rich=config.get("use_rich", True),
            log_format=config.get("log_format"),
        )

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs):
        self.logger.exception(message, **kwargs)

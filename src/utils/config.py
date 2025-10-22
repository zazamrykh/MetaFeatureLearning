"""
Module for working with project configuration.
"""
import configparser
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from functools import lru_cache
from pathlib import Path

# Загружаем .env
load_dotenv()


class EnvInterpolation(configparser.ExtendedInterpolation):
    """
    Custom interpolation that first tries to get value from environment variables.
    """

    def before_get(self, parser, section, option, value, defaults):
        # Если значение в формате ${VAR}, ищем его в окружении
        import re

        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = pattern.findall(value)
        for match in matches:
            env_val = os.getenv(match)
            if env_val is not None:
                value = value.replace(f"${{{match}}}", env_val)
        return super().before_get(parser, section, option, value, defaults)


class Config:
    """
    Class for working with project configuration.
    """

    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser(interpolation=EnvInterpolation())
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.config.read(config_path)

    def get(self, section: str, option: str, fallback: Any = None) -> Any:
        try:
            return self.config.get(section, option, fallback=fallback)
        except Exception:
            return fallback

    def getint(self, section: str, option: str, fallback: Optional[int] = None) -> int:
        val = self.get(section, option, fallback)
        return int(val) if val is not None else fallback

    def getfloat(self, section: str, option: str, fallback: Optional[float] = None) -> float:
        val = self.get(section, option, fallback)
        return float(val) if val is not None else fallback

    def getboolean(self, section: str, option: str, fallback: Optional[bool] = None) -> bool:
        val = self.get(section, option, fallback)
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")

    def getlist(self, section: str, option: str, fallback: Optional[List] = None, delimiter: str = ",") -> List:
        val = self.get(section, option, fallback)
        if val is None:
            return fallback if fallback is not None else []
        return [item.strip() for item in str(val).split(delimiter)]

    def get_all_section(self, section: str) -> Dict[str, str]:
        if not self.config.has_section(section):
            return {}
        return {k: self.get(section, k) for k in self.config[section]}


def _normalize(path: str) -> str:
    return str(Path(path).expanduser().resolve())


@lru_cache(maxsize=None)
def get_config(config_path: str = "config.ini") -> Config:
    """Singleton per normalized file path."""
    return Config(config_path=_normalize(config_path))


def reset_config_cache() -> None:
    """Очистить синглтон(ы) для тестов/релоуда."""
    get_config.cache_clear()

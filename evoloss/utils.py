import logging
import os
import random
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch


def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Безопасное деление тензоров: a / b с защитой от нуля.
    Если |b| < eps, заменяет знаменатель на eps.
    """
    denom = torch.where(torch.abs(b) < eps, torch.full_like(b, eps), b)
    return a / denom


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Стабильный логарифм: log(x + eps), отсечение отрицательных значений."""
    x_clamped = torch.where(x < eps, torch.full_like(x, eps), x)
    return torch.log(x_clamped)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Стабильный корень: sqrt(max(x, eps))."""
    x_clamped = torch.where(x < eps, torch.full_like(x, eps), x)
    return torch.sqrt(x_clamped)


def set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(
    name: str = "evoloss",
    log_file: str = os.path.join("results", "run_log.log"),
    stream: bool = False,
    multiprocess_safe: bool = True,
) -> logging.Logger:
    """Создает и настраивает логгер.

    При multiprocess_safe=True дочерние процессы пишут в отдельные файлы с PID-суффиксом
    (например, `run_log_1234.log`), что исключает гонки записи при параллельной оценке.
    Повторные вызовы корректно обновляют файловый хендлер, если путь изменился.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(processName)s:%(process)d] %(levelname)s %(name)s: %(message)s"
    )

    # Определяем целевой файл с учетом PID для дочерних процессов
    target_log_file = log_file
    if multiprocess_safe and mp.current_process().name != "MainProcess":
        base, ext = os.path.splitext(log_file)
        target_log_file = f"{base}_{os.getpid()}{ext}"

    # Проверяем существующие хендлеры и обновляем файловый при смене пути
    existing_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    existing_stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]

    need_new_file_handler = True
    if existing_file_handlers:
        try:
            current_path = os.path.abspath(existing_file_handlers[0].baseFilename)
            need_new_file_handler = current_path != os.path.abspath(target_log_file)
        except Exception:
            need_new_file_handler = True

    if need_new_file_handler:
        # Удаляем старые файловые хендлеры, если есть
        for h in existing_file_handlers:
            try:
                logger.removeHandler(h)
                h.close()
            except Exception:
                pass
        fh = logging.FileHandler(target_log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if stream and not existing_stream_handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def tensor_constant(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)
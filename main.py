import argparse
import os
from typing import Any, Dict

import torch
import yaml

from evoloss.symbolic_tree import Node
from evoloss.utils import get_logger, set_seed
from evoloss.evolution import Evolution


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def demo_compile_and_run(device: str = "cpu") -> None:
    """
    Демонстрация Фазы 1: компиляция дерева в loss и вычисление.
    Строим дерево, эквивалентное MSE: mean((y_pred - y_true) * (y_pred - y_true))
    """
    # Пример сложного выражения с новыми операторами:
    # if_then_else(y_pred > y_true, safe_log(y_pred + epsilon), (y_true - y_pred)^2)
    cond = Node.operator("gt", Node.terminal("y_pred"), Node.terminal("y_true"))
    right_diff = Node.operator("-", Node.terminal("y_true"), Node.terminal("y_pred"))
    right_sq = Node.operator("*", right_diff, right_diff)
    left_log_arg = Node.operator("+", Node.terminal("y_pred"), Node.terminal("epsilon"))
    left_log = Node.operator("safe_log", left_log_arg)
    expr = Node.operator("if_then_else", cond, left_log, right_sq)
    loss_fn = expr.compile(reduction="mean")

    y_pred = torch.randn(32, 10, device=device)
    y_true = torch.randn(32, 10, device=device)

    loss_val = loss_fn(y_pred, y_true)
    print("Demo expression:", expr.to_string())
    print("Loss value:", float(loss_val.item()))


def main():
    parser = argparse.ArgumentParser(description="EvoLoss: Evolutionary discovery of loss functions")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "mnist_cnn_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger()
    # Установка seed: если указан конкретный int, применяем; иначе эволюция сама установит случайный и сохранит
    seed_val = cfg.get("seed")
    if isinstance(seed_val, (int, float)):
        set_seed(int(seed_val))

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, switching to CPU")
        device = "cpu"

    logger.info("Starting evolutionary search for loss functions (Phase 3)")
    evo = Evolution(cfg)
    evo.run()


if __name__ == "__main__":
    main()
import os
import sys
import traceback

import torch

from evoloss.utils import safe_div, safe_log, safe_sqrt
from evoloss.symbolic_tree import Node
from evoloss.evaluation import evaluate_fitness, FitnessResult
from evoloss.evolution import Evolution


def test_utils():
    a = torch.tensor([1.0, 2.0, -3.0])
    b = torch.tensor([0.0, 1e-9, 2.0])
    div = safe_div(a, b)
    assert torch.isfinite(div).all(), "safe_div produced non-finite"
    x = torch.tensor([-1.0, 0.0, 1.0])
    lg = safe_log(x)
    assert torch.isfinite(lg).all(), "safe_log produced non-finite"
    sq = safe_sqrt(torch.tensor([-1.0, 0.0, 4.0]))
    assert torch.isfinite(sq).all(), "safe_sqrt produced non-finite"


def test_symbolic_tree():
    # MSE-like: (y_pred - y_true)^2
    diff = Node.operator("-", Node.terminal("y_pred"), Node.terminal("y_true"))
    expr = Node.operator("*", diff, diff)
    loss_fn = expr.compile(reduction="mean")
    y_pred = torch.randn(8, 3)
    y_true = torch.randn(8, 3)
    val = loss_fn(y_pred, y_true)
    assert val.ndim == 0, "Loss should be scalar after mean reduction"
    assert torch.isfinite(val), "Loss value should be finite"


def test_evaluation():
    diff = Node.operator("-", Node.terminal("y_pred"), Node.terminal("y_true"))
    expr = Node.operator("*", diff, diff)
    cfg = {
        "dataset": "synthetic",
        "batch_size": 32,
        "epochs": 1,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 123,
        "w1": 1.0,
        "w2": 0.01,
        "w3": 0.1,
        "w4": 0.1,
        "grad_threshold": 100.0,
    }
    res = evaluate_fitness(expr, cfg)
    assert isinstance(res, FitnessResult), "evaluate_fitness should return FitnessResult"
    assert 0.0 <= res.accuracy <= 1.0, "Accuracy out of bounds"
    assert res.complexity >= 1, "Complexity should be >= 1"
    assert isinstance(res.grad_max_norm, float), "grad_max_norm should be float"
    assert res.epoch_to_95 >= 1, "epoch_to_95 should be >= 1 with >=1 epoch"


def test_evolution_small_run():
    cfg = {
        "dataset": "synthetic",
        "batch_size": 32,
        "epochs": 1,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 123,
        "population_size": 4,
        "tournament_size": 2,
        "max_tree_depth": 3,
        "mutation_rate": 0.5,
        "crossover_rate": 0.7,
        "elitism": 1,
        "generations": 1,
        "checkpoint_interval": 1,
        "parallel_eval": False,
        "save_dir": os.path.join("results"),
        "w1": 1.0,
        "w2": 0.01,
        "w3": 0.1,
        "w4": 0.1,
        "grad_threshold": 100.0,
    }
    evo = Evolution(cfg)
    evo.run()
    # Artifacts
    assert os.path.exists(os.path.join("results", "stats.csv")), "stats.csv not created"
    # There should be at least one data row beyond the header
    with open(os.path.join("results", "stats.csv"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 2, "stats.csv should contain header and at least one row"
    assert os.path.exists(os.path.join("results", "best_functions.txt")), "best_functions.txt not exists"
    assert os.path.exists(os.path.join("results", "checkpoints", "gen_1.pkl")), "checkpoint missing"
    assert os.path.exists(os.path.join("results", "plots", "gen_1", "best_loss.png")), "best_loss.png missing"
    assert os.path.exists(os.path.join("results", "plots", "gen_1", "best_dloss.png")), "best_dloss.png missing"
    assert os.path.exists(os.path.join("results", "plots", "gen_1", "best_tree.png")), "best_tree.png missing"


def main():
    tests = [
        ("utils", test_utils),
        ("symbolic_tree", test_symbolic_tree),
        ("evaluation", test_evaluation),
        ("evolution_small_run", test_evolution_small_run),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    else:
        print("All smoke tests passed.")


if __name__ == "__main__":
    main()
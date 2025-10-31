from evoloss.symbolic_tree import Node
from evoloss.evaluation import dispatch_evaluate, FitnessResult


def test_proxy_evaluation_returns_fitness():
    # Simple MSE-like node
    diff = Node.operator("-", Node.terminal("y_pred"), Node.terminal("y_true"))
    expr = Node.operator("*", diff, diff)
    cfg = {
        "evaluation": {"mode": "proxy"},
        "weights": {"accuracy": 1.0, "complexity": 0.01, "gradient": 0.1, "speed": 0.5},
        "seed": 123,
    }
    res = dispatch_evaluate(expr, cfg)
    assert isinstance(res, FitnessResult)
    assert res.complexity >= 1
    assert isinstance(res.grad_max_norm, float)
    assert res.epoch_to_95 == 1
from evoloss.symbolic_tree import Node


def test_compile_simple_add():
    # L = y_pred + 1
    tree = Node(op="+", children=[Node(value="y_pred"), Node(value=1.0)])
    loss = tree.compile(reduction="mean")
    import torch

    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([0, 0])
    val = loss(y_pred, y_true)
    assert torch.isfinite(val), "Compiled loss should produce finite value"
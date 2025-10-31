import os

from evoloss.symbolic_tree import Node
from evoloss.analytics import plot_loss_and_derivative


def test_plot_loss_and_derivative_creates_files(tmp_path):
    # Use a simple expression
    diff = Node.operator("-", Node.terminal("y_pred"), Node.terminal("y_true"))
    expr = Node.operator("*", diff, diff)
    save_dir = tmp_path / "plots_test"
    plot_loss_and_derivative(expr, str(save_dir), y_true_value=1.0)
    assert os.path.exists(os.path.join(str(save_dir), "best_loss.png"))
    assert os.path.exists(os.path.join(str(save_dir), "best_dloss.png"))
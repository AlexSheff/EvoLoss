import os
from evoloss.symbolic_tree import Node
from evoloss.analytics import plot_loss_and_derivative


def test_plot_outputs(tmp_path):
    # Simple function: sigmoid(y_pred)
    tree = Node(op="sigmoid", children=[Node(value="y_pred")])
    out_dir = tmp_path / "plots"
    plot_loss_and_derivative(tree, str(out_dir), y_true_value=1.0)
    assert os.path.exists(out_dir / "best_loss.png")
    assert os.path.exists(out_dir / "best_dloss.png")
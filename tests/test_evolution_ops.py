from evoloss.symbolic_tree import Node
from evoloss.evolution import set_by_path, deep_copy


def test_set_by_path_operator_value_none():
    # (y_pred + 1) * 2
    base = Node(op="*", children=[
        Node(op="+", children=[Node(value="y_pred"), Node(value=1.0)]),
        Node(value=2.0),
    ])
    # Replace right operand in addition with 3.0
    new_sub = Node(value=3.0)
    replaced = set_by_path(base, (0, 1), new_sub)
    # Ensure operator nodes do not carry a value
    assert replaced.op == "*" and replaced.value is None
    assert replaced.children[0].op == "+" and replaced.children[0].value is None
from __future__ import annotations

import os
from typing import Any

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx

from .symbolic_tree import Node
from .utils import get_logger


def node_to_sympy(node: Node) -> sp.Expr:
    yp, yt = sp.symbols("y_pred y_true")
    eps = sp.Symbol("epsilon", positive=True)

    def _convert(n: Node) -> sp.Expr:
        if n.is_terminal():
            if isinstance(n.value, str):
                if n.value == "y_pred":
                    return yp
                if n.value == "y_true":
                    return yt
                if n.value == "epsilon":
                    return eps
                if n.value == "one":
                    return sp.Integer(1)
                if n.value == "mean_y_pred":
                    return sp.Symbol("mean_y_pred")
                if n.value == "std_y_pred":
                    return sp.Symbol("std_y_pred")
                raise ValueError(f"Unsupported terminal for sympy: {n.value}")
            else:
                return sp.Float(float(n.value))
        op = n.op
        cs = n.children or []
        if op == "+":
            return _convert(cs[0]) + _convert(cs[1])
        if op == "-":
            return _convert(cs[0]) - _convert(cs[1])
        if op == "*":
            return _convert(cs[0]) * _convert(cs[1])
        if op == "safe_div":
            return _convert(cs[0]) / (_convert(cs[1]) + eps)
        if op == "relu":
            return sp.Max(_convert(cs[0]), 0)
        if op == "leaky_relu":
            x = _convert(cs[0])
            return sp.Piecewise((x, x >= 0), (sp.Float(0.01) * x, True))
        if op == "sigmoid":
            x = _convert(cs[0])
            return 1 / (1 + sp.exp(-x))
        if op == "tanh":
            return sp.tanh(_convert(cs[0]))
        if op == "safe_log":
            x = _convert(cs[0])
            # Используем Max(x, eps), чтобы избежать log отрицательных значений
            return sp.log(sp.Max(x, eps))
        if op == "safe_sqrt":
            return sp.sqrt(_convert(cs[0]) + eps)
        if op == "sum_classes":
            # Векторная операция суммирования по классам.
            # Для скалярного анализа и построения графиков трактуем как тождественное преобразование.
            return _convert(cs[0])
        if op == "log_softmax":
            # Для целей визуализации аппроксимируем как log(sigmoid(x)).
            # Это неэквивалентно многоклассовому log_softmax, но позволяет избежать падений в SymPy.
            x = _convert(cs[0])
            return sp.log(1 / (1 + sp.exp(-x)))
        if op in {"lt", "gt", "le", "ge", "eq", "ne"}:
            a = _convert(cs[0])
            b = _convert(cs[1])
            if op == "lt":
                return sp.Lt(a, b)
            if op == "gt":
                return sp.Gt(a, b)
            if op == "le":
                return sp.Le(a, b)
            if op == "ge":
                return sp.Ge(a, b)
            if op == "eq":
                return sp.Eq(a, b)
            if op == "ne":
                return sp.Ne(a, b)
        if op == "if_then_else":
            cond = _convert(cs[0])
            t = _convert(cs[1])
            f = _convert(cs[2])
            return sp.Piecewise((t, cond), (f, True))
        raise ValueError(f"Unsupported operator for sympy: {op}")

    return _convert(node)


def simplify_node_expr(node: Node) -> sp.Expr:
    expr = node_to_sympy(node)
    return sp.simplify(expr)


def plot_loss_and_derivative(node: Node, save_dir: str, y_true_value: float = 1.0) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # Логгер для предупреждений визуализации
    results_dir = os.path.abspath(os.path.join(save_dir, os.pardir))
    logger = get_logger(log_file=os.path.join(results_dir, "run_log.log"), stream=False)
    yp, yt = sp.symbols("y_pred y_true")
    eps = sp.Symbol("epsilon", positive=True)
    expr = node_to_sympy(node)
    dexpr = sp.diff(expr, yp)

    # Фиксируем y_true=1 и epsilon=1e-7
    subs = {yt: sp.Float(y_true_value), eps: sp.Float(1e-7)}
    try:
        expr_num = sp.lambdify(yp, expr.subs(subs), "numpy")
    except (TypeError, ValueError) as e:
        logger.warning(f"lambdify(expr) failed: {e}")
        expr_num = lambda x: np.zeros_like(x)
    try:
        dexpr_num = sp.lambdify(yp, dexpr.subs(subs), "numpy")
    except (TypeError, ValueError) as e:
        logger.warning(f"lambdify(dexpr) failed: {e}")
        dexpr_num = lambda x: np.zeros_like(x)

    xs = np.linspace(-5.0, 5.0, 400)
    try:
        ys = np.array(expr_num(xs), dtype=float)
        # Проверка размерности и приведение к одномерному массиву
        if hasattr(ys, 'shape'):
            if len(ys.shape) == 0:  # Скаляр
                ys = np.full_like(xs, ys)
            elif len(ys.shape) > 1 or ys.shape[0] == 1:
                ys = ys.flatten()
                if ys.size == 1:  # Одиночное значение
                    ys = np.full_like(xs, ys[0])
        if ys.size != xs.size:
            ys = np.zeros_like(xs)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.warning(f"Error in expr_num: {e}")
        ys = np.zeros_like(xs)
    try:
        dys = np.array(dexpr_num(xs), dtype=float)
        # Проверка размерности и приведение к одномерному массиву
        if hasattr(dys, 'shape'):
            if len(dys.shape) == 0:  # Скаляр
                dys = np.full_like(xs, dys)
            elif len(dys.shape) > 1 or dys.shape[0] == 1:
                dys = dys.flatten()
                if dys.size == 1:  # Одиночное значение
                    dys = np.full_like(xs, dys[0])
        if dys.size != xs.size:
            dys = np.zeros_like(xs)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.warning(f"Error in dexpr_num: {e}")
        dys = np.zeros_like(xs)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.title("L(y_pred) at y_true=1")
    plt.xlabel("y_pred")
    plt.ylabel("L")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_loss.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, dys)
    plt.title("dL/dy_pred at y_true=1")
    plt.xlabel("y_pred")
    plt.ylabel("dL/dy_pred")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_dloss.png"))
    plt.close()


def plot_expression_tree(node: Node, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    G = nx.DiGraph()

    def _label(n: Node) -> str:
        return n.op if n.op is not None else (str(n.value))

    def _add(n: Node, parent_id: Any | None = None, next_id: list[int] = [0]):
        nid = next_id[0]
        next_id[0] += 1
        G.add_node(nid, label=_label(n))
        if parent_id is not None:
            G.add_edge(parent_id, nid)
        for c in (n.children or []):
            _add(c, nid, next_id)

    _add(node)
    # Укладка графа: пробуем graphviz (если доступен pygraphviz), иначе spring layout
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)
    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, node_color="#e0f3ff", font_size=8, arrows=True)
    plt.title("Expression Tree")
    # Избегаем предупреждения tight_layout: сохраняем с bbox_inches='tight'
    plt.savefig(os.path.join(save_dir, "best_tree.png"), bbox_inches='tight')
    plt.close()
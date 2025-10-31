from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from .utils import safe_div, safe_log, safe_sqrt


def _sum_classes_op(x: torch.Tensor) -> torch.Tensor:
    """
    Суммирование по классовой оси с устойчивостью к различной размерности входа.
    - Если вход имеет форму [batch, num_classes], возвращаем [batch, 1]
    - Если вход имеет форму [batch], превращаем в [batch, 1] (как будто сумма по одной оси)
    - Иначе (скаляр или другое), возвращаем как есть
    """
    if x.dim() >= 2:
        return torch.sum(x, dim=1, keepdim=True)
    if x.dim() == 1:
        return x.unsqueeze(1)
    return x


# Регистрация операторов: имя -> (арность, функция)
class OpSpec:
    def __init__(self, arity: int, func: Callable[..., torch.Tensor]):
        self.arity = arity
        self.func = func


OP_REGISTRY: Dict[str, OpSpec] = {
    # Бинарные арифметические
    "+": OpSpec(2, lambda a, b: a + b),
    "-": OpSpec(2, lambda a, b: a - b),
    "*": OpSpec(2, lambda a, b: a * b),
    "safe_div": OpSpec(2, lambda a, b: safe_div(a, b)),
    # Сравнения (возвращают bool-тензоры)
    "lt": OpSpec(2, lambda a, b: a < b),
    "gt": OpSpec(2, lambda a, b: a > b),
    "le": OpSpec(2, lambda a, b: a <= b),
    "ge": OpSpec(2, lambda a, b: a >= b),
    "eq": OpSpec(2, lambda a, b: a == b),
    "ne": OpSpec(2, lambda a, b: a != b),
    # Унарные нелинейности
    "relu": OpSpec(1, lambda x: torch.relu(x)),
    "sigmoid": OpSpec(1, lambda x: torch.sigmoid(x)),
    "tanh": OpSpec(1, lambda x: torch.tanh(x)),
    "leaky_relu": OpSpec(1, lambda x: torch.where(x >= 0, x, 0.01 * x)),
    "exp": OpSpec(1, lambda x: torch.exp(x)),
    "log": OpSpec(1, lambda x: safe_log(x)),
    "sqrt": OpSpec(1, lambda x: safe_sqrt(x)),
    "sin": OpSpec(1, lambda x: torch.sin(x)),
    "abs": OpSpec(1, lambda x: torch.abs(x)),
    "log_softmax": OpSpec(1, lambda x: F.log_softmax(x, dim=1)),
    # Стабильные функции
    "safe_log": OpSpec(1, lambda x: safe_log(x)),
    "safe_sqrt": OpSpec(1, lambda x: safe_sqrt(x)),
    # Суммирование по классовой оси. Возвращаем размерность с keepdim=True,
    # и обрабатываем одномерные входы, чтобы избежать ошибок размерности.
    "sum_classes": OpSpec(1, _sum_classes_op),
    # Тернарная логика
    "if_then_else": OpSpec(3, lambda cond, t, f: torch.where(cond, t, f)),
    # Тернарный clip
    "clip": OpSpec(3, lambda x, lo, hi: torch.minimum(torch.maximum(x, lo), hi)),
}


TerminalValue = Union[str, float]


@dataclass
class Node:
    """
    Узел символического дерева.
    - Терминал: value in {"y_true", "y_pred"} или float (константа)
    - Оператор: op in {"+", "-", "*", "safe_div"}, children=[left, right]
    """
    op: Optional[str] = None
    value: Optional[TerminalValue] = None
    children: Optional[List["Node"]] = None

    @staticmethod
    def terminal(value: TerminalValue) -> "Node":
        return Node(op=None, value=value, children=None)

    @staticmethod
    def constant(value: float) -> "Node":
        return Node(op=None, value=float(value), children=None)

    @staticmethod
    def operator(op: str, *children: "Node") -> "Node":
        if op not in OP_REGISTRY:
            raise ValueError(f"Unsupported operator: {op}")
        spec = OP_REGISTRY[op]
        if len(children) != spec.arity:
            raise ValueError(f"Operator '{op}' expects {spec.arity} children, got {len(children)}")
        return Node(op=op, value=None, children=list(children))

    def is_terminal(self) -> bool:
        return self.op is None

    def depth(self) -> int:
        if self.is_terminal():
            return 1
        return 1 + max(child.depth() for child in (self.children or []))

    def to_string(self) -> str:
        if self.is_terminal():
            if isinstance(self.value, float):
                return f"{self.value:.6g}"
            return str(self.value)
        # формат по арности
        children = self.children or []
        spec = OP_REGISTRY.get(self.op or "")
        if spec is None:
            raise ValueError(f"Unknown operator {self.op}")
        if spec.arity == 1:
            return f"{self.op}({children[0].to_string()})"
        elif spec.arity == 2:
            if self.op in {"safe_div"}:
                return f"safe_div({children[0].to_string()}, {children[1].to_string()})"
            return f"({children[0].to_string()} {self.op} {children[1].to_string()})"
        elif spec.arity == 3:
            if self.op == "if_then_else":
                return f"if_then_else({children[0].to_string()}, {children[1].to_string()}, {children[2].to_string()})"
            if self.op == "clip":
                return f"clip({children[0].to_string()}, {children[1].to_string()}, {children[2].to_string()})"
            return f"{self.op}({children[0].to_string()}, {children[1].to_string()}, {children[2].to_string()})"
        else:
            raise ValueError("Unsupported operator arity")

    def _eval(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Вычисляет значение узла в контексте тензоров ctx={"y_true": ..., "y_pred": ...}.
        Возвращает тензор той же формы, что и аргументы, с использованием трансляции.
        """
        if self.is_terminal():
            if isinstance(self.value, str):
                # Специальные терминалы
                if self.value == "y_pred":
                    return ctx["y_pred"]
                if self.value == "y_true":
                    # y_true приводим к типу y_pred заранее в compile
                    return ctx["y_true"]
                if self.value == "epsilon":
                    like = ctx.get("y_pred")
                    return torch.as_tensor(1e-7, dtype=like.dtype, device=like.device)
                if self.value == "one":
                    like = ctx.get("y_pred")
                    return torch.as_tensor(1.0, dtype=like.dtype, device=like.device)
                if self.value == "mean_y_pred":
                    return ctx["y_pred"].mean()
                if self.value == "std_y_pred":
                    return ctx["y_pred"].std(unbiased=False)
                raise KeyError(f"Terminal '{self.value}' not found in context")
            else:
                # Числовая константа
                if "y_pred" in ctx:
                    like = ctx["y_pred"]
                else:
                    like = next(iter(ctx.values()))
                return torch.as_tensor(float(self.value), dtype=like.dtype, device=like.device)

        # операторный узел
        children = self.children or []
        spec = OP_REGISTRY.get(self.op or "")
        if spec is None:
            raise ValueError(f"Unknown operator {self.op}")
        args = [c._eval(ctx) for c in children]
        return spec.func(*args)

    def compile(self, reduction: str = "mean") -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Компилирует дерево в функцию Python, совместимую с PyTorch-градиентами.
        Возвращаемая сигнатура: loss(y_pred, y_true) -> torch.Tensor (скаляр).
        reduction: "mean" | "sum" — агрегация по батчу.
        """
        if reduction not in {"mean", "sum", None}:
            raise ValueError("reduction must be 'mean', 'sum', or None")

        def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # Обеспечим совместимость типов
            if y_true.dtype != y_pred.dtype:
                y_true_cast = y_true.to(dtype=y_pred.dtype)
            else:
                y_true_cast = y_true
            ctx = {"y_pred": y_pred, "y_true": y_true_cast}
            out = self._eval(ctx)
            if reduction == "mean":
                return out.mean()
            elif reduction == "sum":
                return out.sum()
            else:
                # без редукции — возвращаем как есть
                return out

        return loss

    def __repr__(self) -> str:
        return f"Node(op={self.op}, value={self.value}, children={self.children})"

    def size(self) -> int:
        """Количество узлов в дереве (для штрафа сложности)."""
        if self.is_terminal():
            return 1
        return 1 + sum(child.size() for child in (self.children or []))
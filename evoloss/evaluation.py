"""PyTorch integration — multi-objective fitness evaluation.

Includes:
- SimpleCNN for MNIST-like input
- evaluate_fitness(node, cfg): compiles loss, trains a model for several epochs,
  collects metrics: Accuracy, Complexity, Gradient Stability, Convergence Speed,
  and returns an aggregated fitness F.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Protocol
import os
import importlib
import numpy as np
import sympy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms

from .symbolic_tree import Node
from .utils import get_logger, set_seed
from .analytics import node_to_sympy


@dataclass
class FitnessResult:
    score: float
    accuracy: float
    complexity: int
    grad_max_norm: float
    grad_has_nan_inf: bool
    epoch_to_95: int


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _prepare_data(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    ds_cfg = cfg.get("dataset", "mnist")
    if isinstance(ds_cfg, dict):
        dataset_type = str(ds_cfg.get("type", "mnist")).lower()
        batch_size = int(ds_cfg.get("batch_size", cfg.get("batch_size", 64)))
    else:
        dataset_type = str(ds_cfg).lower()
        batch_size = int(cfg.get("batch_size", 64))

    # Пользовательский загрузчик данных: 'loader': 'module:function'
    if isinstance(ds_cfg, dict) and ds_cfg.get("loader"):
        loader_spec = str(ds_cfg["loader"])  # формат module:function
        try:
            module_name, func_name = loader_spec.split(":", 1)
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            result = func(cfg)
            if isinstance(result, tuple) and len(result) == 2:
                return result  # (train_loader, val_loader)
        except Exception:
            pass  # откатимся на стандартные варианты ниже

    if dataset_type == "mnist":
        data_dir = cfg.get("data_dir", "./data")
        transform = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        val_size = int(0.2 * len(ds))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        # Синтетические данные для быстрых тестов
        n_train = 512
        n_val = 128
        if isinstance(ds_cfg, dict):
            samples = int(ds_cfg.get("samples", 4096))
            n_train = max(256, int(0.8 * samples))
            n_val = max(64, samples - n_train)
        # Для CNN используем 1 канал, 28x28, 10 классов
        x_train = torch.randn(n_train, 1, 28, 28)
        y_train = torch.randint(0, 10, (n_train,))
        x_val = torch.randn(n_val, 1, 28, 28)
        y_val = torch.randint(0, 10, (n_val,))
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate_fitness(node: Node, cfg: Dict[str, Any]) -> FitnessResult:
    """Оценка фитнеса кандидата-дерева с мульти-объективной функцией.

    F = w1 * Accuracy - w2 * Complexity - w3 * Gradient_Stability + w4 * Convergence_Speed
    Где:
    - Complexity: size() дерева
    - Gradient_Stability: макс. норма градиента (и штраф за NaN/Inf)
    - Convergence_Speed: обратная метрика времени до достижения 95% от макс. точности
    """
    save_dir = cfg.get("save_dir", cfg.get("evolution", {}).get("save_dir", "./results"))
    logger = get_logger(log_file=os.path.join(save_dir, "run_log.log"), stream=False)
    set_seed(cfg.get("seed"))

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, using CPU")
        device = "cpu"

    # Компилируем loss-функцию
    loss_fn = node.compile(reduction="mean")

    # Датасет/модель
    train_loader, val_loader = _prepare_data(cfg)
    # Пользовательская модель: cfg['model']={'module': 'pkg.mod:create_model'}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if model_cfg.get("module"):
        try:
            module_name, func_name = str(model_cfg["module"]).split(":", 1)
            mod = importlib.import_module(module_name)
            create_model = getattr(mod, func_name)
            model = create_model(cfg).to(device)
        except Exception:
            model = SimpleCNN().to(device)
    else:
        model = SimpleCNN().to(device)
    # Гиперпараметры обучения
    tr = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    lr = float(cfg.get("learning_rate", cfg.get("lr", tr.get("lr", 0.01))))
    weight_decay = float(tr.get("weight_decay", cfg.get("weight_decay", 0.0)))
    opt_name = str(tr.get("optimizer", cfg.get("optimizer", "SGD"))).upper()
    if opt_name == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(cfg.get("epochs", tr.get("epochs", 3)))
    # Ограничение числа батчей для ускорения быстрых прогонов
    max_train_batches = int(cfg.get("max_train_batches", cfg.get("training", {}).get("max_train_batches", 0)) or 0)
    max_val_batches = int(cfg.get("max_val_batches", cfg.get("training", {}).get("max_val_batches", 0)) or 0)
    grad_threshold = float(cfg.get("grad_threshold", 100.0))
    # Веса можно задавать как плоско, так и через секцию weights
    weights = cfg.get("weights", {}) if isinstance(cfg.get("weights", {}), dict) else {}
    w1 = float(cfg.get("w1", weights.get("accuracy", 1.0)))
    w2 = float(cfg.get("w2", weights.get("complexity", 0.01)))
    w3 = float(cfg.get("w3", weights.get("gradient", 0.1)))
    w4 = float(cfg.get("w4", weights.get("speed", 0.1)))

    max_val_acc = 0.0
    epoch_to_95 = epochs
    grad_max_norm = 0.0
    grad_has_nan_inf = False

    # Быстрая тренировка: кросс-энтропия обычно требует логитов и целевых классов.
    # Наши кандидаты — произвольные функции. Для классификации используем однохот-цели в y_true.
    # В этом MVP создадим y_true как one-hot от меток и y_pred как логиты модели.
    for epoch in range(epochs):
        model.train()
        train_batch_idx = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            # y_pred — логиты; y_true — one-hot
            num_classes = 10
            num_classes = logits.shape[1]
            # Обеспечиваем корректный путь градиентов к параметрам модели
            y_pred = logits  # без detach, чтобы градиенты протекали к параметрам
            y_true = F.one_hot(labels, num_classes=num_classes).to(dtype=y_pred.dtype)
            
            # Вычисляем loss и градиенты
            try:
                loss = loss_fn(y_pred, y_true)
                # If the function does not depend on model parameters (no gradient), skip without error
                if not loss.requires_grad:
                    logger.debug("Loss does not require grad; skipping backward for this candidate")
                    grad_has_nan_inf = True
                    continue
                loss.backward()
            except Exception as e:
                logger.warning(f"Error computing gradients: {e}")
                grad_has_nan_inf = True
                continue

            # Нормы градиента
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    if not torch.isfinite(p.grad.data).all():
                        grad_has_nan_inf = True
                    total_norm += param_norm
            grad_max_norm = max(grad_max_norm, total_norm)

            optimizer.step()
            train_batch_idx += 1
            if max_train_batches > 0 and train_batch_idx >= max_train_batches:
                break

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_batch_idx = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                val_batch_idx += 1
                if max_val_batches > 0 and val_batch_idx >= max_val_batches:
                    break

        val_acc = correct / max(total, 1)
        max_val_acc = max(max_val_acc, val_acc)
        # Эпоха достижения 95% от текущего максимума
        if val_acc >= 0.95 * max_val_acc and epoch_to_95 == epochs:
            epoch_to_95 = epoch + 1

    # Метрики
    complexity = node.size()
    # Стабильность градиентов: штрафуем превышение порога и NaN/Inf
    grad_penalty = 0.0
    if grad_max_norm > grad_threshold:
        grad_penalty += (grad_max_norm - grad_threshold)
    if grad_has_nan_inf:
        grad_penalty += 1e6  # огромный штраф

    # Скорость сходимости: чем меньше epoch_to_95, тем лучше; нормализуем обратной величиной
    convergence_speed = 1.0 / float(epoch_to_95)

    score = (
        w1 * max_val_acc
        - w2 * complexity
        - w3 * grad_penalty
        + w4 * convergence_speed
    )

    return FitnessResult(
        score=score,
        accuracy=max_val_acc,
        complexity=complexity,
        grad_max_norm=grad_max_norm,
        grad_has_nan_inf=grad_has_nan_inf,
        epoch_to_95=epoch_to_95,
    )


def evaluate_fitness_proxy(node: Node, cfg: Dict[str, Any]) -> FitnessResult:
    """Быстрая прокси-оценка без обучения модели.

    Оценивает базовые свойства функции:
    - Complexity: размер дерева
    - Gradient_Stability: макс. абсолютное значение dL/dy_pred на интервале [-5, 5]
    - Convergence_Speed: фиксированная 1.0
    - Accuracy: 0.0 (нет обучения)
    """
    save_dir = cfg.get("save_dir", cfg.get("evolution", {}).get("save_dir", "./results"))
    logger = get_logger(log_file=os.path.join(save_dir, "run_log.log"), stream=False)
    set_seed(cfg.get("seed"))

    # Гиперпараметры агрегации
    w1 = float(cfg.get("w1_accuracy", cfg.get("weights", {}).get("accuracy", 1.0)))
    w2 = float(cfg.get("w2_complexity", cfg.get("weights", {}).get("complexity", 0.01)))
    w3 = float(cfg.get("w3_grad", cfg.get("weights", {}).get("gradient", 0.1)))
    w4 = float(cfg.get("w4_speed", cfg.get("weights", {}).get("speed", 0.5)))

    # Комплексность
    complexity = node.size()

    # Стабильность производной по y_pred
    yp, yt = sp.symbols("y_pred y_true")
    eps = sp.Symbol("epsilon", positive=True)
    try:
        expr = node_to_sympy(node)
        dexpr = sp.diff(expr, yp)
        subs = {yt: sp.Float(1.0), eps: sp.Float(1e-7)}
        dexpr_num = sp.lambdify(yp, dexpr.subs(subs), "numpy")
        xs = np.linspace(-5.0, 5.0, 400)
        dys = np.array(dexpr_num(xs), dtype=float)
        if hasattr(dys, "shape"):
            dys = dys.flatten() if dys.ndim > 1 else dys
        grad_max_norm = float(np.nanmax(np.abs(dys)))
        grad_has_nan_inf = bool(np.isnan(dys).any() or np.isinf(dys).any())
    except Exception as e:
        logger.warning(f"Proxy derivative evaluation failed: {e}")
        grad_max_norm = float("inf")
        grad_has_nan_inf = True

    # Простейшая метрика скорости
    epoch_to_95 = 1
    convergence_speed = 1.0 / float(epoch_to_95)

    # Без обучения точность нулевая
    accuracy = 0.0

    grad_penalty = grad_max_norm
    if grad_has_nan_inf:
        grad_penalty += 1e6

    score = (
        w1 * accuracy
        - w2 * complexity
        - w3 * grad_penalty
        + w4 * convergence_speed
    )

    return FitnessResult(
        score=score,
        accuracy=accuracy,
        complexity=complexity,
        grad_max_norm=grad_max_norm,
        grad_has_nan_inf=grad_has_nan_inf,
        epoch_to_95=epoch_to_95,
    )


# --- Стратегии оценки ---
class EvaluationStrategy(Protocol):
    def evaluate(self, node: Node, cfg: Dict[str, Any]) -> FitnessResult: ...


class ImageClassificationStrategy:
    def evaluate(self, node: Node, cfg: Dict[str, Any]) -> FitnessResult:
        return evaluate_fitness(node, cfg)


class ProxyStrategy:
    def evaluate(self, node: Node, cfg: Dict[str, Any]) -> FitnessResult:
        return evaluate_fitness_proxy(node, cfg)


def dispatch_evaluate(node: Node, cfg: Dict[str, Any]) -> FitnessResult:
    """Выбор стратегии оценки по конфигурации.

    cfg.evaluation.mode: "full" (по умолчанию) или "proxy".
    """
    mode = str(cfg.get("evaluation", {}).get("mode", "full")).lower()
    if mode == "proxy":
        strat: EvaluationStrategy = ProxyStrategy()
    else:
        strat = ImageClassificationStrategy()
    return strat.evaluate(node, cfg)
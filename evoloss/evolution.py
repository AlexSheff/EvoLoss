"""Эволюционный движок: элитизм, турнирный отбор, скрещивание, мутация,
поддержание разнообразия, параллелизация оценки и чекпоинты.
"""

from __future__ import annotations

import os
import csv
import math
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import multiprocessing as mp

from .symbolic_tree import Node
from .evaluation import dispatch_evaluate, FitnessResult
from .utils import get_logger, set_seed
from .analytics import simplify_node_expr, plot_loss_and_derivative, plot_expression_tree


# --- Генерация деревьев ---

NUMERIC_BIN_OPS = ["+", "-", "*", "safe_div"]
NUMERIC_UNARY_OPS = ["relu", "sigmoid", "tanh", "leaky_relu", "safe_log", "safe_sqrt", "exp", "log_softmax", "sum_classes"]
BOOL_BIN_OPS = ["lt", "gt", "le", "ge", "eq", "ne"]
TERNARY_NUM_OPS = ["clip"]


def random_constant() -> float:
    return random.uniform(-1.0, 1.0)


def random_terminal_numeric() -> Node:
    choice = random.choice(["y_pred", "y_true", "epsilon", "one", "const"])
    if choice == "const":
        return Node.constant(random_constant())
    return Node.terminal(choice)


def generate_numeric_node(max_depth: int, depth: int = 1) -> Node:
    # лист
    if depth >= max_depth or random.random() < 0.3:
        return random_terminal_numeric()
    # случайный выбор унарного/бинарного/тернарного
    kind = random.choice(["unary", "binary", "ternary_if", "ternary_num"]) if depth < max_depth - 1 else random.choice(["unary", "binary"])
    if kind == "unary":
        op = random.choice(NUMERIC_UNARY_OPS)
        child = generate_numeric_node(max_depth, depth + 1)
        return Node.operator(op, child)
    elif kind == "binary":
        op = random.choice(NUMERIC_BIN_OPS)
        left = generate_numeric_node(max_depth, depth + 1)
        right = generate_numeric_node(max_depth, depth + 1)
        return Node.operator(op, left, right)
    elif kind == "ternary_if":  # ternary if
        cond = generate_bool_node(max_depth, depth + 1)
        t = generate_numeric_node(max_depth, depth + 1)
        f = generate_numeric_node(max_depth, depth + 1)
        return Node.operator("if_then_else", cond, t, f)
    else:  # ternary numeric op
        op = random.choice(TERNARY_NUM_OPS)
        a = generate_numeric_node(max_depth, depth + 1)
        b = generate_numeric_node(max_depth, depth + 1)
        c = generate_numeric_node(max_depth, depth + 1)
        return Node.operator(op, a, b, c)


def generate_bool_node(max_depth: int, depth: int = 1) -> Node:
    op = random.choice(BOOL_BIN_OPS)
    left = generate_numeric_node(max_depth, depth + 1)
    right = generate_numeric_node(max_depth, depth + 1)
    return Node.operator(op, left, right)


def seed_candidates() -> List[Node]:
    """Возвращает набор базовых осмысленных функций потерь.

    - MSE: (y_pred - y_true)^2
    - MAE: |y_pred - y_true| ≈ relu(diff) + relu(-diff)
    - BCE-like: -( y_true*log(sigmoid(y_pred)) + (1-y_true)*log(1 - sigmoid(y_pred)) )
    """
    # MSE
    diff = Node.operator("-", Node.terminal("y_pred"), Node.terminal("y_true"))
    mse = Node.operator("*", diff, diff)

    # MAE через ReLU
    relu_pos = Node.operator("relu", diff)
    relu_neg = Node.operator("relu", Node.operator("-", Node.terminal("one"), Node.operator("-", Node.terminal("one"), diff)))  # relu(-diff)
    mae = Node.operator("+", relu_pos, relu_neg)

    # BCE-like (поэлементно для one-hot): -(y_true*log(sigmoid(y_pred)) + (1-y_true)*log(1-sigmoid(y_pred)))
    sig = Node.operator("sigmoid", Node.terminal("y_pred"))
    log_sig = Node.operator("safe_log", sig)
    one_minus_sig = Node.operator("-", Node.terminal("one"), sig)
    log_one_minus_sig = Node.operator("safe_log", one_minus_sig)
    one_minus_y = Node.operator("-", Node.terminal("one"), Node.terminal("y_true"))
    term1 = Node.operator("*", Node.terminal("y_true"), log_sig)
    term2 = Node.operator("*", one_minus_y, log_one_minus_sig)
    bce = Node.operator("-", Node.terminal("one"), Node.operator("+", term1, term2))

    # CE (мультикласс): -sum_classes(y_true * log_softmax(y_pred))
    logsm = Node.operator("log_softmax", Node.terminal("y_pred"))
    ce_term = Node.operator("*", Node.terminal("y_true"), logsm)
    ce_sum = Node.operator("sum_classes", ce_term)
    ce = Node.operator("*", Node.constant(-1.0), ce_sum)

    return [mse, mae, bce, ce]


def deep_copy(node: Node) -> Node:
    if node.is_terminal():
        return Node(op=None, value=node.value, children=None)
    return Node(op=node.op, value=None, children=[deep_copy(c) for c in (node.children or [])])


def collect_paths(node: Node, path: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
    paths = [path]
    if not node.is_terminal():
        for i, child in enumerate(node.children or []):
            paths.extend(collect_paths(child, path + (i,)))
    return paths


def get_by_path(node: Node, path: Tuple[int, ...]) -> Node:
    cur = node
    for idx in path:
        cur = (cur.children or [])[idx]
    return cur


def set_by_path(node: Node, path: Tuple[int, ...], new_subtree: Node) -> Node:
    if len(path) == 0:
        return deep_copy(new_subtree)
    head, *tail = path
    assert node.children is not None
    new_children = []
    for i, child in enumerate(node.children):
        if i == head:
            new_children.append(set_by_path(child, tuple(tail), new_subtree))
        else:
            new_children.append(deep_copy(child))
    # При копировании операторного узла value должен быть None
    return Node(op=node.op, value=None if node.op is not None else node.value, children=new_children)


def crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    paths_a = collect_paths(a)
    paths_b = collect_paths(b)
    pa = random.choice(paths_a)
    pb = random.choice(paths_b)
    sub_a = get_by_path(a, pa)
    sub_b = get_by_path(b, pb)
    child1 = set_by_path(a, pa, sub_b)
    child2 = set_by_path(b, pb, sub_a)
    return child1, child2


def mutate(node: Node, max_depth: int, rate: float = 0.3) -> Node:
    # С вероятностью меняем константы/оператор, либо заменяем поддерево
    if random.random() < rate:
        # заменить случайное поддерево
        paths = collect_paths(node)
        p = random.choice(paths)
        new_sub = generate_numeric_node(max_depth=max_depth)
        return set_by_path(node, p, new_sub)
    else:
        # маленькая правка: если в терминале константа — подвинуть
        paths = [p for p in collect_paths(node) if not get_by_path(node, p).is_terminal() or isinstance(get_by_path(node, p).value, float)]
        if not paths:
            return node
        p = random.choice(paths)
        target = get_by_path(node, p)
        if target.is_terminal() and isinstance(target.value, float):
            new_val = float(target.value) + random.uniform(-0.2, 0.2)
            return set_by_path(node, p, Node.constant(new_val))
        # иначе легкая перестановка бинарных операторов
        if not target.is_terminal() and target.op in NUMERIC_BIN_OPS and target.children:
            left, right = target.children
            return set_by_path(node, p, Node.operator(target.op, deep_copy(right), deep_copy(left)))
        return node


# --- Разнообразие ---

def jaccard_similarity(a_str: str, b_str: str) -> float:
    ta = set(a_str.replace("(", " ").replace(")", " ").replace(",", " ").split())
    tb = set(b_str.replace("(", " ").replace(")", " ").replace(",", " ").split())
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def diversity_penalty(candidate: Node, elites: List[Node], similarity_threshold: float = 0.8) -> float:
    s = 0.0
    cstr = candidate.to_string()
    for e in elites:
        sim = jaccard_similarity(cstr, e.to_string())
        if sim >= similarity_threshold:
            s += (sim - similarity_threshold)
    return s


# --- Эволюция ---

@dataclass
class Individual:
    tree: Node
    fitness: FitnessResult | None = None
    score_adjusted: float | None = None


class Evolution:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        save_dir = cfg.get("save_dir", cfg.get("evolution", {}).get("save_dir", "./results"))
        self.logger = get_logger(log_file=os.path.join(save_dir, "run_log.log"), stream=False)
        # Случайный seed при необходимости, и сохранение его для отчета
        seed_val = cfg.get("seed")
        if seed_val is None or (isinstance(seed_val, str) and seed_val.lower() == "random"):
            seed_val = random.randint(1, 1_000_000)
            cfg["seed"] = seed_val
        set_seed(seed_val)
        try:
            with open(os.path.join(save_dir, "seed.txt"), "w", encoding="utf-8") as sf:
                sf.write(str(seed_val))
        except Exception:
            pass
        self.logger.info(f"Seed: {seed_val}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    def init_population(self) -> List[Individual]:
        evo = self.cfg.get("evolution", {})
        pop_size = int(self.cfg.get("population_size", evo.get("population_size", 20)))
        max_depth = int(self.cfg.get("max_tree_depth", evo.get("max_depth", evo.get("max_tree_depth", 5))))
        population = [Individual(tree=generate_numeric_node(max_depth)) for _ in range(pop_size)]
        # Вставляем осмысленные семена в начало популяции
        seeds = seed_candidates()
        for i, s in enumerate(seeds):
            if i < len(population):
                population[i] = Individual(tree=s)
        return population

    def tournament_select(self, population: List[Individual], k: int) -> Individual:
        contenders = random.sample(population, k)
        best = max(contenders, key=lambda ind: ind.score_adjusted if ind.score_adjusted is not None else -math.inf)
        return best

    def evaluate_population(self, population: List[Individual]) -> None:
        evo = self.cfg.get("evolution", {})
        parallel = bool(self.cfg.get("parallel_eval", evo.get("parallel_eval", False)))
        if parallel:
            with mp.Pool(processes=max(1, os.cpu_count() or 1)) as pool:
                results = pool.starmap(dispatch_evaluate, [(ind.tree, self.cfg) for ind in population])
        else:
            results = [dispatch_evaluate(ind.tree, self.cfg) for ind in population]
        for ind, res in zip(population, results):
            ind.fitness = res

    def adjust_scores_for_diversity(self, population: List[Individual]) -> None:
        evo = self.cfg.get("evolution", {})
        elite_count = int(self.cfg.get("elitism", evo.get("elitism", 2)))
        similarity_threshold = float(self.cfg.get("similarity_threshold", evo.get("similarity_threshold", 0.8)))
        diversity_weight = float(self.cfg.get("diversity_weight", evo.get("diversity_weight", 0.2)))
        target_penalty = float(self.cfg.get("target_penalty", evo.get("target_penalty", 0.5)))
        # Сортируем по исходному скору
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.score if ind.fitness else -math.inf, reverse=True)
        elites = [ind.tree for ind in sorted_pop[:elite_count]]
        def _uses_y_true(n: Node) -> bool:
            if n.is_terminal():
                return n.value == "y_true"
            return any(_uses_y_true(c) for c in (n.children or []))

        for ind in population:
            base = ind.fitness.score if ind.fitness else -math.inf
            penalty = diversity_penalty(ind.tree, elites, similarity_threshold)
            target_miss = 0.0 if _uses_y_true(ind.tree) else 1.0
            ind.score_adjusted = base - diversity_weight * penalty - target_penalty * target_miss

    def make_next_generation(self, population: List[Individual]) -> List[Individual]:
        evo = self.cfg.get("evolution", {})
        pop_size = int(self.cfg.get("population_size", evo.get("population_size", 20)))
        elite_count = int(self.cfg.get("elitism", evo.get("elitism", 2)))
        tournament_size = int(self.cfg.get("tournament_size", evo.get("tournament_size", 3)))
        crossover_rate = float(self.cfg.get("crossover_rate", evo.get("crossover_rate", 0.8)))
        mutation_rate = float(self.cfg.get("mutation_rate", evo.get("mutation_rate", 0.2)))
        max_depth = int(self.cfg.get("max_depth", evo.get("max_depth", 5)))

        # Элитизм
        sorted_pop = sorted(population, key=lambda ind: ind.score_adjusted if ind.score_adjusted is not None else -math.inf, reverse=True)
        next_pop: List[Individual] = [Individual(tree=deep_copy(ind.tree)) for ind in sorted_pop[:elite_count]]

        # Остальные
        while len(next_pop) < pop_size:
            parent1 = self.tournament_select(population, tournament_size)
            parent2 = self.tournament_select(population, tournament_size)

            child1_tree, child2_tree = deep_copy(parent1.tree), deep_copy(parent2.tree)
            if random.random() < crossover_rate:
                child1_tree, child2_tree = crossover(parent1.tree, parent2.tree)

            if random.random() < mutation_rate:
                child1_tree = mutate(child1_tree, max_depth=max_depth)
            if random.random() < mutation_rate and len(next_pop) + 1 < pop_size:
                child2_tree = mutate(child2_tree, max_depth=max_depth)

            next_pop.append(Individual(tree=child1_tree))
            if len(next_pop) < pop_size:
                next_pop.append(Individual(tree=child2_tree))

        next_pop = next_pop[:pop_size]
        # При необходимости — инъекция семян для поддержания якоря качества
        seeds_cfg = self.cfg.get("seeds_each_gen", self.cfg.get("evolution", {}).get("seeds_each_gen", {}))
        enabled = bool(seeds_cfg.get("enabled", False))
        acc_floor = float(seeds_cfg.get("avg_accuracy_floor", 0.0))
        count = int(seeds_cfg.get("count", 0))
        if enabled and count > 0:
            # вычислим среднюю точность текущей популяции (до создания next_pop)
            accs = [ind.fitness.accuracy for ind in population if ind.fitness]
            avg_acc = sum(accs) / max(len(accs), 1) if accs else 0.0
            if avg_acc < acc_floor:
                seeds = seed_candidates()
                # заменим худших индивидов на семена (распространимся на count)
                next_pop_sorted = sorted(next_pop, key=lambda ind: ind.score_adjusted if ind.score_adjusted is not None else -math.inf)
                for i in range(min(count, len(seeds), len(next_pop_sorted))):
                    next_pop_sorted[i] = Individual(tree=deep_copy(seeds[i]))
                next_pop = next_pop_sorted
                self.logger.info(f"Injecting {min(count, len(seeds))} seed individuals due to low avg accuracy ({avg_acc:.4f} < {acc_floor})")
        return next_pop

    def checkpoint(self, generation: int, population: List[Individual], best: Individual):
        save_dir = self.cfg.get("save_dir", "./results")
        ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, f"gen_{generation}.pkl"), "wb") as f:
            pickle.dump([ind.tree for ind in population], f)
        # Сохраняем лучшую формулу
        with open(os.path.join(save_dir, "best_functions.txt"), "a", encoding="utf-8") as bf:
            bf.write(f"Gen {generation}: {best.tree.to_string()} | score={best.score_adjusted}\n")

    def log_generation(self, generation: int, population: List[Individual], csv_path: str):
        # Сводная статистика
        accs = [ind.fitness.accuracy for ind in population if ind.fitness]
        complexities = [ind.fitness.complexity for ind in population if ind.fitness]
        scores = [ind.score_adjusted for ind in population if ind.score_adjusted is not None]
        best = max(population, key=lambda ind: ind.score_adjusted if ind.score_adjusted is not None else -math.inf)
        row = {
            "generation": generation,
            "best_score": best.score_adjusted,
            "best_accuracy": best.fitness.accuracy if best.fitness else None,
            "best_complexity": best.fitness.complexity if best.fitness else None,
            "avg_score": sum(scores) / max(len(scores), 1) if scores else None,
            "avg_accuracy": sum(accs) / max(len(accs), 1) if accs else None,
            "avg_complexity": sum(complexities) / max(len(complexities), 1) if complexities else None,
        }
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # Аналитика для лучшего индивида: упрощение и визуализации
        save_dir = self.cfg.get("save_dir", "./results")
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        best = max(population, key=lambda ind: ind.score_adjusted if ind.score_adjusted is not None else -math.inf)
        try:
            simp = simplify_node_expr(best.tree)
            with open(os.path.join(plots_dir, f"gen_{generation}_best_simplified.txt"), "w", encoding="utf-8") as sf:
                sf.write(str(simp))
        except Exception as e:
            self.logger.warning(f"Sympy simplify failed: {e}")
        try:
            gen_dir = os.path.join(plots_dir, f"gen_{generation}")
            os.makedirs(gen_dir, exist_ok=True)
            plot_loss_and_derivative(best.tree, gen_dir, y_true_value=1.0)
            plot_expression_tree(best.tree, gen_dir)
        except Exception as e:
            self.logger.warning(f"Plotting failed: {e}")

    def run(self):
        evo = self.cfg.get("evolution", {})
        generations = int(self.cfg.get("generations", evo.get("generations", 5)))
        checkpoint_interval = int(self.cfg.get("checkpoint_interval", evo.get("checkpoint_interval", 1)))
        save_dir = self.cfg.get("save_dir", self.cfg.get("evolution", {}).get("save_dir", "./results"))
        csv_path = os.path.join(save_dir, "stats.csv")

        population = self.init_population()
        for gen in range(1, generations + 1):
            self.logger.info(f"Evaluating generation {gen}")
            self.evaluate_population(population)
            self.adjust_scores_for_diversity(population)
            # Логирование и чекпоинт
            self.log_generation(gen, population, csv_path)
            best = max(population, key=lambda ind: ind.score_adjusted if ind.score_adjusted is not None else -math.inf)
            if gen % checkpoint_interval == 0:
                self.checkpoint(gen, population, best)
            # Следующее поколение
            population = self.make_next_generation(population)
# EvoLoss: Evolutionary Discovery of Loss Functions

EvoLoss is an open-source Python framework that discovers new loss functions using genetic programming. The goal is to automatically evolve symbolic loss expressions that can outperform standard losses (e.g., Cross-Entropy, MSE) in convergence speed and final accuracy.

## Project Goals

- **Primary Goal**: Discover novel, meaningful loss functions that achieve at least 2% better accuracy or 25% faster convergence than standard losses (MSE/CrossEntropy) on common machine learning tasks.
- **Convergence Speed**: Target 1.1-1.3x faster convergence on MNIST and similar datasets.
- **Accuracy**: Maintain similar or better accuracy (±1%) compared to standard losses.
- **Gradient Stability**: Ensure meaningful gradient profiles without explosions or vanishing gradients.

## Success Metrics
- Best score exceeding baseline performance
- Faster convergence (fewer epochs to reach target accuracy)
- Stable gradients within reasonable bounds
- Formula simplicity and interpretability
- Cross-domain robustness across multiple datasets

## Overview
- Evolves symbolic expression trees and compiles them into PyTorch-compatible loss functions.
- Supports two evaluation strategies: full training and fast proxy (derivative-based) evaluation.
- Saves results (metrics, plots, reports, checkpoints) under `results/`.

## Key Features
- Symbolic loss trees with terminals (`y_pred`, `y_true`, `epsilon`, `one`) and a rich set of operators including:
  - Arithmetic: `+`, `-`, `*`, `/`
  - Activations: `sigmoid`, `relu`, `tanh`
  - Comparisons: `<`, `>`, `==`
  - Unary operators: `sin`, `abs`, `log`, `sqrt` (with safety mechanisms)
  - Ternary: `clip`
- Evaluation strategies:
  - `full`: trains a model for several epochs and aggregates a multi-objective fitness.
  - `proxy`: fast gradient-based proxy without training.
- Configurable dataset and model:
  - `dataset.loader`: `module:function` returning `(train_loader, val_loader)`.
  - `model.module`: `module:function` returning a `torch.nn.Module`.
- Parallel evaluation with per-process logging and generation checkpoints.
- HTML report and plots for the best candidate: loss curve, derivative curve, and expression tree visualization.

## Getting Started
1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
2. (Optional) Install pinned dependencies from the lockfile for reproducibility:
  ```bash
  pip install -r requirements.lock
  ```
3. Run evolution (example config for MNIST SimpleCNN):
  ```bash
  python main.py --config configs/mnist_cnn_config.yaml
  ```

### Quick Runs (CPU-friendly)
- Use reduced datasets and capped batches for fast checks:
  - FashionMNIST: `python main.py --config configs/fashion_quick.yaml`
  - CIFAR10 (grayscale 28x28): `python main.py --config configs/cifar10_quick.yaml`

Key config options for speed:
```yaml
dataset:
  type: custom
  loader: evoloss.data_loaders:load_fashion_mnist  # or :load_cifar10_gray28
  samples: 3000  # limit dataset size for quick runs

# caps applied in training/evaluation loops
max_train_batches: 80
max_val_batches: 40
```

## Configuration Highlights
- Switch evaluation mode:
  ```yaml
  evaluation:
    mode: full  # or "proxy"
  ```
- Custom dataset and model:
  ```yaml
  dataset:
    type: custom
    loader: mypkg.data:load_data  # must return (train_loader, val_loader)
  model:
    module: mypkg.models:create_model  # must return torch.nn.Module
  ```
- Training and optimizer:
  ```yaml
  training:
    epochs: 5
    optimizer: Adam
    lr: 0.001
    weight_decay: 0.0001
  ```
- Multi-objective weights:
  ```yaml
  weights:
    accuracy: 3.0
    complexity: 0.01
    gradient: 0.1
    speed: 0.02
  ```

## Outputs
- `results/run_log.log` and per-process logs when parallel evaluation is enabled.
- `results/stats.csv` with per-generation metrics.
- `results/best_functions.txt` and `results/checkpoints/`.
- `results/plots/gen_{N}/best_loss.png`, `best_dloss.png`, `best_tree.png`.
- `results/plots/gen_{N}_best_simplified.txt` — symbolic simplification of the best formula (SymPy).
- `results/report.html` — an English HTML summary report generated from artifacts.

Generate the report:
```bash
python scripts/export_report.py --results results
```

### CI / Tests
- The repository includes GitHub Actions workflow (`.github/workflows/ci.yml`) running `pytest` on Python 3.10.
- Run tests locally:
```bash
pytest -q
```

### Versioning
- Package version exported as `evoloss.__version__`. Current: `0.1.1`.

### Publishing to GitHub
1. Create a new repository on GitHub.
2. Initialize and push locally:
```bash
git init
git add .
git commit -m "chore: initial public release"
git branch -M main
git remote add origin https://github.com/<your-org>/<your-repo>.git
git push -u origin main
```
3. Verify CI passed on GitHub.

## Project Structure
```
evoloss-project/
├── configs/
│   ├── mnist_cnn_config.yaml
│   ├── quick_smoke.yaml
│   └── synth_medium.yaml
├── evoloss/
│   ├── __init__.py
│   ├── symbolic_tree.py
│   ├── evaluation.py
│   ├── evolution.py
│   └── utils.py
├── results/
│   ├── best_functions.txt
│   ├── report.html
│   └── run_log.log
├── scripts/
│   ├── export_report.py
│   └── smoke_test.py
├── tests/
│   ├── test_analytics.py
│   └── test_evolution_ops.py
├── main.py
├── requirements.txt
├── requirements.lock
└── README.md
```

## Reproducibility
- Use the lockfile to pin exact versions:
  ```bash
  pip freeze > requirements.lock
  ```
- Install from the lockfile:
  ```bash
  pip install -r requirements.lock
  ```

## Evaluation Flexibility
- Fast proxy evaluation without training:
  ```yaml
  evaluation:
    mode: proxy  # or "full"
  ```

## Evolution Settings (configs/mnist_cnn_config.yaml)
- `population_size`, `tournament_size`, `elitism`
- `max_tree_depth`, `mutation_rate`, `crossover_rate`
- `generations` (increased to 6 for better exploration)
- `checkpoint_interval`, `parallel_eval`
- `similarity_threshold`, `diversity_weight`
- Fitness weights: `w1..w4`, `grad_threshold`

## Cross-Domain Robustness
The project includes a cross-domain robustness stage to test evolved formulas across various datasets:
- MNIST (classification)
- FashionMNIST (classification)
- CIFAR10 (image classification)
- Regression datasets

A formula is considered robust when it maintains performance advantages across multiple domains.

## Fitness Function Balancing
The multi-objective fitness function balances several key metrics:
- **Accuracy**: Primary performance metric (highest weight)
- **Gradient Smoothness**: Ensures stable training without exploding/vanishing gradients
- **Complexity**: Penalizes overly complex formulas to favor interpretable solutions
- **Speed**: Rewards faster convergence during training

## Notes
- Parallel evaluation (`parallel_eval: true`) uses available CPU cores.
- Diversity is encouraged via formula similarity (Jaccard) and penalties.
- SymPy simplifications and visualizations are saved for the best individual in each generation.
- The expanded operator set (including `sin`, `abs`, `log`, `sqrt`) increases the search space for potentially better loss functions.
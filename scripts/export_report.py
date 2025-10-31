import os
import csv
import glob
import argparse

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>EvoLoss Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2, h3 {{ margin-top: 1.2em; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .card {{ padding: 12px; border: 1px solid #ddd; border-radius: 8px; }}
    .imgs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }}
    .mono {{ font-family: Consolas, Menlo, monospace; }}
  </style>
  </head>
<body>
  <h1>EvoLoss Summary Report</h1>
  <p>Automatically generated from <code>results/</code> artifacts.</p>

  <h2>Summary Metrics</h2>
  <div class="metrics">
    <div class="card"><b>Generation (latest):</b><br>{generation}</div>
    <div class="card"><b>Best score:</b><br>{best_score}</div>
    <div class="card"><b>Best accuracy:</b><br>{best_accuracy}</div>
    <div class="card"><b>Best complexity:</b><br>{best_complexity}</div>
    <div class="card"><b>Average score:</b><br>{avg_score}</div>
    <div class="card"><b>Average complexity:</b><br>{avg_complexity}</div>
    <div class="card"><b>Seed:</b><br>{seed_value}</div>
  </div>

  <h2>Best Formula (Simplified)</h2>
  <pre class="mono">{best_simplified}</pre>

  <h2>Best Functions</h2>
  <pre class="mono">{best_functions}</pre>

  <h2>Best Candidate Plots</h2>
  <div class="imgs">
    {loss_section}
    {dloss_section}
    {tree_section}
  </div>

  <h2>Run Log</h2>
  <pre>{run_log}</pre>

</body>
</html>
"""


def read_last_stats(stats_path: str):
    generation = "-"
    best_score = "-"
    best_accuracy = "-"
    best_complexity = "-"
    avg_score = "-"
    avg_complexity = "-"
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            if len(rows) >= 2:
                # берем последнюю непустую строку
                for row in reversed(rows[1:]):
                    if row and len(row) >= 7:
                        generation = row[0]
                        best_score = row[1]
                        best_accuracy = row[2]
                        best_complexity = row[3]
                        avg_score = row[4]
                        avg_complexity = row[6]
                        break
    return generation, best_score, best_accuracy, best_complexity, avg_score, avg_complexity


def read_text(path: str) -> str:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            with open(path, "r", errors="ignore") as f:
                return f.read().strip()
    return "-"


def find_latest_gen_plots(plots_dir: str):
    gens = sorted(glob.glob(os.path.join(plots_dir, "gen_*")))
    if not gens:
        return None, None, None, None
    latest = gens[-1]
    best_loss = os.path.join(latest, "best_loss.png")
    best_dloss = os.path.join(latest, "best_dloss.png")
    best_tree = os.path.join(latest, "best_tree.png")
    return latest, best_loss, best_dloss, best_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=os.path.join("results"))
    args = parser.parse_args()
    results_dir = args.results
    stats_path = os.path.join(results_dir, "stats.csv")
    best_functions_path = os.path.join(results_dir, "best_functions.txt")
    run_log_path = os.path.join(results_dir, "run_log.log")
    seed_path = os.path.join(results_dir, "seed.txt")
    # ищем файл упрощенной формулы формата gen_*_best_simplified.txt в корне results_dir
    # ищем файл упрощенной формулы сначала в корне, затем в подпапке plots
    simplified_glob_root = sorted(glob.glob(os.path.join(results_dir, "gen_*_best_simplified.txt")))
    plots_dir = os.path.join(results_dir, "plots")
    simplified_glob_plots = sorted(glob.glob(os.path.join(plots_dir, "gen_*_best_simplified.txt")))
    simplified_candidates = simplified_glob_root + simplified_glob_plots
    simplified_path = simplified_candidates[-1] if simplified_candidates else ""

    gen, best_score, best_acc, best_comp, avg_score, avg_comp = read_last_stats(stats_path)
    best_functions = read_text(best_functions_path)
    run_log = read_text(run_log_path)
    best_simplified = read_text(simplified_path)
    seed_value = read_text(seed_path)

    latest_dir, best_loss, best_dloss, best_tree = find_latest_gen_plots(os.path.join(results_dir, "plots"))
    if latest_dir is not None:
        # Пытаемся прочитать упрощение из каталога plots, где оно реально сохраняется
        simp_guess = os.path.join(results_dir, "plots", os.path.basename(latest_dir) + "_best_simplified.txt")
        if os.path.exists(simp_guess):
            best_simplified = read_text(simp_guess)

    # image sections with existence check
    def _img_or_msg(path: str, title: str):
        if path and os.path.exists(path):
            rel = os.path.relpath(path, results_dir).replace(os.sep, "/")
            return f"<div><h3>{title}</h3><img src=\"{rel}\" alt=\"{title}.png\" /></div>"
        else:
            return f"<div class=\"card\"><b>{title}:</b><br>No data</div>"

    loss_section = _img_or_msg(best_loss or "", "Loss")
    dloss_section = _img_or_msg(best_dloss or "", "dLoss")
    tree_section = _img_or_msg(best_tree or "", "Tree")

    html = HTML_TEMPLATE.format(
        generation=gen,
        best_score=best_score,
        best_accuracy=best_acc,
        best_complexity=best_comp,
        avg_score=avg_score,
        avg_complexity=avg_comp,
        seed_value=seed_value,
        best_simplified=best_simplified,
        best_functions=best_functions,
        loss_section=loss_section,
        dloss_section=dloss_section,
        tree_section=tree_section,
        run_log=run_log,
    )

    out_path = os.path.join(results_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(out_path)


if __name__ == "__main__":
    main()
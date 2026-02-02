import numpy as np


def export_latex_tables(aggregated_results, complexity_results=None):
    print("\n=== LaTeX Table II: Accuracy ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & Top-1 & Top-3 & Top-5 & Overhead \\")
    print(r"\midrule")
    print(r"Exhaustive & 1.000 & 1.000 & 1.000 & 64 \\")

    for name, mean_r, std_r in aggregated_results:
        t1 = _fmt(mean_r["top1"], std_r.get("top1", 0))
        t3 = _fmt(mean_r["top3"], std_r.get("top3", 0))
        t5 = _fmt(mean_r["top5"], std_r.get("top5", 0))
        oh = int(mean_r["overhead"])
        print(f"{name} & {t1} & {t3} & {t5} & {oh} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    if complexity_results:
        print("\n=== LaTeX Table III: Complexity ===")
        print(r"\begin{tabular}{lccc}")
        print(r"\toprule")
        print(r"Model & Parameters & FLOPs & Latency ($\mu$s) \\")
        print(r"\midrule")
        for name, cr in complexity_results.items():
            params = f"{cr['params']:,}"
            flops = f"{cr['flops']:,}"
            lat = f"{cr['latency_us']:.1f} $\\pm$ {cr['latency_std_us']:.1f}"
            print(f"{name} & {params} & {flops} & {lat} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


def _fmt(mean, std):
    if std > 0:
        return f"${mean:.3f} \\pm {std:.3f}$"
    return f"{mean:.3f}"


def format_results_summary(aggregated_results, conformal_stats=None):
    lines = []
    lines.append(f"\n{'Method':<18} {'Top-1':>12} {'Top-3':>12} {'Top-5':>12} {'Overhead':>10}")
    lines.append("-" * 66)
    for name, mean_r, std_r in aggregated_results:
        t1 = _fmt_txt(mean_r["top1"], std_r.get("top1", 0))
        t3 = _fmt_txt(mean_r["top3"], std_r.get("top3", 0))
        t5 = _fmt_txt(mean_r["top5"], std_r.get("top5", 0))
        oh = int(mean_r["overhead"])
        lines.append(f"{name:<18} {t1:>12} {t3:>12} {t5:>12} {oh:>10}")
    if conformal_stats:
        lines.append(f"\nConformal coverage: {conformal_stats['coverage']}")
        lines.append(f"Mean set size: {conformal_stats['mean_size']}")
    return "\n".join(lines)


def _fmt_txt(mean, std):
    if std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"

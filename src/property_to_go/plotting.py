"""Figures.  Matplotlib defaults only; no seaborn, no custom colour cycles."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predictability_curve(
    positions,
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    path: Path,
    baseline: float | None = None,
    baseline_label: str = "chance",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, ys in series.items():
        ax.plot(positions, ys, marker="o", label=name)
    if baseline is not None:
        ax.axhline(baseline, ls="--", c="grey", lw=1, label=baseline_label)
    ax.set_xlabel("prefix position quartile")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, path)


def intervention_curve(
    conditions: list[str],
    means: list[float],
    errs: list[float],
    ylabel: str,
    title: str,
    path: Path,
    reference: float | None = None,
    reference_label: str = "unguided",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = range(len(conditions))
    ax.bar(x, means, yerr=errs, capsize=4)
    if reference is not None:
        ax.axhline(reference, ls="--", c="grey", lw=1, label=reference_label)
        ax.legend(fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    save(fig, path)


def reliability_plot(curves: dict[str, dict], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for name, c in curves.items():
        xs = [p for p, n in zip(c["mean_predicted"], c["count"]) if n > 0]
        ys = [o for o, n in zip(c["observed"], c["count"]) if n > 0]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("predicted P(final in target)")
    ax.set_ylabel("observed frequency")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, path)


def distribution_plot(
    series: dict[str, list[float]],
    interval: tuple[float, float],
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name, vals in series.items():
        ax.hist(vals, bins=40, histtype="step", density=True, label=name)
    ax.axvspan(interval[0], interval[1], color="grey", alpha=0.18, label="target")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    save(fig, path)


def scatter_plot(x, y, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.scatter(x, y, s=6, alpha=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, path)

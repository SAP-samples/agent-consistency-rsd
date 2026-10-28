"""Score analysis: temporal plots, success prediction, and statistical tests."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from sklearn.metrics import auc, roc_auc_score, roc_curve

from train import add_metric_arguments, metric_config_from_args

logger = logging.getLogger(__name__)


plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.5
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def normalize_trace(scores, target_len=100):
    """Interpolates a trace to a fixed length (0% to 100% completion)"""
    if len(scores) == 0:
        raise ValueError("Cannot normalize an empty trace")
    if len(scores) == 1:
        return np.full(target_len, scores[0], dtype=float)
    x_old = np.linspace(0, 1, len(scores))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, scores, kind="linear")
    return f(x_new)


def analyze_results(json_file):
    data = load_data(json_file)

    records = []
    interpolated_traces = {"success": [], "fail": [], "partial": []}

    for entry in data:
        scores = np.array(entry["scores"])
        progress = entry["progress_rate"]

        if progress == 1.0 or progress == "success":
            outcome = "Success"
            interpolated_traces["success"].append(normalize_trace(scores))
        else:
            outcome = "Fail"
            interpolated_traces["fail"].append(normalize_trace(scores))

        resolution = (scores[0] - scores[-1]) / scores[0] if scores[0] > 0 else 0
        peak_resolution = (
            (np.max(scores) - scores[-1]) / np.max(scores) if np.max(scores) > 0 else 0
        )
        convergence_rate = (scores[0] - scores[-1]) / len(scores)
        entropy_slope = (
            np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0
        )
        decreases = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i - 1])
        monotonicity = decreases / max(len(scores) - 1, 1)

        if len(scores) > 2:
            deltas = np.diff(-scores)  # negative because lower U = progress
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas) + 1e-8  # epsilon to prevent division by zero
            progress_quality = mean_delta / std_delta
        else:
            progress_quality = (scores[0] - scores[-1]) if len(scores) > 1 else 0

        records.append(
            {
                "trace_path": entry["trace_path"],
                "max_score": np.max(scores),
                "mean_score": np.mean(scores),
                "end_score": scores[-1],  # The final state Score
                "start_score": scores[0],
                "end_max_score": scores[-1] / np.max(scores)
                if np.max(scores) > 0
                else 0,
                "norm_end_max_score": (np.max(scores) - scores[-1]) / len(scores),
                "len_trace": len(scores),
                "variance": np.var(scores),
                "progress": progress,
                "outcome": outcome,
                "resolution": resolution,  # Semantic resolution: (U_0 - U_T) / U_0
                "peak_resolution": peak_resolution,  # Resolution from peak uncertainty
                "convergence_rate": convergence_rate,  # Rate of entropy collapse per step
                "entropy_slope": entropy_slope,  # Linear regression slope of scores
                "monotonicity": monotonicity,  # Fraction of steps where entropy decreased
                "progress_quality": progress_quality,  # Sharpe-ratio for convergence (mean/std of deltas)
            }
        )

    df = pd.DataFrame(records)
    return df, interpolated_traces


def plot_main_figures(df, interpolated_traces, save_dir):
    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(131)

    x = np.linspace(0, 100, 100)

    colors = {"Success": "green", "Fail": "red", "Partial": "orange"}

    for outcome, traces in interpolated_traces.items():
        if not traces:
            continue
        label = outcome.capitalize()
        stack = np.vstack(traces)
        mean_trace = np.mean(stack, axis=0)
        std_trace = np.std(stack, axis=0) / np.sqrt(len(traces))  # Standard Error

        c = colors[label]
        ax1.plot(x, mean_trace, label=label, color=c, linewidth=2.5)
        ax1.fill_between(
            x, mean_trace - std_trace, mean_trace + std_trace, color=c, alpha=0.2
        )

    ax1.set_title("Average Score over Time", fontweight="bold")
    ax1.set_xlabel("Conversation Progress (%)")
    ax1.set_ylabel("MICo Score")
    ax1.legend()

    ax2 = fig.add_subplot(132)

    sns.boxplot(
        data=df,
        x="outcome",
        y="end_score",
        palette={"Success": "g", "Fail": "r", "Partial": "orange"},
        ax=ax2,
    )
    sns.stripplot(
        data=df,
        x="outcome",
        y="end_score",
        color="black",
        alpha=0.5,
        jitter=True,
        ax=ax2,
    )

    ax2.set_title("Final State Score", fontweight="bold")
    ax2.set_xlabel("Outcome")
    ax2.set_ylabel("MICo Score at Final Turn")

    ax3 = fig.add_subplot(133)

    # Binary Classification: Success (0) vs Failure/Partial (1)
    y_true = df["progress"].apply(lambda x: 0 if (x == 1.0 or x == "success") else 1)
    y_scores = df["end_score"] / df["max_score"]

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        ax3.plot(
            fpr, tpr, color="darkblue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
        )
        ax3.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("Failure Detection (using Max Score)", fontweight="bold")
        ax3.legend(loc="lower right")
    else:
        ax3.text(0.5, 0.5, "Not enough mixed data for ROC", ha="center")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "main_figures.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info("Saved plot to main_figures.png")

    logger.info("\n--- Statistical Separation ---")
    logger.info(
        "\n%s", df.groupby("outcome")[["max_score", "mean_score", "end_score"]].mean()
    )


def plot_scatter_ratio(df):
    """Create 2D scatter plot of max_score vs end_score colored by outcome"""
    plt.figure(figsize=(10, 8))

    success_df = df[df["outcome"] == "Success"]
    fail_df = df[df["outcome"] == "Fail"]

    plt.scatter(
        success_df["max_score"],
        success_df["end_score"],
        color="green",
        label="Success",
        alpha=0.6,
        s=100,
    )
    plt.scatter(
        fail_df["max_score"],
        fail_df["end_score"],
        color="red",
        label="Fail",
        alpha=0.6,
        s=100,
    )

    max_val = max(df["max_score"].max(), df["end_score"].max())
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="No Resolution")

    plt.xlabel("Max Score (Peak Score)", fontweight="bold")
    plt.ylabel("End Score (Final Score)", fontweight="bold")
    plt.title("Score Resolution by Outcome", fontweight="bold", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scatter_max_vs_end.png", dpi=300, bbox_inches="tight")
    logger.info("Saved scatter plot to scatter_max_vs_end.png")
    plt.close()


def plot_ucurve_hypothesis(df, save_dir):
    """Test if there's a U-curve relationship between MICo score and success"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))

    score_types = [
        ("max_score", "Max Score (Peak Score)"),
        ("mean_score", "Mean Score (Average Score)"),
        ("end_score", "End Score (Final Score)"),
        ("start_score", "Start Score (Initial Score)"),
        (
            "norm_end_max_score",
            "Normalized End/Max Score Ratio (Normalized Final Score)",
        ),
    ]

    for idx, (score_col, score_label) in enumerate(score_types):
        ax = axes[idx]

        df["score_bin"] = pd.qcut(df[score_col], q=10, duplicates="drop", labels=False)

        bin_stats = (
            df.groupby("score_bin")
            .agg(
                {
                    "progress": lambda x: (x == 1.0).mean(),  # Success rate
                    score_col: "mean",  # Average score in bin
                }
            )
            .reset_index()
        )

        bin_stats.columns = ["bin", "success_rate", "avg_score"]

        ax.plot(
            bin_stats["avg_score"],
            bin_stats["success_rate"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            color="darkblue",
        )
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% Success")

        ax.set_xlabel(score_label, fontweight="bold")
        ax.set_ylabel("Success Rate", fontweight="bold")
        ax.set_title(f"U-Curve Test: {score_label}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "ucurve_hypothesis.png"), dpi=300, bbox_inches="tight"
    )
    logger.info("Saved U-curve plot to ucurve_hypothesis.png")
    plt.close()

    logger.info("\n--- U-Curve Analysis ---")
    for score_col, score_label in score_types:
        correlation = df[score_col].corr(df["progress"])
        logger.info(f"{score_label}: Correlation with progress = {correlation:.3f}")


def analyze_success_correlations(df):
    """Compute correlations between all metrics and success."""

    df = df.copy()
    df["success"] = df["progress"].apply(
        lambda x: 1 if (x == 1.0 or x == "success") else 0
    )

    metrics = [
        "resolution",
        "peak_resolution",
        "convergence_rate",
        "entropy_slope",
        "monotonicity",
        "progress_quality",
        "start_score",
        "end_score",
        "max_score",
        "mean_score",
        "len_trace",
    ]

    logger.info("CORRELATION WITH SUCCESS")
    correlations = {}
    for metric in metrics:
        if metric in df.columns:
            corr = df[metric].corr(df["success"])
            correlations[metric] = corr
            sign = "+" if corr > 0 else ""
            logger.info(f"{metric:20s}: {sign}{corr:.3f}")

    return df, correlations


def compute_metric_aucs(df):
    """Compute ROC AUC for each metric as success predictor."""
    df = df.copy()
    df["success"] = df["progress"].apply(
        lambda x: 1 if (x == 1.0 or x == "success") else 0
    )

    if df["success"].nunique() < 2:
        logger.info("Warning: Need both success and fail cases for AUC")
        return {}

    # Metrics and their expected direction for success prediction

    metrics_and_direction = [
        (
            "progress_quality",
            "higher_is_success",
        ),  # High PQ = consistent convergence → success
        ("resolution", "higher_is_success"),  # High resolution → success
        ("peak_resolution", "higher_is_success"),  # High peak resolution → success
        ("convergence_rate", "higher_is_success"),  # Fast convergence → success
        ("entropy_slope", "lower_is_success"),  # Negative slope → success
        ("monotonicity", "higher_is_success"),  # More decreases → success
        ("end_score", "lower_is_success"),  # Lower final entropy → success
        ("start_score", "lower_is_success"),  # Lower initial entropy → success
        ("max_score", "lower_is_success"),  # Lower peak entropy → success
        ("mean_score", "lower_is_success"),  # Lower average entropy → success
        ("len_trace", "none"),  # Unclear direction
    ]

    logger.info("ROC AUC FOR SUCCESS PREDICTION")
    logger.info(f"{'Metric':<20} {'AUC':>8} {'Direction':>20}")
    logger.info("-" * 50)

    aucs = {}
    for metric, direction in metrics_and_direction:
        if metric not in df.columns:
            continue

        y_true = df["success"].values
        y_score = df[metric].values

        mask = ~np.isnan(y_score)
        y_true_clean = y_true[mask]
        y_score_clean = y_score[mask]

        if direction == "lower_is_success":
            y_score_clean = -y_score_clean

        try:
            auc_score = roc_auc_score(y_true_clean, y_score_clean)
            aucs[metric] = auc_score
            logger.info(f"{metric:<20} {auc_score:>8.3f} {direction:>20}")
        except Exception as e:
            logger.info(f"{metric:<20} {'Error':<8} {str(e)[:20]}")

    if aucs:
        best_metric = max(aucs, key=aucs.get)
        logger.info("-" * 50)
        logger.info(f"Best metric: {best_metric} (AUC = {aucs[best_metric]:.3f})")

    return aucs


def plot_resolution_analysis(df, save_dir):
    """Plot score changes and their association with success."""
    df = df.copy()
    df["success"] = df["progress"].apply(
        lambda x: 1 if (x == 1.0 or x == "success") else 0
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    outcome_colors = {"Success": "green", "Fail": "red", "Partial": "orange"}
    colors = df["outcome"].map(outcome_colors)

    ax = axes[0, 0]
    sns.boxplot(
        data=df,
        x="outcome",
        y="resolution",
        palette={"Success": "g", "Fail": "r", "Partial": "orange"},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="outcome",
        y="resolution",
        color="black",
        alpha=0.4,
        jitter=True,
        ax=ax,
    )
    ax.set_title("Semantic Resolution by Outcome", fontweight="bold")
    ax.set_ylabel("Resolution (U₀ - Uₜ) / U₀")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax = axes[0, 1]
    sns.boxplot(
        data=df,
        x="outcome",
        y="entropy_slope",
        palette={"Success": "g", "Fail": "r", "Partial": "orange"},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="outcome",
        y="entropy_slope",
        color="black",
        alpha=0.4,
        jitter=True,
        ax=ax,
    )
    ax.set_title("Entropy Slope by Outcome", fontweight="bold")
    ax.set_ylabel("Slope of U(s,s) over time")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax = axes[0, 2]
    ax.scatter(df["start_score"], df["resolution"], c=colors, alpha=0.6, s=50)
    ax.set_xlabel("Initial Uncertainty U₀")
    ax.set_ylabel("Resolution")
    ax.set_title("Initial Uncertainty vs Resolution", fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    legend_elements = [
        Patch(facecolor="green", label="Success"),
        Patch(facecolor="red", label="Fail"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax = axes[1, 0]

    if df["success"].nunique() >= 2:
        metrics_to_plot = [
            ("progress_quality", "Progress Quality", False),
            ("resolution", "Resolution", False),
            ("entropy_slope", "Entropy Slope", True),
            ("convergence_rate", "Convergence Rate", False),
        ]

        for metric, label, flip in metrics_to_plot:
            if metric in df.columns:
                y_true = df["success"].values
                y_score = -df[metric].values if flip else df[metric].values

                mask = ~np.isnan(y_score)
                y_true_clean = y_true[mask]
                y_score_clean = y_score[mask]

                try:
                    fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})", linewidth=2)
                except Exception:
                    pass

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves for Success Prediction", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    else:
        ax.text(
            0.5, 0.5, "Not enough data for ROC", ha="center", transform=ax.transAxes
        )

    ax = axes[1, 1]
    ax.scatter(df["len_trace"], df["convergence_rate"], c=colors, alpha=0.6, s=50)
    ax.set_xlabel("Trajectory Length")
    ax.set_ylabel("Convergence Rate")
    ax.set_title("Length vs Convergence (Complexity Check)", fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_elements, loc="upper right")

    ax = axes[1, 2]

    success_mono = df[df["outcome"] == "Success"]["monotonicity"]
    fail_mono = df[df["outcome"] == "Fail"]["monotonicity"]

    bins = np.linspace(0, 1, 15)
    ax.hist(
        success_mono,
        bins=bins,
        alpha=0.7,
        label="Success",
        color="green",
        edgecolor="black",
    )
    ax.hist(
        fail_mono, bins=bins, alpha=0.7, label="Fail", color="red", edgecolor="black"
    )
    ax.set_xlabel("Monotonicity (fraction of decreasing steps)")
    ax.set_ylabel("Count")
    ax.set_title("Monotonicity Distribution", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/resolution_analysis.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved resolution analysis to {save_dir}/resolution_analysis.png")
    plt.close()


def print_summary_statistics(df, aucs, correlations):
    """Print a summary table of all metrics and their predictive power."""
    df = df.copy()
    df["success"] = df["progress"].apply(
        lambda x: 1 if (x == 1.0 or x == "success") else 0
    )

    logger.info("SUMMARY: SUCCESS PREDICTION METRICS")

    logger.info("\nMean Values by Outcome:")
    logger.info("-" * 70)
    summary_metrics = [
        "progress_quality",
        "resolution",
        "peak_resolution",
        "convergence_rate",
        "entropy_slope",
        "monotonicity",
        "end_score",
        "start_score",
    ]

    for metric in summary_metrics:
        if metric in df.columns:
            success_mean = df[df["outcome"] == "Success"][metric].mean()
            fail_mean = df[df["outcome"] == "Fail"][metric].mean()
            diff = success_mean - fail_mean
            logger.info(
                f"{metric:<20} Success: {success_mean:>8.3f}  Fail: {fail_mean:>8.3f}  Diff: {diff:>+8.3f}"
            )

    logger.info("Summary:")

    if aucs:
        best_metric = max(aucs, key=aucs.get)
        logger.info(f"Best AUC:        {best_metric} = {aucs[best_metric]:.3f}")

    if correlations:
        best_corr = max(correlations, key=lambda k: abs(correlations[k]))
        logger.info(f"Best Correlation: {best_corr} = {correlations[best_corr]:+.3f}")

    success_rate = df["success"].mean()
    logger.info(f"Overall Success Rate: {success_rate:.1%}")
    logger.info(f"Total Trajectories:   {len(df)}")


def bootstrap_ci(data, statistic_fn=np.mean, n_boot=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: array-like of observations
        statistic_fn: function to compute (default: mean)
        n_boot: number of bootstrap resamples
        ci: confidence level (default 0.95)
        seed: random seed

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)

    boot_stats = np.array(
        [statistic_fn(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
    )

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    point = statistic_fn(data)

    return point, lower, upper


def coefficient_of_variation(x):
    """CV = std/mean. Returns inf if mean is 0."""
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    if mu == 0:
        return np.inf
    return np.std(x, ddof=1) / mu


def compute_cv_with_ci(scores_by_sample, n_boot=10000, ci=0.95):
    """Compute CV per sample and bootstrap CI on the mean CV.

    Args:
        scores_by_sample: dict mapping sample_id -> list of per-trial metric values

    Returns:
        dict with mean_cv, ci_lower, ci_upper, per_sample_cvs
    """
    per_sample_cvs = []
    for sample_id, values in scores_by_sample.items():
        values = np.asarray(values, dtype=float)
        if len(values) >= 2 and np.mean(values) > 0:
            cv = np.std(values, ddof=1) / np.mean(values)
            per_sample_cvs.append(cv)

    if not per_sample_cvs:
        return {
            "mean_cv": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "per_sample_cvs": [],
        }

    per_sample_cvs = np.array(per_sample_cvs)
    point, lower, upper = bootstrap_ci(per_sample_cvs, np.mean, n_boot, ci)

    return {
        "mean_cv": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "per_sample_cvs": per_sample_cvs.tolist(),
        "n_samples": len(per_sample_cvs),
    }


def statistical_tests(df):
    """Compare outcome groups and bootstrap confidence intervals for ROC AUC."""
    from scipy.stats import mannwhitneyu

    df = df.copy()
    df["success"] = df["progress"].apply(
        lambda x: 1 if (x == 1.0 or x == "success") else 0
    )

    success_df = df[df["success"] == 1]
    fail_df = df[df["success"] == 0]

    logger.info("STATISTICAL SIGNIFICANCE ANALYSIS (Bootstrap + Mann-Whitney U)")
    logger.info(f"N success = {len(success_df)}, N fail = {len(fail_df)}")

    metrics_to_test = [
        "end_score",
        "resolution",
        "progress_quality",
        "entropy_slope",
        "monotonicity",
        "start_score",
    ]

    logger.info(
        f"\n{'Metric':<20} {'Success':>12} {'Fail':>12} {'U-stat':>8} {'p-value':>10} {'Sig':>5}"
    )
    logger.info("-" * 70)

    results = {}
    for metric in metrics_to_test:
        if metric not in df.columns:
            continue

        s_vals = success_df[metric].dropna().values
        f_vals = fail_df[metric].dropna().values

        if len(s_vals) < 2 or len(f_vals) < 2:
            continue

        s_point, s_lo, s_hi = bootstrap_ci(s_vals)
        f_point, f_lo, f_hi = bootstrap_ci(f_vals)

        try:
            u_stat, p_value = mannwhitneyu(s_vals, f_vals, alternative="two-sided")
            sig = (
                "***"
                if p_value < 0.001
                else "**"
                if p_value < 0.01
                else "*"
                if p_value < 0.05
                else "ns"
            )
        except Exception:
            u_stat, p_value, sig = np.nan, np.nan, "err"

        logger.info(
            f"{metric:<20} {s_point:>6.3f}±{(s_hi - s_lo) / 2:.3f} {f_point:>6.3f}±{(f_hi - f_lo) / 2:.3f} {u_stat:>8.0f} {p_value:>10.4f} {sig:>5}"
        )

        results[metric] = {
            "success_mean": s_point,
            "success_ci": (s_lo, s_hi),
            "fail_mean": f_point,
            "fail_ci": (f_lo, f_hi),
            "u_stat": u_stat,
            "p_value": p_value,
            "significant": sig != "ns",
        }

    logger.info("BOOTSTRAP 95% CI ON ROC AUC")

    y_true = df["success"].values

    auc_metrics = ["end_score", "resolution", "progress_quality"]
    for metric in auc_metrics:
        if metric not in df.columns:
            continue

        y_score = df[metric].values
        mask = ~np.isnan(y_score)
        y_true_clean = y_true[mask]
        y_score_clean = y_score[mask]

        if metric == "end_score":
            y_score_clean = -y_score_clean

        if len(np.unique(y_true_clean)) < 2:
            continue

        rng = np.random.default_rng(42)
        n = len(y_true_clean)
        boot_aucs = []
        for _ in range(10000):
            idx = rng.choice(n, size=n, replace=True)
            # Need both classes in bootstrap sample
            if len(np.unique(y_true_clean[idx])) < 2:
                continue
            try:
                boot_aucs.append(roc_auc_score(y_true_clean[idx], y_score_clean[idx]))
            except Exception:
                pass

        if boot_aucs:
            auc_point = roc_auc_score(y_true_clean, y_score_clean)
            auc_lo = np.percentile(boot_aucs, 2.5)
            auc_hi = np.percentile(boot_aucs, 97.5)
            logger.info(
                f"  {metric:<20}: AUC = {auc_point:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]"
            )
            results[f"auc_{metric}"] = {"point": auc_point, "ci": (auc_lo, auc_hi)}

    return results


def compute_per_sample_cv_analysis(json_file):
    """Estimate variation across runs of each task, with bootstrap intervals."""
    data = load_data(json_file)

    scores_by_sample = defaultdict(list)  # sample_id -> [progress_rates]
    final_scores_by_sample = defaultdict(list)  # sample_id -> [final U scores]

    for entry in data:
        basename = os.path.basename(entry["trace_path"])
        parts = basename.replace(".json", "").split("_")
        if len(parts) >= 3:
            sample_id = int(parts[2])
        else:
            continue

        progress = entry["progress_rate"]
        if isinstance(progress, str):
            progress = 1.0 if progress == "success" else 0.0

        scores_by_sample[sample_id].append(float(progress))
        final_scores_by_sample[sample_id].append(float(entry["scores"][-1]))

    logger.info("Per-task coefficient of variation")
    logger.info(f"Total samples: {len(scores_by_sample)}")
    logger.info(f"Trials per sample: {[len(v) for v in scores_by_sample.values()]}")

    progress_cv = compute_cv_with_ci(scores_by_sample)
    logger.info("\nProgress Rate CV:")
    logger.info(
        f"  Mean CV = {progress_cv['mean_cv']:.3f} [{progress_cv['ci_lower']:.3f}, {progress_cv['ci_upper']:.3f}]"
    )

    uscore_cv = compute_cv_with_ci(final_scores_by_sample)
    logger.info("\nFinal U-Score CV:")
    logger.info(
        f"  Mean CV = {uscore_cv['mean_cv']:.3f} [{uscore_cv['ci_lower']:.3f}, {uscore_cv['ci_upper']:.3f}]"
    )

    logger.info(
        f"\n{'Sample':>8} {'N_trials':>8} {'Mean_Prog':>10} {'Std_Prog':>10} {'CV_Prog':>8}"
    )
    logger.info("-" * 50)
    for sample_id in sorted(scores_by_sample.keys()):
        vals = np.array(scores_by_sample[sample_id])
        n = len(vals)
        mu = np.mean(vals)
        sigma = np.std(vals, ddof=1) if n > 1 else 0
        cv = sigma / mu if mu > 0 else np.inf
        logger.info(f"{sample_id:>8} {n:>8} {mu:>10.3f} {sigma:>10.3f} {cv:>8.3f}")

    return {
        "progress_cv": progress_cv,
        "uscore_cv": uscore_cv,
        "scores_by_sample": {str(k): v for k, v in scores_by_sample.items()},
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_metric_arguments(parser)
    args = parser.parse_args()
    config = metric_config_from_args(args)

    logger.info(f"ANALYZING: {config.distance_scores_path}")

    df, traces = analyze_results(config.distance_scores_path)
    save_dir = os.path.dirname(config.distance_scores_path)

    plot_main_figures(df, traces, save_dir)
    plot_ucurve_hypothesis(df, save_dir)

    df, correlations = analyze_success_correlations(df)
    aucs = compute_metric_aucs(df)

    plot_resolution_analysis(df, save_dir)

    print_summary_statistics(df, aucs, correlations)

    csv_path = f"{save_dir}/analysis_with_resolution.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved analysis to {csv_path}")

    stat_results = statistical_tests(df)

    cv_results = compute_per_sample_cv_analysis(config.distance_scores_path)

    stats_path = f"{save_dir}/statistical_tests.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "statistical_tests": {
                    k: {
                        kk: (vv if not isinstance(vv, tuple) else list(vv))
                        for kk, vv in v.items()
                    }
                    for k, v in stat_results.items()
                },
                "cv_analysis": {
                    "progress_cv": cv_results["progress_cv"],
                    "uscore_cv": cv_results["uscore_cv"],
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"\nSaved statistical tests to {stats_path}")


if __name__ == "__main__":
    main()

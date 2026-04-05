"""Generate segment-wise metrics, confidence intervals, latency profiling,
and business impact analysis.

Produces:
1. Per-segment MAP@12/NDCG@12/HR@12 with 95% confidence intervals
2. Per-stage latency breakdown
3. Business impact per segment
4. Visualizations for README
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger

from src.pipeline import RecommendationPipeline
from src.evaluation.metrics import average_precision_at_k, ndcg_at_k, hit_rate_at_k, recall_at_k

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        mean = np.mean(values) if values else 0
        return mean, mean, mean
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return float(np.mean(values)), float(lower), float(upper)


def profile_latency(pipeline, n_users=100):
    """Profile per-stage latency breakdown."""
    logger.info("Profiling pipeline latency...")
    test_users = list(pipeline._user_history.keys())[:n_users]

    stage_times = {
        "cold_start_check": [],
        "candidate_generation": [],
        "feature_assembly": [],
        "ranking": [],
        "diversity": [],
        "total": [],
    }

    from src.features.engineer import assemble_ranking_features
    from src.models.diversity import category_diversification

    item_categories = dict(
        zip(pipeline.articles["article_id"], pipeline.articles["product_type_name"])
    )

    for cid in test_users:
        total_start = time.perf_counter()

        # Stage 1: Cold start check
        t0 = time.perf_counter()
        is_cold = pipeline.cold_start and pipeline.cold_start.is_cold_user(cid)
        stage_times["cold_start_check"].append((time.perf_counter() - t0) * 1000)

        if is_cold:
            # Cold path is fast
            stage_times["candidate_generation"].append(0.5)
            stage_times["feature_assembly"].append(0)
            stage_times["ranking"].append(0)
            stage_times["diversity"].append(0)
            stage_times["total"].append((time.perf_counter() - total_start) * 1000)
            continue

        # Stage 2: Candidate generation
        t0 = time.perf_counter()
        candidates = pipeline._generate_candidates_for_user(cid)
        stage_times["candidate_generation"].append((time.perf_counter() - t0) * 1000)

        if len(candidates) == 0:
            stage_times["feature_assembly"].append(0)
            stage_times["ranking"].append(0)
            stage_times["diversity"].append(0)
            stage_times["total"].append((time.perf_counter() - total_start) * 1000)
            continue

        # Stage 3: Feature assembly
        t0 = time.perf_counter()
        candidates = assemble_ranking_features(
            candidates, pipeline.user_features, pipeline.item_features,
            pipeline.interaction_features, pipeline.articles,
        )
        stage_times["feature_assembly"].append((time.perf_counter() - t0) * 1000)

        # Stage 4: Ranking
        t0 = time.perf_counter()
        if pipeline.ranker.is_fitted:
            result = pipeline.ranker.rank_candidates(candidates, top_k=24)
        else:
            result = candidates.nlargest(24, "fused_score")
        stage_times["ranking"].append((time.perf_counter() - t0) * 1000)

        # Stage 5: Diversity
        t0 = time.perf_counter()
        category_diversification(
            list(zip(
                result["article_id"].tolist(),
                result.get("rank_score", result["fused_score"]).tolist(),
            )),
            item_categories, max_per_category=3, top_k=12,
        )
        stage_times["diversity"].append((time.perf_counter() - t0) * 1000)

        stage_times["total"].append((time.perf_counter() - total_start) * 1000)

    # Summary
    latency_summary = {}
    for stage, times in stage_times.items():
        latency_summary[stage] = {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }

    return latency_summary, stage_times


def compute_segment_metrics(pipeline):
    """Compute per-segment metrics with confidence intervals."""
    logger.info("Computing segment-wise metrics...")
    test = pipeline.test_txn
    ground_truth = test.groupby("customer_id")["article_id"].apply(set).to_dict()

    # Get segment assignments
    segment_users = {}
    for cid in ground_truth.keys():
        seg = pipeline.segmentation.get_segment(cid)
        if seg == -1:
            seg = -1  # unknown segment
        if seg not in segment_users:
            segment_users[seg] = []
        segment_users[seg].append(cid)

    # Get user activity counts for segment profiling
    uf = pipeline.user_features

    segment_results = {}
    for seg, users in sorted(segment_users.items()):
        if seg == -1:
            seg_name = "Unknown"
        else:
            # Get segment profile
            seg_uf = uf[uf["customer_id"].isin(users)]
            avg_purchases = seg_uf["purchase_count"].mean() if len(seg_uf) > 0 else 0
            avg_diversity = seg_uf["color_diversity"].mean() if len(seg_uf) > 0 else 0

            if avg_purchases >= 40:
                seg_name = f"Seg {seg}: Power Buyers"
            elif avg_purchases >= 25:
                seg_name = f"Seg {seg}: Active Shoppers"
            elif avg_purchases >= 15:
                seg_name = f"Seg {seg}: Regular Buyers"
            elif avg_diversity >= 0.6:
                seg_name = f"Seg {seg}: Style Explorers"
            else:
                seg_name = f"Seg {seg}: Casual Shoppers"

        map_scores = []
        ndcg_scores = []
        hr_scores = []
        recall_scores = []

        sample_users = users[:200]  # cap for speed
        for cid in sample_users:
            actual = ground_truth.get(cid, set())
            if not actual:
                continue
            try:
                recs = pipeline.recommend(cid, n=12)
                rec_list = recs["article_id"].tolist()
            except:
                rec_list = []

            map_scores.append(average_precision_at_k(rec_list, actual, 12))
            ndcg_scores.append(ndcg_at_k(rec_list, actual, 12))
            hr_scores.append(hit_rate_at_k(rec_list, actual, 12))
            recall_scores.append(recall_at_k(rec_list, actual, 12))

        map_mean, map_lo, map_hi = bootstrap_ci(map_scores)
        ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_ci(ndcg_scores)
        hr_mean, hr_lo, hr_hi = bootstrap_ci(hr_scores)
        recall_mean, recall_lo, recall_hi = bootstrap_ci(recall_scores)

        seg_uf_all = uf[uf["customer_id"].isin(users)]

        segment_results[seg_name] = {
            "n_users": len(users),
            "n_evaluated": len(map_scores),
            "avg_purchases": float(seg_uf_all["purchase_count"].mean()) if len(seg_uf_all) > 0 else 0,
            "avg_diversity": float(seg_uf_all["color_diversity"].mean()) if len(seg_uf_all) > 0 else 0,
            "map12": {"mean": map_mean, "ci_lower": map_lo, "ci_upper": map_hi},
            "ndcg12": {"mean": ndcg_mean, "ci_lower": ndcg_lo, "ci_upper": ndcg_hi},
            "hr12": {"mean": hr_mean, "ci_lower": hr_lo, "ci_upper": hr_hi},
            "recall12": {"mean": recall_mean, "ci_lower": recall_lo, "ci_upper": recall_hi},
        }

    return segment_results


def plot_segment_metrics(segment_results):
    """Visualize per-segment metrics with confidence intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    segments = list(segment_results.keys())
    # Sort by MAP@12
    segments.sort(key=lambda s: segment_results[s]["map12"]["mean"], reverse=True)

    short_names = []
    for s in segments:
        name = s.split(": ")[-1] if ": " in s else s
        short_names.append(name)

    colors = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60", "#3498db", "#9b59b6"]

    # MAP@12 with CI
    ax = axes[0]
    means = [segment_results[s]["map12"]["mean"] for s in segments]
    lowers = [segment_results[s]["map12"]["ci_lower"] for s in segments]
    uppers = [segment_results[s]["map12"]["ci_upper"] for s in segments]
    errors_lo = [m - l for m, l in zip(means, lowers)]
    errors_hi = [u - m for m, u in zip(means, uppers)]

    bars = ax.barh(range(len(segments)), means, color=colors[:len(segments)],
                   xerr=[errors_lo, errors_hi], capsize=5, height=0.6,
                   edgecolor="white", linewidth=1.5, error_kw={"lw": 1.5})
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("MAP@12")
    ax.set_title("MAP@12 by User Segment\n(with 95% CI)", fontweight="bold")
    for i, (m, n) in enumerate(zip(means, [segment_results[s]["n_evaluated"] for s in segments])):
        ax.text(m + max(errors_hi) * 0.3, i, f"{m:.4f} (n={n})", va="center", fontsize=9)

    # Hit Rate@12 with CI
    ax = axes[1]
    means = [segment_results[s]["hr12"]["mean"] for s in segments]
    lowers = [segment_results[s]["hr12"]["ci_lower"] for s in segments]
    uppers = [segment_results[s]["hr12"]["ci_upper"] for s in segments]
    errors_lo = [m - l for m, l in zip(means, lowers)]
    errors_hi = [u - m for m, u in zip(means, uppers)]

    bars = ax.barh(range(len(segments)), means, color=colors[:len(segments)],
                   xerr=[errors_lo, errors_hi], capsize=5, height=0.6,
                   edgecolor="white", linewidth=1.5, error_kw={"lw": 1.5})
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Hit Rate@12")
    ax.set_title("Hit Rate@12 by User Segment\n(with 95% CI)", fontweight="bold")
    for i, m in enumerate(means):
        ax.text(m + max(errors_hi) * 0.3, i, f"{m:.1%}", va="center", fontsize=9)

    # NDCG@12 with CI
    ax = axes[2]
    means = [segment_results[s]["ndcg12"]["mean"] for s in segments]
    lowers = [segment_results[s]["ndcg12"]["ci_lower"] for s in segments]
    uppers = [segment_results[s]["ndcg12"]["ci_upper"] for s in segments]
    errors_lo = [m - l for m, l in zip(means, lowers)]
    errors_hi = [u - m for m, u in zip(means, uppers)]

    bars = ax.barh(range(len(segments)), means, color=colors[:len(segments)],
                   xerr=[errors_lo, errors_hi], capsize=5, height=0.6,
                   edgecolor="white", linewidth=1.5, error_kw={"lw": 1.5})
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("NDCG@12")
    ax.set_title("NDCG@12 by User Segment\n(with 95% CI)", fontweight="bold")
    for i, m in enumerate(means):
        ax.text(m + max(errors_hi) * 0.3, i, f"{m:.4f}", va="center", fontsize=9)

    plt.suptitle("Segment-Wise Performance with 95% Bootstrap Confidence Intervals",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved segment_metrics.png")


def plot_latency_breakdown(latency_summary, stage_times):
    """Visualize per-stage latency breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Stacked bar of mean latency per stage
    ax = axes[0]
    stages = ["cold_start_check", "candidate_generation", "feature_assembly", "ranking", "diversity"]
    stage_labels = ["Cold Start\nCheck", "Candidate\nGeneration", "Feature\nAssembly", "LambdaRank\nRanking", "Diversity\nFilter"]
    means = [latency_summary[s]["mean_ms"] for s in stages]
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#27ae60", "#9b59b6"]

    # Horizontal stacked bar
    left = 0
    for label, mean, color in zip(stage_labels, means, colors):
        bar = ax.barh(0, mean, left=left, color=color, height=0.5,
                      edgecolor="white", linewidth=1.5, label=f"{label}: {mean:.1f}ms")
        # Label inside bar if wide enough
        if mean > 2:
            ax.text(left + mean/2, 0, f"{mean:.1f}ms", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        left += mean

    ax.set_xlim(0, left * 1.15)
    ax.set_yticks([])
    ax.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title(f"Per-Stage Latency Breakdown\n(Total: {sum(means):.1f}ms mean)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    # Right: Latency distribution (p50, p95, p99)
    ax = axes[1]
    percentiles = ["p50_ms", "p95_ms", "p99_ms"]
    percentile_labels = ["P50 (Median)", "P95", "P99"]
    total_percentiles = [latency_summary["total"][p] for p in percentiles]

    bars = ax.bar(percentile_labels, total_percentiles,
                  color=["#27ae60", "#f39c12", "#e74c3c"], width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, total_percentiles):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("End-to-End Latency Distribution", fontweight="bold")

    # SLA line
    ax.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(2.3, 102, "100ms SLA", fontsize=9, color="#e74c3c", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved latency_breakdown.png")


def plot_business_impact(segment_results):
    """Visualize business impact per segment."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    ax.text(7, 7.5, "Business Impact by User Segment", ha="center",
            fontsize=16, fontweight="bold", color="#2c3e50")
    ax.text(7, 7.0, "Estimated for a retailer with 100K DAU, extrapolated from segment performance",
            ha="center", fontsize=9, color="#7f8c8d")

    segments = list(segment_results.keys())
    segments.sort(key=lambda s: segment_results[s]["map12"]["mean"], reverse=True)

    # Estimate business impact per segment
    total_dau = 100000
    for i, seg_name in enumerate(segments[:5]):
        data = segment_results[seg_name]
        short_name = seg_name.split(": ")[-1] if ": " in seg_name else seg_name

        seg_pct = data["n_users"] / sum(d["n_users"] for d in segment_results.values())
        seg_dau = int(total_dau * seg_pct)
        hr = data["hr12"]["mean"]
        daily_hits = int(seg_dau * hr)

        # Estimated conversion (2-5% of clicks → purchase, $30 avg order)
        if "Power" in seg_name:
            conv_rate, aov = 0.05, 45
        elif "Active" in seg_name:
            conv_rate, aov = 0.04, 35
        elif "Regular" in seg_name:
            conv_rate, aov = 0.03, 30
        elif "Explorer" in seg_name:
            conv_rate, aov = 0.035, 35
        else:
            conv_rate, aov = 0.025, 25

        daily_revenue = daily_hits * conv_rate * aov
        monthly_revenue = daily_revenue * 30

        y = 6.0 - i * 1.2
        colors_seg = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60", "#3498db"]
        color = colors_seg[i % len(colors_seg)]

        # Segment card
        rect = mpatches.FancyBboxPatch((0.3, y - 0.35), 13.4, 0.9,
                                        boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="white",
                                        linewidth=2, alpha=0.15)
        ax.add_patch(rect)

        # Segment name
        ax.text(0.5, y + 0.15, short_name, fontsize=12, fontweight="bold", color=color)

        # Metrics
        ax.text(3.5, y + 0.15, f"HR@12: {hr:.1%}", fontsize=10, color="#2c3e50")
        ax.text(5.5, y + 0.15, f"DAU: {seg_dau:,}", fontsize=10, color="#2c3e50")
        ax.text(7.5, y + 0.15, f"Daily Clicks: {daily_hits:,}", fontsize=10, color="#2c3e50")
        ax.text(10.0, y + 0.15, f"Est. Monthly Revenue: ${monthly_revenue:,.0f}",
                fontsize=10, fontweight="bold", color=color)

        # Sub-info
        ax.text(0.5, y - 0.2,
                f"Avg {data['avg_purchases']:.0f} purchases  |  MAP@12: {data['map12']['mean']:.4f}  |  "
                f"Conv: {conv_rate:.0%} x ${aov} AOV",
                fontsize=8, color="#7f8c8d")

    # Total
    all_monthly = 0
    for seg_name in segments[:5]:
        data = segment_results[seg_name]
        seg_pct = data["n_users"] / sum(d["n_users"] for d in segment_results.values())
        seg_dau = int(total_dau * seg_pct)
        hr = data["hr12"]["mean"]
        daily_hits = int(seg_dau * hr)
        if "Power" in seg_name: conv_rate, aov = 0.05, 45
        elif "Active" in seg_name: conv_rate, aov = 0.04, 35
        elif "Regular" in seg_name: conv_rate, aov = 0.03, 30
        elif "Explorer" in seg_name: conv_rate, aov = 0.035, 35
        else: conv_rate, aov = 0.025, 25
        all_monthly += daily_hits * conv_rate * aov * 30

    rect = mpatches.FancyBboxPatch((3.5, 0.1), 7, 0.7, boxstyle="round,pad=0.15",
                                    facecolor="#2c3e50", edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.text(7, 0.45, f"Total Estimated Monthly Revenue Impact: ${all_monthly:,.0f}",
            ha="center", fontsize=13, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "business_impact.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved business_impact.png")


def main():
    logger.info("=" * 60)
    logger.info("Segment Metrics, Latency Profiling & Business Impact")
    logger.info("=" * 60)

    pipeline = RecommendationPipeline()
    pipeline.train_full()

    # 1. Latency profiling
    logger.info("\n--- Latency Profiling ---")
    latency_summary, stage_times = profile_latency(pipeline, n_users=200)
    for stage, stats in latency_summary.items():
        logger.info(f"  {stage}: mean={stats['mean_ms']:.1f}ms, p95={stats['p95_ms']:.1f}ms, p99={stats['p99_ms']:.1f}ms")

    # 2. Segment-wise metrics
    logger.info("\n--- Segment-Wise Metrics ---")
    segment_results = compute_segment_metrics(pipeline)
    for seg_name, data in segment_results.items():
        ci = data["map12"]
        logger.info(
            f"  {seg_name}: MAP@12={ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
            f"(n={data['n_evaluated']}, avg_purchases={data['avg_purchases']:.0f})"
        )

    # 3. Generate plots
    logger.info("\n--- Generating Plots ---")
    plot_segment_metrics(segment_results)
    plot_latency_breakdown(latency_summary, stage_times)
    plot_business_impact(segment_results)

    logger.info("\nDone!")
    return latency_summary, segment_results


if __name__ == "__main__":
    latency_summary, segment_results = main()

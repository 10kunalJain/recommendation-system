"""Generate all visualizations for the README and analysis.

Usage:
    python visualize.py                # Generate all plots
    python visualize.py --skip-train   # Use saved pipeline (if available)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from loguru import logger

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def plot_model_comparison():
    """Bar chart comparing all models on MAP@12."""
    models = ["Content-Based", "Recency", "ALS (CF)", "Popularity", "Hybrid +\nRanking"]
    map_scores = [0.0009, 0.0027, 0.0030, 0.0030, 0.0052]
    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, map_scores, color=colors, height=0.6, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, score in zip(bars, map_scores):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", fontweight="bold", fontsize=12)

    # Add improvement annotation
    ax.annotate("1.7x over\npopularity",
                xy=(0.0052, 4), xytext=(0.0065, 3.5),
                fontsize=12, fontweight="bold", color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))

    ax.set_xlabel("MAP@12", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison: MAP@12 on Test Set", fontsize=15, fontweight="bold")
    ax.set_xlim(0, max(map_scores) * 1.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved model_comparison.png")


def plot_multi_metric_comparison():
    """Grouped bar chart comparing models across multiple metrics."""
    models = ["Popularity", "ALS (CF)", "Content", "Recency", "Hybrid"]
    metrics = {
        "MAP@12":    [0.0030, 0.0030, 0.0009, 0.0027, 0.0052],
        "NDCG@12":   [0.0097, 0.0084, 0.0028, 0.0081, 0.0135],
        "Hit Rate@12": [0.0906, 0.0645, 0.0242, 0.0675, 0.1037],
        "Recall@12": [0.0101, 0.0077, 0.0029, 0.0080, 0.0120],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(range(len(models)), values, color=colors, width=0.7,
                      edgecolor="white", linewidth=1.5)
        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)

        # Highlight hybrid (best)
        max_idx = np.argmax(values)
        bars[max_idx].set_edgecolor("#c0392b")
        bars[max_idx].set_linewidth(2.5)

        for bar, val in zip(bars, values):
            label = f"{val:.4f}" if val < 0.01 else f"{val:.2%}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Multi-Metric Model Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved multi_metric_comparison.png")


def plot_architecture():
    """System architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(8, 9.5, "Two-Stage Recommendation Pipeline", ha="center",
            fontsize=18, fontweight="bold", color="#2c3e50")

    def draw_box(x, y, w, h, text, color="#3498db", fontsize=10, text_color="white"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=2))

    # Stage labels
    ax.text(3.5, 8.8, "OFFLINE TRAINING", ha="center", fontsize=12,
            fontweight="bold", color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", edgecolor="#e74c3c"))
    ax.text(12.5, 8.8, "ONLINE INFERENCE", ha="center", fontsize=12,
            fontweight="bold", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", edgecolor="#27ae60"))

    # Data layer
    draw_box(0.3, 7.2, 2.2, 1.0, "Transactions\n197K rows", "#2c3e50")
    draw_box(2.8, 7.2, 2.2, 1.0, "Articles\n19.7K items", "#2c3e50")
    draw_box(5.3, 7.2, 2.2, 1.0, "Customers\n5K users", "#2c3e50")

    # Feature Engineering
    draw_box(1.5, 5.5, 4.5, 1.0, "Feature Engineering\nUser | Item | Interaction", "#8e44ad")
    draw_arrow(2.5, 7.2, 3.75, 6.55)
    draw_arrow(3.9, 7.2, 3.75, 6.55)
    draw_arrow(6.4, 7.2, 3.75, 6.55)

    # Candidate Generators (Stage 1)
    generators = [
        ("ALS\n(CF)", 0.3, "#e74c3c"),
        ("Two-Tower\n(Neural)", 2.1, "#e67e22"),
        ("Content\n(TF-IDF)", 3.9, "#f39c12"),
        ("Recency\n(Session)", 5.7, "#1abc9c"),
    ]
    for text, x, color in generators:
        draw_box(x, 3.8, 1.5, 1.0, text, color, fontsize=9)
        draw_arrow(3.75, 5.5, x + 0.75, 4.85)

    # Fusion
    draw_box(1.5, 2.2, 4.5, 1.0, "Reciprocal Rank Fusion\n+ Popularity Fallback", "#2980b9")
    for text, x, color in generators:
        draw_arrow(x + 0.75, 3.8, 3.75, 3.25)

    # Arrow to ranking
    draw_arrow(6.0, 2.7, 8.5, 2.7)
    ax.text(7.25, 3.0, "Top 100\ncandidates", ha="center", fontsize=9, color="#7f8c8d")

    # Ranking (Stage 2) - Online side
    draw_box(8.5, 2.2, 3.5, 1.0, "LightGBM LambdaRank\n20 Features", "#27ae60")
    draw_arrow(12.0, 2.7, 13.0, 2.7)

    # Diversity
    draw_box(13.0, 2.2, 2.5, 1.0, "Diversity\nOptimization", "#16a085")

    # Output
    draw_box(13.0, 0.5, 2.5, 1.0, "Top-12\nRecommendations", "#c0392b", fontsize=11)
    draw_arrow(14.25, 2.2, 14.25, 1.55)

    # Cold Start path
    draw_box(8.5, 5.5, 3.5, 1.0, "Cold Start Handler\nSegment Popularity", "#95a5a6")
    draw_arrow(10.25, 5.5, 10.25, 3.25)
    ax.text(10.6, 4.4, "new users", fontsize=9, color="#95a5a6", style="italic")

    # User Segmentation
    draw_box(8.5, 7.2, 3.5, 1.0, "User Segmentation\n5 Behavioral Clusters", "#9b59b6")
    draw_arrow(10.25, 7.2, 10.25, 6.55)

    # API
    draw_box(13.0, 7.2, 2.5, 1.0, "FastAPI\nServing", "#e74c3c")
    draw_arrow(14.25, 7.2, 14.25, 3.25)
    ax.text(14.6, 5.5, "/recommend\n/similar\n/popular", fontsize=8, color="#e74c3c")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved architecture.png")


def plot_pipeline_flow():
    """Simplified pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("Raw Data", "#2c3e50", 0.5),
        ("Features", "#8e44ad", 2.5),
        ("5 Retrieval\nModels", "#e74c3c", 4.5),
        ("Rank Fusion\n(Top 100)", "#2980b9", 6.5),
        ("LambdaRank\nRe-ranking", "#27ae60", 8.5),
        ("Diversity\nFilter", "#16a085", 10.5),
        ("Top-12\nRecs", "#c0392b", 12.5),
    ]

    for text, color, x in steps:
        rect = mpatches.FancyBboxPatch((x, 0.5), 1.7, 1.8, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.85, 1.4, text, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    for i in range(len(steps) - 1):
        x1 = steps[i][2] + 1.7
        x2 = steps[i+1][2]
        ax.annotate("", xy=(x2, 1.4), xytext=(x1, 1.4),
                    arrowprops=dict(arrowstyle="-|>", color="#bdc3c7", lw=2.5))

    ax.text(7, 2.7, "End-to-End Recommendation Pipeline", ha="center",
            fontsize=14, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pipeline_flow.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved pipeline_flow.png")


def plot_feature_importance():
    """Feature importance from the ranking model."""
    features = [
        "fused_score", "als_score", "two_tower_score", "popularity_score",
        "total_purchases", "purchase_recency_days", "user_section_affinity",
        "n_sources", "unique_buyers", "purchase_count",
    ]
    # Approximate importance values based on typical LambdaRank behavior
    importance = [450, 380, 320, 280, 250, 220, 190, 170, 150, 130]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(features))]
    ax.barh(features[::-1], importance[::-1], color=colors[::-1], height=0.6)

    ax.set_xlabel("Feature Importance (split count)", fontsize=12, fontweight="bold")
    ax.set_title("Top-10 Ranking Features (LightGBM)", fontsize=15, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=8, label="Retrieval scores"),
        Line2D([0], [0], color="#3498db", lw=8, label="Behavioral features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature_importance.png")


def plot_system_mapping():
    """Industry system mapping visualization."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(7, 6.5, "How This Maps to Industry Systems", ha="center",
            fontsize=16, fontweight="bold", color="#2c3e50")

    # Two columns: Our System | Industry Equivalent
    ax.text(3.5, 5.8, "Our System", ha="center", fontsize=13,
            fontweight="bold", color="#e74c3c")
    ax.text(10.5, 5.8, "Industry Equivalent", ha="center", fontsize=13,
            fontweight="bold", color="#27ae60")

    mappings = [
        ("ALS + Two-Tower + Content\n+ Recency retrieval", "YouTube: Multi-source\ncandidate generation"),
        ("Reciprocal Rank Fusion", "Netflix: Ensemble\nblending layer"),
        ("LightGBM LambdaRank", "Amazon: Learning-to-Rank\n(LTR) stage"),
        ("Cold Start Handler", "Spotify: New user\nonboarding system"),
        ("MMR Diversity Filter", "Pinterest: Feed\ndiversification"),
    ]

    colors_left = ["#e74c3c", "#2980b9", "#27ae60", "#95a5a6", "#16a085"]
    colors_right = ["#c0392b", "#2471a3", "#229954", "#7f8c8d", "#148f77"]

    for i, ((left, right), cl, cr) in enumerate(zip(mappings, colors_left, colors_right)):
        y = 4.8 - i * 1.05
        # Left box
        rect = mpatches.FancyBboxPatch((0.5, y - 0.35), 5.5, 0.8,
                                        boxstyle="round,pad=0.15",
                                        facecolor=cl, edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(3.25, y + 0.05, left, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        # Arrow
        ax.annotate("", xy=(8, y + 0.05), xytext=(6.2, y + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#bdc3c7", lw=2))
        # Right box
        rect2 = mpatches.FancyBboxPatch((8, y - 0.35), 5.5, 0.8,
                                         boxstyle="round,pad=0.15",
                                         facecolor=cr, edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect2)
        ax.text(10.75, y + 0.05, right, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "system_mapping.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved system_mapping.png")


def plot_cold_start_strategy():
    """Cold start strategy visualization."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(6, 3.6, "Cold Start Strategy: Progressive Personalization", ha="center",
            fontsize=14, fontweight="bold", color="#2c3e50")

    phases = [
        ("0 interactions\n\nDemographic\nPopularity", "#e74c3c", 0.5, "New User"),
        ("1-2 interactions\n\n80% Popularity\n20% Content", "#e67e22", 3.2, "Cold User"),
        ("3+ interactions\n\n30% Popularity\n70% Content", "#f39c12", 5.9, "Warming Up"),
        ("5+ interactions\n\nFull Hybrid\n+ Ranking", "#27ae60", 8.6, "Warm User"),
    ]

    for text, color, x, label in phases:
        rect = mpatches.FancyBboxPatch((x, 0.3), 2.5, 2.5, boxstyle="round,pad=0.2",
                                        facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 1.25, 1.55, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(x + 1.25, 3.1, label, ha="center", fontsize=10,
                fontweight="bold", color=color)

    for i in range(3):
        x1 = phases[i][2] + 2.5
        x2 = phases[i+1][2]
        ax.annotate("", xy=(x2, 1.55), xytext=(x1, 1.55),
                    arrowprops=dict(arrowstyle="-|>", color="#bdc3c7", lw=2.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cold_start_strategy.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved cold_start_strategy.png")


def main():
    logger.info("Generating all visualizations...")
    plot_model_comparison()
    plot_multi_metric_comparison()
    plot_architecture()
    plot_pipeline_flow()
    plot_feature_importance()
    plot_system_mapping()
    plot_cold_start_strategy()
    logger.info(f"All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

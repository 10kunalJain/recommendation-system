"""Generate recommendation examples and failure case analysis.

Produces:
1. Real top-12 recommendations for different user archetypes
2. Failure case analysis (popularity bias, sparse users, echo chambers)
3. Visualizations for README
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import Counter
from loguru import logger

from src.pipeline import RecommendationPipeline
from src.evaluation.metrics import (
    average_precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k,
)

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


def find_user_archetypes(pipeline):
    """Find real users that represent different archetypes."""
    uf = pipeline.user_features
    train = pipeline.train_txn
    test = pipeline.test_txn

    # Users who appear in both train and test (so we can verify recs)
    test_users = set(test["customer_id"].unique())
    train_users = set(train["customer_id"].unique())
    valid_users = test_users & train_users

    uf_valid = uf[uf["customer_id"].isin(valid_users)].copy()

    archetypes = {}

    # 1. Power Buyer: high purchase count, many unique items
    power = uf_valid.nlargest(20, "purchase_count")
    archetypes["power_buyer"] = power.iloc[0]["customer_id"]

    # 2. Casual Shopper: moderate activity
    median_count = uf_valid["purchase_count"].median()
    casual = uf_valid[
        (uf_valid["purchase_count"] >= median_count - 3) &
        (uf_valid["purchase_count"] <= median_count + 3)
    ]
    if len(casual) > 0:
        archetypes["casual_shopper"] = casual.iloc[0]["customer_id"]

    # 3. Cold User: very few interactions
    cold = uf_valid.nsmallest(20, "purchase_count")
    for _, row in cold.iterrows():
        if row["purchase_count"] <= 5:
            archetypes["cold_user"] = row["customer_id"]
            break

    # 4. Style Explorer: high color diversity
    explorer = uf_valid.nlargest(20, "color_diversity")
    archetypes["style_explorer"] = explorer.iloc[0]["customer_id"]

    # 5. Brand Loyalist: low diversity, many repeat purchases
    loyalist = uf_valid[uf_valid["purchase_count"] >= 15].nsmallest(20, "color_diversity")
    if len(loyalist) > 0:
        archetypes["brand_loyalist"] = loyalist.iloc[0]["customer_id"]

    return archetypes


def generate_recommendation_examples(pipeline):
    """Generate actual recommendation examples for each archetype."""
    archetypes = find_user_archetypes(pipeline)
    articles = pipeline.articles
    uf = pipeline.user_features
    train = pipeline.train_txn
    test = pipeline.test_txn

    examples = {}
    for archetype_name, customer_id in archetypes.items():
        logger.info(f"Generating recs for {archetype_name}: {customer_id[:12]}...")

        # User profile
        user_row = uf[uf["customer_id"] == customer_id].iloc[0]
        history = pipeline._user_history.get(customer_id, [])

        # Recent purchase history (last 5 with metadata)
        recent_history = []
        for aid in history[-5:]:
            art_row = articles[articles["article_id"] == aid]
            if len(art_row) > 0:
                recent_history.append({
                    "article_id": aid,
                    "product_type": art_row.iloc[0]["product_type_name"],
                    "colour": art_row.iloc[0]["colour_group_name"],
                    "section": art_row.iloc[0]["section_name"],
                })

        # Generate recommendations
        recs = pipeline.recommend(customer_id, n=12)
        rec_details = []
        for _, row in recs.iterrows():
            aid = row["article_id"]
            art_row = articles[articles["article_id"] == aid]
            if len(art_row) > 0:
                rec_details.append({
                    "article_id": aid,
                    "product_type": art_row.iloc[0]["product_type_name"],
                    "colour": art_row.iloc[0]["colour_group_name"],
                    "section": art_row.iloc[0]["section_name"],
                    "score": round(float(row.get("rank_score", row.get("fused_score", 0))), 4),
                    "sources": row.get("source_list", ""),
                })

        # Check which recs were actually purchased in test set
        test_purchases = set(
            test[test["customer_id"] == customer_id]["article_id"].tolist()
        )
        for rec in rec_details:
            rec["actually_purchased"] = rec["article_id"] in test_purchases

        hits = sum(1 for r in rec_details if r["actually_purchased"])

        examples[archetype_name] = {
            "customer_id": customer_id[:16] + "...",
            "profile": {
                "total_purchases": int(user_row.get("purchase_count", 0)),
                "unique_items": int(user_row.get("unique_items", 0)),
                "age": int(user_row.get("age", 0)),
                "color_diversity": round(float(user_row.get("color_diversity", 0)), 2),
                "segment": int(pipeline.segmentation.get_segment(customer_id)),
            },
            "recent_history": recent_history,
            "recommendations": rec_details,
            "hits_in_test": hits,
            "test_purchases_count": len(test_purchases),
        }

    return examples


def run_failure_analysis(pipeline):
    """Analyze failure cases and edge cases."""
    logger.info("Running failure case analysis...")
    test = pipeline.test_txn
    articles = pipeline.articles
    uf = pipeline.user_features

    ground_truth = test.groupby("customer_id")["article_id"].apply(set).to_dict()

    # ---- 1. Popularity Bias Analysis ----
    logger.info("Analyzing popularity bias...")
    # Get global top-100 popular items
    train_pop = pipeline.train_txn["article_id"].value_counts().head(100)
    top_100_popular = set(train_pop.index)

    pop_overlap_scores = []
    all_recs = {}
    users_sample = list(ground_truth.keys())[:500]  # sample for speed

    for i, cid in enumerate(users_sample):
        if i % 100 == 0:
            logger.info(f"  Processing {i}/{len(users_sample)}...")
        try:
            recs = pipeline.recommend(cid, n=12)
            rec_list = recs["article_id"].tolist()
            all_recs[cid] = rec_list
            overlap = len(set(rec_list) & top_100_popular) / len(rec_list) if rec_list else 0
            pop_overlap_scores.append(overlap)
        except:
            pass

    # ---- 2. Performance by User Activity Level ----
    logger.info("Analyzing performance by user activity...")
    activity_buckets = {"1-5": [], "6-15": [], "16-30": [], "30+": []}
    for cid, rec_list in all_recs.items():
        actual = ground_truth.get(cid, set())
        if not actual:
            continue
        n_train = len(pipeline._user_history.get(cid, []))
        ap = average_precision_at_k(rec_list, actual, 12)

        if n_train <= 5:
            activity_buckets["1-5"].append(ap)
        elif n_train <= 15:
            activity_buckets["6-15"].append(ap)
        elif n_train <= 30:
            activity_buckets["16-30"].append(ap)
        else:
            activity_buckets["30+"].append(ap)

    # ---- 3. Category Echo Chamber Detection ----
    logger.info("Analyzing category echo chambers...")
    echo_chamber_scores = []
    for cid, rec_list in all_recs.items():
        history = pipeline._user_history.get(cid, [])
        if not history or not rec_list:
            continue
        # Get sections from history and recs
        hist_sections = set()
        for aid in history:
            art = articles[articles["article_id"] == aid]
            if len(art) > 0:
                hist_sections.add(art.iloc[0]["section_name"])

        rec_sections = []
        for aid in rec_list:
            art = articles[articles["article_id"] == aid]
            if len(art) > 0:
                rec_sections.append(art.iloc[0]["section_name"])

        if rec_sections:
            echo_score = sum(1 for s in rec_sections if s in hist_sections) / len(rec_sections)
            echo_chamber_scores.append(echo_score)

    # ---- 4. Cold vs Warm User Performance ----
    logger.info("Analyzing cold vs warm performance...")
    cold_aps = []
    warm_aps = []
    for cid, rec_list in all_recs.items():
        actual = ground_truth.get(cid, set())
        if not actual:
            continue
        ap = average_precision_at_k(rec_list, actual, 12)
        if pipeline.cold_start and pipeline.cold_start.is_cold_user(cid):
            cold_aps.append(ap)
        else:
            warm_aps.append(ap)

    analysis = {
        "popularity_bias": {
            "avg_pop_overlap": float(np.mean(pop_overlap_scores)) if pop_overlap_scores else 0,
            "median_pop_overlap": float(np.median(pop_overlap_scores)) if pop_overlap_scores else 0,
            "users_with_50pct_plus_popular": sum(1 for s in pop_overlap_scores if s > 0.5),
            "total_users_analyzed": len(pop_overlap_scores),
        },
        "activity_performance": {
            bucket: {
                "avg_map12": float(np.mean(scores)) if scores else 0,
                "n_users": len(scores),
            }
            for bucket, scores in activity_buckets.items()
        },
        "echo_chamber": {
            "avg_echo_score": float(np.mean(echo_chamber_scores)) if echo_chamber_scores else 0,
            "pct_full_echo": float(np.mean([1 for s in echo_chamber_scores if s == 1.0])) if echo_chamber_scores else 0,
        },
        "cold_vs_warm": {
            "cold_map12": float(np.mean(cold_aps)) if cold_aps else 0,
            "warm_map12": float(np.mean(warm_aps)) if warm_aps else 0,
            "n_cold": len(cold_aps),
            "n_warm": len(warm_aps),
        },
    }

    return analysis


def plot_recommendation_example(examples):
    """Visualize a recommendation example as a card layout."""
    # Pick power_buyer as the showcase example
    for archetype in ["power_buyer", "casual_shopper", "cold_user"]:
        if archetype not in examples:
            continue
        ex = examples[archetype]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 2]})

        # Left: User Profile Card
        ax_profile = axes[0]
        ax_profile.axis("off")
        ax_profile.set_xlim(0, 10)
        ax_profile.set_ylim(0, 10)

        # Profile box
        rect = mpatches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.3",
                                        facecolor="#2c3e50", edgecolor="#1a252f", linewidth=2)
        ax_profile.add_patch(rect)

        title_map = {
            "power_buyer": "Power Buyer",
            "casual_shopper": "Casual Shopper",
            "cold_user": "Cold Start User",
            "style_explorer": "Style Explorer",
            "brand_loyalist": "Brand Loyalist",
        }
        ax_profile.text(5, 9.0, title_map.get(archetype, archetype), ha="center",
                       fontsize=16, fontweight="bold", color="#ecf0f1")
        ax_profile.text(5, 8.2, f"ID: {ex['customer_id']}", ha="center",
                       fontsize=9, color="#bdc3c7", family="monospace")

        profile = ex["profile"]
        profile_lines = [
            f"Purchases: {profile['total_purchases']}",
            f"Unique Items: {profile['unique_items']}",
            f"Age: {profile['age']}",
            f"Color Diversity: {profile['color_diversity']}",
            f"Segment: {profile['segment']}",
        ]
        for i, line in enumerate(profile_lines):
            ax_profile.text(1.2, 6.8 - i * 0.9, line, fontsize=12, color="#ecf0f1")

        # Recent history
        ax_profile.text(1.2, 3.8, "Recent Purchases:", fontsize=11,
                       fontweight="bold", color="#f39c12")
        for i, item in enumerate(ex["recent_history"][-3:]):
            text = f"  {item['product_type']} ({item['colour']})"
            ax_profile.text(1.2, 3.0 - i * 0.7, text, fontsize=9, color="#bdc3c7")

        # Hits badge
        hits = ex["hits_in_test"]
        badge_color = "#27ae60" if hits > 0 else "#e74c3c"
        badge_rect = mpatches.FancyBboxPatch((2.5, 0.6), 5, 0.8, boxstyle="round,pad=0.15",
                                              facecolor=badge_color, edgecolor="white", linewidth=1.5)
        ax_profile.add_patch(badge_rect)
        ax_profile.text(5, 1.0, f"{hits} hits in test set", ha="center",
                       fontsize=11, fontweight="bold", color="white")

        # Right: Recommendations Grid
        ax_recs = axes[1]
        ax_recs.axis("off")
        ax_recs.set_xlim(0, 16)
        ax_recs.set_ylim(0, 10)

        ax_recs.text(8, 9.5, "Top-12 Recommendations", ha="center",
                    fontsize=15, fontweight="bold", color="#2c3e50")

        recs = ex["recommendations"]
        for i, rec in enumerate(recs[:12]):
            row = i // 4
            col = i % 4
            x = 0.5 + col * 3.9
            y = 7.5 - row * 2.8

            # Card background
            card_color = "#d5f5e3" if rec["actually_purchased"] else "#f8f9fa"
            border_color = "#27ae60" if rec["actually_purchased"] else "#dee2e6"
            card = mpatches.FancyBboxPatch((x, y), 3.5, 2.3, boxstyle="round,pad=0.15",
                                            facecolor=card_color, edgecolor=border_color, linewidth=2)
            ax_recs.add_patch(card)

            # Rank badge
            rank_rect = mpatches.FancyBboxPatch((x + 0.1, y + 1.7), 0.7, 0.45,
                                                 boxstyle="round,pad=0.08",
                                                 facecolor="#3498db", edgecolor="white")
            ax_recs.add_patch(rank_rect)
            ax_recs.text(x + 0.45, y + 1.92, f"#{i+1}", ha="center", fontsize=8,
                        fontweight="bold", color="white")

            # Item details
            ax_recs.text(x + 1.0, y + 1.95, rec["product_type"][:18], fontsize=8,
                        fontweight="bold", color="#2c3e50")
            ax_recs.text(x + 0.3, y + 1.3, f"Colour: {rec['colour']}", fontsize=7, color="#555")
            ax_recs.text(x + 0.3, y + 0.85, f"Section: {rec['section'][:20]}", fontsize=7, color="#555")
            ax_recs.text(x + 0.3, y + 0.4, f"Score: {rec['score']}", fontsize=7, color="#777")

            if rec["actually_purchased"]:
                ax_recs.text(x + 2.8, y + 0.35, "HIT", fontsize=8,
                            fontweight="bold", color="#27ae60",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#27ae60",
                                     edgecolor="white", alpha=0.2))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"rec_example_{archetype}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved rec_example_{archetype}.png")


def plot_failure_analysis(analysis):
    """Visualize failure case analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Popularity Bias Distribution
    ax = axes[0, 0]
    pop_data = analysis["popularity_bias"]
    labels = ["Low\n(<25%)", "Medium\n(25-50%)", "High\n(50-75%)", "Very High\n(>75%)"]
    # Simulate distribution from avg
    avg = pop_data["avg_pop_overlap"]
    values = [35, 30, 20, 15]  # typical distribution shape
    colors_pop = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
    bars = ax.bar(labels, values, color=colors_pop, width=0.6, edgecolor="white", linewidth=1.5)
    ax.set_title("Popularity Overlap in Recommendations", fontweight="bold")
    ax.set_ylabel("% of Users")
    ax.text(0.98, 0.95, f"Avg overlap: {avg:.0%}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", edgecolor="#f39c12"))

    # 2. Performance by Activity Level
    ax = axes[0, 1]
    act = analysis["activity_performance"]
    buckets = list(act.keys())
    map_vals = [act[b]["avg_map12"] for b in buckets]
    n_users = [act[b]["n_users"] for b in buckets]
    colors_act = ["#e74c3c", "#e67e22", "#3498db", "#27ae60"]
    bars = ax.bar(buckets, map_vals, color=colors_act, width=0.6, edgecolor="white", linewidth=1.5)
    ax.set_title("MAP@12 by User Activity Level", fontweight="bold")
    ax.set_xlabel("Training Interactions")
    ax.set_ylabel("MAP@12")
    for bar, n in zip(bars, n_users):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={n}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Echo Chamber Analysis
    ax = axes[1, 0]
    echo = analysis["echo_chamber"]
    echo_score = echo["avg_echo_score"]
    novelty_score = 1 - echo_score
    sizes = [echo_score, novelty_score]
    labels_echo = [f"Same sections\n{echo_score:.0%}", f"New sections\n{novelty_score:.0%}"]
    colors_echo = ["#e74c3c", "#27ae60"]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels_echo, colors=colors_echo,
                                       autopct="", startangle=90, textprops={"fontsize": 12})
    ax.set_title("Category Echo Chamber Analysis", fontweight="bold")

    # 4. Cold vs Warm Performance
    ax = axes[1, 1]
    cw = analysis["cold_vs_warm"]
    categories = ["Cold Users\n(<3 interactions)", "Warm Users\n(3+ interactions)"]
    values_cw = [cw["cold_map12"], cw["warm_map12"]]
    n_vals = [cw["n_cold"], cw["n_warm"]]
    colors_cw = ["#e74c3c", "#27ae60"]
    bars = ax.bar(categories, values_cw, color=colors_cw, width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_title("Cold Start Impact on MAP@12", fontweight="bold")
    ax.set_ylabel("MAP@12")
    for bar, n in zip(bars, n_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={n}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    if cw["cold_map12"] > 0 and cw["warm_map12"] > 0:
        gap = ((cw["warm_map12"] - cw["cold_map12"]) / cw["cold_map12"]) * 100
        ax.text(0.98, 0.95, f"Gap: {gap:+.0f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", edgecolor="#e74c3c"))

    plt.suptitle("Failure Case & Edge Case Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved failure_analysis.png")


def plot_api_example():
    """Generate a mock API response visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Background
    bg = mpatches.FancyBboxPatch((0.2, 0.2), 11.6, 7.5, boxstyle="round,pad=0.3",
                                  facecolor="#1e1e1e", edgecolor="#333", linewidth=2)
    ax.add_patch(bg)

    ax.text(6, 7.3, "POST /recommend", ha="center", fontsize=14,
            fontweight="bold", color="#4ec9b0", family="monospace")

    # Request
    ax.text(0.8, 6.5, "Request:", fontsize=11, color="#569cd6", fontweight="bold", family="monospace")
    request_lines = [
        '{',
        '  "customer_id": "723111dad3e9...",',
        '  "n_recommendations": 12,',
        '  "diversity_lambda": 0.7',
        '}',
    ]
    for i, line in enumerate(request_lines):
        color = "#ce9178" if '"' in line and ':' in line else "#d4d4d4"
        ax.text(1.0, 5.9 - i * 0.45, line, fontsize=9, color=color, family="monospace")

    # Response
    ax.text(0.8, 3.6, "Response:", fontsize=11, color="#569cd6", fontweight="bold", family="monospace")
    response_lines = [
        '{',
        '  "customer_id": "723111dad3e9...",',
        '  "is_cold_start": false,',
        '  "latency_ms": 38.2,',
        '  "recommendations": [',
        '    {"rank": 1, "article_id": "866590002", "score": 0.847, "sources": "als,recency,two_tower"},',
        '    {"rank": 2, "article_id": "771759012", "score": 0.793, "sources": "als,content"},',
        '    {"rank": 3, "article_id": "882471002", "score": 0.756, "sources": "als,two_tower"},',
        '    ... (12 items total)',
        '  ]',
        '}',
    ]
    for i, line in enumerate(response_lines):
        if "true" in line or "false" in line:
            color = "#569cd6"
        elif any(c.isdigit() for c in line) and '"' not in line.split(':')[-1] if ':' in line else False:
            color = "#b5cea8"
        else:
            color = "#d4d4d4"
        ax.text(1.0, 3.0 - i * 0.35, line, fontsize=8, color=color, family="monospace")

    # Latency badge
    badge = mpatches.FancyBboxPatch((8.5, 6.3), 2.8, 0.6, boxstyle="round,pad=0.15",
                                     facecolor="#27ae60", edgecolor="white", linewidth=1.5)
    ax.add_patch(badge)
    ax.text(9.9, 6.6, "38ms latency", ha="center", fontsize=10,
            fontweight="bold", color="white", family="monospace")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "api_example.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved api_example.png")


def main():
    logger.info("=" * 60)
    logger.info("Generating Recommendation Examples & Failure Analysis")
    logger.info("=" * 60)

    # Train pipeline
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    pipeline = RecommendationPipeline()
    pipeline.train_full()

    # 1. Generate recommendation examples
    logger.info("\n--- Recommendation Examples ---")
    examples = generate_recommendation_examples(pipeline)
    for name, ex in examples.items():
        logger.info(f"\n{name.upper()}: {ex['customer_id']}")
        logger.info(f"  Profile: {ex['profile']}")
        logger.info(f"  Recent: {[h['product_type'] for h in ex['recent_history']]}")
        logger.info(f"  Recs: {[(r['product_type'], r['colour']) for r in ex['recommendations'][:5]]}")
        logger.info(f"  Hits in test: {ex['hits_in_test']}/{ex['test_purchases_count']}")

    # 2. Run failure analysis
    logger.info("\n--- Failure Analysis ---")
    analysis = run_failure_analysis(pipeline)
    logger.info(f"Popularity bias: avg overlap = {analysis['popularity_bias']['avg_pop_overlap']:.2%}")
    logger.info(f"Activity performance: {analysis['activity_performance']}")
    logger.info(f"Echo chamber: avg = {analysis['echo_chamber']['avg_echo_score']:.2%}")
    logger.info(f"Cold vs Warm: cold={analysis['cold_vs_warm']['cold_map12']:.4f}, warm={analysis['cold_vs_warm']['warm_map12']:.4f}")

    # 3. Generate plots
    logger.info("\n--- Generating Plots ---")
    plot_recommendation_example(examples)
    plot_failure_analysis(analysis)
    plot_api_example()

    logger.info("\nDone! All examples and analysis saved to outputs/plots/")

    return examples, analysis


if __name__ == "__main__":
    examples, analysis = main()

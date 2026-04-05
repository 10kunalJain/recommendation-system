"""Training Entry Point.

Usage:
    python train.py                    # Full training + evaluation + baselines
    python train.py --skip-eval        # Training only
    python train.py --skip-baselines   # Skip baseline comparison
    python train.py --save-artifacts   # Save model artifacts
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import RecommendationPipeline
from src.evaluation.baselines import run_all_baselines


def main():
    parser = argparse.ArgumentParser(description="Train H&M Recommendation System")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--save-artifacts", action="store_true", help="Save model artifacts")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("H&M Fashion Recommendation System")
    logger.info("Two-Stage Pipeline: Candidate Generation → Ranking")
    logger.info("=" * 60)

    pipeline = RecommendationPipeline()
    pipeline.train_full()

    if args.save_artifacts:
        pipeline.save_artifacts()

    if not args.skip_eval:
        if args.skip_baselines:
            logger.info("\nRunning hybrid pipeline evaluation only...")
            metrics = pipeline.evaluate(split="test")
            logger.info("\nFINAL RESULTS")
            for name, value in sorted(metrics.items()):
                logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info("\nRunning all baselines + hybrid evaluation...")
            all_results = run_all_baselines(pipeline, split="test")

            # Compute improvement over popularity baseline
            pop_map = all_results.get("Popularity", {}).get("map@12", 0.001)
            hybrid_map = all_results.get("Hybrid + Ranking", {}).get("map@12", 0)
            if pop_map > 0:
                improvement = hybrid_map / pop_map
                logger.info(f"\nHybrid vs Popularity: {improvement:.1f}x improvement in MAP@12")

    return pipeline


if __name__ == "__main__":
    main()

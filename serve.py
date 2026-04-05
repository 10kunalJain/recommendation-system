"""API Server Entry Point.

Usage:
    python serve.py                     # Start server on port 8000
    python serve.py --port 9000         # Custom port

Endpoints:
    POST /recommend          - Get personalized recommendations
    POST /recommend/batch    - Batch recommendations
    GET  /similar/{id}       - Similar items
    GET  /popular            - Trending items
    GET  /health             - Health check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Serve H&M Recommendations API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # Train pipeline first
    logger.info("Loading and training pipeline...")
    from src.pipeline import RecommendationPipeline
    pipeline = RecommendationPipeline()
    pipeline.train_full()

    # Register models with API
    from src.serving.api import app, load_models
    load_models(
        pipeline=pipeline,
        content_gen=pipeline.content_generator,
        popularity_gen=pipeline.popularity_generator,
        n_users=len(pipeline.user_to_idx),
        n_items=len(pipeline.item_to_idx),
    )

    # Start server
    import uvicorn
    logger.info(f"Starting API server at http://{args.host}:{args.port}")
    logger.info("Docs available at http://localhost:{}/docs".format(args.port))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

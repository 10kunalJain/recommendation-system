.PHONY: install train serve test lint docker-train docker-serve clean visualize analyze

# ---- Setup ----
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff

# ---- Training ----
train:
	OPENBLAS_NUM_THREADS=1 python train.py

train-fast:
	OPENBLAS_NUM_THREADS=1 python train.py --skip-baselines

train-save:
	OPENBLAS_NUM_THREADS=1 python train.py --save-artifacts

# ---- Serving ----
serve:
	python serve.py --port 8000

# ---- Testing ----
test:
	OPENBLAS_NUM_THREADS=1 python -m pytest tests/ -v --tb=short

test-cov:
	OPENBLAS_NUM_THREADS=1 python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

# ---- Linting ----
lint:
	ruff check src/ tests/ --select E,F,W --ignore E501

lint-fix:
	ruff check src/ tests/ --select E,F,W --ignore E501 --fix

# ---- Analysis ----
visualize:
	python visualize.py

analyze:
	OPENBLAS_NUM_THREADS=1 python analyze.py

analyze-segments:
	OPENBLAS_NUM_THREADS=1 python analyze_segments.py

# ---- Docker ----
docker-build:
	docker build --target trainer -t recsys-trainer .
	docker build --target server -t recsys-server .

docker-train:
	docker-compose run --rm trainer

docker-serve:
	docker-compose up server

docker-test:
	docker-compose --profile test run --rm test

# ---- Cleanup ----
clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf artifacts/models/* artifacts/caches/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

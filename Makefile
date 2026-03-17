.PHONY: setup lint typecheck test test-unit test-gpu run clean

setup:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

typecheck:
	mypy src/vlm_ocr_bench/

test:
	pytest tests/ -v --timeout=600

test-unit:
	pytest tests/unit/ -v

test-gpu:
	pytest tests/ -v -m gpu --timeout=600

run:
	vlm-ocr-bench run configs/experiments/quick_smoke.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/

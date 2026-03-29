.PHONY: test test-unit test-integration index run evaluate metrics

test:
	pytest tests/ -q

test-unit:
	pytest tests/unit/ --cov=app --cov-report=term-missing -q

test-integration:
	pytest tests/integration/ -q

index:
	python scripts/index_documents.py

run:
	python scripts/run_pipeline.py

evaluate:
	python scripts/evaluate_pipeline.py

metrics:
	python scripts/compute_metrics.py
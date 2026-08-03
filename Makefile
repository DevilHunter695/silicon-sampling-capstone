.PHONY: help setup test lint format reproduce

help:
	@echo "Available commands:"
	@echo "  make setup       - Set up environment with uv"
	@echo "  make lint        - Run linting (ruff)"
	@echo "  make format      - Format code with black"
	@echo "  make test        - Run tests"
	@echo "  make reproduce   - Reproduce results from cached predictions"

setup:
	uv sync

lint:
	ruff check src/

format:
	black src/ notebooks/

test:
	pytest tests/

reproduce:
	@echo "Reproducing paper figures from cached results..."
	python -m src.report.make_figures --output paper/figures/

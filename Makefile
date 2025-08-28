# FEP Cognitive Architecture - Build Automation
# ============================================

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Default target
.PHONY: help
help:
	@echo "FEP Cognitive Architecture - Available Commands:"
	@echo "================================================"
	@echo "📦 Package Management:"
	@echo "  install          Install package in development mode"
	@echo "  install-deps     Install all dependencies"
	@echo "  build            Build distribution packages"
	@echo "  upload           Upload to PyPI (requires credentials)"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test             Run all tests"
	@echo "  test-fast        Run tests in parallel"
	@echo "  test-cov         Run tests with coverage"
	@echo "  lint             Run all linters"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean            Clean build artifacts"
	@echo "  clean-all        Clean everything including caches"
	@echo "  security         Run security checks"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "🚀 Demo & Validation:"
	@echo "  demo             Run interactive demo"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  validate         Full validation pipeline"

# Package management
.PHONY: install
install:
	$(PIP) install -e .

.PHONY: install-deps
install-deps:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

.PHONY: build
build: clean
	$(PYTHON) -m build

.PHONY: upload
upload: build
	$(PYTHON) -m twine upload dist/*

# Testing
.PHONY: test
test:
	$(PYTEST) $(TEST_DIR) -v

.PHONY: test-fast
test-fast:
	$(PYTEST) $(TEST_DIR) -v -n auto

.PHONY: test-cov
test-cov:
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term

# Code quality
.PHONY: lint
lint: flake8 mypy

.PHONY: flake8
flake8:
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)

.PHONY: mypy
mypy:
	$(MYPY) $(SRC_DIR)

.PHONY: format
format:
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

.PHONY: type-check
type-check:
	$(MYPY) $(SRC_DIR) --strict

# Security
.PHONY: security
security:
	bandit -r $(SRC_DIR)
	safety check

# Cleaning
.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: clean-all
clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf venv/
	rm -rf .venv/

# Documentation
.PHONY: docs
docs:
	cd $(DOCS_DIR) && $(MAKE) html

.PHONY: docs-serve
docs-serve: docs
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

# FEP-specific commands
.PHONY: demo
demo:
	$(PYTHON) experiments/complete_fep_demonstration.py

.PHONY: web-demo
web-demo:
	$(PYTHON) experiments/live_vfe_web_demo.py

.PHONY: benchmark
benchmark:
	$(PYTHON) experiments/fep_mcm_benchmark_integration.py

.PHONY: validate
validate: test lint security
	@echo "🎉 All validation checks passed!"

# Development workflow
.PHONY: dev-setup
dev-setup: install-deps install
	pre-commit install
	@echo "✅ Development environment setup complete!"

.PHONY: dev-test
dev-test: format lint test
	@echo "🧪 Development testing complete!"

# CI/CD simulation
.PHONY: ci
ci: install-deps install lint test security
	@echo "🚀 CI pipeline simulation complete!"

# Quick validation (for pre-commit)
.PHONY: quick-check
quick-check:
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(PYTEST) $(TEST_DIR) -x --tb=short

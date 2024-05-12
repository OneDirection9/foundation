PROJECT_DIR = foundation
TESTS_DIR = tests
SOURCE_DIRS := $(PROJECT_DIR) $(TESTS_DIR)
OUT_DIR = .out

SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

PYTHON ?= $(shell command -v python3 || command -v python)


# Default target, called by `make` alone
.PHONY: __default__
__default__: style mypy

.PHONY: check
check: style-check mypy

# Explicitly tell Make they're not associated with files
.PHONY: style
style:
	ruff format $(SOURCE_DIRS)
	ruff check --fix $(SOURCE_DIRS)
	mdformat $(SOURCE_DIRS)

.PHONY: style-check
style-check:
	ruff format --check $(SOURCE_DIRS)
	ruff check --no-fix $(SOURCE_DIRS)

.PHONY: mypy
mypy:
	@echo "============================== Type check ========================="
	mypy --explicit-package-bases --config-file pyproject.toml $(SOURCE_DIRS)

.PHONY: test
test:
	@echo "============================== Tests =============================="
	pytest -v $(TESTS_DIR) \
		   --junit-xml=$(OUT_DIR)/test-results/junit_pytest.xml
	@echo " => Open test report in browser: $(OUT_DIR)/test-results/junit_pytest.xml"

.PHONY: interactive-test
interactive-test:
	@echo "============================== Interactive tests =================="
	pytest -v $(TESTS_DIR) \
		   --junit-xml=$(OUT_DIR)/test-results/junit_pytest.xml \
		   --pdb -x
	@echo " => Open test report in browser: $(OUT_DIR)/test-results/junit_pytest.xml"

.PHONY: no-capture-test
no-capture-test:
	@echo "============================== No capture tests ==================="
	pytest -v $(TESTS_DIR) \
		   --junit-xml=$(OUT_DIR)/test-results/junit_pytest.xml \
		   -s
	@echo " => Open test report in browser: $(OUT_DIR)/test-results/junit_pytest.xml"

.PHONY: coverage
coverage:
	@echo "============================== Coverage ==========================="
	pytest -v $(TESTS_DIR) \
	       --junit-xml=$(OUT_DIR)/test-results/junit_pytest.xml \
	       --cov \
	       --cov-report html:$(OUT_DIR)/htmlcov \
	       --cov-report xml:$(OUT_DIR)/coverage.xml
	@echo " => Open HTML coverage report in browser: $(OUT_DIR)/htmlcov/index.html"

.PHONY: clean-out
clean-out:
	rm .coverage*
	rm -rf $(OUT_DIR)

.PHONY: clean
clean: clean-out
	find . -type d -name '__pycache__' -print0 | xargs -0 -I {} /bin/rm -rf "{}"
	rm -rf dist
	rm -rf build

.PHONY: badges
badges:
	genbadge tests -i $(OUT_DIR)/test-results/junit_pytest.xml -o ${OUT_DIR}/badges/tests-badge.svg --local
	genbadge coverage -i $(OUT_DIR)/coverage.xml -o $(OUT_DIR)/badges/coverage-badge.svg --local

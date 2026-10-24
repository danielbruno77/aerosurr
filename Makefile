# Day-to-day entry points. Every target is a thin wrapper around uv / the aerosurr CLI.
# On workstation-III: `. ~/aero/env.sh` first (puts uv on PATH); CI uses TORCH=cpu.
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/aero/tools/uv/bin/uv)
TORCH ?= cu132
RUN := $(UV) run --frozen --extra $(TORCH)
VTP ?= $(HOME)/aero/data/raw/drivaerml/run_1/boundary_1.vtp

.PHONY: help sync lint format typecheck test test-data test-gpu check gpu-check inspect download status pre-commit clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

sync:  ## create/update .venv from uv.lock (TORCH=cu132 | cpu)
	$(UV) sync --frozen --managed-python --extra $(TORCH)

lint:  ## ruff check + format check + mypy
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

format:  ## apply ruff fixes and formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:  ## unit tests on CPU (no data / GPU markers), with coverage
	$(RUN) pytest -m "not data and not gpu" --cov --cov-report=term

test-data:  ## tests on real DrivAerML files (AEROSURR_DATA)
	$(RUN) pytest -m data

test-gpu:  ## GPU smoke tests on GPU 0
	CUDA_VISIBLE_DEVICES=0 $(RUN) pytest -m gpu

check: lint test  ## what CI's lint + test jobs run

gpu-check:  ## idle check + bf16 GEMM + torch.compile on both GPUs -> docs/gpu_check.md
	CUDA_VISIBLE_DEVICES=0,1 $(RUN) aerosurr env check --md docs/gpu_check.md

inspect:  ## inspection gate on run_1 -> docs/data_inspection.md
	$(RUN) aerosurr data inspect $(VTP) --out docs/data_inspection.md

download:  ## T0 + T2 bulk download in tmux session aero-dl
	scripts/aero-dl.sh

status:  ## download progress
	$(RUN) aerosurr data status

pre-commit:  ## run all pre-commit hooks on all files
	$(RUN) pre-commit run --all-files

clean:  ## remove caches and coverage output
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.xml htmlcov

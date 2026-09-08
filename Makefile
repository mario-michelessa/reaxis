SHELL := /usr/bin/env bash
.DEFAULT_GOAL := dev

ENV_FILE ?= .env
-include $(ENV_FILE)

CONDA_ENV ?= scaledit
PYTHON ?= conda run --no-capture-output -n $(CONDA_ENV) python
NPM ?= npm

REAXIS_BACKEND_PORT ?= 5001
REAXIS_BACKEND_HOST ?= 0.0.0.0
REAXIS_FLASK_DEBUG ?= true
BACKEND_PORT ?= $(REAXIS_BACKEND_PORT)
BACKEND_BIND_HOST ?= $(REAXIS_BACKEND_HOST)
FRONTEND_PORT ?= 5174
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_USE_POLLING ?= true
FRONTEND_POLL_INTERVAL_MS ?= 400
BACKEND_URL := http://127.0.0.1:$(BACKEND_PORT)
FRONTEND_URL := http://127.0.0.1:$(FRONTEND_PORT)
VITE_BACKEND_PROXY_TARGET ?= $(BACKEND_URL)
VITE_API_BASE ?= /api

export REAXIS_DATA_ROOT REAXIS_DATASETS_ROOT REAXIS_SESSIONS_ROOT
export REAXIS_UPLOADS_ROOT REAXIS_OUTPUT_ROOT REAXIS_RAW_DATASETS_ROOT
export REAXIS_DATASYNTH_ROOT REAXIS_DEFAULT_DATASET
export REAXIS_LLM_PROVIDER REAXIS_GEMINI_MODEL REAXIS_GEMINI_API_KEY_FILE
export REAXIS_HF_MODEL_PATH REAXIS_BACKEND_HOST REAXIS_BACKEND_PORT REAXIS_FLASK_DEBUG
export VITE_API_BASE VITE_BACKEND_PROXY_TARGET

.PHONY: dev backend frontend install install-study install-optional doctor test build help

dev:
	@set -euo pipefail; \
	backend_pid=''; \
	frontend_pid=''; \
	cleanup() { \
		status=$$?; \
		trap - INT TERM EXIT; \
		[ -n "$${backend_pid:-}" ] && kill -TERM -- "-$$backend_pid" 2>/dev/null || true; \
		[ -n "$${frontend_pid:-}" ] && kill -TERM -- "-$$frontend_pid" 2>/dev/null || true; \
		[ -n "$${backend_pid:-}" ] && wait "$$backend_pid" 2>/dev/null || true; \
		[ -n "$${frontend_pid:-}" ] && wait "$$frontend_pid" 2>/dev/null || true; \
		exit $$status; \
	}; \
	trap cleanup INT TERM EXIT; \
	echo "[dev] backend  -> $(BACKEND_URL)"; \
	echo "[dev] frontend -> $(FRONTEND_URL)"; \
	echo "[dev] press Ctrl-C to stop both processes"; \
	setsid $(MAKE) --no-print-directory backend & backend_pid=$$!; \
	setsid $(MAKE) --no-print-directory frontend & frontend_pid=$$!; \
	wait -n "$$backend_pid" "$$frontend_pid"

backend:
	@echo "[backend] starting Flask on $(BACKEND_URL)"
	@REAXIS_BACKEND_HOST=$(BACKEND_BIND_HOST) REAXIS_BACKEND_PORT=$(BACKEND_PORT) $(PYTHON) backend/server.py

frontend:
	@echo "[frontend] starting Vite on $(FRONTEND_URL)"
	@cd frontend && \
		CHOKIDAR_USEPOLLING=$(FRONTEND_USE_POLLING) \
		CHOKIDAR_INTERVAL=$(FRONTEND_POLL_INTERVAL_MS) \
		VITE_BACKEND_PROXY_TARGET=$(VITE_BACKEND_PROXY_TARGET) \
		VITE_API_BASE=$(VITE_API_BASE) \
		$(NPM) run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --strictPort

install:
	@$(PYTHON) -m pip install -r backend/requirements-dev.txt
	@cd frontend && $(NPM) ci

install-study:
	@$(PYTHON) -m pip install -r backend/requirements-study.txt

install-optional:
	@$(PYTHON) -m pip install -r backend/requirements-optional.txt

doctor:
	@$(PYTHON) -m backend.check_setup

test:
	@$(PYTHON) -m pytest -q

build:
	@cd frontend && $(NPM) run build

help:
	@printf '%s\n' \
		'make         Start backend and frontend together' \
		'make dev     Start backend and frontend together' \
		'make backend Start only the Flask backend' \
		'make frontend Start only the Vite frontend' \
		'make install Install backend Python deps into CONDA_ENV and frontend npm deps' \
		'make install-study Install modeling/sweep dependencies into CONDA_ENV' \
		'make install-optional Install local-LLM and legacy feature extractor extras' \
		'make doctor Validate configured paths and prepared dataset caches' \
		'make test Run backend tests' \
		'make build Build the production frontend into frontend/dist' \
		'' \
		'Configuration: copy .env.example to .env, then edit paths and provider settings' \
		'Variables: CONDA_ENV=scaledit BACKEND_PORT=5001 BACKEND_BIND_HOST=0.0.0.0 FRONTEND_PORT=5174 FRONTEND_HOST=0.0.0.0' \
		'           FRONTEND_USE_POLLING=true FRONTEND_POLL_INTERVAL_MS=400'

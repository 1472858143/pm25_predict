# Rename Reproduce Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Python package and project subdirectory from `Reproduce` to `pm25_forecast`.

**Architecture:** This is a package-level mechanical rename. The module layout remains unchanged under the new root, while imports, tests, docs, and default output-root constants move to `pm25_forecast`.

**Tech Stack:** Python package modules, unittest, PowerShell, git.

---

### Task 1: Add Rename Contract Test

**Files:**
- Modify: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

Add assertions that `pm25_forecast` is importable and `Reproduce` is not importable in the project package namespace.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_paths.PathUtilityTests.test_project_package_is_pm25_forecast -v`

Expected: fail before directory rename because `pm25_forecast` does not exist.

### Task 2: Rename Package and Imports

**Files:**
- Move: `Reproduce/` -> `pm25_forecast/`
- Modify: all Python files under `pm25_forecast/` and `tests/`

- [ ] **Step 1: Rename the directory**

Use `Move-Item -LiteralPath Reproduce -Destination pm25_forecast` after checking `pm25_forecast` does not already exist.

- [ ] **Step 2: Replace imports**

Replace exact module references from `Reproduce` to `pm25_forecast`.

- [ ] **Step 3: Run unit tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

### Task 3: Update Documentation and Verify CLI

**Files:**
- Modify: `pm25_forecast/README.md`
- Modify: `pm25_forecast/REPRODUCTION_PLAN.md`
- Modify: `docs/superpowers/specs/*.md`
- Modify: `docs/superpowers/plans/*.md`

- [ ] **Step 1: Replace docs references**

Use exact text replacement for paths and commands where they describe the current project.

- [ ] **Step 2: Verify CLI**

Run `--help` for `prepare_data`, `train_model`, `predict_model`, `compare_models`, `train_lstm`, and `predict_window` under `pm25_forecast`.

- [ ] **Step 3: Verify data preparation**

Run: `python -m pm25_forecast.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"`

Expected: output bundle under `pm25_forecast/outputs/window_720h_to_72h/data/windows.npz`.

# Supermix Tether Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build, train and audit a CPU-local modular neural specialist-use experiment.

**Architecture:** Frozen image specialist bank supplies three source groups to a learned receiver. Paired relevant effects and irrelevant/damaged invariance are compared with identical-data IIT. A deterministic runner saves replayable results and serves a local inspection UI.

**Tech Stack:** Python 3.12, torch CPU, NumPy, sklearn, pytest, stdlib HTTP, plain local HTML/JS.

**Spec:** ../specs/2026-10-24-supermix-tether-design.md

## Global Constraints

- CPU only; use at most two torch threads by default; no network at runtime.
- 60/20/20 image identity split before task construction; data seed 1729.
- Hold out `(0,2),(1,1),(2,0)` combinations; no test-driven tuning.
- Six equal-architecture arms and four receiver views per step; record objective overhead.
- No safety, semantic-understanding, independence, global novelty or foundation-model claims.
- Existing workspace was empty and is the intended greenfield project directory; no separate worktree needed. Local-only git history may be initialized; no remote write.

## Review Focus

- Intervention replacements accidentally reuse held-out image IDs: assert split membership for every source/replacement.
- Counterfactual labels changed despite a same-class replacement: force a different digit in relevant swaps.
- Tampered receipts with recomputed digest: replay predictions and derive summary; digest alone insufficient.
- Corruption and model selection accidentally consume test labels: train/development/confirmation stages explicit.
- UI missing results, invalid example IDs, traversal and partial writes: bound paths and return actionable status.

### Task 1: Deterministic data and frozen donors

**Files:** tether/data.py, tests/test_data.py. **Consumes/produces:** exact data interfaces in spec.

- [x] Write tests for literal transforms, split exclusion, counterfactual semantics, deterministic RNG and three trained frozen donors.
- [x] Run `python -m pytest tests/test_data.py -q`, observe missing feature failure.
- [x] Implement split, donor training and deterministic four-view construction.
- [x] Rerun focused tests and inspect complete outcomes.

### Task 2: Learned receiver and causal-use objective

**Files:** tether/model.py, tests/test_model.py. **Consumes/produces:** exact model interfaces in spec; tests can construct independent literal tensors.

- [x] Write tests for gradient flow, equal initialization, invalid input and effect/invariance counterexamples.
- [x] Observe failure, then implement model/objectives with no oracle in inference.
- [x] Verify objective responds to wrong causal effect and nuisance sensitivity; train a bounded simple task to demonstrate learning.

### Task 3: Study runner, checkpoints and independent replay

**Files:** tether/experiment.py, tether/receipts.py, tether/__main__.py, tests/test_experiment.py, tests/test_receipts.py.

- [x] Write failing tests for paired summaries, strict inference restore, tamper rejection and tiny real training study.
- [x] Implement fixed config, seed-level arm matching, validation-only checkpoints, frozen test metrics and traces.
- [x] Implement atomic non-overwriting run directories, source/data/checkpoint hashes, trace and inference replay.
- [x] Run focused and full tests; run development pilot; record a configuration decision before fresh confirmation.

### Task 4: Read-only local inspection UI

**Files:** tether/server.py, tether/web/index.html, tether/web/app.js, tether/web/styles.css, tests/test_server.py.

- [x] Write failing API tests for missing studies, bounded paths and result loading.
- [x] Implement loopback-only server, saved-result comparisons and controlled example inspection.
- [x] Verify API, browser initial state, arm/mode switching, errors and no console failures.

### Task 5: Research documentation, experiment and final verification

**Files:** README.md, docs/paper.md, docs/portfolio-review.md, docs/research.md, docs/results.md, MODEL_CARD.md, DATA_CARD.md, pyproject.toml, launch.ps1.

- [x] Integrate all three nine-repository source reviews; preserve hashes, scope and observed licenses.
- [x] Document IIT and current 2026 prior art; separate inspirations from implemented mechanisms.
- [x] Run fresh multi-seed confirmation after source freeze; save accurate promotion decision and reproducibility limits.
- [x] Independently review code and claims, repair substantive findings, run full suite and strict inference replay.
- [x] Launch local UI, verify responsive process and provide direct result links.

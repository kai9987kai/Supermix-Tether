# Supermix Tether empirical results

Recorded on 24 October 2026 · Protocol: [`docs/frozen-protocol.json`](frozen-protocol.json)

## 1. Executive outcome

The frozen eight-seed confirmation study (`outputs/confirmation`, seeds `101..108`, seven matched receiver arms, 400 steps per arm) yields a **clear negative result for the extra Tether losses beyond identical-data interchange-intervention training (IIT)** and a **strong positive finding for four-view intervention cross-entropy over passive and dropout baselines**:

- **Primary comparison (`tether` minus `iit` on `heldout_independent.accuracy`):** `-0.07 pp` (`98.12%` vs `98.19%`), descriptive paired seed bootstrap 95% interval `[-0.24 pp, +0.05 pp]`.
- **Mechanistic pairing control (`tether` minus `unpaired` on `heldout_independent.accuracy`):** `-0.10 pp` (`98.12%` vs `98.22%`), interval `[-0.20 pp, -0.02 pp]`.
- **Recorded decision:** `promoted: false`, `candidate_supported_in_this_study: false`, `status: "hypothesis_not_supported_by_all_predefined_checks"`.

### Why the extra losses were unnecessary in this regime

1. **Ordinary training (`passive`) and non-targeted donor masking (`dropout`) remain vulnerable to shortcut cues and single-expert corruption.** On held-out `(source, operation)` pairs with an independent context proxy (`heldout_independent`), `passive` reaches `72.88%` accuracy (`54.96%` correct factual/counterfactual pairs) and drops to `39.58%` when one of the three specialists at the queried source is corrupted (`damage_one`). `dropout` improves held-out accuracy to `79.96%` and single-expert damage accuracy to `74.12%`, but still leaves a `>18 pp` gap below intervention training.
2. **Four-view cross-entropy (`iit`) already saturates the donor-posterior interface.** Simply supervising cross-entropy across the four aligned views (`factual`, `relevant`, `irrelevant`, `damaged`) drives `heldout_independent` accuracy to `98.19%` (compared with ~`97.1%–97.4%` individual donor validation accuracy), achieves `95.97%` correct factual/relevant pairs, reaches `100.00%` irrelevant-swap consistency (`invariance_error < 10^-6`), and maintains `98.22%` accuracy under single-expert corruption (`damage_one`).
3. **Adding explicit probability-effect and invariance penalties (`effect_only`, `invariance_only`, `tether`, `unpaired`) neither improves held-out accuracy nor reduces probability-effect error relative to `iit`.** Because `iit` cross-entropy already pushes counterfactual and invariance errors near the floor imposed by frozen donor errors, the extra squared probability terms add ~`11%` wall-clock training overhead (`16.56 s` per seed for `tether` vs `14.92 s` for `iit`) without improving downstream task decisions.

---

## 2. Predefined gate evaluation (`outputs/confirmation`)

| Predefined check | Target criterion | Observed value (`outputs/confirmation`) | Passed |
|---|---|---|---|
| `primary_interval_above_zero` | `tether - iit` `heldout_independent.accuracy` 95% bootstrap lower bound `> 0` | Mean `-0.07 pp`, interval `[-0.24 pp, +0.05 pp]` | `false` |
| `pairing_beats_unpaired_control` | `tether - unpaired` `heldout_independent.accuracy` 95% bootstrap lower bound `> 0` | Mean `-0.10 pp`, interval `[-0.20 pp, -0.02 pp]` | `false` |
| `clean_mean_noninferior_at_two_points` | `tether - iit` `seen_correlated.accuracy` mean difference `>= -2.00 pp` | Mean `-0.27 pp` (`98.93%` vs `99.19%`) | `true` |
| `irrelevant_consistency_at_least_95_percent` | `tether` mean `heldout_independent.irrelevant_consistency >= 95.0%` | `100.00%` | `true` |
| `positive_donor_utility_above_ten_points` | `tether` mean `heldout_independent.donor_erasure_gap > +10.0 pp` | `+88.18 pp` (`98.12%` intact vs `9.94%` erased) | `true` |

---

## 3. Eight-seed confirmation study (`outputs/confirmation`)

- **Seeds:** `101, 102, 103, 104, 105, 106, 107, 108` (`56` trained receiver checkpoints).
- **Split:** `data_seed = 1729` (`1,078` train, `359` validation, `360` test images). Frozen donor mean validation accuracies across the eight seeds: Expert A (`32` ReLU) `97.11%`, Expert B (`48` tanh) `97.39%`, Expert C (`64` GELU) `97.01%`.
- **Evaluation cohort:** `512` test tasks per condition (`phase = "confirmation"`).
- **Strict replay verification:** `python -m tether replay outputs/confirmation --strict` verified `114` sealed files, `4,312` recomputed trace metrics, and `2,352` restored checkpoint inference arrays within `atol=2e-6, rtol=1e-6`.

### 3.1 Held-out combinations, independent context (`heldout_independent`, primary condition)

| Arm | Factual acc | Relevant acc | Correct pair rate | Irrelevant consistency | Damaged-view acc | Erased acc | Erasure gap | Selected-shuffle acc | Effect error | Invariance error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 72.88% | 72.85% | 54.96% | 99.98% | 39.58% | 9.94% | +62.94 pp | 9.94% | 0.5959 | 0.000001 |
| `dropout` | 79.96% | 79.74% | 65.01% | 99.95% | 74.12% | 9.94% | +70.02 pp | 9.99% | 0.5294 | 0.000000 |
| `iit` | 98.19% | 97.75% | 95.97% | 100.00% | 98.22% | 9.94% | +88.26 pp | 10.08% | 0.0703 | 0.000000 |
| `effect_only` | 98.14% | 97.73% | 95.90% | 100.00% | 98.22% | 9.94% | +88.21 pp | 10.13% | 0.0723 | 0.000000 |
| `invariance_only` | 98.19% | 97.73% | 95.95% | 100.00% | 98.22% | 9.94% | +88.26 pp | 10.11% | 0.0702 | 0.000000 |
| `unpaired` | 98.22% | 97.73% | 95.97% | 100.00% | 98.27% | 9.94% | +88.28 pp | 10.11% | 0.0699 | 0.000000 |
| `tether` | 98.12% | 97.75% | 95.90% | 100.00% | 98.27% | 9.94% | +88.18 pp | 10.11% | 0.0716 | 0.000000 |

### 3.2 Factual accuracy across all seven evaluation conditions

| Arm | `seen_correlated` | `seen_independent` | `seen_reversed` | `heldout_independent` | `heldout_reversed` | `damage_one` | `damage_common` | Mean train time / seed | Mean val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 98.85% | 73.68% | 70.97% | 72.88% | 69.26% | 39.58% | 2.69% | 14.87 s | 0.7434 |
| `dropout` | 99.22% | 80.52% | 77.64% | 79.96% | 77.81% | 74.12% | 2.05% | 15.23 s | 0.6785 |
| `iit` | 99.19% | 98.27% | 97.80% | 98.19% | 97.63% | 98.22% | 0.83% | 14.92 s | 0.1546 |
| `effect_only` | 98.83% | 98.14% | 97.85% | 98.14% | 97.61% | 98.22% | 0.83% | 14.81 s | 0.1595 |
| `invariance_only` | 99.19% | 98.27% | 97.78% | 98.19% | 97.61% | 98.22% | 0.83% | 14.86 s | 0.1547 |
| `unpaired` | 98.75% | 98.27% | 97.97% | 98.22% | 97.66% | 98.27% | 0.83% | 16.26 s | 0.1553 |
| `tether` | 98.93% | 98.17% | 97.85% | 98.12% | 97.61% | 98.27% | 0.83% | 16.56 s | 0.1591 |

### 3.3 Paired seed-level differences (`outputs/confirmation`)

| Comparison | Condition & metric | Mean difference | Descriptive 95% seed bootstrap interval |
|---|---|---:|---|
| `tether - iit` | `heldout_independent.accuracy` | `-0.07 pp` (`-0.0007`) | `[-0.24 pp, +0.05 pp]` |
| `tether - iit` | `heldout_independent.correct_pair_rate` | `-0.07 pp` (`-0.0007`) | `[-0.32 pp, +0.17 pp]` |
| `tether - iit` | `heldout_independent.effect_error` | `+0.0013` | `[+0.0002, +0.0025]` |
| `tether - iit` | `heldout_reversed.accuracy` | `-0.02 pp` (`-0.0002`) | `[-0.22 pp, +0.17 pp]` |
| `tether - iit` | `damage_one.accuracy` | `+0.05 pp` (`+0.0005`) | `[-0.05 pp, +0.17 pp]` |
| `tether - iit` | `seen_correlated.accuracy` | `-0.27 pp` (`-0.0027`) | `[-0.51 pp, -0.07 pp]` |
| `tether - iit` | `seen_independent.accuracy` | `-0.10 pp` (`-0.0010`) | `[-0.20 pp, -0.02 pp]` |
| `tether - unpaired` | `heldout_independent.accuracy` | `-0.10 pp` (`-0.0010`) | `[-0.20 pp, -0.02 pp]` |
| `tether - unpaired` | `heldout_independent.correct_pair_rate` | `-0.07 pp` (`-0.0007`) | `[-0.20 pp, +0.05 pp]` |
| `tether - unpaired` | `damage_one.accuracy` | `+0.00 pp` (`+0.0000`) | `[-0.10 pp, +0.10 pp]` |
| `tether - unpaired` | `seen_correlated.accuracy` | `+0.17 pp` (`+0.0017`) | `[+0.05 pp, +0.34 pp]` |

---

## 4. Disclosed legacy pipeline pilot (`outputs/development`)

Before the methodological review fixes were applied, an initial two-seed (`11, 12`), six-arm (`passive`, `dropout`, `iit`, `effect_only`, `invariance_only`, `tether`) pipeline pilot was recorded in `outputs/development` at local commit `a7283b3`.

### Methodological issues identified and repaired after `outputs/development`

1. **Test-pool exposure during `development` phase:** At commit `a7283b3`, `_batch` selected `"test"` whenever `validation=False`, even when `config.phase == "development"`. In the repaired code (`tether/experiment.py`), `development` evaluates on the validation partition while `confirmation` evaluates on the test partition. No architecture, data split, or hyperparameter (`steps=400`, `lr=0.02`, `effect_weight=1.0`, `invariance_weight=1.0`) was changed after reading `outputs/development`.
2. **Task alignment and corruption retention in `damage_one` and `damage_common`:** At commit `a7283b3`, `damage_one` and `damage_common` drew separate task seeds from `heldout_independent` and copied `batch["damaged"]` only into `batch["donors"]`, so `batch["irrelevant"]` simultaneously repaired the corrupted expert at the queried source (producing artificial `invariance_error` of `1.72` on `damage_common`). The repaired `_batch` reuses the exact `heldout_independent` task draw and retains the queried-source corruption across `donors`, `relevant`, and `irrelevant`.
3. **Addition of the `unpaired` residual control:** Added to separate any effect of exact factual/counterfactual residual pairing from generic marginal squared-error regularization, expanding the recorded decision gate from four checks to five.

`outputs/development` is preserved intact as historical evidence (`tether` and `iit` tied at `97.75%` on `heldout_independent.accuracy`, `candidate_supported_in_this_study: false`). Its `26` sealed artifacts pass byte verification (`verify_files("outputs/development")`); because its summary records the earlier six-arm, four-check decision schema from commit `a7283b3`, running `tether replay outputs/development` requires checking out `a7283b3`.

---

## 5. Interpretation boundaries

- **Single-expert vs common-mode corruption:** Under `damage_one`, the reliability module sees two agreeing specialists and one dissenter at the queried source, allowing all intervention-trained arms (`iit`, `effect_only`, `invariance_only`, `unpaired`, `tether`) to maintain `98.22%–98.27%` accuracy. Under `damage_common`, all three specialists at the queried source are shifted to the same wrong class (`argmax + 1 mod 10`), removing ensemble disagreement; accuracy drops to `0.83%` (the rare cases where donor argmax was off by `-1 mod 10` so the cyclic shift accidentally corrected it).
- **Scope:** All results come from a single 1,797-image dataset (`sklearn.datasets.load_digits`) with a fixed 60/20/20 image-identity split (`data_seed = 1729`) and synthetic posterior-level interventions. They do not establish writer-disjoint generalization, natural corruption robustness, or foundation-model behavior.

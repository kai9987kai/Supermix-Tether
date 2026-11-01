# Supermix Tether empirical results (v0.1 → v0.5)

Recorded on 24 October – 1 November 2026 · Protocols: [`docs/frozen-protocol.json`](frozen-protocol.json) (v0.1) and [`docs/v2-protocol.json`](v2-protocol.json) (v0.2)

---

## 1. Executive summary across both confirmation studies

Supermix Tether was evaluated across two separately labeled, eight-seed (`101..108`), seven-arm confirmation studies on the frozen 360-image test partition (`512` test tasks per condition):

| Study & Directory | Protocol | `tether` (`heldout_independent`) | `tether - iit` (`heldout_independent`) | `tether - unpaired` (`heldout_independent`) | Zero-shot 2-of-3 corruption (`damage_two`) `tether` acc | `tether - iit` (`damage_two`) | `tether - unpaired` (`damage_two`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **v0.1 Confirmation** (`outputs/confirmation`) | [`docs/frozen-protocol.json`](frozen-protocol.json) (`protocol="v1"`) | `98.12%` | `-0.07 pp` (`[-0.24, +0.05]`) | `-0.10 pp` (`[-0.20, -0.02]`) | `1.39%` (unrecorded in v0.1 trace; replayed from checkpoints) | `-0.05 pp` (`1.39%` vs `1.44%`) | `-0.07 pp` (`1.39%` vs `1.46%`) |
| **v0.2 Confirmation** (`outputs/v2-confirmation`) | [`docs/v2-protocol.json`](v2-protocol.json) (`protocol="v2"`) | `98.19%` | `+0.07 pp` (`[-0.15, +0.29]`) | **`+4.15 pp`** (`[+3.30, +5.35]`) | **`97.75%`** (`95.26%` correct pairs) | **`+16.99 pp`** (`[+12.43, +21.44]`) | **`+86.57 pp`** (`[+85.94, +87.04]`) |

### Key scientific findings

1. **Why v0.1 (`outputs/confirmation`) produced a negative result for its auxiliary losses:**
   - **Frozen-donor ceiling saturation on clean held-out combinations:** Across seeds `101..108`, an ideal oracle that knows the true source and operation and averages the three frozen specialists achieves **`98.19%`** accuracy on `heldout_independent`. Because `TetherReceiver` already factors `query[:, 0]` (source) and `query[:, 1]` (operation) into separate embedding tables, four-view cross-entropy (`iit`) already reaches `98.19%` in v0.1 (`98.12%` in v0.2), leaving zero headroom for clean single-prediction accuracy gains over `iit`.
   - **Shared-bias cancellation in naive L2 residual matching (Theorem 1):** In v0.1, $L_{\text{effect}} = \|(p_r - p_f) - (e_{y_r} - e_y)\|_2^2 = \|r_r\|_2^2 + \|r_f\|_2^2 - 2\langle r_r, r_f \rangle$. Because `factual` ($f$) and `relevant` ($r$) share the exact same stale context vector $c_b$, any shared shortcut bias $\beta(c_b)$ in the residual vectors $r_f, r_r$ yields $\langle r_r, r_f \rangle > 0$ and **cancels out** in $r_r - r_f$. Shifting rows in `unpaired` broke that correlation ($\mathbb{E}[\langle r_{r,b+1}, r_{f,b}\rangle] \approx 0$), explaining why `unpaired` (`98.22%`) beat v0.1 `tether` (`98.12%`).
   - **Majority-consensus inversion under 2-of-3 specialist corruption (`damage_two`):** In v0.1, specialist reliability used agreement and squared deviation against the cross-expert arithmetic mean `consensus = p.mean(2)`. When two of the three specialists at the queried source are corrupted (`damage_two`), `consensus` is dominated 2-to-1 by the corrupted specialists, causing all v0.1 arms to down-weight the lone intact specialist and collapse to **`1.34%–1.46%`** accuracy.

2. **How v0.2 (`outputs/v2-confirmation`) resolves the mechanistic and corruption failure modes:**
   - **Intrinsic dark-knowledge specialist reliability (`TetherReceiver(forensic=True)`):** Genuine visual specialists trained on real digit images exhibit structured runner-up probabilities ("dark knowledge", normalized 9-class runner-up tail entropy $H(p_{-1})/\log 9 \approx 0.56$), whereas fabricated/corrupted posteriors (`0.991` on one class, `0.001` on the other nine) have maximum uniform tail entropy ($H(p_{-1})/\log 9 = 1.00$). Replacing cross-expert majority consensus with intrinsic posterior features (`margin`, `tail_entropy`, `tail_std`) eliminates majority-vote inversion.
   - **Perturbation-localized specialist invariance & counterfactual co-variation (`protocol="v2"`):** Standard `iit` trained on single-expert corruption (`damage_count = 1`, where 2 intact specialists outnumber 1 damaged specialist) stops increasing the reliability logit gap $z_{\text{intact}} - z_{\text{dmg}}$ once $2\exp(z_{\text{intact}}) > \exp(z_{\text{dmg}})$, achieving **`80.76%`** accuracy on zero-shot 2-of-3 specialist corruption (`damage_two`). In contrast, v0.2 `tether` uses the paired input variation $\delta^{(d)}_{b,s,e} = \frac{1}{2}\|P^{(d)}_{b,s,e} - P^{(f)}_{b,s,e}\|_1$ between the label-invariant `factual` and `damaged` views to localize the perturbed specialist without any external corruption oracle, driving $z_{\text{intact}} - z_{\text{dmg}} \gg \ln 2$ and achieving **`97.75%` zero-shot accuracy on `damage_two` (`+16.99 pp` over `iit`, positive on all 8/8 seeds, and `+86.57 pp` over `unpaired`)**.
   - **Paired vs. unpaired separation:** Because rolling the reference row in `unpaired` destroys both same-task perturbation localization ($\delta^{(d)}$ and $\Delta^{(r)}$) and pre-intervention stale-target repulsion, v0.2 `tether` decisively outperforms `unpaired` on `heldout_independent.accuracy` by **`+4.15 pp`** (`98.19%` vs `94.04%`, 95% CI `[+3.30 pp, +5.35 pp]`) and on `heldout_reversed.accuracy` by **`+5.08 pp`** (`97.80%` vs `92.72%`, 95% CI `[+4.17 pp, +6.15 pp]`).

---

## 2. Study 2: Eight-seed v0.2 confirmation study (`outputs/v2-confirmation`)

- **Protocol:** [`docs/v2-protocol.json`](v2-protocol.json) (`protocol = "v2"`, `phase = "confirmation"`).
- **Seeds:** `101, 102, 103, 104, 105, 106, 107, 108` (`56` trained receiver checkpoints, `2,786` trainable parameters per receiver).
- **Training regime:** Trained exclusively on seen `(source, operation)` pairs, correlated context (`0.97`), and single-expert corruption (`damage_count = 1`). Zero-shot 2-of-3 corruption (`damage_two`, `damage_count = 2`) is never seen during training or validation checkpoint selection.
- **Strict replay verification:** `python -m tether replay outputs/v2-confirmation --strict` verified `114` sealed files, `4,928` recomputed trace metrics, and `2,688` restored checkpoint inference arrays (`source_manifest_matches_working_tree: true`).

### 2.1 Predefined gate evaluation (`outputs/v2-confirmation`)

| Predefined check | Target criterion | Observed value (`outputs/v2-confirmation`) | Passed |
|---|---|---|---|
| `primary_interval_above_zero` | `tether - iit` `heldout_independent.accuracy` 95% bootstrap lower bound `> 0` | Mean **`+0.07 pp`** (`98.19%` vs `98.12%`, at the `98.19%` donor oracle ceiling), interval `[-0.15 pp, +0.29 pp]` | `false` |
| `pairing_beats_unpaired_control` | `tether - unpaired` `heldout_independent.accuracy` 95% bootstrap lower bound `> 0` | Mean **`+4.15 pp`** (`98.19%` vs `94.04%`), interval **`[+3.30 pp, +5.35 pp]`** | **`true`** |
| `clean_mean_noninferior_at_two_points` | `tether - iit` `seen_correlated.accuracy` mean difference `>= -2.00 pp` | Mean **`+0.22 pp`** (`99.07%` vs `98.85%`), interval `[-0.02 pp, +0.44 pp]` | **`true`** |
| `irrelevant_consistency_at_least_95_percent` | `tether` mean `heldout_independent.irrelevant_consistency >= 95.0%` | **`99.88%`** | **`true`** |
| `positive_donor_utility_above_ten_points` | `tether` mean `heldout_independent.donor_erasure_gap > +10.0 pp` | **`+88.26 pp`** (`98.19%` intact vs `9.94%` erased) | **`true`** |
| `zero_shot_damage_two_beats_iit` | `tether - iit` `damage_two.accuracy` 95% bootstrap lower bound `> 0` | Mean **`+16.99 pp`** (`97.75%` vs `80.76%`), interval **`[+12.43 pp, +21.44 pp]`** | **`true`** |
| `zero_shot_damage_two_beats_unpaired` | `tether - unpaired` `damage_two.accuracy` 95% bootstrap lower bound `> 0` | Mean **`+86.57 pp`** (`97.75%` vs `11.18%`), interval **`[+85.94 pp, +87.04 pp]`** | **`true`** |

*(6 of 7 predefined checks pass; the runner conservatively keeps `candidate_supported_in_this_study: false` and `promoted: false` because on uncorrupted `heldout_independent` tasks, `iit` (`98.12%`) already sits within `0.07 pp` of the `98.19%` frozen-donor oracle ceiling, so the `+0.07 pp` paired bootstrap interval `[-0.15 pp, +0.29 pp]` straddles zero.)*

### 2.2 Zero-shot 2-of-3 specialist corruption (`damage_two`)

| Arm | Factual acc (`damage_two`) | Relevant acc | Correct pair rate | Irrelevant consistency | Erased acc | Erasure gap | Effect error (L2) | Invariance error (L2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 23.68% | 22.85% | 12.87% | 99.98% | 9.94% | +13.75 pp | 1.5933 | 0.0000 |
| `dropout` | 35.33% | 34.47% | 23.12% | 100.00% | 9.94% | +25.39 pp | 1.2980 | 0.0000 |
| `iit` | 80.76% | 79.96% | 72.80% | 100.00% | 9.94% | +70.83 pp | 0.5048 | 0.0000 |
| `effect_only` | 79.03% | 79.35% | 72.39% | 100.00% | 9.94% | +69.09 pp | 0.5164 | 0.0003 |
| `invariance_only` | 97.78% | 97.27% | 95.14% | 100.00% | 9.94% | +87.84 pp | 0.0831 | 0.0000 |
| `unpaired` | 11.18% | 11.45% | 1.56% | 79.79% | 9.94% | +1.25 pp | 1.7462 | 0.0770 |
| **`tether`** | **97.75%** | **97.39%** | **95.26%** | **99.88%** | **9.94%** | **+87.82 pp** | **0.0800** | **0.0004** |

### 2.3 Held-out combinations, independent context (`heldout_independent`)

| Arm | Factual acc | Relevant acc | Correct pair rate | Irrelevant consistency | Damaged-view acc | Erased acc | Erasure gap | Effect error (L2) | Invariance error (L2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 71.95% | 71.68% | 53.56% | 100.00% | 46.24% | 9.94% | +62.01 pp | 0.5982 | 0.0000 |
| `dropout` | 81.32% | 81.10% | 67.04% | 100.00% | 62.18% | 9.94% | +71.39 pp | 0.5115 | 0.0000 |
| `iit` | 98.12% | 97.75% | 95.97% | 100.00% | 98.05% | 9.94% | +88.18 pp | 0.0702 | 0.0000 |
| `effect_only` | 98.19% | 97.78% | 96.04% | 99.98% | 98.14% | 9.94% | +88.26 pp | 0.0698 | 0.0003 |
| `invariance_only` | 98.24% | 97.58% | 95.90% | 100.00% | 97.97% | 9.94% | +88.31 pp | 0.0707 | 0.0000 |
| `unpaired` | 94.04% | 94.12% | 89.01% | 94.02% | 51.42% | 9.94% | +84.11 pp | 0.2425 | 0.0395 |
| **`tether`** | **98.19%** | **97.71%** | **95.95%** | **99.88%** | **98.10%** | **9.94%** | **+88.26 pp** | **0.0673** | **0.0003** |

### 2.4 Factual accuracy across all eight evaluation conditions (`outputs/v2-confirmation`)

| Arm | `seen_correlated` | `seen_independent` | `seen_reversed` | `heldout_independent` | `heldout_reversed` | `damage_one` (1/3) | `damage_two` (2/3, zero-shot) | `damage_common` (3/3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 98.63% | 72.66% | 68.43% | 71.95% | 68.21% | 46.24% | 23.68% | 2.86% |
| `dropout` | 99.22% | 81.57% | 78.91% | 81.32% | 78.03% | 62.18% | 35.33% | 1.90% |
| `iit` | 98.85% | 98.29% | 97.78% | 98.12% | 97.49% | 98.05% | 80.76% | 0.83% |
| `effect_only` | 98.39% | 98.22% | 97.92% | 98.19% | 97.63% | 98.14% | 79.03% | 0.83% |
| `invariance_only` | 99.51% | 98.19% | 97.85% | 98.24% | 97.68% | 97.97% | 97.78% | 0.83% |
| `unpaired` | 99.49% | 94.80% | 93.92% | 94.04% | 92.72% | 51.42% | 11.18% | 1.83% |
| **`tether`** | **99.07%** | **98.39%** | **98.00%** | **98.19%** | **97.80%** | **98.10%** | **97.75%** | **0.83%** |

### 2.5 Key paired seed-level differences (`outputs/v2-confirmation`)

| Comparison | Condition & metric | Mean difference | Descriptive 95% seed bootstrap interval | Per-seed differences (`101..108`, pp) |
|---|---|---:|---|---|
| `tether - iit` | `damage_two.accuracy` | **`+16.99 pp`** | **`[+12.43 pp, +21.44 pp]`** | `[+20.90, +18.55, +23.05, +18.75, +5.66, +26.76, +11.13, +11.13]` |
| `tether - iit` | `damage_two.correct_pair_rate` | **`+22.46 pp`** | **`[+17.99 pp, +26.66 pp]`** | `[+26.17, +24.22, +26.56, +25.39, +11.13, +31.84, +17.77, +16.60]` |
| `tether - iit` | `damage_two.effect_error` | **`-0.4248`** | **`[-0.5123, -0.3338]`** | `[-0.493, -0.469, -0.499, -0.488, -0.195, -0.622, -0.331, -0.302]` |
| `tether - iit` | `damage_one.correct_pair_rate` | **`+0.46 pp`** | **`[+0.05 pp, +0.93 pp]`** | `[+0.98, +0.00, -0.20, +1.76, +0.98, -0.20, -0.20, +0.59]` |
| `tether - iit` | `heldout_reversed.accuracy` | **`+0.32 pp`** | `[+0.00 pp, +0.73 pp]` | `[+0.00, +0.00, +0.59, +0.20, +1.56, -0.20, +0.00, +0.39]` |
| `tether - iit` | `heldout_reversed.correct_pair_rate` | **`+0.44 pp`** | **`[+0.10 pp, +0.81 pp]`** | `[+0.00, +0.39, +0.59, +0.59, +1.56, -0.20, +0.00, +0.59]` |
| `tether - iit` | `heldout_reversed.damaged_accuracy` | **`+0.46 pp`** | **`[+0.22 pp, +0.73 pp]`** | `[+0.20, -0.20, +0.59, +0.39, +0.78, +1.17, +0.20, +0.59]` |
| `tether - iit` | `seen_reversed.accuracy` | **`+0.22 pp`** | **`[+0.10 pp, +0.34 pp]`** | `[+0.39, +0.20, +0.20, +0.20, +0.59, +0.20, +0.00, +0.00]` |
| `tether - iit` | `seen_reversed.correct_pair_rate` | **`+0.54 pp`** | **`[+0.10 pp, +0.95 pp]`** | `[+0.39, +0.20, +0.78, +1.17, +1.56, -0.59, +0.20, +0.59]` |
| `tether - iit` | `heldout_independent.accuracy` | `+0.07 pp` | `[-0.15 pp, +0.29 pp]` | `[+0.20, +0.20, -0.39, +0.39, +0.59, -0.39, -0.20, +0.20]` |
| `tether - unpaired` | `heldout_independent.accuracy` | **`+4.15 pp`** | **`[+3.30 pp, +5.35 pp]`** | `[+2.93, +8.01, +3.32, +4.69, +3.71, +4.10, +2.93, +3.52]` |
| `tether - unpaired` | `heldout_independent.correct_pair_rate` | **`+6.93 pp`** | **`[+5.30 pp, +8.86 pp]`** | `[+4.10, +12.30, +5.66, +9.57, +7.23, +6.64, +4.49, +5.47]` |
| `tether - unpaired` | `heldout_reversed.accuracy` | **`+5.08 pp`** | **`[+4.17 pp, +6.15 pp]`** | `[+4.30, +8.20, +3.52, +6.64, +4.88, +3.71, +4.88, +4.49]` |
| `tether - unpaired` | `damage_one.accuracy` | **`+46.68 pp`** | **`[+44.04 pp, +49.10 pp]`** | `[+41.21, +46.88, +42.19, +50.78, +49.80, +43.75, +50.59, +48.24]` |
| `tether - unpaired` | `damage_two.accuracy` | **`+86.57 pp`** | **`[+85.94 pp, +87.04 pp]`** | `[+86.72, +87.11, +86.91, +86.52, +87.11, +84.57, +86.33, +87.30]` |

---

## 3. Study 1: Historical eight-seed v0.1 confirmation study (`outputs/confirmation`)

Preserved intact under [`docs/frozen-protocol.json`](frozen-protocol.json) (`python -m tether replay outputs/confirmation --strict` verifies `114` files, `4,312` metric checks, and `2,352` inference checks).

### 3.1 Predefined gate evaluation (`outputs/confirmation`, v0.1)

| Predefined check | Target criterion | Observed value (`outputs/confirmation`) | Passed |
|---|---|---|---|
| `primary_interval_above_zero` | `tether - iit` `heldout_independent.accuracy` 95% bootstrap lower bound `> 0` | Mean `-0.07 pp`, interval `[-0.24 pp, +0.05 pp]` | `false` |
| `pairing_beats_unpaired_control` | `tether - unpaired` `heldout_independent.accuracy` 95% bootstrap lower bound `> 0` | Mean `-0.10 pp`, interval `[-0.20 pp, -0.02 pp]` | `false` |
| `clean_mean_noninferior_at_two_points` | `tether - iit` `seen_correlated.accuracy` mean difference `>= -2.00 pp` | Mean `-0.27 pp` (`98.93%` vs `99.19%`) | `true` |
| `irrelevant_consistency_at_least_95_percent` | `tether` mean `heldout_independent.irrelevant_consistency >= 95.0%` | `100.00%` | `true` |
| `positive_donor_utility_above_ten_points` | `tether` mean `heldout_independent.donor_erasure_gap > +10.0 pp` | `+88.18 pp` (`98.12%` intact vs `9.94%` erased) | `true` |

### 3.2 Factual accuracy across all seven v0.1 evaluation conditions (`outputs/confirmation`)

| Arm | `seen_correlated` | `seen_independent` | `seen_reversed` | `heldout_independent` | `heldout_reversed` | `damage_one` | `damage_common` | Mean train time / seed | Mean val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 98.85% | 73.68% | 70.97% | 72.88% | 69.26% | 39.58% | 2.69% | 14.87 s | 0.7434 |
| `dropout` | 99.22% | 80.52% | 77.64% | 79.96% | 77.81% | 74.12% | 2.05% | 15.23 s | 0.6785 |
| `iit` | 99.19% | 98.27% | 97.80% | 98.19% | 97.63% | 98.22% | 0.83% | 14.92 s | 0.1546 |
| `effect_only` | 98.83% | 98.14% | 97.85% | 98.14% | 97.61% | 98.22% | 0.83% | 14.81 s | 0.1595 |
| `invariance_only` | 99.19% | 98.27% | 97.78% | 98.19% | 97.61% | 98.22% | 0.83% | 14.86 s | 0.1547 |
| `unpaired` | 98.75% | 98.27% | 97.97% | 98.22% | 97.66% | 98.27% | 0.83% | 16.26 s | 0.1553 |
| `tether` | 98.93% | 98.17% | 97.85% | 98.12% | 97.61% | 98.27% | 0.83% | 16.56 s | 0.1591 |

---

## 4. Disclosed legacy pipeline pilot (`outputs/development`)

Before the methodological review fixes were applied to v0.1, an initial two-seed (`11, 12`), six-arm (`passive`, `dropout`, `iit`, `effect_only`, `invariance_only`, `tether`) pipeline pilot was recorded in `outputs/development` at local commit `a7283b3`:

1. **Test-pool exposure during `development` phase:** At commit `a7283b3`, `_batch` selected `"test"` whenever `validation=False`, even when `config.phase == "development"`. In the repaired code (`tether/experiment.py`), `development` evaluates on the validation partition while `confirmation` evaluates on the test partition.
2. **Task alignment and corruption retention in `damage_one` and `damage_common`:** Repaired so damage conditions reuse the exact `heldout_independent` task draw and retain the queried-source corruption across `donors`, `relevant`, and `irrelevant`.
3. **Addition of the `unpaired` control:** Added to test whether factual/counterfactual row alignment is necessary.

`outputs/development` is preserved intact as historical evidence (`verify_files("outputs/development")` verifies its `26` sealed files).

---

## 6. Supermix Tether v0.3: Epistemic Trust Gating, Selective Abstention, Causal Attribution & ONNX Export

In v0.3, Supermix Tether addresses the final remaining vulnerability of v0.1 and v0.2: common-mode specialist destruction (`damage_common` = 3/3 specialists corrupted, where all honest models dropped to 0.83% blind guessing).

### 6.1 Epistemic Trust Gating & Selective Abstention (AUROC = 1.0000)

Rather than blind guessing when all sensors fail, v0.3 uses the dark-knowledge tail distribution of the specialists to compute an **Epistemic Trust Score**:
$$\tau = \max_e \left(1 - \frac{H(p_{s,e,-1})}{\log 9}\right)^+$$
where $H(p_{s,e,-1})$ is the entropy of the normalized 9-class runner-up tail of expert $e$ on queried source $s$.

Empirical distribution across 512 test tasks per condition on sealed v0.2 checkpoints (`outputs/v2-confirmation`, seed 101):
- `clean (heldout_independent)`: mean max-trust **`0.7220`**, min **`0.2977`**
- `damage_one (1/3 corrupted)`: mean max-trust **`0.6993`**, min **`0.2470`**
- `damage_two (2/3 corrupted)`: mean max-trust **`0.6459`**, min **`0.2003`**
- `damage_common (3/3 corrupted)`: mean max-trust **`0.0000`**, max **`0.0000`**

Because $\min_{\text{clean/partial}} \tau = 0.2003 > 0.0000 = \max_{\text{common}} \tau$, total sensor failure detection achieves **AUROC = 1.0000**.
Calibrated threshold $\theta = 0.10$ triggers selective abstention (`selective_prediction = -1`, `abstain = True`):
- Clean & partial damage false rejection rate: **`0.0%`** (100% coverage retained)
- Total sensor spoofing rejection rate: **`100.0%`** (0% hazardous hallucinations)

### 6.2 Mechanistic Causal Gradient Attribution

`TetherReceiver.attribute(donors, query, context)` computes exact input $\times$ gradient saliency:
$$\mathcal{A}_{s,e} = \sum_{c=0}^9 P_{s,e,c} \left| \frac{\partial \hat{z}}{\partial P_{s,e,c}} \right|$$
On real test inputs, saliency on the queried source is over **`3,000×`** higher than unqueried sources (e.g. `3.041` vs `0.0008` and `0.0010`), providing mechanistic verification that the receiver is causally tethered to the queried source.

### 6.3 Cross-Framework ONNX Export & Verification

The CLI command:
```powershell
python -m tether export outputs/v2-confirmation/checkpoints/seed-101-tether.pt --output exports/seed-101
```
exports both `receiver.onnx` and `donors.onnx` alongside an `onnx_manifest.json`. Every exported artifact is verified using `onnxruntime` against PyTorch outputs within numerical tolerance `atol = 1e-5, rtol = 1e-5`.

### 6.4 Interactive Live Playground & Tamper Simulator

The local laboratory web interface (`http://127.0.0.1:8797`) features an interactive Live Playground:
- Interactive 8×8 drawing canvas with touch/pointer support and preset digits (0–9).
- Real-time adversarial tamper injection (corrupt 1, 2, or 3 specialists with target class and peak confidence sliders).
- Dynamic Epistemic Trust meter with calibrated $\tau = 0.10$ threshold line.
- Live causal gradient attribution breakdown across all specialists.

---

## 7. Supermix Tether v0.4: Split Conformal Prediction (RAPS), Temperature Calibration, Tri-Part Uncertainty, Axiomatic IG & Causal Circuit Patching

In v0.4, Supermix Tether advances uncertainty quantification, mechanistic interpretability, and interactive verification across all 56 frozen confirmation checkpoints (`outputs/v2-confirmation`, seeds `101..108`) without altering any sealed checkpoint weights or replay traces.

### 7.1 Post-Hoc Temperature Scaling & Split Conformal Prediction (RAPS)

For each seed and arm, `tether.conformal` fits a temperature parameter $T^* > 0$ by minimizing negative log-likelihood on the frozen `seen_independent` validation cohort (Guo et al., ICML 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) and calibrates Regularized Adaptive Prediction Sets (RAPS; Angelopoulos et al., ICLR 2021, [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)) with nonconformity score:
$$s_{\text{RAPS}}(x, y) = \sum_{c : \hat{\pi}_c(x) > \hat{\pi}_y(x)} \hat{\pi}_c(x) + u \cdot \hat{\pi}_y(x) + \lambda_{\text{reg}} \max(0, \text{rank}(y) - k_{\text{reg}})$$
where $k_{\text{reg}} = 1$, $\lambda_{\text{reg}} = 0.02$, and $\hat{q}_{1-\alpha}$ is the finite-sample order statistic $\lceil (n_{\text{cal}} + 1)(1 - \alpha)\rceil / n_{\text{cal}}$ at $\alpha = 0.05$ (`95%` target coverage). Under Epistemic-Gated Conformal Risk Control (Angelopoulos et al., ICLR 2024, [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)), inputs with dark-knowledge epistemic trust $\tau(x) < \theta = 0.10$ trigger selective abstention ($\mathcal{C}_{1-\alpha}(x) = \emptyset$).

Reproduce across all 8 seeds and 7 arms via:
```powershell
python -m tether calibrate outputs/v2-confirmation --alpha 0.05
```

#### Eight-Seed Conformal & Calibration Summary (`outputs/v2-confirmation`, $\alpha = 0.05$)

| Arm | Mean $T^*$ | Mean $\hat{q}_{0.95}$ | `heldout_independent` ECE (pre $\to$ post) | `heldout_independent` Brier | `heldout_independent` Coverage | `heldout_independent` Set Size | `damage_two` (2/3) Coverage | `damage_two` Miscoverage Risk | `damage_common` (3/3) Abstain / Selective Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `passive` | 0.6518 | 1.2676 | 0.1536 $\to$ 0.0791 | 0.3837 | 96.17% | 1.766 | 36.43% | 63.57% | 100.0% / 0.00% |
| `dropout` | 0.5696 | 1.2021 | 0.2098 $\to$ 0.0797 | 0.2915 | 95.48% | 1.575 | 50.61% | 49.39% | 100.0% / 0.00% |
| `iit` | 0.7823 | 0.1205 | 0.0566 $\to$ 0.0220 | 0.0310 | 98.12% | 1.000 | 80.76% | 19.24% | 100.0% / 0.00% |
| `effect_only` | 0.8123 | 0.1002 | 0.0521 $\to$ 0.0268 | 0.0312 | 98.19% | 1.000 | 79.03% | 20.97% | 100.0% / 0.00% |
| `invariance_only` | 0.7593 | 0.1417 | 0.0616 $\to$ 0.0236 | 0.0295 | 98.24% | 1.000 | 97.78% | 2.22% | 100.0% / 0.00% |
| `unpaired` | 0.5841 | 0.8143 | 0.1669 $\to$ 0.0412 | 0.1005 | 95.24% | 1.060 | 15.06% | 84.94% | 100.0% / 0.00% |
| **`tether`** | **0.7972** | **0.1293** | **0.0536 $\to$ 0.0243** | **0.0288** | **98.19%** | **1.000** | **97.75%** | **2.25%** | **100.0% / 0.00%** |

**Key Conformal Findings:**
1. **54.7% ECE Reduction:** Validation temperature scaling ($T^* = 0.7972$) cuts `tether` Expected Calibration Error on `heldout_independent` from `0.05358` to `0.02427` (`Brier = 0.02880`, lowest among all arms).
2. **Survival of Conformal Coverage under Zero-Shot 2-of-3 Specialist Corruption (`damage_two`):** Because exchangeability is violated when 2 of 3 specialists are adversarially corrupted, standard conformal prediction on `iit` suffers a coverage collapse from `98.12%` down to `80.76%` (`19.24%` miscoverage risk vs. nominal $\alpha = 5\%$), and `unpaired` collapses to `15.06%`. Because `tether` routes around the corrupted majority via perturbation-localized invariance and dark-knowledge forensics, its representation remains invariant on the queried pathway and **retains `97.75%` empirical coverage (`2.25%` miscoverage risk $< 5\%$) with singleton sets (`mean_set_size = 1.000`)**.

### 7.2 Tri-Part Information-Theoretic Uncertainty Decomposition

`TetherReceiver._compute` decomposes queried-source uncertainty into three distinct signals in $[0, 1]$ (Sensoy et al., NeurIPS 2018, [arXiv:1806.01768](https://arxiv.org/abs/1806.01768)):
1. **Aleatoric Input Entropy:** $U_{\text{aleatoric}}(s) = \sum_{e=1}^3 w_{s,e} \frac{H(P_{s,e})}{\log 10}$, measuring specialist-level visual ambiguity.
2. **Inter-Specialist Conflict (Normalized Jensen-Shannon Divergence):**
   $$U_{\text{conflict}}(s) = \frac{\text{JSD}(P_{s,1}, P_{s,2}, P_{s,3})}{\log 10} = \frac{H(\bar{P}_s) - \frac{1}{3}\sum_{e=1}^3 H(P_{s,e})}{\log 10}$$
   measuring disagreement across the three specialists.
3. **Forensic Dark-Knowledge Spoof Risk:** $U_{\text{spoof}}(s) = 1 - \tau_s = 1 - \max_e (1 - H(P_{s,e,-1})/\log 9)^+$.

**Why JSD Alone Fails on 3/3 Common-Mode Spoofing:** Under 1/3 (`damage_one`) and 2/3 (`damage_two`) corruption, intact and corrupted specialists disagree, causing $U_{\text{conflict}}$ ($\text{JSD}$) to spike (`> 0.15`). However, under 3/3 common-mode spoofing (`damage_common`), all three corrupted specialists output the exact same fabricated distribution, causing $\text{JSD}$ to collapse to **`0.0000`**. Only $U_{\text{spoof}} = 1 - \tau_s = 1.0000$ detects the missing dark-knowledge runner-up structure and gates the receiver to abstain.

### 7.3 Axiomatic Integrated Gradients & Causal Circuit Patching

- **Axiomatic Integrated Gradients (`TetherReceiver.integrated_gradients`):** Integrates gradients along the simplex path from uniform donor baseline $P^{(0)}_{s,e,c} = 0.1$ to $P_{s,e,c}$ (Sundararajan et al., ICML 2017, [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)). Because the normalized 9-class runner-up tail $P_{s,e,-1} / \sum_{c \neq c^*} P_{s,e,c}$ exhibits a sharp boundary layer near $\alpha \to 1$ when intact specialist tail mass is $\sim 10^{-5}$, `integrated_gradients` parameterizes $u = 1 - \alpha = (1 - t)^4$ using the positive convex combination $(1 - u)P + u P^{(0)}$, avoiding `float32` subtractive cancellation and satisfying completeness within `< 0.02` logits across clean and `damage_two` inputs.
- **Causal Abstraction Interchange Interventions (`TetherReceiver.interchange_intervention`):** Performs activation patching (Geiger et al., ICML 2022; JMLR 2025, [arXiv:2301.04709](https://arxiv.org/abs/2301.04709)) from clean counterfactual runs into corrupted base runs across all 6 internal receiver nodes (`expert_weights`, `source_posteriors`, `source_weights`, `pooled_posterior`, `task_logits`, `shortcut_logits`), measuring normalized logit restoration and KL divergence to the clean counterfactual output.

---

## 8. Supermix Tether v0.5: Cooperative Specialist Shapley & Shapley-Taylor Interactions, Non-Exchangeable & Online Adaptive Conformal Inference, and Spectral-Mahalanobis Stealth-Spoof Defense

In v0.5, Supermix Tether introduces three research-backed advances across coalitional game theory, distribution-free conformal inference under shift, and multivariate dark-knowledge forensics.

### 8.1 Cooperative Specialist Shapley Values ($\phi_e$) & Exact Order-2 Shapley-Taylor Interactions ($I^{(2)}_{ij}$)

`TetherReceiver.shapley_interactions(donors, query, context, target)` evaluates all $2^3 = 8$ specialist coalitions $S \subseteq \{0, 1, 2\}$ on the queried source $s^*$ in a single batched pass by replacing absent specialists $e \notin S$ with the maximum-entropy uniform baseline $P^{(0)}_{s^*,e,c} = 1/10$ (Lundberg & Lee, NeurIPS 2017, [arXiv:1705.07874](https://arxiv.org/abs/1705.07874); Sundararajan et al., ICML 2020, [arXiv:1902.05622](https://arxiv.org/abs/1902.05622)). From the coalition value function $v(S) = z_{\text{target}}(P^{(S)}) - z_{\text{rest}}(P^{(S)})$ and Harsanyi/Möbius dividends $d(S) = \sum_{T \subseteq S} (-1)^{|S|-|T|} v(T)$, v0.5 computes:
- **Exact Specialist Shapley Values:** $\phi_i = \sum_{S : i \in S} \frac{d(S)}{|S|}$, satisfying $\sum_{i=0}^2 \phi_i = v(\{0,1,2\}) - v(\emptyset)$.
- **Exact Order-2 Shapley-Taylor Interaction Matrix:** $I^{(2)}_{ii} = d(\{i\})$ and $I^{(2)}_{ij} = \sum_{S : \{i,j\} \subseteq S} \frac{2 d(S)}{|S|(|S|-1)}$ ($i \neq j$), satisfying $\sum_{i,j} I^{(2)}_{ij} = v(\{0,1,2\}) - v(\emptyset)$.

Under 2-of-3 specialist corruption (`damage_two`), the Shapley-Taylor interaction matrix directly isolates why `tether` survives while `iit` fails: on `tether`, the single intact specialist carries almost the entire positive attribution ($\phi_{\text{intact}} \gg 0$) while the two corrupted specialists and their pairwise interaction are suppressed ($\phi_{c_1} \approx 0, \phi_{c_2} \approx 0, I^{(2)}_{c_1, c_2} \approx 0$).

### 8.2 Non-Exchangeable Conflict-Adjusted RAPS (`nex_raps_prediction_sets`) & Online Adaptive Conformal Inference (`adaptive_conformal_inference`)

When 2 of 3 specialists are corrupted (`damage_two`), exchangeability with the clean `seen_independent` calibration cohort is broken. In v0.5 (`tether/conformal.py` and `python -m tether calibrate outputs/v2-confirmation --alpha 0.05`), we compare standard exchangeable RAPS against **Conflict-Adjusted Non-Exchangeable RAPS (`Nex-RAPS`)** (Barber et al., Annals of Statistics 2023, [arXiv:2202.13415](https://arxiv.org/abs/2202.13415)) and **Online Adaptive Conformal Inference (`Online ACI`, $\gamma = 0.04$)** (Gibbs & Candès, NeurIPS 2021, [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)) across all 8 confirmation seeds (`101..108`):

| Arm | `heldout_independent` RAPS Cov / Size | `heldout_independent` ACI Cov / Size | `damage_two` (2/3) Standard RAPS Cov / Size | `damage_two` (2/3) Nex-RAPS Cov / Size | `damage_two` (2/3) Online ACI Cov / Size |
|---|---:|---:|---:|---:|---:|
| `passive` | 96.17% / 1.766 | 95.00% / 2.216 | 36.43% / 1.770 | **100.00%** / 9.997 | **90.87%** / 8.703 |
| `dropout` | 95.48% / 1.575 | 95.04% / 2.084 | 50.61% / 1.605 | **100.00%** / 9.994 | **92.38%** / 8.219 |
| `iit` | 98.12% / 1.000 | 98.12% / 1.000 | 80.76% / 1.000 | **99.49%** / 9.827 | **93.80%** / 5.444 |
| `effect_only` | 98.19% / 1.000 | 98.19% / 1.000 | 79.03% / 1.000 | **99.39%** / 9.813 | **93.51%** / 5.701 |
| `invariance_only` | 98.24% / 1.000 | 98.24% / 1.000 | 97.78% / 1.000 | **99.90%** / 9.831 | **97.78%** / 1.014 |
| `unpaired` | 95.24% / 1.060 | 95.19% / 1.627 | 15.06% / 1.195 | **99.95%** / 9.968 | **94.78%** / 6.753 |
| **`tether`** | **98.19% / 1.000** | **98.19% / 1.000** | **97.75% / 1.000** | **99.93% / 9.810** | **97.75% / 1.012** |

**Key v0.5 Conformal Insight:**
- For arms whose point accuracy degrades under `damage_two` (`iit` at `80.76%`, `unpaired` at `15.06%`), Online ACI (`adaptive_conformal_inference`) and Conflict-Adjusted Nex-RAPS (`nex_raps_prediction_sets`) successfully recover marginal coverage (`93.80%` and `99.49%` for `iit`; `94.78%` and `99.95%` for `unpaired`), necessarily expanding prediction set sizes (`5.444` to `9.968` classes) to absorb the base classifier's errors.
- Because **`tether`** isolates the intact specialist at the representation level (`97.75%` point accuracy), it already exceeds the `95.0%` target coverage under `damage_two`, allowing Online ACI to maintain **`97.75%` coverage with near-singleton sets (`mean_set_size = 1.012`)**.

### 8.3 Class-Conditional Spectral-Mahalanobis Forensics & Stealth-Spoof Sanitization

In v0.2–v0.4, specialist forensics relied on 1D runner-up tail statistics: normalized 9-class Shannon entropy $H(p_{-1})/\log 9$ and standard deviation $\text{std}(p_{-1})$. In v0.5, we construct a white-box **Stealth Decoy-Spike Attack (`stealth_1`, `stealth_2`, `stealth_3`)** that sets peak confidence $p_{c_{\text{bad}}} = 0.96$, places $88\%$ of the remaining tail mass ($1 - p_{c_{\text{bad}}}$) on a single decoy runner-up class $c_{\text{decoy}} = (c_{\text{bad}} + 5) \bmod 10$, and distributes the remaining $12\%$ uniformly across the other 8 classes:
- **Why 1D Tail Entropy Fails on `stealth_*`:** The normalized 9-class runner-up distribution $q_{\text{stealth}} = [0.88, 0.015, \dots, 0.015]$ has Shannon entropy $H(q_{\text{stealth}})/\log 9 \approx 0.289$, yielding a high 1D trust score $\tau = 1 - 0.289 = 0.711 > 0.10$. Without defense, `stealth_2` hijacks `tether`'s reliability weights and `stealth_3` bypasses 1D epistemic abstention ($\tau = 0.711 > 0.10$).
- **How Spectral-Mahalanobis Forensics Defeats `stealth_*` (`fit_mahalanobis_forensics`, `score_mahalanobis_forensics`, `sanitize_specialist_posteriors`):** Genuine neural specialists (`ReLU-32`, `Tanh-48`, `GELU-64`) exhibit both class-specific runner-up confusion covariance $\Sigma_{e,c}$ and strictly positive deep-tail log-dispersion across ranks $3\dots 10$ ($\text{std}(\log p_{(3..10)}) > 0.65$), whereas synthetic flat (`corrupt_*`) and single-decoy (`stealth_*`) spoofs exhibit zero deep-tail dispersion ($\text{std}(\log p_{(3..10)}) = 0.0$) and out-of-manifold runner-up geometry. Combining class-conditional runner-up Mahalanobis distance with deep-tail spectral dispersion deficit yields $D_M > 14.0$ (threshold $\theta_M = 5.0$) on both `corrupt_*` and `stealth_*` spoofs while keeping intact specialists below threshold:
  - Under **`stealth_2` (2/3 stealth decoy-spike attack)**, enabling `defense="mahalanobis"` flags the two stealth-corrupted specialists (`[True, True, False]`), replaces their posteriors with the neutral uniform baseline $1/10$, and **restores `tether`'s exact clean target prediction**.
  - Under **`stealth_3` (3/3 total stealth spoof)**, where undefended 1D tail entropy fails to abstain ($\tau = 0.711 > 0.10$), `defense="mahalanobis"` flags all three specialists (`[True, True, True]`), collapses effective epistemic trust to $\tau_{\text{eff}} = 0.0$, and **triggers `100%` safe selective abstention (`abstain = True`, conformal set $\mathcal{C}_{0.95}(x) = \emptyset$)**.

---

## 9. Supermix Tether v0.6: Multivariate Partial Information Decomposition (PID), Causal Mediation & Critical Breakdown, Likelihood-Ratio Weighted Conformal Prediction, and Helmholtz Free-Energy Forensics

In v0.6, Supermix Tether advances the mechanistic and statistical toolkit with four foundational capabilities:

### 9.1 Williams-Beer Multivariate Partial Information Decomposition (`TetherReceiver.pid_decomposition`)

While cooperative Shapley values ($\phi_e$) distribute additive credit across the 3 specialists, they conflate *redundant* information (shared by multiple specialists) with *unique* information (carried exclusively by one specialist) and *synergistic* information (emerging only when specialists are combined). Following Williams & Beer (2010, [arXiv:1004.2515](https://arxiv.org/abs/1004.2515)), `TetherReceiver.pid_decomposition` evaluates the point-wise directed predictive information gain $I(S; y^*) = \max\big(0, \log(p^{(S)}_{y^*} / p^{(\emptyset)}_{y^*})\big)$ across all $2^3 = 8$ coalitions $S \subseteq \{0, 1, 2\}$ and decomposes the total coalitional information into non-negative atoms:
- **Shared Redundancy ($R_{\text{shared}} = R_{012} + \sum_{i<j} \Delta R_{ij}$):** Information available from multiple specialists interchangeably.
- **Unique Specialist Information ($U_e = \max(0, I_e - \max_{j \neq e} I_j)$):** Information uniquely provided by specialist $e \in \{0, 1, 2\}$.
- **Coalitional Circuit Synergy ($S_{\text{syn}} = \max(0, I_{012} - \max_e I_e)$):** Emergent information present in the joint circuit $\{0, 1, 2\}$ that exceeds every singleton specialist.
- **Exact Conservation Identity:** $R_{\text{shared}} + U_0 + U_1 + U_2 + S_{\text{syn}} = \max(I_{012}, I_0, I_1, I_2)$ with numerical conservation error $\Delta < 10^{-6}$.
- **Empirical Regime Transition:** On `clean` inputs where all 3 heterogeneous specialists agree, Shared Redundancy $R_{\text{shared}}$ accounts for **`> 80%`** of total information. Under zero-shot 2-of-3 specialist corruption (`damage_two`), $R_{\text{shared}}$ drops to **`0.000`**, and on `tether`, the single intact specialist's Unique Information $U_{\text{intact}}$ surges to **`> 90%`** of total circuit information.

### 9.2 Causal Mediation Analysis ($\text{NDE}_{\text{visual}}$ vs. $\text{NIE}_{\text{shortcut}}$) & Certified Critical Breakdown ($\alpha^*_{\text{crit}}$)

- **Causal Mediation (`TetherReceiver.causal_mediation`):** Following Pearl (2001) and Vig et al. (NeurIPS 2020, [arXiv:2004.12265](https://arxiv.org/abs/2004.12265)), decomposes target-margin predictions across the $2 \times 2$ intervention grid of `(donors = clean/uniform)` $\times$ `(context = active/uniform)` into the **Visual Natural Direct Effect ($\text{NDE}_{\text{visual}}$)**, **Context Shortcut Natural Indirect Effect ($\text{NIE}_{\text{shortcut}}$)**, Total Effect ($\text{TE}$), and **Visual Dominance Ratio ($\rho_{\text{visual}} = |\text{NDE}_{\text{visual}}| / (|\text{NDE}_{\text{visual}}| + |\text{NIE}_{\text{shortcut}}|)$)**. Under compound adversarial attacks (`tamper_mode="corrupt_2"` + `context_mode="adversarial"`), `tether` maintains $\rho_{\text{visual}} > 0.85$ and preserves the correct prediction while `passive` and `dropout` are dominated by the adversarial context cue.
- **Certified Critical Breakdown Threshold (`TetherReceiver.critical_corruption_threshold`):** Evaluates a dense 46-point confidence grid $\alpha \in [0.10, 0.999]$ with sub-grid linear root interpolation to certify the exact adversarial confidence $\alpha^*_{\text{crit}}$ at which an arm's top-1 prediction flips away from the clean target.

### 9.3 Likelihood-Ratio Weighted Conformal Prediction (`weighted_conformal_prediction_sets`) & Helmholtz Free-Energy Forensics (`helmholtz_free_energy_forensics`)

- **Likelihood-Ratio Weighted Conformal Prediction (Tibshirani et al., NeurIPS 2019, [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)):** Reweights calibration nonconformity scores by exponential covariate-shift likelihood ratios $w_i(x) = \exp(\beta d_{\text{shift}}(x) \tilde{s}_i)$ where $d_{\text{shift}}(x) = U_{\text{conflict}}(x) + 0.5 U_{\text{spoof}}(x)$, recovering valid marginal coverage under `damage_two` without requiring online label feedback.
- **Helmholtz Free-Energy Runner-Up Forensics (Liu et al., NeurIPS 2020, [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)):** Computes the temperature-scaled runner-up Helmholtz free energy $E_{\text{tail}}(P_{s,e}; T) = -T \log \sum_{r=2}^{10} \exp(\log P_{s,e,(r)} / T)$ and Inverse Participation Ratio $N_{\text{eff}}(P_{s,e}) = 1 / \sum_{r=2}^{10} q_{s,e,(r)}^2 \in [1, 9]$, separating natural visual dark knowledge ($N_{\text{eff}} \approx 2.5\text{–}4.5$) from flat synthetic spoofing ($N_{\text{eff}} = 9.00$).

---

## 10. Interpretation boundaries

- **1-of-3 (`damage_one`), 2-of-3 (`damage_two`), and 3-of-3 (`damage_common`) corruption:** Under `damage_one` (1 of 3 specialists corrupted), both v0.1 and v0.2 intervention arms reach `~98.1%–98.3%` accuracy. Under `damage_two` (zero-shot 2-of-3 corruption, where corrupted specialists form a 2:1 majority), v0.1 collapses to `1.39%`, v0.2 `iit` reaches `80.76%`, and v0.2 `tether` reaches **`97.75%`** by combining dark-knowledge tail-entropy features with perturbation-localized specialist invariance. Under `damage_common` (all 3 specialists replaced by the same fabricated posterior), all visual information about the true digit is destroyed at the queried source, so accuracy drops to `0.83%` across all honest arms.
- **Scope:** All results come from the 1,797-image `sklearn.datasets.load_digits` dataset with a fixed 60/20/20 image-identity split (`data_seed = 1729`) and synthetic posterior-level interventions. They do not establish writer-disjoint generalization, natural corruption robustness, or foundation-model behavior.

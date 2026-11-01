# Research basis and implementation boundaries

Primary-source metadata and abstracts checked on 24 October 2026. This is a targeted reading log, not a systematic review or a ranking of current methods. Most 2026 entries are arXiv preprints. Their reported findings are attributed to their authors and were not reproduced in Tether.

The implemented question is whether relevant probability-effect matching and irrelevant/damaged invariance help a small receiver beyond identical-data intervention cross-entropy. See the [methods paper](paper.md), [model card](../MODEL_CARD.md) and [recorded results](results.md).

## Direct antecedents

### Geiger et al. (2022): interchange intervention training

[Inducing Causal Structure for Interpretable Neural Networks](https://proceedings.mlr.press/v162/geiger22a.html), ICML, PMLR 162:7324–7338. Authors: Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah Goodman and Christopher Potts.

IIT trains neural behavior under aligned counterfactual state substitutions. Its evaluations already include a digit-based structural task. Tether adopts the principle of supervising changed outputs after an intervention. The `iit` arm is a local posterior-interface control, not a reproduction of those experiments or a formal causal-abstraction guarantee. The additional Tether losses must be evaluated against that strong informational control.

### Wu et al. (2022): causal distillation

[Causal Distillation for Language Models](https://arxiv.org/abs/2112.02505v2), NAACL 2022; arXiv v2, 3 June 2022. Authors: Zhengxuan Wu, Atticus Geiger, Josh Rozner, Elisa Kreiss, Hanson Lu, Thomas Icard, Christopher Potts and Noah D. Goodman.

This work adds causal-computation imitation through IIT to ordinary distillation objectives. It is direct prior art for transferring how a teacher affects outputs, beyond matching features. Tether has no pretrained language teacher, hidden-state distillation or language benchmark. It does not present counterfactual transfer itself as new.

### Angelopoulos et al. (2021, 2024): split conformal prediction (RAPS) and conformal risk control

- [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/2009.14193), ICLR 2021. Authors: Anastasios N. Angelopoulos, Stephen Bates, Jitendra Malik and Michael I. Jordan.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814), ICLR 2024. Authors: Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei and Tal Schuster.

RAPS wraps a classifier to output Regularized Adaptive Prediction Sets $\mathcal{C}_{1-\alpha}(x)$ calibrated on a held-out validation cohort with finite-sample marginal coverage $\mathbb{P}(Y \in \mathcal{C}_{1-\alpha}(X)) \ge 1 - \alpha$ under exchangeability. Conformal Risk Control generalizes set construction to monotone risk functions, including selective abstention. In v0.4 (`tether/conformal.py`), Tether calibrates post-hoc temperature $T^*$ and RAPS quantile $\hat{q}_{1-\alpha}$ strictly on the `seen_independent` validation cohort and gates the conformal set with the dark-knowledge epistemic trust score $\tau$, yielding $\mathcal{C}(x) = \emptyset$ under 3/3 common-mode spoofing while preserving compact singleton sets ($|\mathcal{C}_{0.95}| \approx 1.02$) on clean and 2/3-corrupted inputs.

### Sundararajan et al. (2017) & Geiger et al. (2025): axiomatic integrated gradients & causal abstraction

- [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365), ICML 2017, PMLR 70:3319–3328. Authors: Mukund Sundararajan, Ankur Taly and Qiqi Yan.
- [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://arxiv.org/abs/2301.04709), JMLR 2025. Authors: Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah D. Goodman, Christopher Potts and Thomas F. Icard.

Local first-order input $\times$ gradient saliency can saturate through `Tanh` and `Softmax` nonlinearities when specialist posteriors approach 1.0. In v0.4, `TetherReceiver.integrated_gradients()` integrates gradients along the simplex straight-line path from the maximum-entropy uniform baseline $P^{(0)}_{s,e,c} = 1/10$ to $P_{s,e,c}$, satisfying **Completeness** ($\sum_{s,e,c} \text{IG}_{s,e,c}(P) = F_y(P) - F_y(P^{(0)})$) and **Sensitivity**. Complementing path attribution, `TetherReceiver.interchange_intervention()` executes discrete activation patching across all six internal receiver nodes (`expert_weights`, `source_posteriors`, `source_weights`, `pooled_posterior`, `task_logits`, `shortcut_logits`) to measure exact counterfactual restoration.

### Guo et al. (2017) & Sensoy et al. (2018): temperature calibration & uncertainty decomposition

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), ICML 2017. Authors: Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger.
- [Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/abs/1806.01768), NeurIPS 2018. Authors: Murat Sensoy, Lance Kaplan and Melih Kandemir.

In v0.4, Tether fits post-hoc temperature scaling $T^*$ on validation logits to minimize NLL and decomposes specialist uncertainty into three orthogonal signals: **Aleatoric Entropy** ($\sum_e w_{s,e} H(P_{s,e}) / \log 10$), **Inter-Specialist Conflict** (normalized Jensen-Shannon divergence $\text{JSD}(P_{s,1}, P_{s,2}, P_{s,3}) / \log 10$), and **Forensic Spoof Risk** ($1 - \tau_s$). Crucially, under 3/3 common-mode spoofing, standard ensemble conflict $\text{JSD} = 0$ because all three corrupted experts agree on the fabricated class; only dark-knowledge forensic spoof risk ($1 - \tau_s = 1.0$) detects the attack.

### Lundberg & Lee (2017) & Sundararajan et al. (2020): cooperative Shapley values & Shapley-Taylor coalitional interactions

- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874), NeurIPS 2017. Authors: Scott M. Lundberg and Su-In Lee.
- [The Shapley Taylor Interaction Index](https://arxiv.org/abs/1902.05622), ICML 2020, PMLR 119:9259–9268. Authors: Mukund Sundararajan, Kedar Dhamdhere and Ashish Agarwal.

Because the queried source has $N = 3$ specialists (`ReLU-32`, `Tanh-48`, `GELU-64`), all $2^3 = 8$ coalitions $S \subseteq \{0, 1, 2\}$ can be evaluated exactly in a single batched forward pass (`TetherReceiver.shapley_interactions`) by replacing absent specialists $e \notin S$ with the maximum-entropy uniform baseline $P^{(0)}_{s,e,c} = 1/10$. Using the Harsanyi (Möbius) dividend $d(S) = \sum_{T \subseteq S} (-1)^{|S| - |T|} v(T)$, v0.5 computes both exact Shapley values $\phi_i = \sum_{S : i \in S} \frac{d(S)}{|S|}$ and the Order-2 Shapley-Taylor Interaction Matrix ($I^{(2)}_{ii} = d(\{i\})$, $I^{(2)}_{ij} = \sum_{S : \{i,j\} \subseteq S} \frac{2 d(S)}{|S|(|S|-1)}$ for $i \neq j$), satisfying Exact Efficiency:
$$\sum_{i=0}^2 \phi_i = \sum_{i=0}^2 \sum_{j=0}^2 I^{(2)}_{ij} = v(\{0,1,2\}) - v(\emptyset).$$
Under 2-of-3 specialist corruption (`damage_two`), $I^{(2)}_{ij}$ exposes the exact coalitional voting threshold: `iit` suffers a catastrophic negative pairwise synergy $I^{(2)}_{c_1, c_2} \ll 0$ when both corrupted specialists are active together, whereas `tether` assigns $\phi_{\text{intact}} > 0$ and suppresses both individual and pairwise corrupted contributions ($\phi_{c} \approx 0, I^{(2)}_{c_1, c_2} \approx 0$).

### Barber et al. (2023) & Gibbs & Candès (2021): non-exchangeable & online adaptive conformal inference

- [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415), Annals of Statistics 2023, 51(2):816–845. Authors: Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas and Ryan J. Tibshirani.
- [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170), NeurIPS 2021. Authors: Isaac Gibbs and Emmanuel Candès.

Standard split conformal prediction assumes exchangeability between calibration (`seen_independent`) and test data, which breaks under adversarial 2-of-3 specialist corruption (`damage_two`), causing `iit` coverage to drop to `80.76%` and `unpaired` to `15.06%`. In v0.5 (`tether/conformal.py`), we implement two complementary remedies:
1. **Conflict-Adjusted Non-Exchangeable RAPS (`nex_raps_prediction_sets`):** Uses the observable queried-source specialist conflict $U_{\text{conflict}}(x) = \text{JSD}(P_{s,1}, P_{s,2}, P_{s,3}) / \log 10$ to dynamically inflate the effective conformal threshold $\hat{q}_{\text{eff}}(x) = \hat{q}_{1-\alpha}(1 + \gamma_q U_{\text{conflict}}(x))$ and predictive temperature $T_{\text{eff}}(x) = T^*(1 + \gamma_T U_{\text{conflict}}(x))$ without requiring test labels.
2. **Online Adaptive Conformal Inference (`adaptive_conformal_inference`):** Updates the effective miscoverage target $\alpha_{t+1} = \text{clip}(\alpha_t + \gamma (\alpha - \text{err}_t), 10^{-4}, 1 - 10^{-4})$ across sequential task streams ($\gamma = 0.04$), guaranteeing long-run coverage $\lim_{T\to\infty} \frac{1}{T}\sum_{t=1}^T (1 - \text{err}_t) = 1 - \alpha$ under arbitrary adversarial shifts.

### Lee et al. (2018) & Yin et al. (2018): Spectral-Mahalanobis dark-knowledge forensics & Byzantine sanitization

- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888), NeurIPS 2018. Authors: Kimin Lee, Kibok Lee, Honglak Lee and Jinwoo Shin.
- [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/abs/1803.01498), ICML 2018. Authors: Dong Yin, Yudong Chen, Ramchandran Kannan and Peter Bartlett.

A white-box **Stealth Decoy-Spike Adversary (`stealth_1..3`)** that places $88\%$ of its non-argmax tail mass on a single decoy runner-up class and splits the remaining $12\%$ uniformly across the other 8 classes achieves low runner-up tail entropy $H(p_{-1})/\log 9 \approx 0.289$, yielding a high 1D trust score $\tau = 1 - 0.289 = 0.711 > 0.10$ and fooling 1D tail-entropy defenses. In v0.5, `fit_mahalanobis_forensics` fits class-conditional runner-up covariance $\Sigma_{e,c}$ and deep-tail spectral log-dispersion ($\text{std}(\log p_{(3..10)})$) on the frozen `seen_independent` validation split. Because synthetic flat or single-decoy spoofs have zero deep-tail dispersion ($\text{std}(\log p_{(3..10)}) = 0.0$) and violate class-conditional visual similarity covariance, `score_mahalanobis_forensics` flags both `corrupt_*` and `stealth_*` attacks ($D_M > 5.0$), and `sanitize_specialist_posteriors` replaces flagged specialists with the neutral uniform baseline $1/10$ before receiver inference.

### Williams & Beer (2010), Vig et al. (2020), Tibshirani et al. (2019) & Liu et al. (2020): Multivariate PID, Causal Mediation, Weighted Conformal Prediction & Helmholtz Tail Energy (v0.6)

- [Nonnegative Decomposition of Multivariate Information](https://arxiv.org/abs/1004.2515), 2010. Authors: Paul L. Williams and Randall D. Beer.
- [Investigating Gender Bias in Language Models Using Causal Mediation Analysis](https://arxiv.org/abs/2004.12265), NeurIPS 2020. Authors: Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer and Stuart Shieber.
- [Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019), NeurIPS 2019. Authors: Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candès and Aaditya Ramdas.
- [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759), NeurIPS 2020. Authors: Weitang Liu, Xiaoyun Wang, John D. Owens and Yixuan Li.

In v0.6, `TetherReceiver.pid_decomposition` decomposes point-wise directed target information $I(S; y^*) = \max(0, \log(p^{(S)}_{y^*} / p^{(\emptyset)}_{y^*}))$ across the $2^3 = 8$ specialist coalitions into non-negative Shared Redundancy ($R_{\text{shared}}$), Unique Specialist Information ($U_0, U_1, U_2$), and Coalitional Circuit Synergy ($S_{\text{syn}}$). `TetherReceiver.causal_mediation` separates Visual Natural Direct Effect ($\text{NDE}_{\text{visual}}$) from Context Shortcut Natural Indirect Effect ($\text{NIE}_{\text{shortcut}}$) and computes the exact Critical Breakdown Threshold $\alpha^*_{\text{crit}}$. Finally, `weighted_conformal_prediction_sets` reweights calibration nonconformity scores by exponential conflict likelihood ratios $w(x) = \exp(\beta d_{\text{shift}}(x))$, and `helmholtz_free_energy_forensics` evaluates runner-up tail Helmholtz free energy $E_{\text{tail}}(P_{s,e}; T)$ and inverse participation ratio $N_{\text{eff}} \in [1, 9]$.

## Current context: useful lessons, different implemented mechanisms

| Primary source and checked version | Source-level observation | Consequence for Tether |
|---|---|---|
| Armstrong, Ayoobi and Mukherjee, [*Decodable But Not Detachable*](https://arxiv.org/abs/2608.10214v1), 10 August 2026 | Their LLM study distinguishes domain decodability from selectively detachable neuron groups and reports different patterns at different data granularities. | Keep predictive information, selective intervention effects and task accuracy separate. Tether does not search LLM neurons or establish a corresponding data-granularity law. |
| Tang et al., [*Modular TTT: Rethinking Test-Time Training as Composable Modules*](https://arxiv.org/abs/2608.07110v1), 7 August 2026 | Exposes the fast learner, loss, learning rate, decay and normalization as separately composable design choices. | Use ablations to distinguish objective components. Tether has frozen inference parameters and implements no fast-weight inner loop or test-time updates. |
| Cao et al., [*An Empirical Study and Theoretical Explanation on Task-Level Model-Merging Collapse*](https://arxiv.org/abs/2603.09463v1), 10 March 2026 | Reports task combinations with poor merges and a relation to representational incompatibility, with a rate-distortion analysis. | Do not assume adding specialists creates a useful combined model. Tether combines outputs; it neither merges parameters nor tests that theoretical bound. |
| Siddika, Hossen, Mallick and Jannesari, [*Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning*](https://arxiv.org/abs/2601.17616v3), revised 17 October 2026 | SETA separates unique/shared expert components, tests subspace/sign agreement and protects shared knowledge through anchoring. | Reliability and compatible reuse merit measurement. Tether does not implement SETA, sequential task acquisition, sparse parameter sharing or anti-forgetting anchors. The checked version is v3; v2 is marked withdrawn in arXiv history. |
| Nath et al., [*How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning*](https://arxiv.org/abs/2610.03119v1), 2 October 2026 | AMSC selects and weights policies using online task similarity based on Wasserstein embeddings and sparsemax support. | Choosing a source and choosing how much to use it are distinct questions. Tether learns supervised softmax routes; it uses no policy library, RL interaction or task-similarity estimator. |

These connections are design inferences. None of the papers validates Tether's architecture, coefficients, corruption model or expected performance. Their parameter scales, tasks and evaluation protocols differ substantially.

## Portfolio lineage

The [27-repository portfolio index](portfolio-review.md) links selected-file reviews in [reviews/](reviews/). The review records accessible commits, inspected paths and failures where possible. It does not assert that every source file was audited or that upstream model results were independently reproduced.

The immediate project-level motivations are unused graft/consolidation contributions in Expanse, decodable-but-unused state reported in Prometheus, and the value of controlled stream/transplant experiments in Ghost and Morpheus. Other reviewed projects already cover simulation, memory transfer, experimental planning, routing, evidence ledgers and cross-system composition. Tether therefore makes a trained receiver objective the experimental center instead of naming those existing capabilities as new.

No upstream implementation was copied into Tether. Paper references and portfolio inspirations are intellectual attribution. Dependencies and dataset attribution are recorded separately in [THIRD_PARTY.md](../THIRD_PARTY.md).

## What would change the conclusion

The predefined analysis compares Tether with the same-input `iit` arm. Effect-only and invariance-only arms test component necessity. A seventh `unpaired` arm retains the marginal residual penalties while breaking correct factual/relevant pairing; it distinguishes paired effects from generic extra supervised pressure. No pairing-specific benefit should be claimed without comparison with that control. A null result should report unchanged or worse held-out behavior even if training loss or route visualizations improve. An apparent gain should be checked for clean-data regressions, incorrect consistent answers, brittle donor dependence and sensitivity to corruption design.

Future work could add stronger passive feature-alignment baselines, different donor datasets, natural image shifts, writer-disjoint partitions, larger intervention spaces and explicitly recurrent tasks. Those are proposals; no such controls or capabilities are implemented by calling the current study a confirmation run.

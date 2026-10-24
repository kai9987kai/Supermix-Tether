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

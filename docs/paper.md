# Supermix Tether: testing useful specialist dependence with paired effects and invariance

Design and methods paper · 24 October 2026 · v0.1

## Abstract

A specialist can supply recognizable information without contributing useful computation to a receiver. Conversely, a receiver can become sensitive to a specialist without improving its answers. Supermix Tether studies that distinction in a small CPU-local neural system. Three frozen visual classifiers supply probabilities for three handwritten digit images. A learned receiver routes sources, weights experts and applies a requested digit operation while a proxy context cue creates a competing shortcut. We compare seven identical receiver architectures, including intervention supervision, relevant-effect matching, nuisance invariance and an unpaired-effect control. Evaluation separates held-out source/operation pairs, proxy shifts, targeted swaps, donor erasure and synthetic corruption. This paper specifies the mechanism and evidence boundaries. It reports no empirical superiority; study outcomes are recorded separately in [results](results.md).

## 1. Motivation and relation to prior work

Selected-file reviews of 27 portfolio repositories identify a recurring practical concern: a new branch, decodable representation or successful alignment fit need not explain downstream gains. Those reviews are documented in the [portfolio index](portfolio-review.md) and [individual reports](reviews/). They are source inspections, not replications. Tether adopts the resulting experimental question without copying upstream implementations or treating their reported results as independently verified.

[Interchange intervention training (IIT)](https://proceedings.mlr.press/v162/geiger22a.html) trains neural counterfactual behavior against an aligned causal model. [Causal distillation](https://arxiv.org/abs/2112.02505) applies that principle to transferring a teacher's computation. These are direct methodological antecedents. Internal swaps, intervention supervision and digit-based compositional tasks are not new contributions claimed by Tether.

The implemented `iit` arm is an **IIT-style control at the donor-posterior interface**. It is not a reproduction of the original paper's architectures, benchmarks, internal alignment procedure or zero-loss abstraction result. Tether asks a narrower question: do explicit probability-effect and invariance losses add useful behavior beyond cross-entropy on identical intervention examples?

Recent [work separating decodability from parametric detachability](https://arxiv.org/abs/2608.10214) motivates keeping representational availability separate from selective causal effects. Its large-language-model findings do not establish what a small digit receiver will do. Other current modular-learning directions and the mechanisms deliberately omitted from Tether are listed in [research.md](research.md).

## 2. Data and acquisition task

The implementation loads the 1,797-example [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Each input has 64 grayscale features. Dividing pixels by 16 is a fixed transformation requiring no fitted held-out statistics. Stratified image-identity splitting with seed 1729 produces 1,078 training, 359 validation and 360 test examples before any task sampling. This is a within-dataset partition rather than a writer-disjoint evaluation.

For one task, sample three image IDs from the declared partition and a query `(s,o)`. The source index `s` is 0, 1 or 2; the operation `o` is identity, successor modulo ten, or mirror `9 - d`. The deterministic task generator supplies labels for supervision and evaluation. The receiver has no access to those labels during inference.

Acquisition excludes `(0,2)`, `(1,1)` and `(2,0)`. All sources and operations appear elsewhere, so the held-out test concerns their pairing, not an unseen primitive or an arbitrarily deep program. Sampling can reuse images within a batch or across tasks; task rows are not independent new data acquisitions.

The context is a one-hot answer proxy. During acquisition it matches the factual target with probability 0.97; otherwise it names a different class. Evaluation uses correlated, independently sampled and always-incorrect proxies. The same factual context is retained after a donor intervention, so a relevant swap often makes the original proxy stale. The design therefore exposes a competition between visual evidence and shortcut use.

## 3. Frozen specialists and learned receiver

Each seed trains three multilayer perceptrons on training images only: `64→32→10` with ReLU, `64→48→10` with tanh and `64→64→10` with GELU. Default donor training uses 180 AdamW steps. Validation scores are reported without donor checkpoint selection. The donor bank is frozen before receiver training. Test posteriors may be cached for later evaluation; test labels do not guide donor updates.

The same bank evaluates all three source images, producing `P ∈ R^(B×3×3×10)`. These are heterogeneous redundant classifiers with common data lineage, rather than independent knowledge domains.

The receiver has 2,762 trainable parameters at its default hidden width of 24. For each source and expert, a reliability MLP reads the posterior, normalized entropy, agreement with the local mean posterior, squared deviation from that mean and a learned expert identity. A softmax produces expert weights, whose weighted sum forms each source posterior. Learned query embeddings and an MLP produce source weights. The pooled posterior passes through a learned operation-specific `10×10` matrix and bias. A learned context MLP supplies an additive shortcut logit vector.

There is no fixed correct-source mask or arithmetic output mapping in this inference path. Query indexing selects learned operation parameters. Receiver gradients are detached at the donor boundary. Inspection exposes routing weights and branch outputs, but the weights alone are not causal explanations.

## 4. Four aligned views

Each acquisition batch contains:

1. **Factual:** original donor probabilities, query, context and target `y`.
2. **Relevant:** all three expert vectors at the queried source are replaced by a different-digit image from the same partition; the new target is `y_r`.
3. **Irrelevant:** a different source is replaced by another same-partition, different-digit image; the target remains `y`.
4. **Damaged:** one expert at the queried source receives a fabricated confident posterior; the target remains `y`.

For single-expert corruption, the fabricated class is that expert's predicted argmax plus one modulo ten. It has probability 0.991 and all other classes have probability 0.001. No true label is used to choose that class. A corruption can therefore correct a pre-existing error. A separate test changes all three expert posteriors at the selected source to a common fabricated class. This common-mode stress removes the redundancy that can help identify a lone dissenter.

All four views retain the original query and context. Synthetic posterior changes are controlled interface interventions; they are not naturally occurring image shifts or evidence of robustness to real sensor failure.

## 5. Objectives and matched controls

Let `p_f`, `p_r`, `p_i` and `p_d` denote receiver softmax vectors on the four views, and `e_y` a one-hot target. All expectations below are batch means; squared norms sum over output classes.

```text
L_CE = (CE(f,y) + CE(r,y_r) + CE(i,y) + CE(d,y)) / 4

L_effect = E || (p_r - p_f) - (e_y_r - e_y) ||²

L_invariance = (E ||p_i - p_f||² + E ||p_d - p_f||²) / 2

L_tether = L_CE + λ_effect L_effect + λ_invariance L_invariance
```

Both coefficients default to 1. Gradients flow through factual and changed predictions. The losses encourage a specific relevant change and discourage nuisance changes; they do not prove a unique internal causal mechanism. In particular, effect matching can encourage sharper probabilities, and its tradeoff with calibration requires separate measurement.

| Arm | Four receiver evaluations | Added loss |
|---|---|---|
| Passive | Factual repeated four times | None |
| Dropout | Factual with four deterministic donor masks | None |
| IIT-style | Four aligned intervention views | None |
| Effect only | Same as IIT-style | `L_effect` |
| Invariance only | Same as IIT-style | `L_invariance` |
| Tether | Same as IIT-style | Both |
| Unpaired | Same as IIT-style | Invariance plus row-shifted relevant-residual effect |

For the mechanistic control, write `r_f = p_f - e_y` and `r_r = p_r - e_y_r`. Tether's effect loss is `E ||r_r - r_f||²`. The unpaired arm cyclically shifts `r_r` by one batch row before subtracting `r_f`, while preserving Tether's cross-entropy and invariance. Thus the marginal squared residual terms remain the same and the correctly paired cross-term is broken. A pairing-specific benefit requires a comparison against this control; beating IIT-style cross-entropy alone cannot distinguish pairing from generic extra supervised pressure. A one-row batch cannot break pairing and should not be used to support that comparison.

Dropout replaces a quarter of row/source/expert positions with uniform posteriors in each view; the four masks cover every position once and consume no RNG state. It does not inspect targets or the queried source. All arms share receiver initialization within a seed, training task draws, donor bank, step count and batch size. The extra losses are not FLOP-matched; training time and view count are recorded. Passive and dropout do not receive the same intervention information as IIT-style and Tether.

## 6. Selection and evaluation

Default receiver acquisition uses Adam at learning rate 0.02, 400 steps, batch size 128 and gradient-norm clipping at 5. Every 50 steps and at the final step, checkpoint selection evaluates cross-entropy on a frozen **seen-pair, independent-context validation** cohort. The minimum validation loss selects a checkpoint. Neither held-out combinations nor test scores select it.

Seven frozen conditions cover seen correlated/independent/reversed context, held-out independent/reversed context, single-expert damage and common-mode damage. In the corrected evaluator, development uses validation images and confirmation uses test images. Damage conditions reuse the `heldout_independent` task cohort, replace the factual donor input with the corrupted view, and retain the queried-source corrupted expert posteriors across the `relevant` and `irrelevant` views so an irrelevant swap does not simultaneously repair the damaged specialist.

Reported metrics include factual and relevant accuracy, correct factual/relevant pairs, irrelevant consistency, damage accuracy, selected-source shuffle accuracy, donor-erased accuracy and the intact-minus-erased accuracy gap. Probability effect error and irrelevant probability-invariance error are retained. The reported `invariance_error` concerns the irrelevant view; the **training** invariance penalty additionally includes damaged-view changes.

The primary comparison is paired seed-level Tether minus IIT-style accuracy on held-out pairs with independent context. Tether minus unpaired is a separate mechanistic control for correct effect pairing. Bootstrap resampling operates on seed differences, not individual task rows. The interval is descriptive and unstable with few seeds. Fixed data lineage, shared image partition and repeated task ingredients limit independence even across receiver seeds.

The runner records predefined checks on the primary descriptive interval, clean-data regression, irrelevant consistency and donor utility. The exact executed gate is saved with each study and interpreted in [results](results.md); mechanistic pairing claims additionally require the unpaired comparison. These are finite-study decision rules, not statistical safety guarantees. The runner always records `promoted: false`. A `confirmation` phase label does not itself verify fresh seeds or preregistration; that separation requires an externally recorded configuration decision and fresh run inputs.

An earlier engineering pipeline pilot at source revision `a7283b3` evaluated the same test-image pool. Initial coefficients of 1 and the 400-step budget were fixed before that pilot, and no data, model or training hyperparameters were selected from its accuracies. The pilot is not independent validation. Corrected phase separation and a fresh-seed study address that workflow issue; they do not create an untouched independent dataset or erase prior exposure. The results report retains the chronology.

## 7. Receipts and reproducibility

Saved studies include source, dataset and split fingerprints; configuration and runtime versions; selected donor/receiver state dictionaries; per-seed/arm histories; full output traces; and file hashes. The source fingerprint covers project Python, local web assets and packaging metadata. It does not make a hermetic record of operating-system binaries, numerical libraries or hardware.

Ordinary replay verifies bytes, recomputes trace metrics and reconstructs paired summaries and decisions. Strict replay additionally checks the source snapshot, rebuilds models from configurations and state dictionaries, regenerates evaluation tasks and compares inference numerically. It does not rerun optimization. Hashes detect disagreement with a recorded manifest; they are not independent attestation of the correctness of that manifest or its scientific interpretation.

## 8. Limits and falsification

Success would require better useful behavior beyond identical-data IIT-style training, with the predicted relevant effects and acceptable nuisance/damage behavior. A larger erasure gap alone can indicate fragility. High irrelevant consistency can mean the receiver is consistently wrong. High effect fidelity without improved held-out answers would answer a different question from task improvement. These outcomes should remain separate.

The experiment does not test arbitrary donor architectures, unseen datasets, writer-disjoint recognition, novel digit operations, recurrent reasoning or language-model integration. The receiver sees compact class posteriors rather than raw images or high-dimensional donor hidden states. Donor errors are correlated and corruption is synthetic. Full-bank common-mode failure can destroy task information, so invariance is not a promise that missing information can be recovered.

Current scope does not include test-time training, parameter merging, policy reuse, continual learning, conformal prediction or a learned abstention policy. No claim is made about biological memory, consciousness, semantic understanding or real-world safety. General method novelty and superiority remain open questions.

## Data attribution and results status

Source dataset: E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). The UCI source lists CC BY 4.0. Tether applies normalization, new image partitions and generated tasks; see [DATA_CARD.md](../DATA_CARD.md) and [THIRD_PARTY.md](../THIRD_PARTY.md).

This document is a methods description. Completed measurements, source revision, run directories, replay evidence and the evidence-bounded outcome are reported in [results.md](results.md) when available. A negative or null result is a valid outcome of the study.

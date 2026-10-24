# Supermix Tether v0.1 design

Date: 2026-10-24. Status: implementation authorized by the user's `you decide` reply. Architecture and resource choices below are the builder's decisions. CPU-local Python 3.12, PyTorch, NumPy and scikit-learn already available on this PC; no remote model, paid API or GPU required.

## Intent

Portfolio reviews repeatedly expose a gap between information decoded from a donor and donor information that actually changes a receiver's decisions. Tether asks: **does matching relevant intervention effects while suppressing irrelevant and damaged-expert effects improve usable specialist dependence under cue shifts and withheld task combinations, beyond identical-data interchange-intervention training?**

This is a small trainable modular model and a research instrument. It is not a general language model, a biological model, an established breakthrough, or a reproduction of all 27 upstream projects. Existing IIT and causal distillation are direct prior art. The candidate contribution is an explicitly tested dual relevant-effect/invariance objective with heterogeneous redundant donors, equal-information controls and corruption tests. Novelty and superiority remain hypotheses.

## Approaches considered

1. Larger heterogeneous graft/foundation-model composite: expensive locally and strongly overlaps Archimedes, Expanse and Poseidon. Reject for v0.1.
2. Another risk-aware router, causal experiment planner or evidence ledger: useful but already covered by Supermix, Poseidon, Mnemorph, CROSSWIRE and FAULTLINE. Reject as the central contribution.
3. Trainable specialist-use conversion: selected by the builder under delegated authority. Addresses an observed portfolio failure and allows a bounded real-data experiment with strong existing-method controls.

## Data and task

Use sklearn's bundled 1,797 handwritten 8x8 digits; split image identities into stratified 60/20/20 train/validation/test BEFORE constructing any task. This is a new within-dataset split, not the official UCI writer-disjoint protocol. Save exact IDs and dataset hashes. Train three heterogeneous visual classifiers on training images only. Their probabilities for three independently sampled source images form a tensor `[batch, source=3, expert=3, class=10]`. All experts share the dataset lineage; they are not independent sources of evidence.

Query `[source, operation]` requests identity, next-digit modulo ten, or mirror `9-digit` on one source. Hold out pairs `(0,2),(1,1),(2,0)` from receiver acquisition; all individual sources and operations remain observed. Frozen validation/test draws include seen and held-out pairs separately. Context is a deliberately correlated answer proxy (0.97 during acquisition); evaluate correlated, independent and always-wrong proxies. No test labels enter training, corruption choice, model selection or calibration.

Generate four aligned views: factual; relevant (replace all experts at selected source with outputs from a different training image, retaining the old proxy); irrelevant (replace a different source, same target); damaged (replace one expert's posterior with a wrong confident posterior, same target). Relevant targets come from the training/validation/test task oracle as appropriate. Corruptions are controlled synthetic interventions on outputs of real image classifiers, not evidence of real-world robustness.

## Model and objective

Frozen donors: three small neural classifiers with different hidden widths/activations. Receiver: learned query/source routing; learned expert reliability weighting from probabilities, agreement and entropy; learned task transformation matrices; a learned shortcut branch. No hard-coded correct source selector or output transformation in inference. All arms have identical receiver architecture and initialization per seed.

`passive`: four repeated factual CE views. `dropout`: four factual CE views with deterministic donor dropout. `iit`: mean CE on the four intervention views. `effect_only`: IIT + relevant effect loss. `invariance_only`: IIT + irrelevant/damaged invariance. `tether`: IIT + both. Match batch, steps, four receiver view evaluations, donor training and cached donor outputs; report extra objective cost and measured time, not equal FLOPs.

For probability vectors p, relevant loss is `mean(sum(((p_relevant-p_factual) - (onehot(y_relevant)-onehot(y)))^2))`; invariance loss is the mean squared probability change for irrelevant and damaged views. Coefficients fixed before confirmation, initially effect=1 and invariance=1. Existing-method comparison is Tether vs IIT, not merely Tether vs passive. Honest null or negative results are valid deliverables.

## Interfaces

`tether.data`: `load_digits_split(seed=1729) -> dict` keys `images` float tensor [1797,64], `labels` long tensor, `train/validation/test` long index tensors, `manifest` JSON-able dict. `task_targets(digits,operations) -> long tensor`. `make_batch(posteriors, labels, indices, n, seed, split='train', cue='correlated', pairs='seen', damage_count=1) -> dict` with `donors/relevant/irrelevant/damaged` float tensors [n,3,3,10], `query` long [n,2], `context` float [n,10], `target/relevant_target` long [n], `image_ids` long [n,3], `replacement_ids` long[n], `manifest` dict. It must be deterministic without mutating global RNG. `fit_donors(split, seed, steps=180) -> (bank, posteriors, metrics)`; bank is torch module, posteriors [1797,3,10], metrics JSON-able. `DonorBank` lives in data module, checkpoint loading uses state_dict and config, never arbitrary pickle objects.

`tether.model`: `TetherReceiver()`; `forward(donors,query,context) -> logits[n,10]`; `inspect(...) -> dict` tensors including `source_weights`, `expert_weights`, `logits`; `objective(model,batch,arm,effect_weight=1.,invariance_weight=1.) -> (loss, diagnostics)`; diagnostics numerical, contains CE/effect/invariance. All donor/query/context inputs validated at the public boundary. Frozen donor parameters receive no receiver gradients. Seed initialization controlled externally with torch.manual_seed.

`tether.experiment`: `StudyConfig` dataclass; `run_study(config, output_dir, progress=None) -> dict`; `evaluate_model(model,batch) -> dict`; immutable outputs contain config, source fingerprint, dataset/split manifest, complete per-seed/arm rows, inference traces and checkpoint hashes. Receipt replay recomputes metrics from saved trace vectors, and strict replay reruns checkpoint inference against regenerated frozen batches. Summary uses seed-level paired descriptive bootstrap; samples within seed are not independent experimental replications. Decision is exploratory/unpromoted unless the predefined primary gate is met on fresh confirmation seeds. Never call a few seeds confirmatory proof.

`tether.server`: loopback only, read-only HTTP service with `/api/results`, `/api/example?index=0&arm=tether&mode=factual`, static local UI. Shows completed saved results; inference uses selected saved checkpoint and test examples only. Does not train on browser input. Root served files restricted to `tether/web`. Failed/missing studies return clear state. No remote assets.

## Evidence, tests and completion

Meaningful tests: image-ID disjointness; held-out pair exclusion; true targeted swap and irrelevant invariance with literal targets; deterministic draws/RNG isolation; invalid dimensions/NaNs/indices; objective counterexample (wrong relevant effect penalized, irrelevant sensitivity penalized); receiver gradients and learned useful dependence; donor freeze; save/load inference equality; receipt metric and checkpoint tampering rejection; paired seed statistics; bounded real training smoke; HTTP traversal/missing study/error behavior; browser inspect controls.

Train a development pilot, freeze configuration, then run a fresh multi-seed study. Report passive/IIT/Tether results, damage sensitivity, held-out accuracy, donor-erasure gap, and limitations. Freeze all source before generating final receipts; inference replay validates bytes and model behavior, not scientific truth. Include README, architecture/prior-art paper, all 27 source reviews, model/data cards and Windows launcher. Launch and verify the local UI. No publication requested.

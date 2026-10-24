# Portfolio review: memory, modularity, and evidence systems

Review date: 24 October 2026. Scope: public GitHub state for nine named repositories. This is a read-only source review, not an execution, replication, model evaluation, security audit, or legal determination.

## Method and access

The GitHub connector was used first. For every repository, repository metadata, the main branch head, the root directory, recursive file tree, README, and at least one meaningful implementation or design artifact were fetched. All nine were public and accessible; there were no repository access failures. README blob SHAs fetched through the default branch matched a second fetch pinned to the recorded head. Implementation files were fetched at immutable head SHAs.

Depth is deliberately uneven: this review reads small engines closely and samples larger monorepos at the mechanisms relevant to a new Supermix project. Statements labelled README evidence are project-authored reports, not independently reproduced findings. No tests, applications, checkpoints, data downloads, or training jobs were run. Source links below are immutable revision links.

## Revision and licensing inventory

| Repository | Verified default branch and head | Head author date (UTC) | Observed license |
| --- | --- | --- | --- |
| [Supermix](https://github.com/kai9987kai/Supermix) | main, 6ea1c2cc41e3705b4b899b764a001cec180d155d | 2026-09-12 18:07:51 | Root LICENSE: MIT, copyright 2026 Kai Piper |
| [FLY-DIAMOND-NEXUS](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS) | main, 0c8d4100b21d8889bacdef1a9c6dfc20b3424242 | 2026-09-21 08:33:02 | Root LICENSE: MIT; bundled malecns/LICENSE.md is GPL v3, so the tree is not uniformly covered by the root MIT declaration |
| [QuantumBot](https://github.com/kai9987kai/QuantumBot) | main, 42e4fd637da2df1da5241da484ebc2c625050e72 | 2026-09-18 18:31:33 | Root LICENSE: MIT |
| [Causeway](https://github.com/kai9987kai/Causeway) | main, e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac | 2026-10-05 22:07:06 | Root LICENSE: MIT |
| [MOLT](https://github.com/kai9987kai/MOLT) | main, 8ac04b1839b82d4e4d83166f1938bc6e1150197a | 2026-10-05 20:47:01 | Root LICENSE: MIT |
| [MEMORY-CARRIER-OBSERVATORY](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY) | main, 523c10b46fa26acd53cd8468f95d3cbf55011437 | 2026-10-06 08:22:54 | Root LICENSE: MIT |
| [Mnemorph](https://github.com/kai9987kai/Mnemorph) | main, 4fc3353b4aa4ea3e2c7b24798121ee65deda6939 | 2026-10-07 10:07:34 | Root LICENSE: MIT |
| [TESSERA](https://github.com/kai9987kai/TESSERA) | main, 85e740452824a0ae089332e47d3473db879ba70c | 2026-10-07 13:13:35 | Root LICENSE: MIT |
| [NEUROGRAFT](https://github.com/kai9987kai/NEUROGRAFT) | main, 2301f0af16a92a9722608c868e4719f1898ca0fe | 2026-10-21 20:54:03 | No root LICENSE present in fetched tree; LICENSE_MATRIX.md explicitly says no publication license selected |

This inventory records observed files rather than inferring rights to weights, data, vendored code, or derivatives. NEUROGRAFT documentation also says no external publication occurred, although the repository is now public; that text is stale relative to observed hosting state.

## 1. Supermix

Inspected: README.md (selected current release and evaluation sections), source/nexus_conformal_stopping.py, source/nexus_active_inference.py, source/nexus_risk_control.py (opening contract, policy grid, API surface), source/route_policy_lab.py and source/route_policy_protocol.py (opening contracts and API surfaces), source/nexus_epistemics.py (authority boundary and verifier metadata), source/score_fusion.py (design contract and consensus implementation), docs/V90_DIFFUSION_OF_THOUGHT.md (selected sections), LICENSE. The whole 396 MB repository was not audited.

Mechanisms observed:
- A mixed local-first runtime, training, benchmarking, desktop packaging, and model-export workspace.
- Margin-based stop advice explicitly returns no risk certificate. Its calibration method fits an empirical quantile of unlabeled margins, and terminal budget exhaustion does not prove correctness.
- A separate dependency-free selective-risk module does use labelled frozen records: either simultaneous one-sided Clopper-Pearson bounds over a fixed policy grid or disjoint development selection followed by held-out calibration. It declares exchangeability and independent labels as assumptions rather than established facts. It does not control live routing or deployment.
- Route-policy analysis is descriptive agreement-subset analysis, not off-policy causal estimation. Protocol preflight freezes stateful design declarations but does not execute or authorize a study.
- Active inference is a hand-authored risk/information-gain action heuristic, not measured posterior inference learned from outcomes.
- Score fusion calibrates heterogeneous signals by percentile ranks and gates near-constant signals. Its CombMNZ-style consensus multiplier counts supporting signals; the inspected function does not establish independence or correct for shared provenance.

Lessons: keep answer authority, calibration, runtime control, and model promotion separate; measure actual budgets and fixed problem sets. There is already substantial evidence and routing infrastructure, so a generic evidence ledger, uncertainty router, active-inference planner, or risk-aware stopping wrapper would repeat existing work.

Important discrepancy: the V90 design document asserts distribution-free risk guarantees and typical 30–60% reasoning FLOP savings, while the current margin-stopping source explicitly says it provides no certificate and calls savings potential step-budget fractions, not measured FLOPs. Treat these design claims as unsupported by this module. README sections also document generator/cap/exposure confounds in earlier model comparisons; historical release labels should not substitute for receipts.

Sources: [README](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/README.md), [stopping](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/nexus_conformal_stopping.py), [labelled risk control](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/nexus_risk_control.py), [route analysis](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/route_policy_lab.py), [protocol](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/route_policy_protocol.py), [fusion](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/score_fusion.py), [V90 document](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/docs/V90_DIFFUSION_OF_THOUGHT.md).

## 2. FLY-DIAMOND-NEXUS

Inspected: root README.md, DIAMOND-SIM-FLY/README.md (current expansion sections), DIAMOND-SIM-FLY/docs/fly-lab-22-design.md, DIAMOND-SIM-FLY/src/fly-supermix-client.js, DIAMOND-SIM-FLY/src/fly-brain-engine.js (22-module construction, normalized coupling, external-advice validation and application), root LICENSE, malecns/LICENSE.md header.

Mechanisms observed: educational browser worlds plus a synthetic fly-inspired specialist ensemble. The 22-module class extends the 16-module system with six bounded functional specialists. Incoming coupling weights are L1-normalized per destination independently of graph size; editing one projection preserves the destination budget. The sidecar client distinguishes telemetry from inference-ready status, rejects stale asynchronous responses, and ties expiry to the observation tick. Engine advice is a four-action vector with influence at most 0.35 and lifetime at most 30 ticks. Immediate threat evidence reduces its effective influence.

Lessons: asynchronous advice needs freshness and residency evidence; adding modules must not silently increase total influence. Separate heuristic specialists, biological inspiration, actual imported connectome data, and measured advantage. Reusing a shared graph or adding another specialist ring would duplicate this project.

Limits: root README still says sixteen modules, while nested README and code contain 22. The design document requests a v92 no-load guarantee; the current nested README instead documents v91 residency checks without an atomic no-load guarantee. This review verified client/engine constraints, not a running model service. The README reports that a single Antennal Lobe baseline beat the sixteen-module system in an older comparison, which cautions against architecture-size claims; that experiment was not rerun.

Sources: [current nested README](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/blob/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/DIAMOND-SIM-FLY/README.md), [engine](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/blob/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/DIAMOND-SIM-FLY/src/fly-brain-engine.js), [client](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/blob/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/DIAMOND-SIM-FLY/src/fly-supermix-client.js), [design](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/blob/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/DIAMOND-SIM-FLY/docs/fly-lab-22-design.md), [vendored license](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/blob/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/malecns/LICENSE.md).

## 3. QuantumBot

Inspected: README.md (mechanism, comparison, ablation and limits sections), quantumbot/policies.py, quantumbot/benchmark.py, LICENSE.

Mechanisms observed: interchangeable quantum, MLP, linear, scripted-greedy and random policies share a bounded observation/action contract. The quantum policy delegates to a small classical statevector/density-matrix engine; it is not quantum hardware. Classical baselines use explicit parameter budgets; random policies require a supplied RNG. Benchmark conditions train on identical seed lists and comparisons join results by seed before bootstrapping paired differences and performing permutation tests. Summary performance uses the final fifth of generations, with learning-curve AUC and elapsed time also recorded.

Lessons: a matched simple baseline can defeat an attractive mechanistic story; preserve negative and null results. Parameter counts, optimization budgets, initial bias, world exposure, and actual computation need distinct reporting.

Overlap to avoid: another evolving forager, tiny quantum controller, architecture ablation dashboard, or multi-seed comparison wrapper.

Limits: reported default quantum performance ties the stronger linear/scripted baselines rather than establishing quantum advantage; its default circuit is unentangled. README performance, speed, Qiskit correctness, and initialization-trap numbers were not reproduced in this review. Parameter matching is approximate for architectures with discrete hidden widths, and equal steps do not establish equal compute.

Sources: [README](https://github.com/kai9987kai/QuantumBot/blob/42e4fd637da2df1da5241da484ebc2c625050e72/README.md), [policies](https://github.com/kai9987kai/QuantumBot/blob/42e4fd637da2df1da5241da484ebc2c625050e72/quantumbot/policies.py), [benchmark](https://github.com/kai9987kai/QuantumBot/blob/42e4fd637da2df1da5241da484ebc2c625050e72/quantumbot/benchmark.py).

## 4. Causeway

Inspected: README.md, src/model.js, LICENSE.

Mechanisms observed: a synthetic watershed policy workbench with twelve fixed grower archetypes, a 52-week horizon, four policies and four assumption lenses. Rain, heat, market pulses, initial practice and adoption decisions use keyed pseudorandom events based on seed, week, actor and channel. This preserves alignment when previous policy decisions make trajectories diverge. Simulation state includes water, habitat, adoption, allocation, yield and trust. Contrasts pair final outcomes by seed; intervals are mean difference plus/minus 1.96 standard errors.

Lessons: common event keys are stronger than merely sharing an initial RNG seed when interventions alter control flow. Source notes and model rules should be visible separately from fitted evidence. Sensitivity lenses expose dependence on modelling assumptions.

Overlap to avoid: generic evidence-to-experiment workbench, shared-resource policy simulator, configurable assumptions, run ledger.

Limits: hand-authored coefficients and archetypes, no bundled observations or calibration; its Monte Carlo intervals quantify this simulator rather than policy uncertainty in a real basin. Notes linked in the evidence library do not validate the implemented watershed.

Sources: [README](https://github.com/kai9987kai/Causeway/blob/e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac/README.md), [model](https://github.com/kai9987kai/Causeway/blob/e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac/src/model.js).

## 5. MOLT

Inspected: README.md, src/engine.js, LICENSE.

Mechanisms observed: paired synthetic cohorts compare a trace stored in three head cells against a trace stored in nine body cells. Fixed schedules train cue-associated scalar sums/counts; interventions blank selected carriers, replace turnover cells or graft opposite-trained donor traces. Test choices are generated by a fixed confidence rule applied to trace alignment. A separate finite food pool and energy/death process couples cohort outcomes. The same schedule, decision uniforms, intervention orders and cohort orders are reused across arms, while arm-specific resource stocks may diverge. Bootstrap units are paired seeds.

Lessons: retention and usefulness are different measurements; persistent information can become harmful after a cue reversal. Record post-intervention state separately from behavioural readout and ecological survival.

Overlap to avoid: tissue/head loss, regrowth memory retention, distributed traces, cell grafting, shared-resource toy organisms.

Limits: the head intervention is structurally arranged to remove head storage and spare body storage. That is an engineered mechanism demonstration, not discovery that biological body cells retain memory. No learned neural policy or organism model. Exports aggregate seed/arm and exposure outcomes rather than full event histories. Reported large reference effects were not rerun.

Sources: [README](https://github.com/kai9987kai/MOLT/blob/8ac04b1839b82d4e4d83166f1938bc6e1150197a/README.md), [engine](https://github.com/kai9987kai/MOLT/blob/8ac04b1839b82d4e4d83166f1938bc6e1150197a/src/engine.js).

## 6. MEMORY-CARRIER-OBSERVATORY

Inspected: README.md, src/engine.js, LICENSE.

Mechanisms observed: four trace channels (internal, body, habitat, colony) provide fixed weighted votes; the colony trace is explicitly shared by the twelve subjects in each seed. Six transfer contexts drop channels or reverse the cue. Removal studies use the same latent mapping, traces and test-decision noise. The full 16-subset by six-context study computes per-seed inclusion-exclusion coefficients, giving main and higher-order interactions on the accuracy scale. A leave-one-pair-out range exposes influential seed blocks. The next-probe suggestion ranks disagreement across five fixed hand-authored hypotheses; it does not update those hypotheses from results.

Lessons: an ablation result depends on context and interactions; shared signals create dependence; intervals and influence checks answer different questions. Decomposition on an outcome scale is not a unique decomposition of the underlying mechanism.

Overlap to avoid: another carrier-removal matrix, interaction atlas, context transfer, fixed-hypothesis probe recommender or memory-substrate observatory.

Limits: trace quality is an illustrative saturating function of training exposure, not neural training; votes and readout are engineered. Normal intervals are descriptive over simulation seeds and correlated cells, with no multiplicity correction. Run IDs are FNV fingerprints rather than authentication.

Sources: [README](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY/blob/523c10b46fa26acd53cd8468f95d3cbf55011437/README.md), [engine](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY/blob/523c10b46fa26acd53cd8468f95d3cbf55011437/src/engine.js).

## 7. Mnemorph

Inspected: README.md, src/core/compiler.js, LICENSE. Other subject, inference, receipt and intervention files were inventoried, not independently read in full.

Mechanisms observed: a typed bounded assay compiler enumerates valid operation sequences from a grammar, caps candidate evaluation at 256, scores expected information gain over a supplied hypothesis bank, penalizes invalid/all-arms-lost outcomes, and divides by an operation/tick cost proxy. The grammar can retain useful completed protocols when a before/after Brier improvement reaches 0.01. Validation blocks known auditor/ground-truth fields in public compiler inputs. README describes matched frozen-state forks, withheld-intervention scoring and six strategies, with adaptive revised grammar compared against frozen-grammar information gain under matched candidate/assay budgets.

Lessons: experiment selection is itself a model whose assumptions and budget need controls. Predict before observing; separate design search from withheld prediction and auditor scoring. Data-flow restrictions are weaker than semantic secrecy.

Overlap to avoid: information-gain assay selection, revisable intervention grammar, memory archaeology, adaptive experiment compiler, hidden-mechanism identification in a small synthetic family.

Limits: five predefined hand-coded mechanism families; no biophysical model. README explicitly notes that public replay wiring may reveal labels to source-informed readers. The confirmatory configuration is only an unlocked template; the smoke profile checks wiring and reproducibility rather than providing powered confirmation. This review did not verify the full subject/inference loop.

Sources: [README](https://github.com/kai9987kai/Mnemorph/blob/4fc3353b4aa4ea3e2c7b24798121ee65deda6939/README.md), [compiler](https://github.com/kai9987kai/Mnemorph/blob/4fc3353b4aa4ea3e2c7b24798121ee65deda6939/src/core/compiler.js).

## 8. TESSERA

Inspected: README.md, src/engine.js (interpreter inventory, exactCertificate, mineCommons, expiry, acquisition and frozen-transfer loops), LICENSE. Benchmark manifests were inventoried, not separately validated.

Mechanisms observed: expression-tree evolution over a two-input four-bit machine. Candidate subtrees are canonicalized by all 256 input/output pairs. Open opcode admission needs one lineage/family; ratified admission needs at least two single-founder lineages and two task families. Unused opcodes expire after two epochs. Transfer clones retained vocabulary, starts matched fresh populations and examples for each unseen task, freezes vocabulary, and runs the same number of population/generation slots. The transferred object is a symbolic function table.

Lessons: sharing needs a semantic contract and acquisition/transfer separation; new population initialization prevents inherited individuals from masquerading as transfer. Independent lineage is a property of the constructed provenance rules, not automatic empirical independence.

Overlap to avoid: program-library learning, reusable opcode commons, exhaustive four-bit DSL certificates, independent-lineage publication, frozen vocabulary transfer.

Limits: exact certificates establish equivalence only over the specified finite machine. They do not establish general usefulness or open-domain correctness. Composition split is public and fixed. Shorter opcodes change effective search geometry and execution cost; equal slots are not equal compute. No neural search or live code execution. Pilot effects were not reproduced.

Sources: [README](https://github.com/kai9987kai/TESSERA/blob/85e740452824a0ae089332e47d3473db879ba70c/README.md), [engine](https://github.com/kai9987kai/TESSERA/blob/85e740452824a0ae089332e47d3473db879ba70c/src/engine.js).

## 9. NEUROGRAFT

Inspected: README.md (base benchmark, current navigation/SR releases, evidence and limitations), neurograft/dynamics.py, LICENSE_MATRIX.md; recursive inventory of training/evaluation/controller modules. Current navigation/SR training implementations and saved reports were not fully audited.

Mechanisms observed: four source graphs have separate sparse signed relation transforms, structural node features and GRU state updates. Pooling feeds learned query/key/value routing, excludes self routes, retains a null route and selects top-k cross-source senders. Synthetic bridges remain separate from immutable source graph records. Legacy readout uses the human module; an optional zero-initialized residual reads all modules, preserving initial legacy logits. An optional action-conditioned predictor forecasts the next observation. This is an actual differentiable trainable modular receiver, not only a diagram or control panel.

README also describes a delayed split-cue XOR task, no-graft/no-recurrence/rewired controls, held-out checkpoint replay, a frozen-Qwen interface, live teacher-assisted navigation, frozen observation-only controller studies, rotation-augmented readout studies, predictive recovery and successor-representation/replay experiments. These are adjacent to any proposed new modular-skills project.

Lessons: preserve source data provenance while treating learned bridges as artificial; check interventions against every readout path. In a model with all-source residual readout, disabling graph grafts does not remove direct donor access. Raw learned policy, legal masking, observed-map planning and teacher help need separate scores.

Overlap to avoid: graph grafting, delayed memory integration, source lesions, cross-source route telemetry, zero-gated frozen-LLM adapter, another synthetic navigational successor/replay mechanism.

Limits: README reports authentic normalized worm input but synthetic fly/human fixtures, randomly initialized tiny Qwen tests, no loaded pretrained Qwen, no established biological topology advantage. The original five-seed graft benchmark is unstable (one seed solves, four at chance) and its interval crosses zero. Observation-map and map-only controls both solve 24/24 scenarios in a reported study, while raw policy solves 0/24; symbolic assistance is not evidence of neural learning. Those source-reported values were not rerun. Licensing is unresolved for code publication and some source tables; public availability is not a license grant.

Sources: [README](https://github.com/kai9987kai/NEUROGRAFT/blob/2301f0af16a92a9722608c868e4719f1898ca0fe/README.md), [trainable dynamics](https://github.com/kai9987kai/NEUROGRAFT/blob/2301f0af16a92a9722608c868e4719f1898ca0fe/neurograft/dynamics.py), [license matrix](https://github.com/kai9987kai/NEUROGRAFT/blob/2301f0af16a92a9722608c868e4719f1898ca0fe/LICENSE_MATRIX.md).

## Cross-portfolio implications

The portfolio already covers:
1. Trace location, persistence, destructive intervention and regeneration.
2. Carrier interactions and transfer-context effects.
3. Adaptive assay selection and grammar revision.
4. Symbolic reusable functions and independent-lineage admission.
5. Trainable cross-source graph routing, source lesions and frozen-model interfaces.
6. Local evidence receipts, selective answer authority, labelled static risk calibration and stateful experiment preflight.
7. Multi-agent and quantum/classical policy baselines with deterministic paired evaluation.

A new project should import these as requirements and lessons, not assemble renamed versions as its novelty claim. There is no evidence here that a single unified “brain” made from all repositories would improve capability.

## Candidate missing question

A comparatively unoccupied question in the inspected code is:

**When specialists share evidence or copy one another, and feedback is observed mainly for the system's own selected answers, can a receiver learn when agreement is real corroboration and maintain useful selective accuracy after a source changes?**

This is an inference-and-feedback question rather than memory transfer, organism simulation, causal assay discovery, opcode sharing or graph grafting. Supermix already has consensus counting and frozen labelled risk evaluation, but the inspected fusion function does not correct for common evidence ancestry and the risk module explicitly assumes independent labelled calibration records. TESSERA uses lineage as an admission rule, but does not estimate the decision value of correlated specialist agreement. Absence here is scoped to inspected files, not a proof that no related code exists anywhere in the portfolio.

A minimal CPU-local instrument could use typed classification or bounded symbolic tasks with known source-error ancestry, delayed/missing labels and controlled reliability changes. It should compare majority/weighted vote, independence-assuming combination, covariance or provenance-aware combination, frozen calibration, and an auditor-only full-label diagnostic. Random audit labels must be collected independently of acceptance so that a controller cannot appear calibrated merely by avoiding hard cases. Independent-label and shared-error settings are necessary null controls. Report coverage, accepted risk, calibration, audit-label cost, detection delay and source-group sensitivity separately.

The interesting mechanism would be learning effective evidence units and purchasing a small unbiased audit stream under shared error and shift; a new ledger or confidence score is not the contribution. An initial synthetic benchmark would establish software behaviour only. Real model-output or real-data tracks would be needed before interpreting it as a practical Supermix improvement.

## Critique of the alternative donor-intervention receiver direction

Proposed direction reviewed: train a tiny modular receiver with paired donor interventions to make frozen specialist skills causally useful on held-out compositions; use missing/shuffled/corrupt-donor controls, plus a real-data digits track.

**This is feasible and researchable, but “trainable modular receiver with ablations” alone is already substantially occupied by NEUROGRAFT.** Its source has learned sparse bridges, recurrence, all-source residual readout, delayed split cues, source interventions and a differentiable frozen-language interface. TESSERA already measures frozen acquired vocabulary on unseen compositions. Replacing graph messages with expert vectors and calling them donors changes the implementation, not automatically the research question.

A sharper distinct question would be: **Can intervention-consistency training teach a receiver the semantic effect of a frozen donor, so that its counterfactual output changes remain correct on unseen donor/receiver compositions, rather than merely exploiting donor identity or answer-correlated shortcuts?**

For this sharper direction:
- Freeze independently trained donors and verify their own skill first. Keep donor training data and the receiver's composition test split disjoint at the right semantic unit.
- Separate three outcomes: receiver task accuracy, donor usefulness, and correct mediated counterfactual response. A performance drop after removal establishes dependence, not that the intended skill was transferred.
- Include donor identity permutation, semantically equivalent donor re-encoding, label-preserving input changes, wrong-but-confident donor outputs, irrelevant competent donors, and matched corrupted channels. Compare to a receiver with equal input bandwidth and parameters.
- A missing/shuffled donor is often off-distribution. Train/evaluate controls that match channel distribution, and measure whether the receiver responds to the donor's task-relevant change rather than its availability.
- Pair interventions before scoring and ensure the receiver never sees hidden intervention labels, task answers, test split identity or oracle donor selection.
- Compare ordinary supervised fusion, concatenation/attention, randomized intervention training and a donor-free receiver under matched examples/optimizer budgets. Resource fairness needs actual timing as well as parameters.
- A digits track adds non-synthetic pixels but does not by itself establish open-world composition, causal semantics or general intelligence. Hold out combinations or transformation families, not only random images.
- Avoid direct code reuse from NEUROGRAFT until its publication license is clarified; conceptual reuse is enough for a clean CPU-local implementation.

Recommendation: if selecting this direction, center the project on **semantic intervention consistency and compositional generalization of a frozen specialist interface**, with donor-use ablations as controls. Do not center it on the existence of modules, grafts, memory, a receiver, a dashboard, or a ledger. Novelty remains a candidate claim requiring primary-literature comparison.


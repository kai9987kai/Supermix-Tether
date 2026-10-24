# Systems repository review

Review date: 24 October 2026. Research-only, selected-file source inspection through the connected GitHub service. No upstream checkout, installation, execution, integration, or benchmark reproduction was performed. Branches were resolved once and files were read at those immutable commits. This report distinguishes observed code from upstream documentation claims.

## Snapshot and access ledger

All nine repositories were accessible through GitHub metadata, branch and recursive-tree reads. No requested file read failed. The Flipper tree has no README or license file, so its review uses source and application metadata. Recursive tree responses were not truncated. Larger source files were inspected selectively, as stated below.

| Repository | Reviewed branch | Pinned commit | Observed licensing |
|---|---|---|---|
| [universal-modder](https://github.com/rehan-remade/universal-modder) | main | `c1dff21663665dd17af289744a301b66e369605e` | MIT root LICENSE |
| [REA](https://github.com/morluto/rea) | main | `bc9efc9279ea924031bcdc62e4fae936c392fd23` | MIT root LICENSE; external providers/dependencies separate |
| [Odysseus](https://github.com/odysseus-dev/odysseus) | dev, repository default | `a0c93f38ef196aff5c2bdb90ba2c64419254d012` | AGPL-3.0-or-later per README, AGPL v3 root LICENSE |
| [NexusFlow](https://github.com/kai9987kai/nexusflow) | main | `3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec` | MIT |
| [Flipper periodic NTAG emulator](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator) | main | `98fd7478253f9041195adc8eedfa95eabe084aff` | No license observed in complete small tree |
| [quibble-builds](https://github.com/7coil/quibble-builds) | master | `f89481a1a3b81abfcbc1d43a2d539204101201aa` | LGPL v3 root LICENCE; boot.cpp permits v3 or later; bundled material has additional notices |
| [SVideo](https://github.com/7coil/svideo) | master | `d4b34633e92be30fd6395da1ffde21d6cac96917` | MIT root LICENCE |
| [Supermix-Poseidon](https://github.com/kai9987kai/Supermix-Poseidon) | main | `72d42218ae92cee5c9668a090f9d31d8d9961681` | Original code/Tidal MIT; upstream model/data/dependency terms separate |
| [FAULTLINE](https://github.com/kai9987kai/FAULTLINE) | main | `b7ae1b9e97a3258cbc44c44e928d7e937487a9f8` | MIT |

Odysseus README calls dev the newest branch and main the more curated branch; this review uses the default dev, not an inferred main snapshot. The user's requested main snapshots for universal-modder and FAULTLINE were followed.

## 1. universal-modder

Inspected: [README](https://github.com/rehan-remade/universal-modder/blob/c1dff21663665dd17af289744a301b66e369605e/README.md) introduction, workflow and knowledge-base sections; [um/kb.py](https://github.com/rehan-remade/universal-modder/blob/c1dff21663665dd17af289744a301b66e369605e/um/kb.py), especially lines 167–297; [LICENSE](https://github.com/rehan-remade/universal-modder/blob/c1dff21663665dd17af289744a301b66e369605e/LICENSE). Selected source sections, not all CLI or engine playbooks.

Mechanism: agent skills, a CLI, asset-service adapters and engine playbooks organize a concrete discover/build/run/verify/package workflow. Its shared knowledge base stores versioned field notes with routes, verification status and symptom/cause/fix records. The observed search implementation is weighted lexical matching over title/game/engine/route/tags/tools/body, with query phrase handling and matching snippets. It is not a learned semantic memory. The note validator checks front matter, route/status categories, required sections, template remnants, potential secrets/decompiled dumps, links and media.

Lesson for Supermix: reuse the discipline of recording an experiment's exact environment/version, failed routes and observed verification. Searchable experience records need schema validation and provenance before reuse.

Overlap to avoid: another general agent skill bundle, game-modding orchestrator or lexical field-note store. None would supply a distinct learning mechanism.

Limitations: no games were launched and community notes were not independently validated. MIT covers the repository's software; owned game files, decompiled content, generated assets and paid service outputs remain separate rights/contracts.

## 2. REA

Inspected: [README](https://github.com/morluto/rea/blob/bc9efc9279ea924031bcdc62e4fae936c392fd23/README.md) introduction, local analysis/provider/requirements sections; complete [reconstruction-readiness design](https://github.com/morluto/rea/blob/bc9efc9279ea924031bcdc62e4fae936c392fd23/docs/reconstruction-readiness.md); [LICENSE](https://github.com/morluto/rea/blob/bc9efc9279ea924031bcdc62e4fae936c392fd23/LICENSE). This is documentation-level architecture review, not provider/runtime source audit.

Mechanism: local static/runtime reverse-engineering workflows exposed through MCP and CLI, with typed evidence and explicit provider/platform limits. The readiness document makes an unusually important separation: observed native/JavaScript static-analysis journeys are distinct from a wholly synthetic nine-stage report evaluator. Its evaluator checks evidence commitments, contradictions, stale digests, incomplete authority, partial-order semantics, obligation closure and replay consistency. It does not independently observe the events represented by supplied observations.

Lesson: a replayable or internally consistent report is evidence about a computation and its inputs, not automatically evidence that a real-world event occurred. Preserve unknown/truncated/unsupported statuses, and distinguish actual executed journeys from synthetic evaluator conformance.

Overlap to avoid: a generic evidence ledger, reconstruction wrapper or unified MCP interface as the claimed scientific contribution.

Limitations: the inspected document explicitly excludes fixture runtime behavior, native MCP parity and end-to-end reconstruction closure from its static readiness result. No native provider was installed or exercised. External decompilers and inspected application assets retain separate licenses.

## 3. Odysseus

Inspected: complete [README](https://github.com/odysseus-dev/odysseus/blob/a0c93f38ef196aff5c2bdb90ba2c64419254d012/README.md), complete [agent turn contract](https://github.com/odysseus-dev/odysseus/blob/a0c93f38ef196aff5c2bdb90ba2c64419254d012/docs/AGENT_TURN_CONTRACT.md), root [LICENSE](https://github.com/odysseus-dev/odysseus/blob/a0c93f38ef196aff5c2bdb90ba2c64419254d012/LICENSE). Architecture/design review only; no live deployment or full agent runtime audit.

Mechanism: self-hosted AI workspace combining chat/agents, tools/MCP, research, documents, email, calendar, local/API models and comparisons. The turn contract separates capability selection from permission. It freezes immutable serialized schemas and enforces required capabilities contained in offered capabilities contained in executable capabilities. Missing capabilities stop visibly; compound requests cannot finish after one family. Dispatcher membership and runtime authorization are separate checks, and output ownership is tracked across streamed/canonical rendering and saved history.

Lesson: experiments should freeze the actual admissible action/query inventory, distinguish unavailable operations from failed inference, and retain resource/schema/producer identity through resume. User-facing status should reflect real completion.

Overlap to avoid: another local assistant workspace, broad workflow orchestrator or tool router. Those are infrastructure, not a new experimental question.

Limitations: deterministic vocabulary classification is not semantic understanding, remote providers can fail after registration, and the document does not claim all families passed live checks. AGPL code should not be casually copied into a separately MIT-labeled implementation; an independent implementation of general design lessons is the clean boundary.

## 4. NexusFlow

Inspected: complete [README](https://github.com/kai9987kai/nexusflow/blob/3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec/README.md); selected [nexusflow/lang.py](https://github.com/kai9987kai/nexusflow/blob/3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec/nexusflow/lang.py) class inventory, Executor initialization around 919–1000 and simulation methods around 10996–11094; [LICENSE](https://github.com/kai9987kai/nexusflow/blob/3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec/LICENSE). The language/runtime is 13,546 lines; this was a targeted review, not all builtins.

Mechanism: a parsed DSL/IR executes agent simulations, state/metrics/channels, pipeline control, models, exports and polyglot/Windows steps. Executor explicitly stores runs and operation histories across these domains. Simulation supports single-thread ticking or per-agent chunks through ThreadPoolExecutor, records worker statistics, and awaits futures.

Lesson: declarative experiment plans and narrow typed adapters can make protocols inspectable, but scientific meaning must reside in the experiment and estimator. A language which can call a mechanism does not validate it.

Overlap to avoid: another portfolio orchestration DSL or a large registry of disconnected tools. CPU-local experiments can use ordinary Python plus a small JSON protocol without recreating a whole language.

Limitations: README labels the science modules coarse prototypes and 3D support primarily metadata/export. No determinism or parallel equivalence was tested here; thread locks and a seed field alone do not prove schedule-independent outcomes.

## 5. Flipper-Zero-Periodic-NTAG-Emulator

Inspected: complete [ntag213_rotator.c](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator/blob/98fd7478253f9041195adc8eedfa95eabe084aff/ntag213_rotator.c), [application.fam](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator/blob/98fd7478253f9041195adc8eedfa95eabe084aff/application.fam) and root tree. No README or license exists in the inspected tree.

Mechanism: a Flipper app periodically replaces a seven-byte UID, uses a fixed manufacturer byte plus hardware randomness, stops/frees the previous NFC listener when rotating, and starts an ISO14443-3A listener. A one-second event queue drives a roughly ten-minute countdown, with screen progress and allocation/start counters.

Lesson: identity, lifecycle and state continuity are separate concerns. A changed outward identifier is not equivalent to transferred memory or a newly learned policy. Resource cleanup and observable error states matter at interfaces.

Overlap to avoid: treating identifier rotation as cognition or importing this as a software-only NTAG hardware capability.

Limitations: source uses the generic ISO14443-3A protocol; this review does not establish a complete NTAG213 memory/command emulator, successful reader interaction, or hardware reliability. The success counter follows listener allocation/start path, not a verified NFC exchange. Hardware SDK compilation and hardware operation were not attempted. No observed license means no assumption of reuse permission.

## 6. quibble-builds

Inspected: [README](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/README.md), [LICENCE](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/LICENCE), complete [amd64 build workflow](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/.github/workflows/build-amd64.yml), selected [src/boot.cpp](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/src/boot.cpp) header/version-dispatch and EFI-driver-loading sections. boot.cpp has 5,439 lines; it was not fully audited.

Mechanism: an extensible Windows EFI loader with filesystem drivers, version-specific loader blocks, and EFI protocol boundaries. Source chooses Windows loader layouts by version and rejects unsupported versions. The build workflow checks out the workflow commit and produces debug/release EFI and Btrfs/NTFS driver artifacts named by commit SHA.

Lesson: compatibility must be explicit and versioned. Unsupported model/query interfaces should be rejected before execution. A runnable experiment package should bind artifacts to the actual source identity.

Overlap to avoid: an OS boot or hardware integration project. These mechanisms are useful deployment discipline, not evidence for learning or cognition.

Limitations: upstream README calls it a proof of concept and lists missing signature verification, ASLR, registry recovery and other work. No firmware, disk, boot entry or machine configuration was modified. Root LGPL and component notices (FreeType, font, compression libraries, GNU-EFI/HarfBuzz) require separate consideration if code/binaries are redistributed.

## 7. SVideo

Inspected: complete [README](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/README.md), complete [src/FFmpeg.ts](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/src/FFmpeg.ts), complete [src/ProjectBuilder.ts](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/src/ProjectBuilder.ts), [LICENCE](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/LICENCE).

Mechanism: converts videos to Scratch projects using tiled spritesheet costumes rather than one costume per frame. FFmpeg mpdecimate removes nearby duplicate frames, while a reverse-time index preserves playback repetition; scaling, padding, subtitles and audio export precede template injection of costume/audio metadata.

Lesson: compression is useful only if the task-relevant reconstruction contract remains explicit. For an experimental memory system, report both saved storage/calls and what decisions or observations can no longer be reconstructed.

Overlap to avoid: repackaging media compression or sprite export as semantic memory or general reasoning.

Limitations: the commit is from 2022. This review did not build the TypeScript package, run FFmpeg or validate Scratch output. MIT source terms are separate from codec/binary distribution and input-media rights.

## 8. Supermix-Poseidon: substantial overlap surface

Inspected: [README](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/README.md), [LICENSE](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/LICENSE), [THIRD_PARTY.md](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/THIRD_PARTY.md), research index, controlled-adaptation and Stage 9 reports, CAUSEWAY design; selected source below. This is a broad selected-file review, not a full audit of hundreds of modules/tests.

Observed mechanisms:

- [information_seeking.py](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/poseidon/information_seeking.py): targeted vs uniform-random probes at equal step budgets, isolated candidate adapters, anchor/return/survival admission checks. The probe score combines a heuristic epistemic drive, hand-set action bonuses and base logits. Its reported uncertainty reduction is a clipped observation-derived scalar, not posterior entropy over alternative models.
- [hypothesis_engine.py](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/poseidon/hypothesis_engine.py), especially 199–335: named rival hypotheses and caller-supplied discriminating predictions, experiment records, match-based tests and hashed receipts. Expected information is binary prior entropy multiplied by caller severity. Bayes factors are derived from severity/match outcomes, not an explicit likelihood model for candidate experiments.
- [discovery_lab/causal_experiments.py](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/poseidon/discovery_lab/causal_experiments.py): five preset buoyancy/support/enzyme hypotheses updated from event-string matches and support/falsification counts.
- [causal_scm.py](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/poseidon/causal_scm.py): hand-written TidePool equations, residual shock estimates, action projections and counterfactual regret attribution. This is an explicit assumed structural model, not a procedure for discovering which of multiple observationally equivalent graphs is identifiable.
- [nexus_omega/causal_efe.py](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/poseidon/nexus_omega/causal_efe.py): model-score planning with disagreement penalty and threshold action guard. Current module documentation explicitly says the guard is uncalibrated and supplies no distribution-free bound/safety guarantee. The older Stage 9 report calls the same form Conformal-SPI and uses certification language; that historical claim must not be adopted as a verified guarantee.

The current README is appropriately cautious about frozen core actions, selected artifacts, uncalibrated death targets, one-step planning support and unpromoted candidates. The research index contains both measured studies and ambitious historical concept reports; their status must be checked at the implementation/evaluation layer.

Lesson: active probes, causal words, rival hypotheses and receipts are already present. A new project needs an experimentally distinct estimator or decision rule, with negative controls and an explicit model-misspecification boundary.

Overlap to avoid: another TidePool cognitive coordinator, memory hierarchy, heuristic epistemic action bonus, preset hypothesis counter, symbolic vital projector, or receipt-only laboratory. A likelihood-based, costed intervention scheduler with query-specific identifiability and valid adaptive error control would be materially stronger than the inspected mechanisms, but absence elsewhere in this large repository is not proven by this limited review.

Limitations: no model weights were downloaded or executed; upstream performance and test counts were not reproduced. Poseidon code/Tidal MIT does not relicense SmolLM2, smol-smoltalk, dependencies or cited repositories. THIRD_PARTY identifies the upstream model/data revisions and Apache-2.0 terms separately.

## 9. FAULTLINE: exact current scope

Inspected: complete [README](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/README.md), complete [faultline/core.py](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/faultline/core.py), complete [docs/design.md](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/docs/design.md), complete [faultline/study.py](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/faultline/study.py), complete [faultline/receipts.py](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/faultline/receipts.py), fixtures function inventory/parity interface, selected [docs/paper.md](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/docs/paper.md) mathematical and boundaries sections, LICENSE. No empirical receipt was replayed locally.

Actual research question: fixed-budget selection of small odd voting panels from fully observed component predictions under correlated/higher-order failures. Inputs carry complete binary predictions and known condition labels, with distinct calibration and test rows.

The four implemented selectors exhaustively enumerate panels:

1. accuracy minimizes summed pooled individual errors.
2. pairwise minimizes the worst-condition second factorial-moment bound.
3. pooled minimizes empirical pooled majority error.
4. joint minimizes empirical worst-condition majority error.

Selection takes only calibration rows; test outcomes are evaluated after freezing panels. Exact Fraction keys and lexical IDs make selection deterministic. Metrics include average/worst-condition error, unanimous wrong rates, pair co-failure matrix, complete observed failure signatures and test witnesses. Costs separate full-component calibration from panel deployment and search-row evaluations. Dataset bounds are 3–12 components and 12,000 total rows.

Its identifiability result is specific: equal marginal and pairwise moments can mask different majority error, shown by three-bit even/odd parity. It is not causal graph identification or a procedure that learns latent common causes.

Its experiments are fixed synthetic scenarios (shared, independent, unseen common-cause shift) and paired-seed panel comparisons. The study runner computes descriptive percentile bootstrap intervals over seed differences. No adaptive choice of which observation/intervention to purchase is present in the reviewed computation modules. No sequential stopping, e-process, conformal risk certificate or anytime-valid risk control is implemented there. The design/paper explicitly disclaim causal identification and sequential/conformal guarantees.

Its receipts are stronger than a bare checksum: source files are hashed; imported datasets are bound; verification recomputes demo/study/parity/analyze outputs and rejects rehashed false result payloads. This verifies deterministic recomputation under supplied data/source, not label truth, semantic split independence or deployment safety.

Upstream confirmation tables report joint-selection benefit in the shared scenario and substantial failure under unseen shift; those are repository-reported synthetic results, not reproduced results in this review.

Lesson: full dependence matters, but empirical failure signatures cannot protect against unobserved common causes. A new project should operate upstream of this selection problem or answer another question.

Overlap to avoid: another joint-failure hypergraph selector, parity demonstration, static sensor ensemble, descriptive bootstrap lab or source-bound receipt wrapper.

## Synthesis boundary for a distinct project

The systems/tools repositories primarily contribute engineering constraints: explicit interfaces, bounded resources, immutable identities, accurate failure status, reconstruction contracts and evidence discipline. They should not be added as extra named cognitive subsystems.

FAULTLINE does not occupy the full space of active experiments or causal identifiability. Poseidon already occupies the vocabulary and basic scaffolds of both. A distinct candidate can therefore target a precise gap: given rival finite causal mechanisms that agree observationally, choose the next affordable intervention using action-specific observation likelihoods, and decide whether a particular downstream decision is identifiable under the permitted intervention set. Retain irreducible equivalence classes and abstain when all permitted experiments leave conflicting decisions.

To be distinct and falsifiable, such a candidate should compare equal-cost random/balanced/information-seeking/decision-focused acquisition, include useless interventions and true non-identifiability controls, test omitted-mechanism misspecification, and measure decision regret and experiment cost rather than general intelligence. Statistical guarantees require their own assumptions and derivation; heuristic entropy, a 1.96 multiplier, repeated test success and SHA-256 receipts do not confer them.

This is a source-grounded possible direction, not a global novelty claim. Current primary literature review remains necessary before choosing the final mechanism.

## Critique of the proposed donor-intervention receiver direction

The subsequently proposed direction is a compact trainable Supermix receiver distilled from three frozen image specialists, using relevant donor swaps for effect alignment and irrelevant donor swaps for invariance. Task queries select source and transformation; held-out source-by-transformation combinations and spurious-cue reversals test systematic causal use. This is materially distinct from the mechanisms inspected here: FAULTLINE freezes a voting panel from labeled failure tables, while Poseidon's selected causal/probing modules use assumed dynamics, action bonuses or match-based hypothesis records. None of those inspected paths trains a receiver with the proposed dual interchange-effect and irrelevant-swap objective. This is a bounded source observation, not proof of absence from every Poseidon module or every repository.

Useful study constraints: split underlying digit-image identities before generating transformed/triple examples; keep all variants of an image in one split; freeze donor training and receiver development before test; construct intervention pairs without using test labels for selection; count donor calls, receiver forwards, optimizer steps and trainable parameters separately. A same-triple augmentation control must consume the same examples and forward budget. Include relevant-only and irrelevant-only objective ablations, no-donor/shuffled-donor controls, and report normal accuracy alongside intervention accuracy and spurious-cue reversal. Improved observational accuracy alone cannot establish causal donor use. Any novelty language should acknowledge IIT/causal distillation and limit the candidate contribution to the particular dual objective and measured portfolio gap.


# Portfolio review and Tether design lineage

Reviewed live on 24 October 2026 through the connected GitHub service. The three detailed reports cover all 27 supplied repositories. Every repository was accessible. Default branches and branch heads were resolved through repository/branch metadata, and selected files were read at the pinned commits below.

This is a selected-file source and design review, not a full code audit, benchmark reproduction, checkpoint evaluation or verification of every upstream claim. Large repositories were sampled around relevant mechanisms. A repository-reported result remains an upstream claim unless independently reproduced. The Flipper repository contains no README or license in its inspected tree, so its review uses source and application metadata. Connector URL encoding issues for two slash-containing branch names were resolved; they were not repository access failures.

The detailed reports were copied byte-for-byte from the completed research reports:

- [Supermix, developmental systems and donor-use review](reviews/supermix.md): nine repositories, including the Expanse evidence that motivates Tether.
- [Memory, modularity and evidence review](reviews/memory.md): nine repositories, including NEUROGRAFT and TESSERA.
- [Systems, tools, Poseidon and FAULTLINE review](reviews/systems.md): nine repositories, including third-party tools and overlap boundaries.

The root research originals remain intact. These documentation copies preserve the original inspection depth, source links, caveats and upstream licensing observations.

## The selected research question

Tether asks whether matching task-relevant donor intervention effects while suppressing irrelevant and damaged-expert effects improves **correct task-specific donor use** on unseen source/operation combinations and proxy-cue shifts, beyond an identical-data interchange-intervention baseline.

The gap is narrower than adding modules or making a receiver sensitive to donor removal. A donor may contain decodable information that the receiver ignores; a receiver may depend on a donor for the wrong reason; and corruption may reduce accuracy without showing that the intended donor skill was used. Tether therefore separates ordinary task accuracy, correct relevant counterfactual response, nuisance invariance, and damage sensitivity.

Several portfolio mechanisms already occupy adjacent ground:

- **NEUROGRAFT already has a trainable differentiable modular receiver**, sparse cross-source bridges, recurrent states, readout alternatives and source interventions. Modules, routing and donor ablations are not Tether's distinctive claim.
- **TESSERA already tests composition** by freezing acquired symbolic vocabulary and starting matched fresh searches on unseen compositions. A composition split alone is not a new research contribution.
- **Supermix already provides risk and evidence mechanisms**, including frozen labelled selective-risk calibration, advisory stopping and route experiment preflight. A new confidence wrapper or ledger is not the selected mechanism.
- **Supermix-Poseidon already contains planning, probes, rival hypotheses and causal-model scaffolds.** Tether does not recreate an agent coordinator or call heuristic epistemic scores calibrated inference.
- **FAULTLINE already studies dependence in specialist failures** and fixed voting-panel selection. Tether's donors share training lineage; architecture heterogeneity does not establish independent evidence.

Interchange Intervention Training and causal distillation are direct external prior art, described in [the detailed review](reviews/supermix.md#follow-up-critique-modular-receiver-with-causal-use-training). The candidate contribution is the particular relevant-effect/invariance objective and controlled study, not the invention of neural interventions, model composition or causal distillation. Its utility and novelty remain hypotheses.

## Complete pinned inventory

Each row names a distinct lesson and the concrete Tether design choice it informed. Pinned artifact links identify an inspected source or design file; detailed reports record additional paths and limits. These are review snapshots, not assertions that the same heads will remain current.

### Developmental systems and Supermix ancestry

| Supplied repository | Reviewed default branch | Pinned head SHA | Distinct lesson mapped to Tether |
| --- | --- | --- | --- |
| [kai9987kai/NexusSearch](https://github.com/kai9987kai/NexusSearch) | main | 538edcba595889090946a51ce03f72ac9fb72202 | [Exact inspectable reference](https://github.com/kai9987kai/NexusSearch/blob/538edcba595889090946a51ce03f72ac9fb72202/docs/SEARCH.md): use literal digit-task oracles and frozen data identities before judging learned behavior. |
| [kai9987kai/prometheus-alpha](https://github.com/kai9987kai/prometheus-alpha) | claude/epic-tesla-l7brlx | 417009e5548453338761732152b7b8ec30ffa79a | [State can be available without behavioral use](https://github.com/kai9987kai/prometheus-alpha/blob/417009e5548453338761732152b7b8ec30ffa79a/src/prometheus/tissue.py): test predicted donor effects instead of equating decodability with useful transfer. |
| [kai9987kai/3d-animal-simulator-Hybrid-Agent](https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent) | main | b7dd934770380a28717ddc82e3dfa7f127732a75 | [Freeze the baseline before interventions](https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent/blob/b7dd934770380a28717ddc82e3dfa7f127732a75/src/experiments.js): construct aligned views from the same factual batch and keep observation noise separate. |
| [kai9987kai/morpheus](https://github.com/kai9987kai/morpheus) | claude/zealous-euler-y2xqul | 5db6f98d605b80245ed9ff5e2cc9d42151830a14 | [A stable intervention output is not an enduring target change](https://github.com/kai9987kai/morpheus/blob/5db6f98d605b80245ed9ff5e2cc9d42151830a14/src/morpheus/tissue.py): separate immediate counterfactual fidelity from held-out learned generalization. |
| [kai9987kai/Supermix-Expanse-v2](https://github.com/kai9987kai/Supermix-Expanse-v2) | main | 33e38a279c86bccdb1e5680f15f193d61a3f71cb | [Module activity and loss reduction can miss behavioral usefulness](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/EVOKED_CNS_DESIGN.md): compare generated task outcomes and causal-use measures against matched controls. |
| [kai9987kai/GhostInTheMachine](https://github.com/kai9987kai/GhostInTheMachine) | main | 239f5de6452f78f3960de07c393268eeac142d03 | [Signal timing and closed-loop dependence need separate interventions](https://github.com/kai9987kai/GhostInTheMachine/blob/239f5de6452f78f3960de07c393268eeac142d03/src/ghost_v7/engine.py): distinguish relevant swaps, irrelevant swaps and damage rather than merging them into one sensitivity score. |
| [kai9987kai/Supermix-expanse](https://github.com/kai9987kai/Supermix-expanse) | main | 3fd091e9cc3387b2c17da5ce7514831d77e2af44 | [Extra training can explain apparent graft gains](https://github.com/kai9987kai/Supermix-expanse/blob/3fd091e9cc3387b2c17da5ce7514831d77e2af44/expanse/V2_DESIGN.md): match receiver architecture, initialization, examples and update/view budgets across objective arms. |
| [kai9987kai/GenesisEngine](https://github.com/kai9987kai/GenesisEngine) | main | a43099ef21e91b75b812e3eb883cb4f3813ab7c7 | [Inherited structure differs from learned state](https://github.com/kai9987kai/GenesisEngine/blob/a43099ef21e91b75b812e3eb883cb4f3813ab7c7/src/genesis/neural.py): freeze pretrained visual donors and train the receiver interface separately. |
| [kai9987kai/supermix-archimedes](https://github.com/kai9987kai/supermix-archimedes) | main | 1952cf1c92cb506fff60f9ee0c055c2365988e5e | [Dimensional compatibility is not skill transfer](https://github.com/kai9987kai/supermix-archimedes/blob/1952cf1c92cb506fff60f9ee0c055c2365988e5e/archimedes/build_archimedes.py): verify donor competence, receiver behavior and restoration independently. |

Detailed scope, reported negative results and licensing boundaries: [supermix.md](reviews/supermix.md).

### Memory, specialist ensembles and modular reception

| Supplied repository | Reviewed default branch | Pinned head SHA | Distinct lesson mapped to Tether |
| --- | --- | --- | --- |
| [kai9987kai/Supermix](https://github.com/kai9987kai/Supermix) | main | 6ea1c2cc41e3705b4b899b764a001cec180d155d | [Risk calibration has an explicit frozen-labelled scope](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/nexus_risk_control.py): keep measured donor-use outcomes separate from risk certificates, runtime authority and promotion. |
| [kai9987kai/FLY-DIAMOND-NEXUS](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS) | main | 0c8d4100b21d8889bacdef1a9c6dfc20b3424242 | [More specialists must not silently increase influence](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/blob/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/DIAMOND-SIM-FLY/src/fly-brain-engine.js): fix donor count and receiver capacity across arms; normalize learned expert weighting. |
| [kai9987kai/QuantumBot](https://github.com/kai9987kai/QuantumBot) | main | 42e4fd637da2df1da5241da484ebc2c625050e72 | [A simple matched baseline can defeat an architectural story](https://github.com/kai9987kai/QuantumBot/blob/42e4fd637da2df1da5241da484ebc2c625050e72/quantumbot/policies.py): compare passive, dropout and ordinary intervention training; report actual time separately from parameter matching. |
| [kai9987kai/Causeway](https://github.com/kai9987kai/Causeway) | main | e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac | [Shared random events need explicit alignment after paths diverge](https://github.com/kai9987kai/Causeway/blob/e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac/src/model.js): use local deterministic batch generators and identical intervention draws across matched arms. |
| [kai9987kai/MOLT](https://github.com/kai9987kai/MOLT) | main | 8ac04b1839b82d4e4d83166f1938bc6e1150197a | [Retained information may become harmful after a world change](https://github.com/kai9987kai/MOLT/blob/8ac04b1839b82d4e4d83166f1938bc6e1150197a/src/engine.js): test correlated, independent and always-wrong contextual proxies separately. |
| [kai9987kai/MEMORY-CARRIER-OBSERVATORY](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY) | main | 523c10b46fa26acd53cd8468f95d3cbf55011437 | [Removal effects depend on context and interactions](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY/blob/523c10b46fa26acd53cd8468f95d3cbf55011437/src/engine.js): retain relevant-only and invariance-only objective ablations rather than judging only the combined loss. |
| [kai9987kai/Mnemorph](https://github.com/kai9987kai/Mnemorph) | main | 4fc3353b4aa4ea3e2c7b24798121ee65deda6939 | [Predictions and budgets should precede observed outcomes](https://github.com/kai9987kai/Mnemorph/blob/4fc3353b4aa4ea3e2c7b24798121ee65deda6939/src/core/compiler.js): freeze coefficients and evaluation draws before fresh seed studies; separate auditor targets from inference inputs. |
| [kai9987kai/TESSERA](https://github.com/kai9987kai/TESSERA) | main | 85e740452824a0ae089332e47d3473db879ba70c | [Composition needs a separate frozen transfer phase](https://github.com/kai9987kai/TESSERA/blob/85e740452824a0ae089332e47d3473db879ba70c/src/engine.js): hold out source/operation pairs while acknowledging that symbolic-library transfer already exists upstream. |
| [kai9987kai/NEUROGRAFT](https://github.com/kai9987kai/NEUROGRAFT) | main | 2301f0af16a92a9722608c868e4719f1898ca0fe | [Trainable modular reception is already implemented](https://github.com/kai9987kai/NEUROGRAFT/blob/2301f0af16a92a9722608c868e4719f1898ca0fe/neurograft/dynamics.py): center Tether on correct task-specific effects and invariances, not the existence of a learned donor interface. |

Detailed scope, documentation discrepancies and licensing boundaries: [memory.md](reviews/memory.md).

### Systems, workflow, planning and failure dependence

| Supplied repository | Reviewed default branch | Pinned head SHA | Distinct lesson mapped to Tether |
| --- | --- | --- | --- |
| [rehan-remade/universal-modder](https://github.com/rehan-remade/universal-modder) | main | c1dff21663665dd17af289744a301b66e369605e | [Reusable field notes require schema and verification status](https://github.com/rehan-remade/universal-modder/blob/c1dff21663665dd17af289744a301b66e369605e/um/kb.py): retain source versions, failed checks and exact experiment configuration in durable reports. |
| [morluto/rea](https://github.com/morluto/rea) | main | bc9efc9279ea924031bcdc62e4fae936c392fd23 | [Internal report consistency differs from observing a real event](https://github.com/morluto/rea/blob/bc9efc9279ea924031bcdc62e4fae936c392fd23/docs/reconstruction-readiness.md): state precisely what trace recomputation and inference replay verify. |
| [odysseus-dev/odysseus](https://github.com/odysseus-dev/odysseus) | dev | a0c93f38ef196aff5c2bdb90ba2c64419254d012 | [Capabilities and completion need an explicit contract](https://github.com/odysseus-dev/odysseus/blob/a0c93f38ef196aff5c2bdb90ba2c64419254d012/docs/AGENT_TURN_CONTRACT.md): show unavailable/missing studies clearly and freeze the actual task/query inventory. |
| [kai9987kai/nexusflow](https://github.com/kai9987kai/nexusflow) | main | 3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec | [A declarative pipeline does not establish scientific validity](https://github.com/kai9987kai/nexusflow/blob/3fb4be7e7f9a523bcd817bd2bd94a2d7855229ec/nexusflow/lang.py): keep Tether's protocol small and inspectable instead of adding a general orchestration language. |
| [djpiper28/Flipper-Zero-Periodic-NTAG-Emulator](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator) | main | 98fd7478253f9041195adc8eedfa95eabe084aff | [An outward identity is separate from underlying state](https://github.com/djpiper28/Flipper-Zero-Periodic-NTAG-Emulator/blob/98fd7478253f9041195adc8eedfa95eabe084aff/ntag213_rotator.c): keep image identity, donor identity, source role and checkpoint identity distinct in task construction and receipts. |
| [7coil/quibble-builds](https://github.com/7coil/quibble-builds) | master | f89481a1a3b81abfcbc1d43a2d539204101201aa | [Compatibility must be versioned and unsupported interfaces rejected](https://github.com/7coil/quibble-builds/blob/f89481a1a3b81abfcbc1d43a2d539204101201aa/src/boot.cpp): validate donor configurations, tensor contracts and checkpoint/source identity before replay. |
| [7coil/svideo](https://github.com/7coil/svideo) | master | d4b34633e92be30fd6395da1ffde21d6cac96917 | [Compression needs a reconstruction contract](https://github.com/7coil/svideo/blob/d4b34633e92be30fd6395da1ffde21d6cac96917/src/ProjectBuilder.ts): distinguish saved trace-vector metric replay from rerunning checkpoint inference. |
| [kai9987kai/Supermix-Poseidon](https://github.com/kai9987kai/Supermix-Poseidon) | main | 72d42218ae92cee5c9668a090f9d31d8d9961681 | [Planning and epistemic probes already exist but have bounded guarantees](https://github.com/kai9987kai/Supermix-Poseidon/blob/72d42218ae92cee5c9668a090f9d31d8d9961681/poseidon/nexus_omega/causal_efe.py): use counterfactual task supervision, not another heuristic planner, as the selected experimental mechanism. |
| [kai9987kai/FAULTLINE](https://github.com/kai9987kai/FAULTLINE) | main | b7ae1b9e97a3258cbc44c44e928d7e937487a9f8 | [Joint failures can defeat marginal/pairwise reliability](https://github.com/kai9987kai/FAULTLINE/blob/b7ae1b9e97a3258cbc44c44e928d7e937487a9f8/faultline/core.py): report one-, two- and three-expert damage separately, including common-mode corruption, without claiming donor independence. |

Detailed scope, access evidence and licensing boundaries: [systems.md](reviews/systems.md).

## What is reused and what is not

Tether is a clean local implementation informed by these design lessons. **No upstream source code, model weights, datasets or vendored licensed components from the 27 reviewed repositories were copied into Tether.** The copied files in this directory are the newly authored review reports themselves. Tether uses its installed scientific dependencies and sklearn's bundled handwritten-digit dataset under their own terms.

This is not a combination of 27 models. Some supplied repositories are search engines, development tools, a bootloader, a media converter or a hardware application; pretending they were neural components would erase their actual contribution. Their influence is experiment discipline, explicit interfaces, reconstruction contracts and careful interpretation.

License observations in the detailed reports are not grants of reuse rights. In particular, NEUROGRAFT has no selected root publication license in the reviewed snapshot; Flipper has no observed license; Odysseus has AGPL terms; quibble has LGPL/component terms; Fly bundles GPL-licensed malecns material; and the Expanse/Poseidon/Archimedes lines have separate model/data/dependency conditions. These materials were reviewed conceptually and were not imported.

## Evidence boundary

This index records the design rationale and source review. It reports no Tether experiment metrics or superiority finding. Successful execution, useful learned donor reliance, held-out composition, robustness, source freezing, inference replay and any promotion decision each require their own saved evidence. A checksum verifies bytes; replay verifies a computation under stated inputs; neither proves scientific truth or general semantic understanding.


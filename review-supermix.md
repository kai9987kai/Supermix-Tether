# Public portfolio review: Supermix, memory and synthetic-agent systems

Reviewed 24 October 2026 using the connected GitHub plugin. This is a read-only source review of nine repositories assigned to this research subtask. No models, tests or experiments were run, no weights were downloaded, and no GitHub state was changed. Reported experimental outcomes below are repository-reported evidence, not independent replication. A README, selected code/design artifact and license were fetched for each repository; individual boundaries are recorded below.

## Snapshot inventory

| Repository | Verified public default branch | Verified branch head | Observed license |
|---|---|---|---|
| [NexusSearch](https://github.com/kai9987kai/NexusSearch) | `main` | `538edcba595889090946a51ce03f72ac9fb72202` | MIT |
| [prometheus-alpha](https://github.com/kai9987kai/prometheus-alpha) | `claude/epic-tesla-l7brlx` | `417009e5548453338761732152b7b8ec30ffa79a` | MIT |
| [3d-animal-simulator-Hybrid-Agent](https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent) | `main` | `b7dd934770380a28717ddc82e3dfa7f127732a75` | Apache-2.0; vendored dependencies retain their own notices |
| [morpheus](https://github.com/kai9987kai/morpheus) | `claude/zealous-euler-y2xqul` | `5db6f98d605b80245ed9ff5e2cc9d42151830a14` | MIT |
| [Supermix-Expanse-v2](https://github.com/kai9987kai/Supermix-Expanse-v2) | `main` | `33e38a279c86bccdb1e5680f15f193d61a3f71cb` | MIT source; composite model/data terms |
| [GhostInTheMachine](https://github.com/kai9987kai/GhostInTheMachine) | `main` | `239f5de6452f78f3960de07c393268eeac142d03` | MIT |
| [Supermix-expanse](https://github.com/kai9987kai/Supermix-expanse) | `main` | `3fd091e9cc3387b2c17da5ce7514831d77e2af44` | MIT source; composite model/data terms |
| [GenesisEngine](https://github.com/kai9987kai/GenesisEngine) | `main` | `a43099ef21e91b75b812e3eb883cb4f3813ab7c7` | MIT |
| [supermix-archimedes](https://github.com/kai9987kai/supermix-archimedes) | `main` | `1952cf1c92cb506fff60f9ee0c055c2365988e5e` | MIT repository source; source weights need their own license review |

Access succeeded for all nine repositories through `github_get_repo`, `github_fetch`, and `github_fetch_file`. Encoding `/` as `%2F` in branch/tree URLs failed for the two `claude/...` defaults. Reading the unencoded branch path succeeded; querying their recursive trees by the resulting commit SHA also succeeded. This was a connector URL-shape limitation, not an inaccessible repository. Branch heads were fetched from the branch endpoint; reading a repository README alone was not treated as proof of head identity.

## NexusSearch

**Inspected:** root tree, `README.md`, `docs/SEARCH.md` through line 220, and `LICENSE`. [Pinned search design](https://github.com/kai9987kai/NexusSearch/blob/538edcba595889090946a51ce03f72ac9fb72202/docs/SEARCH.md).

**Mechanism:** dependency-free C11 immutable JSONL snapshots. The active query path combines exact typed filtering, scanned BM25 text ranking, exhaustive cosine vector ranking and deterministic reciprocal rank fusion. IDs break ties. A separate scalar numeric scan acts as a reference. Original JSON and executed-work counters support inspection. Caller-supplied vectors are required: it does not embed text itself.

**Reusable lesson:** preserve an exact, inspectable reference before adding an approximate or learned mechanism. For a new model, a task compiler and exact solver can fill the role the scalar/exhaustive references fill here. Freeze corpus identity, distinguish ranking scores from calibrated probabilities, and publish measured work.

**Overlap to avoid:** another local search/RAG interface, indexed evidence browser, immutable record store, or deterministic ranking dashboard would repeat an established capability. Standalone ANN/quantization modules are explicitly distinguished from the active snapshot path.

**Limits:** this inspection did not audit C implementation, memory safety, retrieval quality or performance. The documentation explicitly limits input sizes and states that snapshot durability, streaming and experimental storage modules should not be overclaimed. The HTTP interface is a local development interface.

## Prometheus-alpha

**Inspected:** README methods/results/limitations, recursive source/preregistration inventory, full `src/prometheus/tissue.py`, and `LICENSE`. [Pinned cell rule](https://github.com/kai9987kai/prometheus-alpha/blob/417009e5548453338761732152b7b8ec30ffa79a/src/prometheus/tissue.py).

**Mechanism confirmed in code:** a 40-site, 16-channel one-dimensional PyTorch NCA with a shared two-layer update rule. Cue input is gated by simulated head identity; behavioral response is a head-weighted response channel. Asynchronous update masks are generated independently per worm seed and reused between conditions. A fixed passive voltage diffusion rule and alive mask govern local propagation. The learned rule remains frozen within each simulated life, so within-life adaptation resides in state.

**Repository-reported experiments:** paired versus explicitly unpaired conditioning, complete head removal and regrowth, channel/region transplants, compiled state patterns, repeated amputation, fission, turnover and noise training. Particularly relevant: the README reports high linear decoding accuracy in emergent rules whose regrown heads do not behaviorally use the information. It also reports private, sometimes oppositely interpreted codes across learned rules.

**Reusable lesson:** causal sufficiency and behavioral use require interventions, not just decoding. Share update noise explicitly across controlled arms; keep selected training outcomes distinct from held-out generalization.

**Overlap to avoid:** state-memory transfer, lesion/regrowth memory, compiled engrams, private neural codes, NCA organism development, robustness versus turnover/noise, or a claims ledger alone.

**Limits:** claims remain finite synthetic experiments; the review did not recompute any outcome or verify every preregistration timestamp. Only two lineages per family are reported. Voltage is passive and hidden channels offer an easy alternative memory substrate; the authors explicitly restrict the interpretation accordingly.

## 3d-animal-simulator-Hybrid-Agent / EvoSim 6.1

**Inspected:** README, source inventory, full `src/experiments.js`, and the license header. [Pinned experiment implementation](https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent/blob/b7dd934770380a28717ddc82e3dfa7f127732a75/src/experiments.js).

**Mechanism confirmed in code:** interventions run the actual agent simulation from a frozen snapshot. Each baseline/intervention pair receives matching future process seeds; intervention setup draws are separated from the subsequent stream. `finally` restores the live snapshot and paused state. Outputs retain raw endpoints, paired differences, original configuration and starting snapshot. A separate sensor RNG generates illustrative observation noise and false positives without consuming the simulation RNG.

**Reusable lesson:** separate latent truth, noisy observation and intervention outcomes. Snapshot restoration belongs in a guaranteed cleanup path. Shared initial seeds are useful but do not imply the same subsequent random events after arms diverge.

**Overlap to avoid:** ecosystem sandbox, local biodiversity sensors, paired intervention dashboard, inherited-agent ecology, energy-aware hunting and resource renewal.

**Limits:** illustrative ecology, short rollouts and small replicate sets. Detection coverage is explicitly not statistical confidence. No browser or deterministic-continuation tests were executed in this review; the implementation was inspected statically.

## Morpheus

**Inspected:** README, source/preregistration inventory, full `src/morpheus/tissue.py`, and `LICENSE`. [Pinned cell rule](https://github.com/kai9987kai/morpheus/blob/5db6f98d605b80245ed9ff5e2cc9d42151830a14/src/morpheus/tissue.py).

**Mechanism confirmed in code:** batched 2D NCA with 13 channels, a shared 57→32→18 network, local identity/Sobel/Laplacian perception, random half-cell updates, fixed gap-junction voltage diffusion and a per-cell forward-model head. The comparator is actual visible change minus the firing-weighted prediction, supplied on the next step. Optional clamp/add/set voltage interventions and comparator gain are explicit parameters.

**Repository-reported experiments:** own versus transplanted prediction error, anatomical program compilation, repeated wounds, and recursive edits evaluated by naive, fixed-audit and sequential e-value audit loops. The README already includes an anytime-valid betting e-process under e-LOND, with a separate held-out auditor and an anti-forgetting anchor. It says a compiled head is maintained after release but generally not remembered through a new amputation.

**Reusable lesson:** an intervention may create a stable output without rewriting a system's enduring target. Compare separate stages of growth, persistence and regeneration. Audited self-improvement must report real gains and false adoptions as well as nominal statistical discipline.

**Overlap to avoid:** bioelectric/NCA self-model synthesis, anatomical compilers, prediction-error ownership, online-FDR/e-value self-improvement audits and evidence-led browser laboratories are already central mechanisms.

**Limits:** no outcomes or statistical validity were independently reproduced. Small synthetic tissues and few learned rules; the e-audit depends on its proposer, starting rule and clipping scale. The reviewed source is the cell engine, not the audit implementation.

## Supermix-Expanse-v2

**Inspected:** README including completed pilot sections, source/CNS-lab inventory, `expanse/EVOKED_CNS_DESIGN.md`, `expanse/checkpoints/mixture-20260927-v2/REPORT.md`, `LICENSE`, and `LICENSE-MODEL.md`. [Pinned evoked design](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/EVOKED_CNS_DESIGN.md), [pinned mixture result](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/checkpoints/mixture-20260927-v2/REPORT.md).

**Mechanism:** beyond the earlier Expanse grafts, this repository contains a configurable CNS lab: real/rewired/random/empty graphs, sign and strength controls, persistent/reset state, static/dynamic gates and locked hypotheses. Evoked readout writes `gate(h) * (F(u(h)) - F(0))`, removing a zero-drive constant-bias shortcut. Both terms remain differentiable and token-local. The design explicitly requires a parameter-matched dense adapter before a stronger confirmatory claim.

**Reported evidence:** a six-run pilot has evoked shuffled slightly ahead of evoked real; no promotion and no generated-answer benchmark. The completed source-mixture pilot reports parent/natural/balanced all at 4/72 strict final-answer accuracy. Most failures are parsed wrong numbers rather than format rejections. Balanced sampling slightly reduces pooled dev loss but regresses on fresh examples. Its report discloses unequal token exposure and one-seed/short-budget limitations. These are inspected reports, not rerun evidence.

**Reusable lesson:** loss reduction, module presence and circuit sensitivity are different evidence from correct behavior. Constant offsets can mimic useful computation. Lock checkpoint/data/code/cohort/order/batch identities, preserve historical verdicts, and refuse to repair absent provenance by invention.

**Overlap to avoid:** additional connectome grafts, dynamic gates, zero-drive subtraction, source-mixture balancing, another arithmetic/paraphrase diagnostic, immutable receipts or a paired training study by themselves.

**Limits:** the evoked report explicitly says almost 90% of write power remains in the mean. Anatomy superiority is not supported under the tested protocols. Some historical H-CNS-004 evidence becomes inconclusive under newer cohort checks. Do not claim all prior supported verdicts have complete modern provenance.

**License:** source is MIT. The repository's model-license file records inherited BioMedLM/OpenRAIL-M restrictions, Qwen Apache-2.0, and connectome CC BY 4.0 attribution. Reusing weights or derived data is a separate decision from reusing source.

## GhostInTheMachine

**Inspected:** README including current findings and retractions, source inventory, `src/ghost_v7/engine.py` through line 240, and `LICENSE`. [Pinned v7 engine](https://github.com/kai9987kai/GhostInTheMachine/blob/239f5de6452f78f3960de07c393268eeac142d03/src/ghost_v7/engine.py).

**Mechanism confirmed in inspected code:** recurrent synthetic agent with learned forward models, efference copy, comparator injection, action selection and a closed-loop world or cross-yoked recorded stream. v7 decomposes comparator mismatch into a fixed ridge-predicted state-feedback term and an innovation residual. Either stream can be replaced independently. Separate seeded RNGs handle network, action and authorship choices. Default live parts algebraically recover the original mismatch.

**Repository-reported evidence:** transplants and desynchronization locate a persistence effect in the comparator stream; the peak follows the comparator's target rather than an intrinsically special self population. The README retains a retraction of the v3 emergent-self interpretation and failed replication of its stronger causal-emergence component. The supported claim is operational dynamics, not consciousness.

**Reusable lesson:** metric validity must be tested on ground-truth systems and appropriate nulls before interpreting synthetic internal dynamics. Causal transplantation can distinguish signal amplitude from timing and closed-loop dependence. Keep failed and retracted claims visible.

**Overlap to avoid:** authorship versus yoking, prediction-error transplantation, feedback/innovation decomposition, persistence as an agency/consciousness proxy, a null ladder or claims ledger as the headline novelty.

**Limits:** static inspection only; no estimator, statistical test or replication was rerun. The v7 engine excerpt ends near record construction. Descriptive state predictability alone does not establish a useful task capability.

## Supermix-expanse

**Inspected:** README, source inventory, full `expanse/V2_DESIGN.md`, `LICENSE`, and `LICENSE-MODEL.md`. [Pinned consolidation design](https://github.com/kai9987kai/Supermix-expanse/blob/3fd091e9cc3387b2c17da5ce7514831d77e2af44/expanse/V2_DESIGN.md).

**Mechanism:** heterogeneous frozen branches and donor experts feed a v93/Archimedes-derived language trunk. The v2 consolidation design adds three zero-gated native latent blocks with 512-dimensional shared latents, top-2-of-8 experts and learned memory slots. A training-only fusion bank aligns signals from Archimedes, Qwen-derived experts, BioMedLM-derived experts, Omni, FlyCore and CNS. Fusion freezes and then disappears for a teacher-free final phase.

**Evidence distinction:** the design file's current-validation section is older than the README's completed-run interpretation. The README reports that zeroing the new native blocks removes almost none of the v2 gain: extra training, rather than the proposed consolidation computation, explains the improvement. It similarly reports donor/FlyCore ablations with no measurable contribution. This is the central negative result a new Supermix architecture should address.

**Reusable lesson:** appending a latent interface and training it does not establish causal skill consolidation. Match extra optimization, freeze common evaluation cohorts, measure generated correctness and ablate the new computation after training. Alignment loss and teacher/source labels can create plausible but unused interfaces.

**Overlap to avoid:** multi-source latent fusion, learned router consensus, zero-gated adapters, teacher-free bake-out, sparse graph grafts or a larger mixed checkpoint as the new project's core idea.

**Limits:** no model loaded or benchmark rerun. The source uses a narrow context and finite synthetic corpora. Reported code/BioMed generalization is weak. Composite licenses match the separate v2 repository's restrictions; this report is not an independent legal analysis.

## GenesisEngine

**Inspected:** README, source inventory, full `src/genesis/neural.py`, and `LICENSE`. [Pinned controller code](https://github.com/kai9987kai/GenesisEngine/blob/a43099ef21e91b75b812e3eb883cb4f3813ab7c7/src/genesis/neural.py).

**Mechanism confirmed in code:** differentiated cells create sensor/interneuron/motor nodes; anatomy determines available organs. A bounded continuous-time recurrent controller is wired from distance and receptor/effector compatibility. Optional Hebbian or reward-gated eligibility plasticity changes phenotype weights, with relaxation toward inherited initial weights and clipping. Sensorless/motorless development has meaningful functional consequences rather than a fixed controller relabeled by anatomy.

**Reusable lesson:** separate inherited structure from learned state and measure the consequences of their coupling. Keep a deterministic headless engine independent of rendering. Full RNG snapshots and paired experiments are already available at the project level.

**Overlap to avoid:** indirect developmental genome→body→brain ecology, artificial organisms, local plasticity, lineage/evolution dashboards or paired experiments as the principal novelty.

**Limits:** an abstract synthetic system. The README explicitly leaves fair direct-versus-developmental encoding and richer ecologies for future work. No biology, replay or experiment tests were executed here.

## Supermix Archimedes

**Inspected:** README architecture/build/evaluation sections, source inventory, `archimedes/build_archimedes.py` through line 240, the first 120 lines of the graft receipt, and `LICENSE`. [Pinned build code](https://github.com/kai9987kai/supermix-archimedes/blob/1952cf1c92cb506fff60f9ee0c055c2365988e5e/archimedes/build_archimedes.py).

**Mechanism confirmed in code:** v93 trunk, v87 cross-dimensional expert grafting, a PyTorch FlyCore calibrated against recorded JS votes, heuristic fly-to-CNS node insertion and independent Omni v48/v38 branches. Build receipts record source file hashes, fit/calibration statistics, node assignments, parameter counts and a measured logit-difference check. Fly calibration updates role bias and vigor parameters before later joint training.

**Useful caution:** the build docstring says every graft is function-preserving, but the README explicitly discloses the stored build's nonzero maximum logit difference (about 0.3197). The code records that value but the inspected script does not assert it is zero. The exact-value receipt field was outside the first 120 receipt lines inspected here, so this number is a README disclosure rather than an independently verified receipt value.

**Reusable lesson:** preserve measured discrepancies even when they contradict a design invariant. Source compatibility, dimensional alignment and calibration agreement are each separate from downstream skill transfer. A 44/48 small internal probe is not broad language-model evidence.

**Overlap to avoid:** cross-dimensional expert lifting, dormant expert waking, 22-brain controller grafting, multi-output local Gradio labs or source-hash receipts alone.

**Limits:** no checkpoint downloaded and no source license of each external weight checked. The MIT repository license alone does not resolve all external model/data terms. Fly roles and biological analogues are explicitly heuristic computational abstractions.

## Missing research question suggested by this group

**Can a small model learn an interface that turns available donor information into causally used, compositionally reusable computation, rather than merely fitting an aligned representation or absorbing more training data?**

This is an inference from converging gaps in the inspected portfolio, not a claim of global novelty. Prometheus reports decodable state that regrowth ignores; Expanse reports grafts and latent consolidation blocks whose removal costs little or nothing; Archimedes shows dimensional lifting and calibration without broad independent task evidence. The portfolio already contains sophisticated audits, source fusion and interventions. A new model should make causal skill use its research target.

A CPU-local experiment could use tiny independently trained donors for deterministic finite-state transformations or short symbolic programs, each with randomized private input/state/output coordinates. A compact recipient must compose their operations on held-out program structures. It can learn interfaces from passive examples or from a matched-budget curriculum of controlled donor interventions. Donor weights and the trunk remain fixed where possible, keeping the question focused on interface learning rather than more optimization or capacity.

The decisive metric is exact execution on held-out compositions, jointly with a prespecified donor-intervention effect: wrong donor state or semantics must change the recipient in the predicted way; the correct translated state must rescue it. Also measure task-irrelevant swaps, shuffled interfaces, random donors, equal-capacity native adapters and direct end-to-end learning. Report the best behavior achievable without donor access, and enforce information splits so the recipient cannot bypass donor computation by memorizing all primitive mappings.

Important design caveat: forcing a donor to be necessary by hiding information trivially in the donor does not show superior computation. Include a transparent native baseline with the same observable information and parameter/optimization budget. Likewise a high probe score, a large ablation drop or a rescue in isolation can be misleading; the held-out behavior and semantic counterfactual prediction must agree.

Suitable novelty wording is **a new controlled Supermix study of converting accessible information into causally used compositional skills**. Adjacent work on model stitching, causal abstraction/interchange interventions, representation alignment and modular learning must be researched before claiming a new method. The portfolio review above recommends a research direction, not an implementation or an empirical result; a limited primary-literature check follows.

## Follow-up critique: modular receiver with causal-use training

The parent proposed a compact neural modular receiver, paired donor interventions for causal-use credit, a real-data digit specialist plus recurrent symbolic tasks, withheld compositions, and absent/shuffled/damaged donor tests. This fits the portfolio gap and could train on CPU, but **counterfactual supervision of internal swaps is established prior art**.

- [Interchange Intervention Training, Geiger et al., ICML 2022](https://proceedings.mlr.press/v162/geiger22a.html) trains aligned neural representations to match a causal model's counterfactual output after a source/base interchange. Its experiments already include MNIST-PVR, navigation and systematic generalization. Consequently digits, internal swaps and withheld compositions are insufficient as a novelty claim.
- [Causal Distillation for Language Models, Wu et al.](https://arxiv.org/abs/2112.02505) adds an IIT-style objective to make a student's causal computations resemble its teacher. A receiver learning donor causal responses must compare with this idea explicitly.
- [Addressing divergent representations from causal interventions on neural networks, Grant et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/133e588e1429f9f1e25b215da145580e-Abstract-Conference.html) studies intervention-induced distribution shifts and distinguishes harmless null-space shifts from behaviorally pernicious pathways. This is a current reason to prefer matched-source interventions and to measure whether patched states remain near the natural distribution.
- [Revisiting Model Stitching in the Foundation Model Era, Mai et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_Revisiting_Model_Stitching_In_the_Foundation_Model_Era_CVPR_2026_paper.html) demonstrates that stitch training objectives matter and that penultimate feature matching can outperform naive intermediate alignment. A serious passive-alignment baseline should use a competent objective, not just an arbitrarily weak projection.

These are current primary-source abstract-level checks, not a systematic literature review. No claim of algorithmic novelty follows from them.

### A narrower mechanism worth investigating

Study **selective causal-use allocation under donor redundancy and damage**, rather than maximizing donor dependence. A train-time paired intervention estimates each donor's task-conditional effect signature: which output changes when a semantically relevant donor state is replaced, and which should stay invariant. A small learned credit head predicts whether this donor's contribution is useful on the current example; that prediction controls a constrained router or learning credit. Require directional counterfactual correctness, not merely a loss increase when the donor is removed. Add a fixed intervention budget and test whether allocating it where useful beats spending it uniformly.

The hypothesis is that a receiver can acquire **appropriate** causal dependence: use a complementary correct donor, remain invariant to irrelevant donor swaps, bypass a redundant/damaged donor when an equally informed native path can solve the task, and signal uncertainty when no valid route remains. The paired-credit allocation mechanism is a candidate extension whose novelty still needs checking; the scientific contribution can instead be an unusually clean answer to when graft use helps versus creates fragility.

Minimum comparisons should share receiver capacity, task data, donor data, training steps and intervention count: task-only training, donor dropout, strong passive feature matching, ordinary IIT/counterfactual distillation, uniform intervention training, and learned selective allocation. Match information as well as parameter count. A receiver denied the raw input while a native baseline is given it is not a fair efficiency comparison.

Use digits as one real-input boundary, not the whole novelty story. Split original source images by writer/example identity before deriving sequential compositions; otherwise the same visual items can leak across program splits. Validate semantic donor swaps separately from absent, noisy and damaged donors: those are different interventions. Train-time corruption patterns and test-time failure patterns should differ if the claim is robustness to unseen donor damage.

The most informative failure result would be: better behavioral composition with no new causal-use fidelity, or higher causal-use fidelity with no behavioral gain. Preserve that separation. Replacing the headline with an aggregate score would conceal the very gap this new project is meant to resolve.

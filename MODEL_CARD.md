# Model card: Supermix Tether (v0.1, v0.2 & v0.3)

## Identity and intended use

Supermix Tether is a CPU-local experimental receiver for frozen handwritten-digit specialists. Its purpose is to compare training objectives for useful specialist dependence under controlled interventions, proxy shifts, and single-expert (`damage_one`), majority-expert (`damage_two`), and common-mode (`damage_common`) corruption. In v0.3, it adds calibrated epistemic trust gating, selective safe abstention, mechanistic causal gradient attribution, and cross-framework ONNX export. It is suitable for inspecting a finite modular-learning experiment and reproducing its saved inference.

It is not a general language model or a model released for operational digit recognition. No broad reasoning, semantic-understanding, biological, consciousness or real-world safety capability is claimed. See [results](docs/results.md) for the recorded v0.1 (`outputs/confirmation`) and v0.2 (`outputs/v2-confirmation`) studies.

## Architecture

The default receiver has **2,762 trainable parameters** in `v1` (`forensic=False`) and **2,786 trainable parameters** in `v2` (`forensic=True`) at hidden width 24. It consumes three sources × three expert posteriors × ten classes, a source/operation query, and a ten-class proxy context vector. Its learned components are:

- Query embeddings and a softmax source router.
- Expert reliability weighting: in `v1` (`forensic=False`), from posteriors, entropy, ensemble agreement/deviation and expert identity; in `v2` (`forensic=True`), from posteriors, entropy, top-1/top-2 margin, normalized 9-class runner-up tail entropy (`H(p_-1)/log 9`), runner-up tail standard deviation, and expert identity.
- Operation-specific transformation matrices and biases.
- An additive context shortcut MLP.

All source selection and output transformations in receiver inference are learned parameters. The deterministic task oracle lives in data generation and scoring. Frozen donor inputs are detached from the receiver's gradient graph.

Three separately parameterized visual classifiers use hidden widths 32/48/64 and ReLU/tanh/GELU. The same donor bank handles all three source images, and all donors learn from the same image partition. They provide redundant heterogeneous predictions with shared lineage, not independent evidence.

## Training

Default donor training: 180 AdamW steps, learning rate 0.01, weight decay 0.0001, minibatches up to 256. Default receiver training: 400 Adam steps, batch size 128, learning rate 0.02, gradient-norm clipping at 5 and two CPU threads.

The seven arms share receiver architecture and per-seed initialization. Passive/dropout use factual examples; the `iit`, `effect_only`, `invariance_only`, `tether` and `unpaired` arms share the four aligned labeled intervention views. All use four receiver evaluations per step. Extra objective arithmetic is not FLOP-matched.

The `unpaired` arm preserves Tether's cross-entropy and invariance but cyclically shifts relevant prediction residuals across batch rows before effect matching. It preserves marginal squared-error terms while breaking the correct pairing. A claim that pairing matters requires a favorable comparison with this control; ordinary IIT-style improvement is insufficient.

Tether adds relevant probability-effect matching and irrelevant/damaged probability invariance, each with default coefficient 1. This builds on [IIT](https://proceedings.mlr.press/v162/geiger22a.html) and [causal distillation](https://arxiv.org/abs/2112.02505); it is not a claim to invent internal swaps or causal supervision. The `iit` arm is an IIT-style local control rather than a full reproduction of that literature.

## Data and selection

The [data card](DATA_CARD.md) describes the 1,797-image dataset, stratified image split, synthetic tasks and all interventions. Receiver acquisition withholds `(source,operation)` pairs `(0,2)`, `(1,1)` and `(2,0)`. Checkpoint selection minimizes cross-entropy on a frozen validation cohort using seen pairs and independent context. Test labels and held-out-pair scores do not choose a checkpoint. Corrected development evaluation uses the validation pool; confirmation uses the test pool. Fresh seeds reuse this same dataset and partition.

An earlier engineering pilot evaluated the test pool. It is disclosed as a pipeline pilot, not independent validation; no data, model or training hyperparameters were selected from its accuracies. See [results](docs/results.md) for its source revision and the corrected fixed study.

The phase label `confirmation` is descriptive metadata. Fresh seeds, fixed coefficients and a recorded pre-run configuration decision must be maintained by the experiment workflow; the flag itself does not enforce preregistration.

## Evaluation and interpretation

Evaluation covers proxy shifts, withheld pairs, relevant/irrelevant swaps, fabricated single-expert/common-mode corruption, donor erasure and selected-source shuffling. Saved metrics distinguish accuracy, correct factual/counterfactual pairs, effect fidelity, nuisance invariance and erasure sensitivity. Seed-level paired bootstrap intervals are descriptive; small seed sets do not establish a general performance advantage.

A large donor-erasure gap can arise from fragility. High irrelevant consistency can occur when both predictions are wrong. In v0.1 and v0.2, raw softmax outputs lacked an abstention mechanism; v0.3 introduces calibrated dark-knowledge epistemic trust gating $\tau = \max_e (1 - H(p_{s,e,-1})/\log 9)^+$ with selective abstention ($\theta = 0.10$, AUROC = 1.0000 on total sensor failure). Routing weights describe a computation; exact causal attribution is provided in v0.3 via `model.attribute()` input $\times$ gradient saliency maps.

The task has one operation on one selected source. It does not test multi-step program execution, recurrent memory, unseen operations, writer-disjoint recognition or cross-dataset transfer. Full-bank corruption may remove all usable visual information. Synthetic corruption behavior does not validate robustness to real data or hardware failures.

## Checkpoints, replay and status

Each saved arm/seed checkpoint contains configuration, donor and receiver state dictionaries, source/data manifests and selection history. Loading uses reconstructed module classes and `torch.load(..., weights_only=True)`. Runtime versions and hashes are recorded, but dependencies and hardware are not hermetically pinned.

Full model weights for both v0.1 and v0.2 across all 8 seeds and all 7 arms are hosted on Hugging Face:
- **Hugging Face Model Repository:** [https://huggingface.co/Kai9987kai/Supermix-Tether](https://huggingface.co/Kai9987kai/Supermix-Tether)

```powershell
python -m tether replay outputs/v2-confirmation --strict
python -m tether replay outputs/confirmation --strict
```

Strict replay checks artifact integrity, regenerated task identity and restored inference within recorded tolerances. It does not reproduce optimization or validate scientific interpretation. The runner never automatically promotes a checkpoint. No externally published model or performance certification is implied.

## Licensing and attribution

No license has yet been assigned to newly written Tether source or generated model artifacts. Do not infer a grant from an upstream repository's license. Dataset and dependency terms are listed in [THIRD_PARTY.md](THIRD_PARTY.md). The source reviews are cited in [docs/portfolio-review.md](docs/portfolio-review.md); no upstream model weights or portfolio source code are included in the model.

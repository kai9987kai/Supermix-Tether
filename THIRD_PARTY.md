# Third-party data, software and research attribution

Checked 24 October 2026. Newly written Tether source has **no project license assigned yet**. This document records third-party provenance and terms; it does not grant a license to Tether or replace an upstream license text.

## Dataset

E. Alpaydin and C. Kaynak (1998). *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).

The [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) lists **Creative Commons Attribution 4.0 International**; [license terms](https://creativecommons.org/licenses/by/4.0/). Tether accesses the 1,797-image copy bundled with scikit-learn, normalizes pixels, creates new image-ID partitions and generates tasks/interventions. See [DATA_CARD.md](DATA_CARD.md) for those changes. The dataset creators do not endorse Tether.

## Runtime dependencies

These packages are dependencies installed separately; their source has not been vendored into this project.

| Dependency | Role | Upstream license reference |
|---|---|---|
| PyTorch | Neural models, optimization and tensor serialization | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE); BSD-style terms with component notices |
| NumPy | Trace metrics, hashes and descriptive bootstrap summaries | [NumPy LICENSE.txt](https://github.com/numpy/numpy/blob/main/LICENSE.txt); BSD three-clause terms |
| scikit-learn | Bundled digits loader and stratified image split | [scikit-learn COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING); BSD three-clause terms |

Pytest is an optional development dependency, and setuptools is a build dependency. Installed distributions and their transitive libraries may include additional notices. The table is not a complete redistribution audit or a substitute for retaining the notices in any packaged dependencies. The project currently declares version ranges in `pyproject.toml`; run receipts record actual installed versions rather than claiming a locked environment.

## Papers and portfolio sources

IIT, causal distillation and the 2026 modular-learning papers are research references listed in [docs/research.md](docs/research.md). No code, pretrained checkpoints, paper figures or paper datasets were copied from those works.

The [portfolio index](docs/portfolio-review.md) and [selected-file reports](docs/reviews/) document 27 upstream repositories reviewed for mechanisms and overlap. Review access and observed licenses are recorded there. Source inspection is not independent reproduction of their experiments. Tether does not bundle those repositories' source, weights, generated corpora or connectome data, and their licenses do not automatically govern newly written Tether code.

The source's implementation and documentation attribution should remain separate from any later license or publication decision by the project owner.

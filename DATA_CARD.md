# Data card: digits and generated Tether tasks

## Source and attribution

Tether uses `sklearn.datasets.load_digits`, whose documentation identifies 1,797 images, ten classes, 64 integer features and an 8×8 image shape. It describes this bundled data as a copy of the UCI handwritten-digits test set. Tether does not download or use the full UCI training/testing collection. [scikit-learn source documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

Dataset citation: **E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49)**. The [UCI dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) lists **CC BY 4.0**. Tether changes normalization, partitions and task construction; those changes are described here. Attribution does not imply endorsement by the dataset creators.

## Preprocessing and split

Original feature values are divided by 16 into float32 vectors. This fixed scaling fits no training or held-out statistics. A stratified split of image IDs is created before tasks or donor training, with default split seed 1729:

| Partition | Image IDs | Use |
|---|---:|---|
| Training | 1,078 | Donor updates and receiver acquisition tasks |
| Validation | 359 | Report donor validation performance, select receiver checkpoints and run development evaluation |
| Test | 360 | Frozen confirmation evaluation and its saved-example inspection |

This is a new **within-dataset image-identity split**. It does not recreate the original UCI writer-disjoint protocol. Image-ID separation does not imply writer independence. The same default partition is shared across experimental seeds, so fresh receiver/donor seeds are not fresh datasets.

`split.json` records exact sorted IDs, feature/label hashes, partition hashes and the split method. The observed default normalized image-plus-label hash at documentation time was `8b3b6ae90863430bdef06a3f7d1ec2f6a905a53d3375655b18cf63780ad98772`; a saved study's manifest is the authority for its actual input bytes.

## Donor-derived inputs

Three visual networks are trained only on training IDs. They have different widths and activations but share images, labels and the classification objective. Their outputs for every image are cached after fitting. Test output caching is inference only; test labels do not choose donor parameters or corruption classes.

Each receiver task samples three IDs, with replacement, from one partition. The three experts score each image. Receiver input has shape `[batch,3 sources,3 experts,10 classes]`; experts are repeated across source groups rather than retrained nine times. The receiver never receives raw image pixels during acquisition.

## Labels and task support

Query `(source,operation)` requests:

| Operation index | Target |
|---:|---|
| 0 | Selected image's digit |
| 1 | Selected digit plus one, modulo ten |
| 2 | Nine minus selected digit |

Acquisition draws from six pairs and excludes `(0,2)`, `(1,1)` and `(2,0)`. The three withheld pairs test recombination of familiar source and operation factors. They do not test new visual classes, new operations or deeper programs. Task labels are exact functions of supplied dataset labels; no label noise is added to the oracle.

The context cue is a one-hot proxy for the factual answer. `correlated` is correct with probability 0.97, `independent` is uniformly sampled and `reversed` chooses an always-incorrect class. These names describe generator controls, not naturally observed spurious features.

## Aligned interventions

| View | Change | Supervision |
|---|---|---|
| Factual | No change | Original task target |
| Relevant | Replace all expert vectors at the selected source with another same-partition, different-digit image | Recompute target from replacement digit |
| Irrelevant | Replace a different source with another same-partition, different-digit image | Keep original target |
| Damaged | Replace selected expert posteriors at the queried source with fabricated confident vectors | Keep original target |

Every replacement stays within its declared image pool. The context remains the original factual proxy even after the relevant target changes. Both types of image replacement force a different underlying digit, rather than merely a different image ID.

Damage choices use a separate seeded stream. A fabricated class is based on a donor argmax shifted by one, not on the true digit. Its posterior assigns 0.991 to that class and 0.001 to each other class. Thus damage is not guaranteed to make a previously wrong donor worse. Single-expert damage changes one of three specialists; common-mode damage changes all three to a common fabricated class. Source, expert and corruption dependence should be considered when interpreting results.

Erasure uses a uniform posterior. Selected-source shuffling rolls that source's expert group across the batch; some rolled items may share a digit. These interventions measure changes at a posterior interface, not naturally occurring visual corruptions.

## Leakage controls and limits

Image splits precede task construction. Donor fitting uses training labels; receiver checkpoint selection uses seen-pair, independent-context validation labels. The corrected development evaluator also draws from validation IDs, while confirmation draws from test IDs. Test labels define and score those frozen tasks and display their ground truth. They do not choose corruption classes, training updates or selected checkpoints.

An earlier engineering pipeline pilot evaluated test IDs before phase-specific evaluation was corrected. It is not independent validation. No data, model or training hyperparameters were selected from those accuracies, and the [results report](docs/results.md) retains the pilot's chronology and limitations. A later fresh-seed study reuses the same 1,797-image dataset; it must not be described as a new untouched dataset.

Task draws use local deterministic generators without mutating the global sampling RNG. Source image IDs, replacement IDs, queries and manifests allow regeneration. Repeated task rows reuse a finite image pool, so a large `eval_size` is not a large independent sample of people or environments. The dataset's demographic coverage is not established by this experiment, and results should not be used as a population handwriting estimate.

This dataset is real imagery; operations, proxy cues and interventions are synthetic. No newly collected human data, cloud model output or external portfolio corpus enters Tether. See [the model card](MODEL_CARD.md), [methods](docs/paper.md) and [results](docs/results.md) for the scope of downstream conclusions.

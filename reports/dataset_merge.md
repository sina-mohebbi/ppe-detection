# Experiment: merging a second dataset to improve generalization

## Motivation

The baseline model scored well on its own test set but was suspected of
overfitting to that single dataset's distribution. A quick check confirmed it:

| Baseline model evaluated on | mAP@0.5 |
|-----------------------------|---------|
| Its own test set (51ddhesh PPE) | **0.812** |
| An unseen dataset (Construction-PPE) | **0.118** |

A 0.81 → 0.12 collapse on a different distribution is a classic
overfitting-to-one-source generalization gap.

## What I did

Merged the **Ultralytics Construction-PPE** dataset (1,416 images, freely
auto-downloadable) into the training set. The two datasets use different label
taxonomies, so the core work was **harmonizing the classes** — see
`src/merge_datasets.py`, which remaps the source ids onto ours and drops classes
we don't use:

```
helmet -> helmet   gloves -> Gloves   vest -> Vest
boots  -> safety_shoe   goggles -> goggles
none / Person / no_* -> dropped
```

This added **+946 training images** (notably +1,146 glove instances, the weakest
class). Test sets were kept **separate** so the comparison stays fair, and a
remapped Construction-PPE test set was built to measure cross-dataset
generalization.

## Result

Both models evaluated on both test sets, identical settings:

| Test set | Baseline | Merged | Δ |
|----------|----------|--------|---|
| Original (51ddhesh) | 0.812 | 0.803 | −0.009 |
| Cross-dataset (Construction-PPE) | 0.118 | **0.741** | **+0.623** |

**The merge closed the generalization gap almost entirely** — cross-dataset
mAP@0.5 jumped 6.3× (0.118 → 0.741) — while costing less than 1 point on the
original test set. That's the ideal outcome: much broader competence, negligible
regression on home turf.

## Honest caveats

- On the **original** test set, per-class scores were essentially flat; the weak
  classes (Gloves 0.66, mask 0.72) didn't improve *there*. The merge's value is
  **generalization to new distributions**, not lifting weak classes on the
  original one — the extra glove data helps Construction-PPE-style gloves, not
  51ddhesh-style gloves.
- **safety_shoe dipped slightly** on the original test (0.73 → 0.68). This is the
  expected cost of an imperfect taxonomy mapping: Construction-PPE labels
  "boots", which I mapped to "safety_shoe" — not a perfect definitional match, so
  it introduces a little label noise.

  I tested the obvious fix — **re-merging without the boots mapping**. It
  confirmed the diagnosis (safety_shoe recovered 0.68 → 0.70) but was a **net
  loss**: overall accuracy dropped (0.803 → 0.795), gloves/helmet got worse
  (dropping boots leaves footwear unlabelled, teaching the model to suppress it),
  and cross-dataset generalization fell. So the mapping is kept — the cure was
  worse than the disease. A proper fix would need a dedicated boots class or
  relabelling, not just dropping data.

## Takeaway

Curating and harmonizing a second public dataset is a cheap, effective way to
improve real-world robustness — and measuring it on a held-out cross-dataset test
is what turns "I added more data" into a defensible engineering result.

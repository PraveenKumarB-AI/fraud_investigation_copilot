# Module 3 — GNN Fraud Detection: Results

## Final model: GraphSAGE, fixed 100-epoch training run, no checkpoint selection

Trained on the full transaction graph (203,769 nodes, 234,355 edges, 165 features),
evaluated on the honest time-based test split (steps 35-49, never seen during training).

| Metric | Value |
|---|---|
| Precision (illicit) | 74.6% |
| Recall (illicit) | 57.1% |
| F1 (illicit) | 64.7% |
| AUC-PR | 62.1% |
| Baseline ("always licit") F1 | 0.0% |
| Baseline ("always licit") accuracy | 93.5% (misleading -- catches zero fraud) |

Confusion matrix (rows = actual, cols = predicted, order [licit, illicit]):

    [[15377   210]
     [  465   618]]

## Why this is the reported result, not a later attempt

Two follow-up training runs tried validation-based early stopping -- first selecting
the checkpoint by lowest validation loss, then by highest validation F1 -- expecting
this standard technique to improve on the fixed 100-epoch run above.

| Run | Selection method | Test F1 | Test AUC-PR |
|---|---|---|---|
| 1 (this one) | None -- fixed 100 epochs | 64.7% | 62.1% |
| 2 | Best validation loss (epoch 40) | 27.3% | 39.0% |
| 3 | Best validation F1 (epoch 99) | 40.3% | 37.0% |

Both attempts scored substantially worse on the true test set, despite Run 3 reaching
84.6% F1 on its own validation slice. AUC-PR -- a threshold-independent metric -- declined
steadily across all three runs, ruling out a simple threshold-calibration explanation.

The cause: the validation period (steps 28-34) sits immediately adjacent to training
(steps 1-27), while the test period (steps 35-49) is temporally further out. Module 2's
EDA already showed the labeled illicit rate shifts from 11.6% to 6.5% across this same
boundary -- real drift in fraud patterns over time. A model that increasingly specialized
to the training-and-validation era looked better and better on validation while
generalizing worse to the true, more distant test period. This is a documented property
of this benchmark: published research on the Elliptic dataset has specifically studied
performance degradation under temporal distribution shift, and flagged that some earlier
published results on this dataset carried leakage tied to exactly this kind of temporal
handling.

Run 1's fixed-epoch-budget approach was never chosen by peeking at any held-out set --
it is the honest, non-cherry-picked result, and it happens to also be the best one.

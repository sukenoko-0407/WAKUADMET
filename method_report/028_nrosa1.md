# Methodology Report

## Description of the Model

### Model Type and Architecture

The submission uses **Chemprop**, implementing a **message passing neural network (MPNN)** for molecular property prediction.

* **Learning paradigm:** Multitask learning
* **Input:** Molecular graphs derived from SMILES, with standard Chemprop atom and bond features
* **Architecture:**

  * Shared message passing layers to learn a molecular representation
  * Feed-forward prediction layers producing outputs for all ExpansionRx endpoints

All endpoints are predicted jointly by a **single multitask model**, allowing shared representations across tasks.

### Training Procedure

Models were trained **from random initialisation** using only the data provided in the ExpansionRx challenge. No external data, pre-training, or transfer learning was used.

Key training and model settings (selected via extensive manual hyperparameter tuning):

* **Optimizer and schedule:** Noam learning rate schedule (init LR = 1e-6, max LR = 1e-3, final LR = 1e-4)

* **Batch size:** 256

* **Epochs:** Up to 100 (no early-stopping patience)

* **Weight decay:** 1e-4

* **Loss:** Evidential loss

* **MPNN configuration:**

  * Message passing depth: 6
  * Message hidden dimension: 2048
  * Aggregation: normalised aggregation

* **Feed-forward network:**

  * Number of layers: 1
  * Hidden dimension: 256
  * Batch normalisation enabled
  * Dropout: 0.1

* Multiple multitask models were trained on different **cross-validation splits**

* **Final predictions** are the mean ensemble of these models

## Performance Comments

* Larger MPNN configurations improved performance, while keeping the feed-forward network small helped limit overfitting on the relatively small dataset. This may be related to Chemprop’s multihot atom and bond representations being sparse.

* Multiple data augmentation strategies were explored but did not improve performance. Our best attempt involved randomly combining subgraphs via summation. This reduced overfitting but led to test performance matching the baseline and resulted in a lower leaderboard ranking.

* Atom-level representations were substantially more informative than bond-level representations. Any modification that degraded atom features led to a significant drop in performance.

* Correlation between cross-validation performance and leaderboard results was generally good; however, some models performed notably better or worse on the leaderboard despite having cross-validation performance within statistical error.
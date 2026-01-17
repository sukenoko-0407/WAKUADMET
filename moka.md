# OpenADMET ExpansionRx Challenge - Prediction Scripts

This repository describes the submission to the OpenADMET-ExpansionRx Blind Challenge for the `moka` user.

Author: Davide Boldini

Date: January 2026

## Overview

The modelling pipeline splits the challenge dataset into four endpoint categories, each targeting specific ADME properties. Each category is modelled independently, using multitasking only between tasks belonging to the same group.

All models were built using the `MATCHA` package (to be open-sourced soon).

### Endpoint Categories

1. MPPB, MBPB, MGMB

2. HLM CLint, MLM CLint

3. LogD, KSOL

4. Caco-2 Permeability Papp A>B, Caco-2 Permeability Efflux

## Methodology

### Model Architectures

The final ensemble combines predictions from five complementary architectures:

- **GatedGCN** with a virtual node for enhanced global context (https://arxiv.org/abs/2003.00982)
- **GraphGPS** as the transformer-based architecture to provide a different inductive bias (https://arxiv.org/abs/2205.12454)
- **GIN** with laplacian and random-walk positional encodings (https://arxiv.org/pdf/1810.00826)
- **Chemprop** with RDKIT descriptor concatenation (https://pubs.acs.org/doi/10.1021/acs.jcim.5c02332)
- **ModernSNN** with concatenation of ECFP, Avalon and RDKIT descriptors (see below)

Each model except ModernSNN was pretrained, then finetuned on the challenge data. ModernSNN was trained directly on the challenge data using the strategy `Type B` (see in `Finetuning Strategies`).

### ModernSNN

It's a MLP with a bunch of features to (try) to make it more performant than a standard MLP/XGBoost, while maintaining the flexibility of pure Torch implementations.

- The backbone is based on Self Normalizing Networks (https://arxiv.org/abs/1706.02515)

- Instead of vanilla linear layers, it uses "ensemble"-style linear layers as in this paper: https://arxiv.org/abs/2410.24210

- Uses deep lasso regularization to handle high input feature dimensionality (https://arxiv.org/pdf/2311.05877)

- Has a bunch of other tricks from RealMLP (https://arxiv.org/abs/2407.04491) like robust scaling, cosine cyclical LR schedule etc 

### Pretraining strategies

Two pretraining datasets were used:

#### Large
Sparse multitask dataset encompassing regression and classification on 3D QM properties, RDKIT descriptors, ADME data and biochemical assays. Curriculum was used to adjust task importance during training.

#### ADME
Multitask regression-only dataset with ADME-related tasks.

GatedGCN, GraphGPS and GIN were pretrained using the first dataset, Chemprop was pretrained using the second one. ModernSNN did not undergo pretraining.

Both ADME and Large had a mix of public and internal data.

### Finetuning strategies

Two finetuning strategies were used:

#### Type A
Simply finetune on the challenge endpoints, grouped as indicated above

#### Type B
Finetune on the grouped challenge endpoints, while having as auxiliary task related ADME tasks from internal sources and most significant MOE descriptors for those endpoints, identified via XGBoost + SHAP. Curriculum was used to adjust task importance during training.

#### Details

- **Optimizer**: AdamW with learning rate 1e-3, weight decay 1e-4
- **Pretraining LR**: 1e-4 with model-specific decay schedules across the architecture, e.g. making earlier layers with lower LRs.
- **Epochs**: 200 maximum with early stopping (patience=20)
- **Batch Size**: 32
- **Loss Function**: Mean Absolute Error (MAE) in multitask framework

Strategy A or B was selected depending on the endpoint and / or algorithm.

Appropriate transformations were applied per endpoint (log10p for most targets, none for LogD).

Each architecture trains an ensemble of 10 models. Each model per algorithm uses different random initializations, and uses a different slice of the training data, obtained via 10-fold CV. The validation set for each CV split is used for early stopping. The CV seeds are different for each algorithm to improve inter-model variability

**Final Prediction**: Simple averaging across all model predictions

### Model selection and performance evaluation

- **Splits**: I used a time-based sliding window splitting scheme, using the molecular IDs as a proxy for time. The procedure sorts the dataset according to the IDs, then e.g. uses the first 50% of the data as train, and the subset between 50% and 60% as test. Then, repeats training using 60% of the data, and testing using the 60% - 70% set and so forth.  The idea is to simulate model deployment and prospective inference, as that matches the train/test setting of the challenge. The procedure is repeated 5 times and performance metrics are averaged across splits
- **Consistency with leaderboard**: Generally speaking, performance improvements using the splitting scheme described above matched improvements on the leaderboard
- **Hyperparameter tuning**: No systematic hyperparameter tuning was performed for finetuning or pretraining
- **Model seletion**: I tried different pretraining approaches (e.g. increase in model capacity, data curation, data sources, loss curriculum etc) and selected the best ones based on the performance on the sliding window performance
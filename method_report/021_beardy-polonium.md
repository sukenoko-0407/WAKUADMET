# Graph Transformer for Multi-Task ADMET Prediction

## Overview

This notebook demonstrates training a **Graph Transformer** model for multi-task ADMET property prediction.

The primary goal is to test the limits of this particular GNN architecture, even though other methods (ChemPropV2, TabPFN, fingerprint-based RF/XGBoost, and LLM-embedding models) are also known to perform well. We intentionally focus on a single architecture rather than ensembling different model types.

**Key Results:**
- This single-model example achieves approximately **0.62 MA-RAE** on the leaderboard
- The full submission uses an ensemble of ~~seven~~ nine models with varying configurations.
- Only competition data was used (no external data was used)
- Random 80/20 split performs best on the test set (pseudo-time and scaffold splits were evaluated but did not generalize well)
- Performance is largely insensitive to exact hyperparameter choices when parameters are within reasonable ranges

## Installation

```bash
pip install git+https://github.com/pgniewko/gt-pyg
```

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Model | GraphTransformerNet |
| GT Layers | 4 |
| Hidden Dimension | 128 |
| Attention Heads | 8 |
| Normalization | Batch Norm |
| GT Aggregators | sum, mean |
| Readout Aggregators | sum, mean, max, std |
| Dropout | 0.3 |
| Activation | GELU |
| Gating | Enabled |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 2000 |
| Batch Size (train) | 256 |
| Batch Size (eval) | 1024 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Scheduler | Cosine annealing with linear warmup |
| Warmup Epochs | 25 |
| Validation Split | 80% train / 20% val (random) |
| Gradient Clipping | max_norm = 5.0 |

## Loss Function

Combined multi-objective loss with gradient clipping:

| Component | Weight | Parameters |
|-----------|--------|------------|
| MAE (RAE-style) | 1.0 | Normalized by per-task MAD |
| Huber | 0.25 | delta = 0.5 |
| Pearson Correlation | 0.25 | 1 - correlation |
| Kendall's Tau | 0.1 | tau_temp = 2.0, sampled pairs |
| R² | 0.1 | SSE / Var(y) |

## Prediction Pipeline

1. **Per-endpoint checkpoint selection**: Best model checkpoint saved independently for each endpoint based on validation normalized MAE.
2. **Post-training calibration**: Linear calibration (y = a·ŷ + b) fitted on validation set per endpoint. **Important**: Although we noticed that calibration does not improve the MA-RAE per se, it improves other metrics, especially R².
3. **Prediction clipping**: Outputs clipped to training range ± 20% (DELTA = 0.2).

## Featurization
One of the things we noticed is that aggressive molecule sanitization may lead to a drop in performance. This is because there are pairs of compounds that differ only by a small structural detail, such as chirality. Therefore, we ensure that molecular featurization preserves stereochemical information (among other properties). Removing stereochemical features (R/S, cis/trans) creates duplicate SMILES with different target values. This issue was identified through memorization tests on small batches.

## Endpoints

Nine ADMET properties (log-transformed where applicable):

| # | Original Name | Log Name | Transform |
|---|--------------|----------|-----------|
| 1 | LogD | LogD | None |
| 2 | KSOL | LogS | log10((val+1) × 1e-6) |
| 3 | HLM CLint | Log_HLM_CLint | log10(val+1) |
| 4 | MLM CLint | Log_MLM_CLint | log10(val+1) |
| 5 | Caco-2 Papp A>B | Log_Caco_Papp_AB | log10((val+1) × 1e-6) |
| 6 | Caco-2 Efflux | Log_Caco_ER | log10(val+1) |
| 7 | MPPB | Log_Mouse_PPB | log10(val+1) |
| 8 | MBPB | Log_Mouse_BPB | log10(val+1) |
| 9 | MGMB | Log_Mouse_MPB | log10(val+1) |
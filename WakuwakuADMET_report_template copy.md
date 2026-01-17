# OpenADMET ExpansionRx Challenge - Methodology Report

**Account Name**: WakuwakuADMET

---

## 1. Model Description

- **Algorithm**:
  - Molecular Graph Neural Networks
    - Chemprop
    - AttentiveFP
    - DimeNet

- **Training Strategy**:
  - Single-task

---

## 2. External Data

- **External Data Sources**: In-house (not public)

| Endpoint | External Data Use |
|-----------|---------------|
| LogD | Yes |
| KSOL | Yes |
| MLM CLint | XX |
| HLM CLint | XX |
| Caco-2 Permeability Efflux | Yes |
| Caco-2 Permeability Papp A>B | No |
| MPPB | XX |
| MBPB | XX |
| MGMB | XX |

---

## 3. Performance Comments

<!-- 必須項目：学習・評価時の性能に関するコメント -->

- Performance between train/validation is consistent


<!-- 例:
- Internal CV metrics are comparable to leaderboard scores for LogD
- Higher variance observed for MPPB, MBPB, MGMB endpoints
- Performance between train/validation is consistent
例1：順当に収束（ベースライン）

Training/validation ともに改善して収束した、というコメント

During training, the validation metric improved steadily for the first ~N epochs and then plateaued. Early stopping selected the checkpoint at epoch X, which consistently gave the best validation score across folds. The gap between training and validation remained small, suggesting limited overfitting.

例2：過学習が出た（早期終了・正則化）

train は上がるのに valid が落ちる、の典型

We observed overfitting after around epoch X: training loss kept decreasing while the validation metric stopped improving (and slightly degraded). We mitigated this by enabling early stopping and increasing regularization (dropout / weight decay), which stabilized validation performance.

例3：CVごとのブレが大きい（データ分布・分割依存）

fold間のばらつき＝データが難しい/分割が効いている

Cross-validation results showed noticeable variance across folds (std ≈ …). This suggests sensitivity to the validation split and potentially heterogeneous data. The final submission uses an ensemble over folds to reduce variance.

例4：学習率やスケジュールが効いた

LRを変えたら良くなった、など

A smaller learning rate with a warmup + cosine schedule improved validation stability. With a constant LR we saw oscillations in the validation metric, while the scheduled LR produced smoother convergence and slightly better peak performance.


例6：リーク・バグ・分布ずれに気付いた

「気付き→対処」が書けると強い（誠実＋再現性）

An initial training run produced unusually high validation scores; after investigation we found data leakage via [feature/target timing/group split]. After fixing the split (group/time-aware CV), validation performance dropped to a more realistic level and became consistent.
-->

---

## 4. Ensemble Strategy

- **Aggregation Method**: Simple average
- **Model Diversity**: Different CV folds

---

## 5. Additional Features / Molecular Representations

- **Fingerprints**: Not used
- **Descriptors**: Not used
- **Learned Embeddings**: AIMNet2 (hoge, hoge, hoge)

---

## 6. Data Preprocessing

- **Target Transformation**: log10 for all endpoints except LogD
- **Zero/Missing Value Handling**: Replace 0 with smallest non-zero value
- **SMILES Standardization**: RDKit canonicalization

---

## 7. Loss Function / Validation / Split Strategy

- **Loss Type**: MSE
- **Cross-Validation**: 5-fold CV
- **Split Method**: Random split
- **Early Stopping**: Yes
- **Scheduler**: hoge

---

## 8. Negative Results / What Didn't Work

<!-- オプション項目：試したが効果がなかった手法 -->

- For some endpoints (CacoA>B, MPPB, hoge), our in-house data did not improve the validation R-squared — either as additional training data or as pretraining data. We suspect this may be driven by differences in wet-lab assay protocols, leading to a distribution shift between datasets.

---

## 9. References

- Chemprop: https://github.com/chemprop/chemprop, https://arxiv.org/abs/2003.03123
- Attentivefp: https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
- DimeNet: https://arxiv.org/abs/2003.03123
- AIMNet2: https://chemrxiv.org/engage/chemrxiv/article-details/6763b51281d2151a022fb6a5


# OpenADMET ExpansionRx Challenge - Methodology Report

**Author**: [Your Name]
**Date**: [Month Year]

---

## 1. Model Description

<!-- 必須項目：使用したモデル・アルゴリズムの概要を記載 -->


- **Algorithm**:
  <!-- 例: Chemprop v2, Graph Transformer, TabPFN v2, Random Forest, XGBoost など -->
  - Chemprop
  - AttentiveFP
  - DimeNet

- **Training Strategy**:
  <!-- 例: Multi-task / Single-task -->
  - Single-task

- **Brief Summary**:
  <!-- モデルの概要を1-3文で記載 -->
  - hogehoge

---

## 7. External Data

<!-- 推奨項目：事前学習の有無・外部データソース -->

- **External Data Sources**: In-house (not public)

| Property | External Data Use |
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

## 2. Additional Training Steps

<!-- 必須項目：追加で行った学習ステップを記載 -->

- Data augmentation with in-house/external ADME datasets
-

<!-- 例:
- Pretraining on external ADME datasets
- Data augmentation
- De-noising on training set
- Task-specific optimization
-->

---

## 3. Performance Comments

<!-- 必須項目：学習・評価時の性能に関するコメント -->

- Performance between train/validation is consistent
-

<!-- 例:
- Internal CV metrics are comparable to leaderboard scores for LogD
- Higher variance observed for MPPB, MBPB, MGMB endpoints
- Performance between train/validation is consistent
-->

---

## 4. Model Architecture Details

<!-- 推奨項目：アーキテクチャの詳細 -->

| Component | Configuration |
|-----------|---------------|
| Message Passing Depth | |
| Hidden Dimension | |
| FFN Layers | |
| Dropout | |
| Aggregation | |

<!-- その他のアーキテクチャ詳細があれば記載 -->

---

## 5. Ensemble Strategy

<!-- 推奨項目：アンサンブル手法 -->

- **Number of Models**:
- **Aggregation Method**:
  <!-- 例: Simple average, Weighted average, Winsorized mean -->
- **Model Diversity**:
  <!-- 例: Different random seeds, Different architectures, Different CV folds -->

---

## 6. Features / Molecular Representations

<!-- 推奨項目：使用した分子表現・特徴量 -->

- **Fingerprints**:
  <!-- 例: ECFP4, FCFP4, Avalon, ERG -->
- **Descriptors**:
  <!-- 例: RDKit-2D, Mordred, ChemAxon -->
- **Learned Embeddings**:
  <!-- 例: CheMeleon, MiniMol, CLAMP -->
- **Other**:
  <!-- 例: Jazzy, ADMET-AI predictions -->

---



## 8. Data Preprocessing

<!-- 推奨項目：データ前処理 -->

- **Target Transformation**:
  <!-- 例: log10 for all endpoints except LogD -->
- **Zero/Missing Value Handling**:
  <!-- 例: Replace 0 with half of smallest non-zero value -->
- **SMILES Standardization**:
  <!-- 例: RDKit canonicalization, salt removal -->
- **Other**:

---

## 9. Loss Function

<!-- 推奨項目：使用した損失関数 -->

- **Loss Type**:
  <!-- 例: MSE, MAE, Huber, Bounded-MSE -->
- **Multi-task Weighting**:
  <!-- 例: Uniform, Inverse frequency, Task-specific scaling -->
- **NaN Handling**:
  <!-- 例: NaN masking -->

---

## 10. Validation / Split Strategy

<!-- 推奨項目：バリデーション手法・データ分割方法 -->

- **Split Method**:
  <!-- 例: Random 80/20, 5-fold CV, Time-based split, Butina clustering, Scaffold split -->
- **Cross-Validation**:
  <!-- 例: 5x5 CV, 10-fold CV -->
- **Early Stopping**:
  <!-- 例: Patience=15 on validation MAE -->

---

## 11. Hyperparameter Optimization

<!-- 推奨項目：ハイパーパラメータ最適化手法 -->

- **HPO Performed**: Yes / No
- **Method**:
  <!-- 例: Ray Tune + ASHA, Grid search, Manual tuning -->
- **Number of Trials**:
- **Key Findings**:

---

## 12. Endpoint Categories / Task Grouping

<!-- オプション項目：エンドポイントのグループ化方法 -->

<!-- 例:
1. MPPB, MBPB, MGMB
2. HLM CLint, MLM CLint
3. LogD, KSOL
4. Caco-2 Permeability Papp A>B, Caco-2 Permeability Efflux
-->

---

## 13. Tools / Frameworks

<!-- オプション項目：使用したツール・フレームワーク -->

| Tool | Purpose |
|------|---------|
| | |
| | |

<!-- 例:
| Chemprop v2 | MPNN implementation |
| Ray Tune | Hyperparameter optimization |
| PyTorch Lightning | Training framework |
| MLflow | Experiment tracking |
-->

---

## 14. Negative Results / What Didn't Work

<!-- オプション項目：試したが効果がなかった手法 -->

-
-

<!-- 例:
- Pseudo-time split did not generalize well
- Curriculum learning caused catastrophic forgetting
- AutoGluon was outperformed by TabPFN
-->

---

## 15. Known Limitations

<!-- オプション項目：既知の制限事項 -->

-
-

<!-- 例:
- Model trained on drug-like small molecules only
- No calibrated uncertainty estimates
-->

---

## 16. Future Work

<!-- オプション項目：今後の改善案 -->

-
-

---

## 17. Reproducibility

<!-- オプション項目：再現性情報 -->

- **Random Seed**:
- **Environment**:
  <!-- 例: Python 3.11, PyTorch 2.1, Chemprop 2.2 -->
- **Hardware**:
  <!-- 例: NVIDIA A100, CPU only -->

---

## 18. References

<!-- オプション項目：参考文献 -->

-
-

<!-- 例:
- Chemprop: https://github.com/chemprop/chemprop
- CheMeleon: https://zenodo.org/...
-->

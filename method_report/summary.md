# OpenADMET ExpansionRx Method Report Summary

## 記載項目一覧

各Method Reportから抽出した主要な記載項目：

| # | 項目 | 説明 |
|---|------|------|
| 1 | ML Architecture | 使用したモデルアーキテクチャ（GNN、Transformer、Classical ML等） |
| 2 | Molecular Features | 分子特徴量・表現（Fingerprints、Descriptors、Embeddings等） |
| 3 | Pre-training | 事前学習の有無と手法 |
| 4 | Additional Data | 追加データ使用（Public/Proprietary） |
| 5 | Multi-task Strategy | マルチタスク学習の戦略（Single/Multi/Task Grouping） |
| 6 | Ensemble Method | アンサンブル手法 |
| 7 | Data Preprocessing | データ前処理・変換 |
| 8 | HPO | ハイパーパラメータ最適化手法 |
| 9 | Validation Strategy | 検証戦略（CV split方式等） |
| 10 | Performance Comments | 性能に関するコメント・観察 |

---

## 各チームの情報抽出

### 001_pebble

| 項目 | 内容 |
|------|------|
| ML Architecture | Graph Neural Networks (複数アーキテクチャのアンサンブル) |
| Molecular Features | 記載なし |
| Pre-training | 記載なし |
| Additional Data | 一部のendpointにおいて、Curated public + Proprietary ADMET data |
| Multi-task Strategy | Task group別に学習: {LogD}, {KSOL}, {MLM, HLM}, {Caco-2 Papp, ER}, {MPPB, MGMB, MBPB} |
| Ensemble Method | Greedy Forward Stepwise (Caruana, out-of-fold CV predictions) |
| Data Preprocessing | 記載なし |
| HPO | Optuna (temporal/clustered splits) |
| Validation Strategy | Temporal (compound-ID based) and clustered splits |
| Performance Comments | 追加ADMETデータで予測安定化。Validation accuracyとblind test setの相関は必ずしも高くない。一部のタスク群モデルでは、性能向上のために他の物性データを追加タスクとして取り入れました。たとえば、多くのモデルは、対象タスクに加えて LogD と pKa も予測するように学習されています。Expansion の学習セットで十分に表現されていない化学空間にある化合物について、ローカルで学習したモデルは予測の分散が非常に大きくなる傾向があることを観察しました。追加の ADMET データも用いて学習したモデルは、これらの化合物に対してより安定した予測を示しました。 |

検討項目候補：
- Ensemble: Greedy Forward Stepwise Ensemble
- Validation: Temporal / Clustered Splits
- Multi-task: YES
- HPO: Optuna

---

### 002_campfire-capillary

| 項目 | 内容 |
|------|------|
| ML Architecture | MPNN + Graph Transformer (Deep Learning) + AutoGluon (Classical ML: CPU/GPU learners) |
| Molecular Features | scikit-fingerprints による classical molecular representations |
| Pre-training | External datasets + internal proprietary data での事前学習 |
| Additional Data | External datasets + Internal proprietary data |
| Multi-task Strategy | Multi-task (endpoint-specific weighting) |
| Ensemble Method | NES (Neural Ensemble Search) like method (DL + Classical ML ensemble)→Greedy法の発展形 |
| Data Preprocessing | 記載なし |
| HPO | Ray Tune + HyperBand (DL), AutoGluon (Classical) |
| Validation Strategy | 5x5 cross-validation + time-based holdout splits |
| Performance Comments | 一部endpointにてExternal datasets使用。重み付けしたmulti-task trainingで精度改善。CV scoresとheld-out performanceの相関が低いendpointあり。モデルが安定した段階で、提供された学習データ全体を用いてモデルを再学習。 |

検討項目候補：
- ML: AutoGluon (classical ML)
- Ensemble: NES-like ensemble
- Validation: 5x5 cross-validation + time-based holdout splits
- Multi-task: YES (task-specific weighting)
- HPP: Ray Tune + HyperBand
- Fingerprint: scikit-fingerprints

---

### 003_overfit

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop-based multi-task/single-task models |
| Molecular Features | なし（追加descriptors/features未使用） |
| Pre-training | Fine-tuned with pre-trained Chemprop models |
| Additional Data | Proprietary datasets |
| Multi-task Strategy | Multi-task and Single-task models |
| Ensemble Method | 記載なし |
| Data Preprocessing | Log10 transform (LogD以外)、CLint/permeability: 0→half of smallest、PPB: 0→10^(-6) |
| HPO | Light optimization (<20 models) |
| Validation Strategy | 記載なし |
| Performance Comments | 追加データが必ずしも性能向上に寄与しない。Leaderboard最適化は誤解を招く可能性あり |

検討項目候補：
- ML: KERMT (pretrained ChemProp)
- Multi-task: YES (Singleも併用)

---

### 004_shin-chan

> 論文化予定

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop d-mpnn (multi-task) |
| Molecular Features | Calculated: TPSA (RDKit), ESOL, MD-derived properties |
| Pre-training | Pretraining on 108 public ADME tasks (~450k data points) |
| Additional Data | Public datasets (Polaris, Novartis, Wang2015, Wang2023, ...) |
| Multi-task Strategy | Task affinity grouping: Cluster A (LogD, LogS, MPPB, MBPB, MGMB), Cluster B (LogD, HLM CLint, MLM CLint), Single-task (Caco-2 Efflux, Caco-2 A>B) |
| Ensemble Method | 5 models with different random seeds, predictions averaged |
| Data Preprocessing | Log10 (clearance/permeability), logit transform (protein binding) |
| HPO | Bayesian HPO with 2-fold CV |
| Validation Strategy | Difficulty-biased splitting (Nearest Neighbor Jaccard distances on Morgan FP) |
| Performance Comments | Custom L1-uncertainty based loss function使用。当初は、大規模なマルチタスクモデルのアンサンブルを一括で学習するアプローチに基づいていましたが、事前学習／ファインチューニング（pretraining/fine-tuning）方式の方がより良い性能を示しました。ESOL値を計算し、Endpointと相関する場合には、ファインチューニング時の補助タスクとして使用。ExpansionRx データセットに対して、異なる溶媒中で分子動力学（MD）シミュレーションを実施し、関心のある複数のプロパティ（例：PSA）を導出しました。 |

検討項目候補：
- ML: Chemprop v2.2.1
- Molecular Features: TPSA, ESOL, MD-derived Properties
- Transform: Logit (for % unit)
- Validation: Nearest Neighbor Jaccard distances on Morgan FP (hold-out strategy)
- Multi-task: YES (grouping)
- HPP: Bayesian HPO with 2-fold CV
- Loss: Custom L1-uncertainty based loss

---

### 005_moka

| 項目 | 内容 |
|------|------|
| ML Architecture | GatedGCN + GraphGPS + GIN + Chemprop (4アーキテクチャ) |
| Molecular Features | RDKit descriptors, Jazzy descriptors, MOE descriptors (SHAP-selected) |
| Pre-training | Large (3D QM + ADME), ADME-only, CheMeleon |
| Additional Data | Public + Internal data (Large/ADME pretraining datasets) |
| Multi-task Strategy | 4 endpoint categories: {MPPB, MBPB, MGMB}, {HLM CLint, MLM CLint}, {LogD, KSOL}, {Caco-2 Papp A>B, Caco-2 Efflux} |
| Ensemble Method | 10 models per architecture (different seeds, 10-fold CV), simple averaging |
| Data Preprocessing | Log10p transform (LogD以外) |
| HPO | Default hyperparameters (HPO attempted but noisy) |
| Validation Strategy | Time-based sliding window (molecular IDs as time proxy) |
| Performance Comments | Pretrained > from scratch。Descriptor/FP vs graph-based methods: low correlation in residuals. 学習中にタスク重要度を調整するためカリキュラム（curriculum）を使用。一部のendpointでは、Jazzy 記述子と、各エンドポイントに対して重要度の高い MOE 記述子（XGBoost + SHAP により同定）も同時にファインチューニング。学習中のタスク重要度調整にカリキュラムを使用。一部のエンドポイント（例：CACO）では、MOE 記述子や Jazzy のような補助タスク追加が特に有効。記述子／フィンガープリント系手法は概してグラフ系手法に劣りました。ただし残差の相関は低いことが多く、アンサンブル戦略の洗練にはさらなる検討が必要です。SMILES ベース手法も概してグラフ系に劣りました。このモダリティでは、より大規模な事前学習が必要です。ファインチューニング設定の HPO（ハイパーパラメータ最適化）も試しましたがノイズが大きく、リーダーボード上の改善にはつながりませんでした。試行回数を増やせば改善する可能性はあるものの、得られる利得は小さいだろうと考えています。 |

検討項目候補：
- ML: GatedGCN, GraphGPS, GIN, MATCHA package (近日公開)
- Additional datasets: Large, ADME, Chemeleon
- Validation: Temporal Sliding Window (compound-id based)
- Multi-task: YES (grouping), Original Fintuning Strategy (addition of auxiliary tasks)

---

### 006_beetroot

| 項目 | 内容 |
|------|------|
| ML Architecture | ChemProp v2 |
| Molecular Features | Mordred, Jazzy, ECFP (Task Affinity Group basis) |
| Pre-training | CheMeleon, AZ weights, in-house ADMET multi-task model, Foundational modelling (10M de novo molecules) |
| Additional Data | Curated public data only (no proprietary) |
| Multi-task Strategy | Task Affinity Grouping (TAG), per-endpoint & multi-task optimization |
| Ensemble Method | Multiple ChemProp approaches ensemble, weighted averaging |
| Data Preprocessing | De-noising, explicit task-level masking during pretraining |
| HPO | 記載なし |
| Validation Strategy | 記載なし |
| Performance Comments | Bivalent moleculesで高variance。Staged freezing strategy、masked-objective training使用。独自の de novo 化学空間でのマルチタスク事前学習：独自パイプラインで生成した数百万の de novo 分子でも事前学習を実施した。具体的には、二価（bivalent）の ExpansionRx 分子を明示的に同定して切断し、ペイロード／リンカー断片を組合せ的に列挙した。REINVENT は、コンペの低分子と列挙した二価分子を用いて、強化学習により別途学習した。大規模なフィルタリングとクリーニングにより、化学的に妥当な構造のみを残した。生成プロセスは、モデル崩壊や無関係な化学空間へのドリフトを避けつつ、コンペに関連する化学に意図的に制約した。この「精査されつつ多様」な 1,000 万化合物コーパスにより、コンペ空間に対する表現学習が可能となり、（CheMeleon のような）ランダムサンプルでの基盤 Chemprop 事前学習や狭いデータセットよりも、より汎化しやすい初期表現を得られる可能性がある。 |

検討項目候補：
- ML: Chemprop v2 
- Ensemble: multiple different ChemProp approaches
- Weight Reuse: Chemeleon, in-house multitask Chemprop
- Validation: Temporal Sliding Window (compound-id based)
- Multi-task: YES (loss weighting)
- Loss: Bounded-MSE

---

### 007_rced_nvx

| 項目 | 内容 |
|------|------|
| ML Architecture | ChemProp 2.2 MPNN + TabPFN v2 + RF, LightGBM, XGBoost |
| Molecular Features | ECFP4, FCFP4, Avalon FP + RDKit-2d, Mordred, ADMET-AI predictions + CheMeleon, MiniMol, CLAMP embeddings |
| Pre-training | 記載なし（learned embeddings使用） |
| Additional Data | Public data only |
| Multi-task Strategy | 記載なし |
| Ensemble Method | Weighted average ensemble (no HPO) |
| Data Preprocessing | 記載なし |
| HPO | なし |
| Validation Strategy | Internal cross-validation |
| Performance Comments | Internal CV metrics ≈ leaderboard (LogD)。PPB endpointsで内部/LB scoreに差異あり |

検討項目候補：
- ML: ChemProp 2.2, TabPFN v2
- Molecular Features: Many features

---

### 008_tibo

> sub-account of rced_nvx

| 項目 | 内容 |
|------|------|
| ML Architecture | Classical ML (RF, XGB, LightGBM) + ChemProp v2 MPNN + TabPFN v2 |
| Molecular Features | ECFP4, FCFP4, Avalon FP + RDKit-2d, Mordred, ADMET-AI + CheMeleon, MiniMol, CLAMP embeddings |
| Pre-training | MapLight-TDC, CaliciBoost refactored |
| Additional Data | Challenge data + curated public data only |
| Multi-task Strategy | 記載なし |
| Ensemble Method | Weighted average ensembling |
| Data Preprocessing | Data augmentation: C-MixUp, RIGR (resonance forms enumeration) |
| HPO | 記載なし |
| Validation Strategy | 記載なし |
| Performance Comments | RIGR data augmentationで大幅向上（training set約3倍）。TDC top submissions didn't generalize well |

検討項目候補：
- Data augmentation: RIGR, C-MixUp

---

### 009_crh201

| 項目 | 内容 |
|------|------|
| ML Architecture | ChemProp v2 (Foundational modelling + de novo generation) |
| Molecular Features | Mordred, Jazzy, ECFP (TAG basis) |
| Pre-training | CheMeleon/AZ weights, in-house data, Foundational modelling (REINVENT + 10M de novo molecules) |
| Additional Data | Public data reconciled (HLM, MLM CLint) |
| Multi-task Strategy | Task Affinity Grouping (TAG), per-endpoint & multi-task |
| Ensemble Method | 記載なし |
| Data Preprocessing | De-noising |
| HPO | 記載なし |
| Validation Strategy | 記載なし |
| Performance Comments | Test/training performance consistent。Bivalent moleculesで高variance |

検討項目候補：
- ML: Chemprop v2 
- Pretrain: Large-scale multitask pretraining
- Weight Reuse: Chemeleon-like Chemprop
- Multi-task: YES (Task Affinity Grouping (TAG))
- Loss: Bounded-MSE

---

### 010_Artichoke

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop MPNN (CheMeleon-initialized) + LightGBM, CatBoost, XGBoost, ExtraTrees, Ridge, TabPFN + kNN predictors |
| Molecular Features | RDKit descriptors, Morgan/MACCS/atom-pair/torsion FP, Chemprop embeddings, Boltz-2 protein binding predictions |
| Pre-training | Large-scale multitask pretraining on ChEMBL, Polaris, TDC, Wang2015, PAMPA等 |
| Additional Data | Curated public ADMET datasets |
| Multi-task Strategy | Endpoint-specific fine-tuning with task-weighted losses |
| Ensemble Method | Stacked meta-learners, learned gating networks, adaptive ensembling |
| Data Preprocessing | 記載なし |
| HPO | 記載なし |
| Validation Strategy | 記載なし |
| Performance Comments | Multitask pretraining provides strong baseline。Similarity-aware local corrections critical for permeability/binding. RDKit の物理化学ディスクリプタ、複数種のフィンガープリント群（Morgan、MACCS、atom-pair、torsion）、Chemprop の学習埋め込み、さらに Boltz-2 によるタンパク結合親和性予測を含む、豊富な特徴量設計を行いました。エンドポイント間の転移とスタッキングとして、相関のある ADMET タスクの予測値を下流モデルの特徴量として再利用しました。適応的アンサンブルとして、不確実性推定、モデル間の不一致（disagreement）指標、信頼性特徴量を用い、ニューラル・木系・ローカルモデルの間を動的にゲーティングしました。記述子ベースの木モデルと残差学習器（residual learners）は一貫して MAE を低減し、ニューラルモデルが取りこぼす非線形効果を捉えます。類似度ベースの局所補正は、透過性や結合系エンドポイントで特に重要で、scaffold 分割や分布外（out-of-distribution）分割において大きな改善をもたらします。誤差を意識したゲーティングとスタック型アンサンブルは、固定比率のブレンドを上回る性能を示し、化学的に多様なテスト化合物に対して頑健で安定した性能を実現します。 |

検討項目候補：
- ML: Chemprop v2 + classical ML
- Molecular Features: Many features
- Pretrain: Large-scale multitask pretraining
- Weight Reuse: Chemeleon
- Ensemble: Adaptive ensembling, stacking

---

### 011_HybridADMET

| 項目 | 内容 |
|------|------|
| ML Architecture | Uni-Mol2 (Transformer) + PAMNet (GNN) + Fingerprints (MACCS, ErG, PubChem) |
| Molecular Features | MACCS, ErG, PubChem fingerprints |
| Pre-training | Uni-Mol2 pre-trained checkpoints (84M or 164M) |
| Additional Data | Official training data only (no external/private) |
| Multi-task Strategy | Target-specific training data combinations (table provided) |
| Ensemble Method | End-to-end training of all components |
| Data Preprocessing | Organizer's tutorial pipeline |
| HPO | Model selection based on public leaderboard |
| Validation Strategy | 5-fold scaffold split |
| Performance Comments | Fingerprint branch improves Efflux/Papp。LDS, GAFF2 atom types didn't work |

---

### 012_yanyn

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop + CheMeleon + KERMT (multi-task models) |
| Molecular Features | 記載なし |
| Pre-training | CheMeleon foundation model |
| Additional Data | Public data for logD (single-task) |
| Multi-task Strategy | Multi-task (except logD: single-task) |
| Ensemble Method | Models below test MAE cutoff threshold only |
| Data Preprocessing | 記載なし |
| HPO | 記載なし |
| Validation Strategy | 10-fold cross-validation |
| Performance Comments | 記載なし |

---

### 014_Universal15

| 項目 | 内容 |
|------|------|
| ML Architecture | KERMT (GNN) |
| Molecular Features | RDKit 2D Descriptors (cuik_molmaker) |
| Pre-training | 記載なし |
| Additional Data | Extensive public datasets (LogD, KSOL, HLM, MLM, Caco-2, MPPB等) + additional endpoints for multitasking |
| Multi-task Strategy | 6 tasks: 20 endpoints multi-task。MPPB/MBPB: single-task。MGMB: 7 tasks multi-task |
| Ensemble Method | 記載なし |
| Data Preprocessing | Ref[1]のstrategies |
| HPO | 記載なし |
| Validation Strategy | 記載なし |
| Performance Comments | GNN + RDKit >> tree-based + FP + RDKit。QM features直接追加は効果なし |

---

### 015_Gashaw

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop 2.2.1 MPNN + XGBoost, LightGBM, CatBoost, RF, ExtraTrees, Ridge, Huber (stacking) |
| Molecular Features | MPNN embeddings + target-specific descriptors, Morgan FP (2048), MACCS keys |
| Pre-training | なし（from scratch） |
| Additional Data | Provided data only (no external pre-training) |
| Multi-task Strategy | Single-task (most), Sub-multitask (LogD+LogS, PPB+BPB+MPB) |
| Ensemble Method | 5 CV folds averaged |
| Data Preprocessing | Standardization, MAD-based outlier removal (3.5), SMILES enumeration (1-5x) |
| HPO | Manual hyperparameter tuning |
| Validation Strategy | Scaffold-based 5-fold CV |
| Performance Comments | Single-task > multi-task for most endpoints。ChemProp best, but stacking outperformed for Caco-2 ER |

---

### 016_UncertainTea

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop 2.2.1 multitask + TabPFN v2.5 |
| Molecular Features | RDKit 2D, ECFP4, 3D, Avalon, ERG + ChemAxon pKa/logP/logD + Jazzy + CheMeleon, Minimol embeddings + AZ-pred |
| Pre-training | CheMeleon foundation model |
| Additional Data | 記載なし |
| Multi-task Strategy | Chemprop: multitask across all endpoints |
| Ensemble Method | Winsorized mean over multiple strategies |
| Data Preprocessing | Feature reduction (drop constant, correlated >0.85) |
| HPO | 記載なし |
| Validation Strategy | 5-fold CV for TabPFN feature selection |
| Performance Comments | Public ADMET model predictions as features didn't help。ECFP4 not used in final models |

---

### 018_aglisman

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop v2 MPNN (MLP/MoE/Branched FFN variants) |
| Molecular Features | 記載なし（molecular graphs from SMILES） |
| Pre-training | 評価したが使用せず（curriculum learning negative result） |
| Additional Data | ExpansionRx only (KERMT, PharmaBench evaluated but not used) |
| Multi-task Strategy | Multi-task (9 endpoints simultaneously) |
| Ensemble Method | 25 models (5 Butina splits x 5 CV folds), mean aggregation |
| Data Preprocessing | Log10 transform, SMILES canonicalization, salt removal, duplicate averaging |
| HPO | Ray Tune ASHA ~2000 trials |
| Validation Strategy | 5x5 CV (Butina clustering + k-fold), 12% temporal holdout during HPO |
| Performance Comments | Top 2.8% (10th/356)。Curriculum learning with external data caused catastrophic forgetting |

---

### 019_c-test

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop + KERMT (multi-task models) |
| Molecular Features | Morgan fingerprints evaluated |
| Pre-training | 記載なし |
| Additional Data | Public data for logD |
| Multi-task Strategy | Multi-task (except logD: single-task) |
| Ensemble Method | Low test MAE models only |
| Data Preprocessing | 記載なし |
| HPO | Varied hyperparameter configurations |
| Validation Strategy | 記載なし |
| Performance Comments | Chemprop > KERMT (consistency)。KERMT poor on ER/Papp。Morgan FP suboptimal |

---

### 020_Hydra

| 項目 | 内容 |
|------|------|
| ML Architecture | Graph Transformer with gating mechanism (gt-pyg) |
| Molecular Features | Standard atom/bond features (stereochemistry preserved) |
| Pre-training | なし |
| Additional Data | Challenge data only |
| Multi-task Strategy | Multi-task, multi-endpoint |
| Ensemble Method | 5 GNN models, averaged predictions |
| Data Preprocessing | Log transform |
| HPO | Explored but insensitive to exact choices |
| Validation Strategy | 80/20 random split (scaffold/pseudo-time didn't generalize) |
| Performance Comments | Multi-task learning + endpoint-wise checkpoint selection + robust loss scaling most beneficial |

---

### 021_beardy-polonium

| 項目 | 内容 |
|------|------|
| ML Architecture | Graph Transformer (GraphTransformerNet from gt-pyg) |
| Molecular Features | Standard features (stereochemistry preserved) |
| Pre-training | なし |
| Additional Data | Competition data only |
| Multi-task Strategy | Multi-task |
| Ensemble Method | 9 models ensemble |
| Data Preprocessing | Log transform, post-training linear calibration, prediction clipping |
| HPO | Insensitive to exact choices |
| Validation Strategy | Random 80/20 split |
| Performance Comments | ~0.62 MA-RAE。Aggressive sanitization degrades performance (stereochemistry important) |

---

### 022_martin

| 項目 | 内容 |
|------|------|
| ML Architecture | ChemProp v2 |
| Molecular Features | Mordred, Jazzy, ECFP (TAG basis) |
| Pre-training | CheMeleon/AZ weights, in-house data, Foundational modelling (10M de novo) |
| Additional Data | Public data (HLM, MLM CLint reconciled) |
| Multi-task Strategy | Task Affinity Grouping |
| Ensemble Method | 記載なし |
| Data Preprocessing | De-noising |
| HPO | 記載なし |
| Validation Strategy | 記載なし |
| Performance Comments | Test/training consistent。Bivalent molecules high variance |

---

### 023_robusta

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop GNN |
| Molecular Features | MolGraph + Maplight features (Morgan, Avalon, ErG, MACCS) |
| Pre-training | 記載なし |
| Additional Data | Public: Galapagos Polaris, Novartis 300K + CDD calculated logD/pKa |
| Multi-task Strategy | Multi-task |
| Ensemble Method | 5 random splits averaged |
| Data Preprocessing | Chemical curation (Datautils), log10 transform |
| HPO | Manual > Raytune TPE |
| Validation Strategy | 5 random splits |
| Performance Comments | Manual HPO on ExpansionRx only → port to multitask。Data curation + public ADME data improved performance |

---

### 024_chundu05

| 項目 | 内容 |
|------|------|
| ML Architecture | Heterogeneous Meta-Ensemble: XGBoost/CatBoost/LightGBM/RF + TabPFN + ChemProp GNN |
| Molecular Features | MOE descriptors, 3D-conformer shape descriptors, ADMET-AI Oracle features |
| Pre-training | Biogen ADME, TDC benchmarks (GNN pre-training) |
| Additional Data | Biogen ADME, TDC |
| Multi-task Strategy | Target-Specific Expert Selection |
| Ensemble Method | SLSQP weight optimization, Selective Calibration |
| Data Preprocessing | Adversarial Weighting, Cation Shift (-3.0 log for quaternary ammonium) |
| HPO | 記載なし |
| Validation Strategy | Chronological validation splits |
| Performance Comments | R2=0.58, MA-RAE=0.57。LogD R2=0.82, MBPB R2=0.78 |

---

### 025_temal

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop MPNN |
| Molecular Features | 記載なし |
| Pre-training | 記載なし |
| Additional Data | Public datasets included |
| Multi-task Strategy | Multi-head predictions (1 head per endpoint) |
| Ensemble Method | 25 models, weighted average (based on best validation MAE) |
| Data Preprocessing | Log transform, scaling, basic sanitization (duplicates, metals, salt removal) |
| HPO | 記載なし |
| Validation Strategy | Stratified k-fold (90/10) |
| Performance Comments | Chemprop > custom GNN。Combining data or removing outliers worsened performance |

---

### 026_okidoki

| 項目 | 内容 |
|------|------|
| ML Architecture | Custom DNN (PyTorch) |
| Molecular Features | ECFP4 (2048 bits) + descriptastorus (RDKit 2D normalized) + CheMeleon + ADMET predictions (Peteani et al.) |
| Pre-training | 記載なし |
| Additional Data | ADMET predictions as features |
| Multi-task Strategy | Single multi-task model (all properties simultaneously) |
| Ensemble Method | 10 models (random 95% data + early stopping on 5%) |
| Data Preprocessing | 記載なし |
| HPO | 記載なし |
| Validation Strategy | Early stopping on validation MA-RAE |
| Performance Comments | ADMET predictions as features helped。CheMeleon features: small improvement。Architecture had limited influence |

---

### 027_echo

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop + CheMeleon + KERMT (multi-task) |
| Molecular Features | 記載なし |
| Pre-training | CheMeleon foundation model |
| Additional Data | Large-scale public data for logD |
| Multi-task Strategy | Multi-task (except logD: single-task) |
| Ensemble Method | Top 5 by CV MAE, averaged |
| Data Preprocessing | 記載なし |
| HPO | Different hyperparameter configurations |
| Validation Strategy | Cross-validation MAE |
| Performance Comments | Table showing MAE per endpoint for top 3 models |

---

### 028_nrosa1

| 項目 | 内容 |
|------|------|
| ML Architecture | Chemprop MPNN |
| Molecular Features | Molecular graphs (standard Chemprop atom/bond features) |
| Pre-training | なし（random initialization） |
| Additional Data | ExpansionRx only |
| Multi-task Strategy | Multi-task (all endpoints jointly) |
| Ensemble Method | Mean ensemble of CV models |
| Data Preprocessing | 記載なし |
| HPO | Extensive manual tuning |
| Validation Strategy | Cross-validation splits |
| Performance Comments | Larger MPNN + small FFN best。Data augmentation (subgraph summation) didn't help。Atom-level features critical |

---

### 029_vaishnavi53

| 項目 | 内容 |
|------|------|
| ML Architecture | CheMeleon MPNN + XGBoost, LightGBM, CatBoost, GBR (stacking) |
| Molecular Features | Enhanced descriptors (264D): RDKit (63), 3D conformer (8), stereochemistry (4), physicochemical (8), MACCS (167), Jazzy (26) + Morgan FP (2048) + ErG FP (315) |
| Pre-training | Chemeleon pre-trained weights |
| Additional Data | 記載なし |
| Multi-task Strategy | 記載なし |
| Ensemble Method | Snapshot ensemble (5 snapshots x 5 folds x 3 seeds = 75 models), winsorized mean, global isotonic calibration |
| Data Preprocessing | StandardScaler, RobustScaler, stratified sampling (10 bins), prediction clipping |
| HPO | 記載なし |
| Validation Strategy | 5-fold stratified K-fold (target binning) |
| Performance Comments | Isotonic calibration reduced OOF MAE 2-5%。Jazzy + ErG FP added electrostatic/pharmacophoric info |

---

## 集計サマリー

### ML Architecture 使用頻度

| アーキテクチャ | 使用チーム数 |
|--------------|------------|
| Chemprop MPNN | 22 |
| Classical ML (XGBoost/LightGBM/CatBoost/RF) | 10 |
| TabPFN | 5 |
| Graph Transformer | 3 |
| KERMT | 4 |
| Uni-Mol2 | 1 |
| PAMNet | 1 |
| Custom DNN | 2 |

### Pre-training 使用状況

| Pre-training手法 | 使用チーム数 |
|-----------------|------------|
| CheMeleon foundation model | 10 |
| External ADMET datasets | 8 |
| AZ published weights | 3 |
| Foundational modelling (de novo) | 3 |
| なし（from scratch） | 6 |

### Additional Data 使用状況

| データ種別 | チーム数 |
|----------|--------|
| Challenge data only | 8 |
| Public data only | 12 |
| Public + Proprietary | 7 |

### Ensemble Method

| 手法 | チーム数 |
|-----|--------|
| Simple/Weighted averaging | 18 |
| Stacking/Meta-learning | 4 |
| Caruana method | 1 |
| Snapshot ensemble | 1 |

### Multi-task Strategy

| 戦略 | チーム数 |
|-----|--------|
| Full multi-task | 12 |
| Task affinity grouping | 6 |
| Single-task main | 3 |
| Hybrid (endpoint-dependent) | 6 |

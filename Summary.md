# OpenADMET ExpansionRx Challenge - 参加者レポート要約

## 対象レポート（参加者名）
- campfire-capillary
- moka (Davide Boldini)
- uncertaintea
- aglisman (Alec Glisman)
- crh201 / martin（同一内容）
- overfit
- rced_nvx (Ruel Cedeno, PhD)
- hydra
- beardy-polonium

※ 計9ファイル、8参加者（crh201とmartinは同一内容）

---

## 1. レポートに含まれる項目

各参加者のレポートに記載されている主な項目を整理する。

### 共通して含まれる項目
- **Model Description / Model Architecture**（使用したモデル・アルゴリズム）
- **Additional Training Steps**（追加の学習ステップ）
- **Performance Comments / Observations**（性能に関するコメント）

### 参加者別の特徴的な項目

| 参加者 | 特徴的な項目 |
|--------|-------------|
| campfire-capillary | ハイパーパラメータ最適化手法、クロスバリデーション方法 |
| moka | Endpoint Categories（タスクの分類）、Pretraining/Finetuning strategies、Model selection方法 |
| uncertaintea | Features（特徴量の詳細）、試したが効果がなかったアプローチ |
| aglisman | Table of Contents、Data Sources、Data Harmonization、Evaluation詳細、HPO詳細、Curriculum Learning Study (Negative Result)、Known Limitations、Future Work、Reproducibility |
| crh201/martin | De-noising手法、Task Affinity Groupings (TAG)、Foundational modelling |
| overfit | データ前処理（Log変換、0値の処理） |
| rced_nvx | 論文参照、使用した特徴量の詳細リスト |
| hydra | Loss function詳細（MAE + Huber）、Endpoint毎のチェックポイント選択 |
| beardy-polonium | Jupyter Notebook形式、Key Results概要 |

### aglismanのレポートが最も詳細
- 目次（Table of Contents）付き
- mermaidによるフローチャート図
- 詳細なデータソースと品質評価
- 評価メトリクスの詳細なテーブル
- 再現性に関する情報（環境・シード・アーティファクト）
- 参考文献リスト

---

## 2. 使用されたModel / Algorithm / Architecture

### アルゴリズム一覧

| 参加者 | Deep Learning | Classical ML | Foundation Model |
|--------|--------------|--------------|------------------|
| campfire-capillary | Multi-task MPNN, Graph Transformer | AutoGluon (CPU/GPU learners) | - |
| moka | GatedGCN + virtual node, GraphGPS, GIN, Chemprop | ModernSNN (MLP-based) | - |
| uncertaintea | Chemprop multitask | TabPFN v2.5 | CheMeleon |
| aglisman | Chemprop v2 MPNN (MLP, MoE, Branched FFN) | - | CheMeleon |
| crh201/martin | Chemprop v2 | - | CheMeleon, 独自Foundation model |
| overfit | Chemprop (multi-task/single-task) | - | 事前学習済みChemprop |
| rced_nvx | MPNN (ChemProp 2.2) | Random Forest, LightGBM, XGBoost | TabPFN v2, CheMeleon, MiniMol, CLAMP |
| hydra | Graph Transformer (gt-pyg) | - | - |
| beardy-polonium | Graph Transformer | - | - |

### 主要なモデルアーキテクチャ詳細

#### Chemprop（多数の参加者が使用）
- Message Passing Neural Network (MPNN) ベース
- SMILES入力 → 分子グラフ → メッセージパッシング → 予測
- Multi-task学習に対応

#### Graph Transformer（hydra, beardy-polonium, campfire-capillary）
- **hydra**: gt-pyg実装、Gating mechanism付き
- **beardy-polonium**: 単一アーキテクチャに焦点、約0.62 MA-RAE達成

#### Graph Neural Networks
- **GatedGCN + Virtual Node**（moka）: グローバルコンテキストの強化
- **GraphGPS**（moka）: Transformer-based アーキテクチャ
- **GIN + Positional Encodings**（moka）: Laplacian/Random-walk positional encodings付き

#### ModernSNN（moka独自）
- Self Normalizing Networks ベースのMLP
- Ensemble-style linear layers
- Deep lasso regularization
- RealMLPの技法（robust scaling, cosine cyclical LR等）

#### TabPFN（uncertaintea, rced_nvx）
- TabPFNRegressor v2 / v2.5
- 特徴量数制限あり（2000 features）
- AutoGluonより高速で高性能との報告

#### FFN Variants（aglisman）
- **MLP**: 標準的なFFN（最終モデルで採用）
- **MoE (Mixture of Experts)**: Gating network付き
- **Branched**: Shared trunk + task-specific branches

### アンサンブル手法

| 参加者 | アンサンブル構成 |
|--------|-----------------|
| campfire-capillary | Deep Learning + Classical ML の異種モデルアンサンブル |
| moka | 5アーキテクチャ × 10モデル = 50モデル、単純平均 |
| uncertaintea | 3 Chemprop + 2 TabPFN、Winsorized mean |
| aglisman | 5 Butina splits × 5 CV folds = 25モデル、平均 |
| rced_nvx | Weighted average ensemble（MPNN + TabPFN + RF/LightGBM/XGBoost） |
| hydra | 5 Graph Transformer モデル、平均 |
| beardy-polonium | 9モデルのアンサンブル |

### 分子表現・特徴量

| 参加者 | 使用した分子表現 |
|--------|-----------------|
| campfire-capillary | scikit-fingerprints |
| moka | ECFP, Avalon, RDKIT descriptors |
| uncertaintea | RDKit (2D/3D descriptors, ECFP4, Avalon, ERG), ChemAxon, Jazzy, CheMeleon embeddings, Minimol, AZ-pred |
| aglisman | SMILESからの分子グラフ（Chemprop内部表現） |
| crh201/martin | Mordred, Jazzy, ECFP（TAGベースで選択） |
| rced_nvx | ECFP4, FCFP4, Avalon, RDKit-2d, Mordred, ADMET-AI predictions, CheMeleon, MiniMol, CLAMP |

---

## 3. Model情報以外の要約

### 事前学習（Pre-training）

- **外部データセット活用**（campfire-capillary, moka, crh201/martin, overfit）
  - 公開データ + 社内データの組み合わせが多い
  - ADME関連の公開データセット：KERMT, PharmaBench
- **CheMeleon事前学習モデル**（uncertaintea, aglisman, crh201/martin, rced_nvx）
  - Zenodoから自動ダウンロード可能
- **独自のFoundation Model**（crh201/martin）
  - REINVENTで10M分子をde novo生成
  - Mordred fingerprints予測でCheMeleon-like事前学習
- **外部データなし**（hydra, beardy-polonium）
  - 競技データのみ使用でも良好な性能を達成
- **公開データのみ使用**（rced_nvx）
  - 社内データなしでアンサンブル構築

### データ前処理

- **Log変換**
  - LogD以外のエンドポイントはlog10変換が一般的
- **0値・欠損値の処理**（overfit）
  - CLint/Permeability: 0を最小非ゼロ値の半分に置換
  - PPB: 0を10^(-6)に設定（検出限界考慮）
- **SMILES標準化**（aglisman）
  - RDKitで正規化、塩除去
- **De-noising**（crh201/martin）
  - 先行研究の手法でトレーニングセットのノイズ除去

### ハイパーパラメータ最適化

| 参加者 | 手法 | 試行数 |
|--------|------|--------|
| campfire-capillary | Ray Tune + HyperBand | 不明 |
| moka | なし（系統的なチューニングは未実施） | - |
| aglisman | Ray Tune + ASHA | ~500-2000 |
| overfit | 軽度の最適化 | <20 |
| rced_nvx | なし | - |
| hydra | 探索したが性能は比較的不感 | 不明 |
| beardy-polonium | なし（妥当な範囲内では不感） | - |

### 評価・バリデーション手法

- **クロスバリデーション**
  - 5×5 CV（campfire-capillary, aglisman）
  - 10-fold CV（moka）
  - 内部CV（rced_nvx）
- **Time-based split**
  - Sliding window方式（moka）
  - Molecule Name順でtemporal split（aglisman）
  - Pseudo-time split評価（hydra, beardy-polonium）→ 汎化せず不採用
- **Random split**
  - 80/20 split（hydra, beardy-polonium）→ 最も汎化
- **Butina clustering split**（aglisman）
  - 分子の構造的多様性を考慮した分割
- **Scaffold split**（hydra, beardy-polonium）→ 汎化せず不採用

### Loss Function

| 参加者 | Loss関数 |
|--------|---------|
| aglisman | Weighted MSE with NaN masking |
| hydra | MAE + Huber loss（endpoint毎のmedian absolute deviation でスケーリング） |
| crh201/martin | Bounded-MSE |

### 試したが効果がなかったアプローチ

| 参加者 | 効果がなかった手法 |
|--------|-------------------|
| uncertaintea | 公開ADMETモデル予測を特徴量として使用、Delta learning、TabPFNのfine-tuning、AutoGluon |
| aglisman | Curriculum learning（外部データ利用）、補足データセット統合 |
| overfit | リーダーボード最適化のみに基づくモデル選択 |
| hydra | Pseudo-time split、Scaffold split |
| beardy-polonium | Pseudo-time split、Scaffold split |

### 発見・知見

- **ECFP4は最終モデルに不採用**（uncertaintea）
  - Count-based fingerprintsは性能低下
- **TabPFN > AutoGluon**（uncertaintea）
  - 秒単位で4時間のAutoGluonより高性能
- **外部データが必ずしも有効でない**（overfit, aglisman）
  - オーバーフィット防止には有効だが、予測性能向上には寄与しないケースあり
- **CV性能とリーダーボード性能の相関が低いエンドポイントあり**（campfire-capillary, rced_nvx）
  - 特にMPPB, MBPB, MGMBで変動が大きい
- **Catastrophic forgetting**（aglisman）
  - Curriculum learningで一部エンドポイントのR²が負に
- **Random splitが最も汎化**（hydra, beardy-polonium）
  - Pseudo-time/Scaffold splitは汎化しなかった
- **ハイパーパラメータに比較的不感**（hydra, beardy-polonium）
  - 妥当な範囲内であれば性能に大きな差なし
- **Multi-task学習が有効**（hydra）
  - Endpoint毎のチェックポイント選択との組み合わせが効果的
- **Robust loss scaling有効**（hydra）
  - Median absolute deviationでのタスク毎スケーリング

### ツール・フレームワーク

| ツール | 用途 | 使用参加者 |
|--------|------|-----------|
| Ray Tune | ハイパーパラメータ最適化 | campfire-capillary, aglisman |
| AutoGluon | 自動ML | campfire-capillary |
| PyTorch Lightning | 学習フレームワーク | aglisman |
| MLflow | 実験管理 | aglisman |
| MATCHA | モデリングパイプライン | moka |
| scikit-fingerprints | 分子記述子生成 | campfire-capillary |
| gt-pyg | Graph Transformer実装 | hydra |

### 参考文献・論文

- **rced_nvx**: J. Chem. Inf. Model. 2025, 65 (24), 13115-13131（ASAP-Polaris-OpenADMET Challengeの論文）

---

## まとめ

- **Chempropが最も広く使用されている**（8参加者中6人）
- **Graph Transformerも複数参加者が採用**（hydra, beardy-polonium, campfire-capillary）
- **アンサンブル手法が一般的**（複数モデルの平均、Weighted average）
- **事前学習済みモデル（CheMeleon等）の活用**が複数参加者で見られる
- **外部データなしでも競争力のある結果**（hydra, beardy-polonium）
- **外部データの活用は効果が限定的**という報告が複数あり
- **Random splitが最も汎化する**という報告（Pseudo-time/Scaffold splitは不採用）
- **ハイパーパラメータ最適化は限定的**（多くの参加者が系統的なチューニングなし）
- **レポートの詳細度は参加者により大きく異なる**（aglismanが最も詳細）

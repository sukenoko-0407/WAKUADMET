 **Model Report for OpenADMET \+ ExpansionRx Blind Challenge**      	

**1\. Description of the Model**

**1.1 Overview and Assumptions**

This ADMET prediction pipeline employs a **hybrid ensemble approach** combining deep learning and traditional machine learning methods to predict nine pharmacokinetic endpoints from molecular SMILES representations. My working assumption is that the training data was obtained from a single laboratory under similar/identical experimental conditions, resulting in minimal label noise—a common challenge in ADMET modeling. This assumption allowed me to focus primarily on evaluating the performance of different molecular representations and ML algorithms rather than extensive noise-handling strategies.

**1.2 Molecular Representations Evaluated**

Various molecular representations were systematically tested:

| Representation | Description | Performance |
| :---- | :---- | :---- |
| **Fingerprints** | Morgan/ECFP4, FCFP4, MACCS, Avalon | Strong baseline for classical ML |
| **RDKit Descriptors** | 80+ physicochemical, topological, electronic features | Good for interpretability |
| **MPNN Embeddings** | Learned representations from Chemprop | Best graph-level features |
| **ChemBERTa** | Transformer-based molecular embeddings | Competitive but not superior |

**Key Finding:** Best results were obtained by concatenating MPNN-learned features with target-specific molecular descriptors. This hybrid approach leverages both the expressive power of graph neural networks and domain knowledge encoded in curated descriptors.

**1.3 Model Architecture**

Single task and multitask and sub-multitask models were trained using ChemProp 2.2.1. The architecture features as follows.

* **Message Passing Layers:** 3-6 depth (adaptive based on dataset size)  
* **Hidden Dimensions:** 256-768 (adaptive)  
* **Aggregation:** Mean pooling for molecular representation  
* **Prediction Head:** 2-4 layer FFN with dropout (0.10-0.25) and batch normalization  
* **Descriptor Integration:** Target-specific descriptors concatenated before prediction head

For endpoints where MPNN underperformed, a stacking ensemble of classical ML models (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Ridge, Huber) with fingerprint features was employed.

**1.4 Training Strategies**

**Single-task, sub-multitask, and multitask models** were systematically compared:

| Configuration | Endpoints | Rationale |
| :---- | :---- | :---- |
| Single-task | LogD, Log\_HLM\_CLint, Log\_MLM\_CLint, Log\_Caco\_Papp\_AB, Log\_Mouse\_PPB, Log\_Mouse\_BPB | Best performance for most endpoints |
| Sub-multitask | LogD \+ LogS | LogS is left-skewed; benefits from related LogD signal |
| Sub-multitask | Log\_Mouse\_PPB \+ BPB \+ MPB | Log\_Mouse\_MPB has limited data; leverages related binding endpoints |
| Classical ML Ensemble | Log\_Caco\_ER | Imbalanced distribution; stacking outperformed MPNN |

**Key Observation:** Single-task models outperformed multitask configurations for most endpoints, except LogS (KSOL) and Log\_Mouse\_MPB (MGMB), where data limitations or distributional challenges favored sub-multitask learning. These models were selected based on their leaderboard MAE values.

**1.5 Additional Processing Steps**

* **Molecular Standardization:** Largest fragment selection, charge neutralization, canonical SMILES  
* **Outlier Removal:** MAD-based filtering (threshold: 3.5)  
* **Data Augmentation:** SMILES enumeration (1-5× based on dataset size)  
* **Cross-Validation:** Scaffold-based 5-fold CV for realistic generalization estimates  
* **Scaling:** StandardScaler or RobustScaler depending on target distribution  
* **Ensemble Averaging:** Predictions averaged across 5 CV folds

**No external pre-training** was used; all models were trained from scratch on the provided data.

**2\. Performance Comments**

**2.1 Training Observations**

* **Convergence:** Most MPNN models converged within 50-100 epochs with early stopping (patience=15-20)  
* **Scaffold vs. Random Split:** Scaffold-based CV yielded 15-30% higher RMSE than random splits, providing more realistic performance estimates  
* **Augmentation Impact:** SMILES augmentation reduced validation RMSE by 10-20% for small datasets (\<300 samples)

**2.2 Endpoint-Specific Findings**

| Endpoint | Best Approach | Notes |
| :---- | :---- | :---- |
| LogD | MPNN \+ Integrated Descriptors | Smooth convergence, well-behaved distribution |
| LogS | Sub-multitask with LogD | Left-skewed distribution; \~10% improvement over single-task |
| Log\_HLM\_Clint | MPNN \+ Integrated Descriptors | Consistent performance across folds |
| Log\_MLM\_Clint | MPNN \+ Integrated Descriptors | Similar to HLM with species variation |
| Log\_Caco\_Papp\_AB | MPNN \+ Integrated Descriptors | Generally well-predicted |
| Log\_Caco\_ER | Classical ML Stacking | Imbalanced distribution; required density-based sample weighting |
| Log\_Mouse\_PPB | MPNN \+ Residual Refinement | 5-15% improvement from refinement stack |
| Log\_Mouse\_BPB | MPNN \+ Residual Refinement | Benefits from robust scaling |
| Log\_Mouse\_MPB | Sub-multitask (PPB+BPB+MPB) | Limited data; 5-8% improvement from multitask |

**2.3 Key Insights**

ChemProp excels for most endpoints in single-task and sub-multitask settings, confirming its status as state-of-the-art. However, stacking classical ML algorithms with fingerprints and target-specific descriptors outperformed ChemProp for some endpoints (notably Log\_Caco\_ER).

This finding underscores a critical insight: there is no single molecular representation or ML model that works optimally for every ADMET endpoint. The diverse biochemical mechanisms underlying different pharmacokinetic properties necessitate endpoint-aware modeling strategies that match representation and algorithm to the specific prediction task.

**2.4 Summary**

This methodology employs a flexible, endpoint-aware system that adapts based on dataset size, target distribution, and endpoint relationships. The combination of MPNN-based molecular learning, domain-specific descriptors, scaffold-stratified CV, and strategic use of classical ML ensembles provides a comprehensive solution that acknowledges the heterogeneous nature of ADMET prediction challenges.


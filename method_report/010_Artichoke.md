## **Methodology Report**

### **Description of the Model**

**Model type**  
 We developed a **unified hybrid modeling framework** combining **multitask graph neural networks (Chemprop MPNNs, CheMeleon-initialized)** with **descriptor-based machine learning models** (LightGBM, CatBoost, XGBoost, ExtraTrees, ridge/elastic-net regression, TabPFN) and extensive **similarity-based kNN predictors**. Across endpoints, predictions are integrated using **stacked meta-learners**, **learned gating networks**, and **local similarity-aware corrections**, enabling adaptive blending of global and local signals.

**Additional steps beyond standard training**

* **Large-scale multitask pretraining** on curated public ADMET datasets (ChEMBL, Polaris, TDC, Wang2015, PAMPA, multispecies clearance and protein-binding data), followed by **endpoint-specific fine-tuning** with task-weighted losses.

* **Rich feature engineering**, including RDKit physicochemical descriptors, multiple fingerprint families (Morgan, MACCS, atom-pair, torsion), Chemprop learned embeddings, and **Boltz-2 protein binding affinity predictions**.

* **Cross-endpoint transfer and stacking**, where predictions from correlated ADMET tasks are reused as features for downstream models.

* **Similarity-aware modeling**, with kNN predictors and local linear or ridge corrections applied in descriptor, fingerprint, and latent embedding spaces.

* **Adaptive ensembling**, using uncertainty estimates, disagreement metrics, and reliability features to dynamically gate between neural, tree-based, and local models.

* **Efficiency-focused inference**, storing lightweight metadata (coefficients, PCA/SVD transforms, gating parameters) and refitting only fast components at test time.

### **Performance Comments**

* **Multitask Chemprop pretraining provides a strong global baseline** across all endpoints, especially for low-data tasks (e.g., protein binding and clearance).

* **Descriptor-based tree models and residual learners consistently reduce MAE**, capturing nonlinear effects missed by neural models.

* **Similarity-aware local corrections are critical** for permeability and binding endpoints, yielding substantial gains under scaffold and out-of-distribution splits.

* **Error-aware gating and stacked ensembling outperform fixed blends**, delivering robust, stable performance across chemically diverse test compounds.


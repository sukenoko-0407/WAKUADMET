# Methodology Report

## **Description of the Model**

### **Overview**

Our approach employs a sophisticated ensemble architecture combining deep learning message-passing neural networks with gradient boosting methods, enhanced by advanced molecular featurization and statistical calibration techniques.

### **Core Architecture Components**

#### **1\. ChemProp Message-Passing Neural Network Ensemble**

**Model Type:** Graph Neural Network with Snapshot Ensemble and Multi-Seed Training

**Architecture Details:**

* **Base Model:** Chemeleon message-passing network pre-trained on molecular property prediction tasks  
* **Message Passing Layer:** Bond-level message passing that captures molecular graph structure  
* **Aggregation:** Mean aggregation across atomic representations  
* **Prediction Head:** Feed-forward neural network (FFN) for regression with untransformed output scaling

**Training Strategy:**

* **Multi-Seed Ensemble:** Training with 3-5 different random seeds  to capture model variance  
* **K-Fold Cross-Validation:** 5-fold stratified splitting for robust generalization  
* **Snapshot Ensembling:** Cyclic cosine annealing learning rate schedule with snapshots saved at cycle endpoints  
  * Cycle length: 10 epochs  
  * Learning rate range: 0.0001 to 0.001  
  * 5 complete cycles per training run  
* **Early Stopping:** Patience of 40 epochs monitoring validation loss

**Total Models per Property:** 75 snapshots (3 seeds × 5 folds × 5 snapshots per fold)

#### **2\. Gradient Boosting Ensemble Stack**

**Model Type:** Heterogeneous stacking ensemble with robust meta-learning

**Base Learners:**

* **XGBoost:** 1000 trees, max depth 8, learning rate 0.03  
* **LightGBM:** 1000 iterations, learning rate 0.03  
* **CatBoost:** 1000 iterations, learning rate 0.03  
* **Gradient Boosting Regressor:** 300 estimators, max depth 5

**Meta-Learner:** Huber Regressor (robust to outliers, epsilon=1.2, alpha=0.0001)

**Stacking Strategy:**

* 7-fold stratified cross-validation for out-of-fold predictions  
* Meta-features include: individual model predictions, median, mean, standard deviation, and prediction range  
* RobustScaler normalization for meta-feature scaling

### **Molecular Featurization**

Our approach leverages multiple complementary molecular representations:

#### **For ChemProp Models:**

1. **Enhanced Molecular Descriptors (264 dimensions):**  
   * **RDKit Descriptors (63):** Valence electrons, PEOE VSA, SlogP VSA, EState VSA, rotatable bonds, CSP3 fraction  
   * **3D Conformer Descriptors (8):** Asphericity, eccentricity, spherocity, PBF, radius of gyration, inertial shape factor, surface area, volume  
   * **Stereochemistry Descriptors (4):** Stereocenter counts, bridgehead atoms, spiro atoms  
   * **Additional Physicochemical (8):** Molecular weight, heavy atoms, TPSA, heteroatom counts, heterocycle counts  
   * **MACCS Keys (167):** Structural fingerprint patterns  
   * **Jazzy Descriptors (26):** Novel charge-based and electrostatic features  
     * Charge distribution: max/min/mean/std Gasteiger charges, charge sums and ranges  
     * Polarizability: molar refractivity, TPSA, Labute surface area  
     * Electronic structure: valence electrons, aromatic rings, conjugated bonds  
     * Dipole moments: approximate dipole magnitude and components  
2. **Morgan Fingerprints (2048 bits):** Radius 2, circular substructure patterns  
3. **Extended-Reduced Graph (ErG) Fingerprints (315 bits):** Reduced graph representations capturing pharmacophoric patterns

**Total Feature Dimension:** 2,627 (264 \+ 2048 \+ 315\)

#### **For Gradient Boosting Models:**

1. **ECFP Fingerprints:** Morgan fingerprints (radius 3, 2048 bits)  
2. **Solubility-Focused Features (32):** Molecular weight, LogP, hydrogen bond donors/acceptors, TPSA, aromatic rings, heavy atoms, lipophilicity indices  
3. **Rich Descriptors (200):** MACCS keys \+ physicochemical properties  
4. **Atom Pair Fingerprints (2048 bits):** Hashed atom pair fingerprints

### **Prediction Aggregation and Calibration**

#### **Winsorized Mean Ensemble:**

* Winsorization at 10th and 90th percentiles across all snapshot predictions  
* Reduces influence of outlier models while preserving diversity  
* Applied to 75 snapshot predictions per test molecule

#### **Global Isotonic Calibration:**

* Isotonic regression fitted on out-of-fold predictions during training  
* Monotonic transformation that corrects systematic prediction biases  
* Applied post-ensemble to final predictions

#### **Prediction Clipping:**

* Final predictions clipped to training target distribution percentiles (0.5th to 99.5th)  
* Prevents extrapolation beyond observed chemical space

### **Additional Training Steps**

1. **Pre-trained Initialization:** Chemeleon message-passing weights downloaded from Zenodo repository (trained on large-scale molecular property datasets)  
2. **Feature Scaling:**  
   * StandardScaler for ChemProp descriptors  
   * RobustScaler for gradient boosting features (robust to outliers)  
3. **Stratified Sampling:** Target binning into 10 bins for stratified K-fold splits, ensuring balanced distribution across folds  
4. **Model Diversity Analysis:** Correlation matrix computed across base models to ensure complementary predictions (lower correlation improves stacking performance)

## **Performance Comments**

### **Training Performance Observations**

#### **ChemProp Multi-Seed Ensemble:**

* **Out-of-Fold MAE Improvement:** Global isotonic calibration consistently reduced OOF MAE by 2-5% across properties  
* **Snapshot Diversity:** Cyclic learning rate schedule produced snapshots with varying generalization characteristics, with later snapshots typically showing lower validation loss  
* **Seed Variance:** Different random seeds captured complementary error patterns, with correlation coefficients between seed predictions ranging from 0.92-0.96  
* **Convergence:** Most folds converged within 50-60 epochs, with early stopping preventing overfitting

#### **Gradient Boosting Stack:**

* **Ensemble Training MAE:** Achieved training MAE \< 0.34 target through meta-learning  
* **Base Model Correlation:** Low to moderate correlation (0.65-0.85) between base learners indicated good diversity for stacking  
* **Feature Importance:** ECFP and solubility features showed highest importance for XGBoost, while MACCS keys and atom pair fingerprints contributed most to CatBoost and LightGBM

#### **Overall System Performance:**

* **Calibration Effect:** Isotonic regression provided smooth monotonic corrections without overfitting  
* **Prediction Uncertainty:** Standard deviation across ensemble predictions provided reliable uncertainty estimates, with higher uncertainty correlating with test molecules farther from training distribution  
* **Computational Efficiency:** GPU acceleration enabled training completion within reasonable timeframes (3-4 hours per property on NVIDIA GPUs)

### **Key Success Factors**

1. **Complementary Representations:** Combining graph-based (ChemProp) and descriptor-based (gradient boosting) approaches captured both structural and physicochemical aspects of molecular properties  
2. **Robust Aggregation:** Winsorized mean and isotonic calibration provided stable predictions less sensitive to individual model failures  
3. **Enhanced Featurization:** Jazzy descriptors and ErG fingerprints added electrostatic and pharmacophoric information not captured by standard fingerprints, improving prediction of ADMET properties with strong charge-dependent components  
4. **Rigorous Validation:** Multi-seed, multi-fold cross-validation with proper stratification ensured unbiased performance estimates and prevented overfitting to training data


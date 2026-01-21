# OpenADMET Challenge Report

# Description of the Model

## 	TYPE OF MODEL AND ARCHITECTURE \- 

The final model is a **Heterogenous Meta-Ensemble** that utilizes a **Target-Specific Expert Selection** architecture. Rather than relying on a single global, algorithm, the model selects specific “Domain Experts” for each ADMET endpoint from four distinct modeling paradigms developed across 13 phases:

1. **Classical ML Ensembles:** Optimized blends of XGBoost, CatBoost, LightGBM, and Random Forest.   
2. **Bayesian Transformers:** Integration of **TABPFN**, a transformer-based model designed for tabular data, specifically utilized for complex protein-binding targets.  
3. **Deep Learning(GNNs):** Multi-task **Message Passing Neural Networks(ChemProp)** with **Bounded MSE Loss** to capture molecular graph relationships.  
4. **Hybrid Physics-ML:** Integration of industrial **MOE descriptors, 3D-conformer shape descriptors**( Radius of Gyration, Asphericity), and **ADMET-AI Oracle features**(Transfer Learning).  
   	  
   The final architecture employs a **Selective Calibration** strategy, where endpoints are chosen from either a statistically calibrated distribution ( to match training set physics) or a raw model-driven correction (to preserve non-linear chemical gradients).   
   

   ## ADDITIONAL STEPS AND OPTIMIZATION-

   		Beyond standard training, the following advanced optimization steps were implemented:  
     
* **Adversarial Weighting:** A binary classifier calculated Density Ratio Weights(p/1-p) for training samples, prioritizing molecules that chemically resemble the test set.  
* **Transfer Learning( Pre-training):** GNN components were pre-trained using external datasets including the **Biogen ADME** set and **Therapeutic Data Commons (TDC)** benchmarks to learn general chemical grammar before specializing on the target data.   
* **Ensemble Weight Optimization:** Final model weights were determined via **SLSQP(Sequential Least Squares Programming)** to minimize Mean Absolute Error (MAE) on chronological validation splits.   
* **Domain-Informed Post-Processing:** A “**Cation Shift”** was implemented using RDKit SMARTS patterns to apply a \-3.0 log unit correction to the LogD of quaternary ammonium cations, correcting a known systematic bias in statistical lipophilicity models. 

# 

# Performance Observations

The modeling framework demonstrated significant accuracy across the ADMET spectrum:

* **Overall Accuracy:** The final Selective Calibration strategy achieved an overall R2 of 0.58 and Overall MA-RAE of 0.57.  
* **Lipophilicity Breakthrough:** The hybridization of ML with chemical intuition resulted in a LogD R2 of 0.82 and Spearman of 0.89, indicating nearly perfect rank-ordering.   
* **Binding Precision:** The use of TabPFN allowed MBPB to reach a stable 0.78 R2 and a really low 0.14 MA-RAE.  
* **Distribution Alignment:** Statistical Calibration successfully stabilized “noisy” targets, bringing KSOL to R2 of 0.51 and HLM CLint to R2 of 0.41 by correcting for prediction variance.   
* **Generalization:** A high Spearman R of 0.76 across the test set confirms the model’s robustness in prioritizing lead compounds, effectively balancing graph-based learning with physical anchors. 


# OpenADME submission

## Summary:

We trained various multi-task d-mpnn's (chemprop) models on a collection of public datasets. Initially the approach was based on one go, ensembles of large multi-task models, but a pretraining/fine-tuning approach demonstrated better performance.

Datasets (public data only) that were used for pre-training included (but were not limited to) the data from the full dataset Polaris challenges \[1\], Novartis \[2\], and additional datasets \[3, 4\]. Additionally, calculated properties were added:

* TPSA (rdkit)  
* ESOL \[5\] (for the competition data only), ESOL values were calculated for all compounds in the ExpansionRx dataset, including the blinded test set and used as an auxiliary task during fine-tuning when ESOL values correlated with the challenge tasks.  
* MD derived properties, molecular dynamics simulations were performed in different solvents to derive different properties of interest (e.g. PSA) for the ExpansionRx dataset.

The final set of models were trained on 108 public ADME tasks filtered by compound overlap (min 50 shared compounds). All training data and models will be made available at a later date.

*Target transformations*: log10 for clearance/permeability endpoints; logit transform (log10(x/(100-x))) for protein binding percentages or fractions (only for pretrained data); LogD/LogS already on log scale.

*Held-out set creation strategy*: Difficulty-biased splitting using Nearest Neighbor Jaccard distances on Morgan fingerprints \- held-out sets enriched with compounds farther from training distribution.

Based on these splits/datasets, the final submission was created in the following way:

Chemprop v2.2.1 was used to train models. Next to the hyperparameter optimization, a custom L1-uncertainty based loss function was used. \[6\]

1) Pretrained models were created based on 108 public ADME datasets (\~450k data points).  
2) Task affinity grouping based on Spearman correlation values of the challenge tasks:  
   *Example grouping for the majority of models (grouping was also optimized):*  
   \- Cluster A: LogD, LogS, MPPB, MBPB, MGMB (solubility/protein binding)  
   \- Cluster B: LogD, HLM CLint, MLM CLint (metabolism)  
   \- Single-task: Caco-2 Efflux, Caco-2 A\>B (permeability)  
   Bayesian HPO was performed, including a 2-fold CV to determine optimal epochs.  
3) The Best model was retrained using the HPO parameters on the full ExpansionRx data.  
   For the final submission an ensemble was used, we used 5 models with different random seeds. Predictions were averaged prior to submission.

Further details will be provided in a subsequent publication, along with code, data and predicted properties – pending approval.

References

1. Goossens, Kenneth, et al. "ChemProp multi-task models for predicting ADME properties in the Polaris challenge." (2025).  
2. Peteani, Giulia, et al. "Application of machine learning models for property prediction to targeted protein degraders." *Nature communications* 15.1 (2024): 5764\.  
3. Wang, J.-B.; Cao, D.-S.; Zhu, M.-F.; Yun, Y.-H.; Xiao, N.; Liang, Y.-Z. In Silico Evaluation of logD7.4 and Comparison with Other Prediction Methods. J. Chemom. 2015, 29 (7), 389–398.  
4. Wang, Y.; Xiong, J.; Xiao, F.; Zhang, W.; Cheng, K.; Rao, J.; Niu, B.; Tong, X.; Qu, N.; Zhang, R.; Wang, D.; Chen, K.; Li, X.; Zheng, M. LogD7.4 Prediction Enhanced by Transferring Knowledge from Chromatographic Retention Time, Microscopic pKa and logP. J. Cheminform. 2023, 15 (1), 76\.  
5. Lawrenz, Morgan, et al. "A computational physics-based approach to predict unbound brain-to-plasma partition coefficient, Kp, uu." Journal of Chemical Information and Modeling 63.12 (2023): 3786-3798.  
6. Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018\.
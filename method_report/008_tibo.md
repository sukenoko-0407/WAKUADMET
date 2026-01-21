# OpenADMET-ExpansionRx
Model report for my contribution (LB nickname: 'tibo') to the OpenADMET-ExpansionRx Blind Challenge (Nov.2025-Jan.2026).

## Methodology Report

I participated alongside 'rced_nvx' [https://github.com/cedenoruel/OpenADMET-ExpansionRx-Blind-Challenge/blob/main/README.md] and therefore performed overlapping pre-processing techniques.
The overall approach is similar to 'rced_nvx's submission to the *ASAP-Polaris-OpenADMET Challenge* described in our recent publication:

Deep Learning vs Classical Methods in Potency and ADME Prediction: Insights from a Computational Blind Challenge.  
*J. Chem. Inf. Model. 2025, 65 (24), 13115–13131.*

- [JCIM Article](https://doi.org/10.1021/acs.jcim.5c01982) 
- [ChemRxiv Preprint](https://doi.org/10.26434/chemrxiv-2025-64fcb-v3)

In brief, initial results were obtained though preliminary data analysis and pre-processing, followed by performance comparison of Classical ML (RF, XGB, LightGBM) vs Deep Learning (ChemProp v2 MPNN, TabPFN v2) techniques.

Intermediate results were obtained through task-specific feature engineering (see below) and refactoring of top-performing Therapeutics Data Commons (ADMET Group) submissions [https://tdcommons.ai/benchmark/admet_group/overview/].
Notably, we refactored the code from *MapLight-TDC* [https://github.com/maplightrx/MapLight-TDC?tab=readme-ov-file] and *CaliciBoost* [https://github.com/Calici/CaliciBoost/tree/main].

- Structural fingerprints: ECFP4, FCFP4, Avalon
- Physchem descriptors: RDKit-2d, Mordred, ADMET-AI predictions
- Learned embeddings: CheMeleon, MiniMol, CLAMP

Data augmentation was attempted with *C-MixUp* (https://arxiv.org/abs/2210.05775) and *RIGR* (https://chemrxiv.org/engage/chemrxiv/article-details/68d35179f4163037700edd38).

Final results were obtained through weighted averable ensembling of our intermediate submissions.

## Data Availability

We only leveraged the data provided for the challenge, as well as our curation of publicly available data. No proprietary data was used.

## Performance Comments

- My most significant performance boost was observed after performing data augmentation with RIGR. 
This technique allowed to almost triple the size of the training set by enumerating all possible resonance forms of the training set's molecules.
- Surprisingly, without extensive hyperparameter optimization or feature selection, the refactored top-performing TDC submissions did not generalize well to this challenge.
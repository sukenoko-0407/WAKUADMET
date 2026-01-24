Methodology Report

Description of the Model

A Chemprop MPNN with multi-head predictions was chosen as the architecture. 

Public datasets were also included. Basic dataset sanitization was performed and included the removal of duplicates, metals, salt. Other newly introduced endpoints were predicted separately (except HLM Clint).

The labels were log-transformed and scaled. An ensemble of 25 models was used, each trained with one prediction head per endpoint. For each endpoint, the model with the lowest validation MAE of the stratified k-fold (90/10) was saved. The final prediction was generated from the saved best model heads, and a weighted average (based on best validation MAE) was used for submission. 

Performance comments

Chemprop models outperform a custom GNN implementation. The custom overfitted single model outperformed an ensemble of custom GNNs on the intermediate leaderboard. Combining data into one endpoint or removal of outlier/ activity cliffs (Tanimoto Similarity \>0.8, SALI \>3.0) worsens performance.

\[1\] LogD: https://github.com/nanxstats/logd74  
\[2\] ADME: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00160  
\[3\] LogD Dataset: https://moleculenet.org/datasets-1  
\[4\] CYP-Inhibition: https://pubs.acs.org/doi/abs/10.1021/acs.molpharmaceut.2c00962  
\[5\] Caco-2 Permeability: https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.4c00946
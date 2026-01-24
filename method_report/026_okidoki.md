# Methodology Report

# Description of the Model

- The architecture is a single multi-task model predicting all properties at the same time  
- **Input features are ECFP4 fingerprints (2048 bits, bit-based), descriptastorus descriptors (normalized RDKit 2D descriptors), chemeleon feature vectors and predicted ADMET properties using public data** ( Peteani et al. (2024), this model I trained with ECFP4 fingerprints+descriptastorus features)  
- The model itself is a custom implementation of a deep neural network using PyTorch  
- The model has a dimension of the hidden layer of 2048 and 5 layers  
- Trained with AdamW, cyclical learning rate, batch size of 256, weight decay of 0.01, no dropout  
- Huber loss was used, scaled by the MAD for each feature  
- In the end an ensemble of ten models is generated, each leveraging random 95% of the data and early stopping on the other 5%.   
- Early stopping done based on validation Macro-averaged RAE

# Performance comments

\-Inclusion of predicted ADMET properties lead to quite some increase in model performance.  
\-Checkpoint selection for each feature did not lead to improvements over single checkpoint selection for overall MAE  
\-Addition of chameleon features led to some additional performance, but not much  
Architecture itself had limited influence on model performance


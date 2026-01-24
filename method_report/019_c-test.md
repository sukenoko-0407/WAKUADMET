Methodology Report

# **Description of the Model**

We trained multiple multi-task models using varied hyperparameter configurations, based on both Chemprop and  Kermt. All tasks were jointly optimized within a multi-task learning framework, except for logD—owing to the availability of public data, it was trained independently as a single-task model. For the final predictions, we formed an ensemble by selecting only those models that achieved consistently low test Mean Absolute Error (MAE), ensuring high predictive reliability.

# **Performance comments**

Chemprop demonstrated stronger consistency between training and test set performance, as reflected by higher correlation. In contrast, Kermt yielded poor results on both ER and Papp tasks. Morgan fingerprints were also evaluated but led to suboptimal predictive performance. 
## **Methodology Report**

### **Description of the Model**

The submission uses an ensemble of five graph neural network models, with final predictions obtained by averaging their outputs. Each model is a bespoke graph transformer with a gating mechanism, implemented using an open-source framework: [gt-pyg](https://github.com/pgniewko/gt-pyg). The models were trained in a multi-task, multi-endpoint setting, jointly predicting all target properties.

Training was performed on an 80/20 random train–validation split for about 1000 epochs. Validation performance was monitored per endpoint, and for test-time inference, the best checkpoint was selected independently for each endpoint. The loss function combined MAE and Huber loss, averaged uniformly across endpoints, with additional task-wise scaling based on the median absolute deviation.

Only the challenge data was used; no external data was used for pre-training or fine-tuning. Hyperparameter optimization was explored, but performance was largely insensitive to exact parameter choices. Alternative splitting strategies (pseudo-time and scaffold splits) were evaluated but did not generalize well and were not used.

### **Performance Comments**

Lower validation MA-RAE consistently corresponded to better test-set performance and was therefore used for model selection. Overall, performance benefited most from multi-task learning, endpoint-wise checkpoint selection, and robust loss scaling, rather than extensive hyperparameter tuning.


# Methodology Report

 

# **Description of the Model**

Multiple multi-task models were trained using different hyperparameter configurations, leveraging  Chemprop, CheMelenon foundation model and Kermt. 

All tasks were jointly optimized in a multi-task learning framework, with the exception of logD, which—due to the availability of large-scale public data—was trained separately as a single-task model.

For final predictions, we use the cross-validation MAE as a guide and average the top 5 ensemble models.

# **Performance comments**

| model | LogD\_mae | LogS\_mae | Log\_Caco\_ER\_mae | Log\_Caco\_Papp\_AB\_mae | Log\_HLM\_CLint\_mae | Log\_MLM\_CLint\_mae | Log\_Mouse\_BPB\_mae | Log\_Mouse\_MPB\_mae | Log\_Mouse\_PPB\_mae |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| top1 | 0.21985644375265878 | 0.2500435469453153 | 0.11561921649076613 | 0.16413080873648767 | 0.2633534217339963 | 0.29859104829659955 | 0.11362264479700722 | 0.10556906108710552 | 0.144636 |
| top2 | 0.20179033719648293 | 0.23852138227763148 | 0.11078067385857168 | 0.1586382757628695 | 0.25249612654563286 | 0.2853641803696477 | 0.1054277786123969 | 0.09302 | 0.1329576470587332 |
| top3 | 0.19424534929784917 | 0.23936910020974486 | 0.11182359072445212 | 0.1546566172941035 | 0.2483425275947228 | 0.28244638091100754 | 0.10535289791347761 | 0.093108 | 0.13321490391462862 |


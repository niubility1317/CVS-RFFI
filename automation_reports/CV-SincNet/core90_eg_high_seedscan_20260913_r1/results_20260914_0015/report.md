# 完整EG＋高学习率seed扫描：当前完整可用结果

截止2026-09-14T00:15:43.736967+08:00，5/6完成；本报告全部为source V最终评分，目标测试尚未执行。

每场景27000样本，四场景108000决策/模型，source RX1/3/4/6/8，day1/2/3。全部模型lr=0.0004、scratch E200，final_only。目标基准已有研究者接触，本轮为探索性后续训练。

|模型|状态|最后完整epoch|最新动作epoch|
|---|---|---|---|
|V2_EG_HIGH_ADV0_seed392005|completed|200|200|
|V2_EG_HIGH_ADV0_seed392006|completed|200|200|
|V2_EG_HIGH_ADV0_seed392007|completed|200|200|
|V2_EG_HIGH_ADV035_seed392005|completed|200|200|
|V2_EG_HIGH_ADV035_seed392006|completed|200|200|
|V2_EG_HIGH_ADV035_seed392007|active|172|173|


## accuracy（%）

|模型|clean|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|LEO均值|
|---|---|---|---|---|---|
|V2_EG_HIGH_ADV0_seed392005|98.433|92.833|89.463|89.607|90.635|
|V2_EG_HIGH_ADV0_seed392006|98.219|93.641|90.330|90.089|91.353|
|V2_EG_HIGH_ADV0_seed392007|98.048|93.785|91.226|90.974|91.995|
|V2_EG_HIGH_ADV035_seed392005|98.285|93.237|90.267|90.133|91.212|
|V2_EG_HIGH_ADV035_seed392006|98.189|94.319|91.026|90.996|92.114|


## macro_f1（%）

|模型|clean|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|LEO均值|
|---|---|---|---|---|---|
|V2_EG_HIGH_ADV0_seed392005|98.433|92.825|89.399|89.548|90.591|
|V2_EG_HIGH_ADV0_seed392006|98.217|93.635|90.276|90.015|91.309|
|V2_EG_HIGH_ADV0_seed392007|98.048|93.811|91.257|91.003|92.024|
|V2_EG_HIGH_ADV035_seed392005|98.285|93.240|90.234|90.072|91.182|
|V2_EG_HIGH_ADV035_seed392006|98.187|94.316|90.980|90.934|92.076|


## 分组统计

均值±样本标准差；adv0.35若未完成三个seed，不视为完整三seed结果。

|adv|已完成seed数|LEO指标|均值|SD|
|---|---|---|---|---|
|0.0|3|accuracy|91.328|0.681|
|0.0|3|macro_f1|91.308|0.717|
|0.35|2|accuracy|91.663|0.637|
|0.35|2|macro_f1|91.629|0.632|


## 验证与文件

全量解析1172个epoch、57448条动作；已完成540000预测独立重算，详见full_source_audit.json。

scene_metrics.csv包含总体Precision/Recall/F1与floor；per_rx.csv、per_day.csv、per_tx.csv及confusion_matrices.csv保存全部分组结果。epoch_metrics_all_rows.csv保存全部现有epoch，matrix_progress.csv保存本次进度。

训练完成不等于目标测试完成。最后一行健康训练继续，本次没有启动、停止或重跑实验。

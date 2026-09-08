# 五组A1实验训练与测试状态（2026-09-09）

## 结论

截至北京时间约01:17，两组指定checkpoint续训已完成E200及四场景测试，状态为ARTIFACTS_COMPLETE_SCORE_VERIFIED；三组R3从零训练分别到E177/E182/E183，仍为RUNNING。本次没有重启、停止或加载旧权重重做prediction。

两组既有prediction均绑定本行final_ssdg.pth，共672000条，clean和三个leo_*_weak各168000条；已核对行名/run ID和覆盖数，并独立调用现有scorer复算，所有指标与已保存score.json相同。训练CSV各完整200行，pipeline记录训练exit=0及SCORED_PENDING_ANALYSIS。状态由产物独立读回确定，不只依赖dispatcher退出码。

## 已完成测试结果

以下是既有checkpoint继承实验的描述性结果，协议状态为PROTOCOL_VALIDITY_UNVERIFIED，不能列入干净泛化或晋级证据。

|场景|A1_REFERENCE|A1_FAST_SEQUENTIAL|Fast−Reference|
|---|---:|---:|---:|
|clean|76.7339%|76.6214%|−0.1125pp|
|leo_clear_weak|68.8899%|68.9202%|+0.0304pp|
|leo_low_elev_weak|66.9327%|66.9345%|+0.0018pp|
|leo_rain_weak|67.0774%|67.1673%|+0.0899pp|
|LEO三场景均值|67.6333%|67.6740%|+0.0407pp|

不能将这组单seed微小差异作为加速版性能提升证据；也未在本次状态/测试请求中完成全量收敛、机制与资源分析。各类准确率、正确数、完整路径和测试覆盖核验保存在`analysis/a1_five_test_status_20260909.json`。

## Checkpoint权限边界

两组直接来源都是历史ADV3B02_CORE90_SOFT_E200。旧记录核对了args、split_info、样本数、seed及模型参数兼容，但这些不等于2026-09-08新协议要求的实际物理ID、全部祖先及source-only选模证明。本次仅读取和复核已经存在的prediction/score，没有重新加载该来源用于正式训练或预测。来源暂按CHECKPOINT_PROVENANCE_UNVERIFIED处理，其结果按PROTOCOL_VALIDITY_UNVERIFIED保留；未证实污染，不将来源不明擅自改写为已污染。既有产物保留，不回流调参或选择。

## R3待完成测试

|行|已完成|状态|当前每轮|
|---|---:|---|---:|
|R3_STRUCTURE_CONTROL|177/200|RUNNING|约4.52分钟|
|R3_REFERENCE|182/200|RUNNING|约4.37分钟|
|R3_FAST_SEQUENTIAL|183/200|RUNNING|约4.42分钟|

三组从随机初始化开始，无外部checkpoint/teacher来源；本轮EMA由学生内存副本创建。现有dispatcher PID3850390仍存活，实际release的runner在训练结束后核验本行E200权重，再调用prediction和独立scorer；本次已核查该接续逻辑仍在等待，无需新增或重复测试进程。将固定E200输出用于测试，不使用中途checkpoint。

按目前速度，剩余训练约1.3—1.8小时，三组测试预计北京时间03:00—04:00陆续闭合；这是估算，不是已完成状态。训练尚未结束，不能提前报告R3测试成绩。后续若存在新的技术失败，应保留现场并按当前授权处理。

本次交付仅包含测试分数和核验元数据；大型checkpoint、原始IQ、prediction全文、完整训练日志及异常包保留N607原路径，未复制或覆盖。运行代码仍为旧续训release c53cabea和R3 release 9b16413e；报告提交不改变活跃训练代码。

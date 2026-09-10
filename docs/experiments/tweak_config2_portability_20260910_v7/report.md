# Tweak V7：稳定欧氏距离与source侧选模

状态：LOCAL_VERIFIED / RELEASE_PENDING。用户于2026-09-10授权修复并重新发布。

## 修复范围与实验假设

| ID | 问题与处理 | 状态 | 验证 |
|---|---|---|---|
| V7-1 | `torch.cdist`默认MM路径发生近邻相消；全部评分使用显式差分欧氏范数，校准均值/半径以float64累计 | verified | 同一近邻查询在1/32样本batch上预测一致；旧代码把32个第二类样本全部错分第一类，新实现通过 |
| V7-2 | 公共`shared_triplet_loss`统一平方L2，修正文档陈旧miner描述 | verified | Eq.(1)独立手算3.1，旧代码1.1失败，新代码通过 |
| V7-3 | 固定source训练池监控替代最低hard loss的LR/checkpoint排序 | verified | 高source识别率优先于margin-floor loss；test区NaN投毒不影响source监控；probe完整状态重置和best恢复测试通过 |
| V7-4 | 保存预测与opaque ID，再统计accuracy，保留用于独立复算的校准状态 | implemented | 本地预测与truth对齐、结果可复算、拒绝覆盖通过；正式20行独立scorer须待训练完成 |
| V7-5 | 新release/run/log；保留V1—V6 | pending | 本地测试、N607真实数据前反传和启动读回 |

V6审计中的uniform缩尺度方向对所有只保留严格hard的目标都成立，并非全量miner独有，也不能单凭该方向证明更换miner能够解决塌缩。当前没有足够证据推出作者的准确miner；V7保留V6平方L2严格hard集合、网络和物理split，只修复已实证的欧氏距离相消，并修复选模对最小尺度的偏好。不新增normalization、CE或未经验证的loss。

source监控固定读取Config2每设备训练逻辑位置0…255作参考、256…505作监控样本（一次前向读到511，最后6帧不计入M=10聚合）；两部分互斥，均属于既有75%训练池，不访问25%测试帧。它是训练侧监控准确率，不是独立validation或泛化性能。采用相同Algorithm1/2和M=10；LR仍五个各1epoch探测、要求训练loss下降，优先source监控准确率，确定性loss/LR打破平局；选择后从相同初始权重重新训练100epoch，每epoch用相同source监控指标保留best。论文未公开该选模规则，明确记为UNPUBLISHED_DEFAULT，不声称严格数值等价。

Config2 source、设备1…10、四配置单校准4×4及四配置联合校准4行、seed=20260908、SGD momentum=.9、batch64=8×8、N=10%、M=10均保持。

## 发布路径与技术停止范围

N607根：`/home/szu2070436088/2510044040/CV-SincNet`。release=`releases/tweak_config2_portability_20260910_v7/source`，output=`runs/tweak_config2_portability_20260910_v7/official_config2_full`，log=`logs/tweak_config2_portability_20260910_v7/train.log`。数据沿用`datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup`。

只允许处理本run的确定性技术故障/非法输入/输出碰撞；低性能不停止。既有数据、checkpoint和日志保留。完整结果后与论文同row比较；source监控进步不代表论文数值复现。

## 本地验证

三个已确认回归先RED再GREEN；含source边界、预测拒绝覆盖和状态重置/最佳checkpoint恢复的相关测试共36项，编译通过。真实Config2 source诊断在同一seed、LR=.001下执行3,000batch，16组梯度有限，source训练池监控准确率在step1/100/1000/3000为16.0%/26.4%/56.0%/55.6%。最后loss=.101575，中心距=.033419、半径=.055210；这并不证明消除塌缩，V7保留source表现最佳的epoch，后续以正式完整结果判断。

冻结V6 checkpoint在修复后的source监控上为62.8%，说明非常小的embedding差异仍保有身份信号；不能单凭小绝对尺度断言已丢失全部可分信息。此前报告把评分可重复等同于排除数值评分错误、把strict-hard缩尺度方向归为全量miner独有，结论过强，本报告据回归实证修正。独立P0/P1审查无发布阻断，提醒监控不是独立validation、正式预测仍须完整后独立评分。

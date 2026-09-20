# RIEI优化版原系数+等权星地CE修正版

用户修正：其余损失系数遵循原RIEI优化版，额外星地身份CE系数与clean一致，均为1。旧r01采用0.5+0.5，保留为已替代产物，不续训其权重。CVCNN原实验不变。

新RIEI从零训练，seed299、阈值0.9、E200、STAR相同数据/物理ID划分、Loo拼接增强及采样/LR/EMA口径。第一步L1=CE_TX_clean+CE_TX_Loo+0.65PL+CE_RX+0.0001Norm；第二步冻结EC/RC，L2=1.2MI−1.2IE+0.0001Norm，只更新FED。RX/MI/IE/Norm仍仅在源clean有标签及源无标签上计算，SG辅助目标仅身份CE；无标签TX真值不进入训练。PL自E30启用、阈值0.9、按选中视图平均，未改变。

原系数证据：CVS发布承载面scripts/launchers/run_cvs_fixed_riei_drift_ratio_sweep.sh第249–257行，baselines/riei_fd/train.py及losses.py。优化版MI1.2、IE1.2（损失负号）、Norm0.0001，RX CE1。

实际唯一训练差异：RIEI监督身份clean/SG系数各从0.5改为1，并追加实际系数字段日志。目标RX4只评估，E200 EMA主结果/student辅助，不按目标结果选模。旧RIEI保留日志、checkpoint、partial predictions及SUPERSEDED.json；不影响旧组CVCNN和其他任务。

验证：针对性验证两项身份CE实际权重各1、总损失分解、E29关闭/E30启用PL、有限参数更新、checkpoint读回。一次针对差异的独立P0/P1审查。状态证据见verification.json、review.json、stop_evidence.json及remote_inspect.json。

新目录和output root独占，唯一dispatch launch.lock。代码在STAR独立Git仓库提交（无remote）；登记镜像到CVS Git承载面并推送。模型scratch，全部继承来源为空，EMA仅本次student派生。

发布VERIFIED：调度器PID147712，row状态QUEUED。代码commit e5261fe；本地验证clean/SG身份CE系数各1。旧RIEI已停止保留，CVCNN未停止。当前GPU每卡2个名额已占满，等待空位自动启动；未宣称实际训练或完成。

# F 四组冻结模型：仅 clean 测试

按用户最新要求只测 clean，practical 队列已取消并保留中途产物。本 run 不重新训练，复用 r04 四个固定20轮 final.pt 和 checkpoint 内默认头；每组 clean168000 物理样本，预测固定后独立 CPU truth-last 评分。最多3个 GPU 预测 worker，另1个 CPU scorer。

需要明确：r04 实际训练使用 clean＋satellite 拼接，卫星CE权重0.68；每组记录568320个卫星分支样本，其中371968个实际叠加信道。这里只修改测试范围，不追溯改写训练配方。

四组源域扩种子决定已在目标评分前固定，WEIGHT、GEOMETRY、BOTH均未通过，因此初始四组各保留1个模型seed，不根据目标分数扩种子。

旧程序把clean预测数组暂存内存，七个视图全结束才落npz，取消时没有完整预测文件。因此本run单独重新预测clean，保留旧目录，不做模型更新、头拟合或选择性重训。实际命令、环境、输出、指标与停止规则见experiment.json。

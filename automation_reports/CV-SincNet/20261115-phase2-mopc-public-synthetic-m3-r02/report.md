# MoPC 仓库公式合成数值定位

针对 clean ProtoNet→MoPC 的真实非有限故障，在不访问真实数据、query 或 truth 的条件下，用随机合成输入和模型 seed 1737 至 1766 检查公开仓库公式，寻找能稳定复现的数值边界。沿用现有 ProtoNet 平方特征桥接、原训练预算和固定确定性设置；每个 seed 比较无裁剪与仅诊断用全局裁剪 1。裁剪改变优化更新，不能作为原方法或声称真实故障已解决。

使用 `20261115-phase2-mopc-clean-failed-savedb-support-m3-r01-release` 中的 `group2_mopc_synthetic_diagnostic --semantics public_code --seed <seed>...`；GPU 5、6、7 各一进程，启动前余量至少 6000 MiB，不影响其他 owner。输出独占 `runs/20261115-phase2-mopc-public-synthetic-m3-r02/gpu{5,6,7}.log`，不产生预测。仅观测合成非有限和梯度，不能直接证明真实任务同因。

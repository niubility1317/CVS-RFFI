# MoPC 仓库版 N20 四阶段合成数值诊断

真实 clean 故障出现在 N20 的第 3 个注册阶段，而此前随机合成仅覆盖 N10 的两个阶段。现对随机合成输入和 ProtoNet 模型 seed 1731–1760，使用 K10/N20、公开仓库目标、四个阶段分别检查无裁剪与诊断裁剪 1，查找可复现的非有限及其前置梯度/平方特征增长。随机输入和模型不使用实际数据、query、truth，不生成预测。裁剪改变优化器，仅为诊断，不冒充作者原版。

N607 GPU 5、6、7 各一进程，每进程十个 seed，启动前空闲至少 6000 MiB，不影响其他 owner；执行 `python -m experiments.pamf_phase2_practical.group2_mopc_synthetic_diagnostic --semantics public_code --new-count 20 --seed ...`，固定确定性设置及原其他预算。输出独占 `runs/20261115-phase2-mopc-public-n20-synthetic-m3-r01`，分析非有限必须区分仅 FP32 范数统计溢出和梯度元素真实非有限；合成结果不能直接证明真实故障同因。

2026-11-15：30 个 seed 的仓库版无裁剪和诊断裁剪分支均完成四阶段、每支 320 更新，未复现非有限。仅说明该随机合成范围内的结果，不证明真实故障修复。

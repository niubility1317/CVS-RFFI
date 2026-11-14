# MoPC 仓库语义合成数值检查

用户要求修复并继续所有 MoPC 实验。对原先可使 ProtoNet→MoPC 论文语义发散的随机输入构造，使用当前用户指定的 `public_code` 语义测试 seed 1731–1736；每 seed 从相同随机初始化分别运行无裁剪和诊断性全局梯度裁剪 1.0。保留每支完整阶段预算与 FP32 历史检查，记录非有限、损失、平方特征和梯度。GPU 5、6、7 各顺序处理两个 seed；固定确定性设置，无真实数据、query、truth 或预测。每 GPU 启动前无其他 owner 且空余至少 6000 MiB。

命令 `python -m experiments.pamf_phase2_practical.group2_mopc_synthetic_diagnostic --semantics public_code --seed <seed1> --seed <seed2>`，GPU 通过 `CUDA_VISIBLE_DEVICES` 指定，进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。裁剪改变优化器，不等价于作者仓库原版；合成案例不能证明真实任务已修复。代码以本报告所属提交固定。

VERIFIED：6 个 seed 的未裁剪和裁剪两支均完成 160 次更新，无非有限；裁剪支每次均发生裁剪，不能称与原优化轨迹相同。seed1731 在 paper 语义下曾出现真实梯度非有限，在本次 public_code 下未复现；这不能证明真实故障根因。各进程已退出，无 query/truth/预测。

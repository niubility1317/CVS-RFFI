# MoPC 平方特征合成故障与裁剪候选

在 GPU7 对已观察到合成非有限的随机 seed 1731、ProtoNet 到 MoPC 平方距离桥接，顺序重新初始化两次，比较无裁剪与全局裁剪 1.0。paper 损失、20 epoch、support 形状、源/新类数、C seed 42、FP32 原 guard 与固定确定性选项保持，记录 loss、平方特征和梯度边界。输入与参数均为随机合成，不读任何真实数据、query/truth，不生成预测。裁剪改变优化器，仅作为诊断候选，不能称为原作者仓库版本或据此推断真实任务故障已修复。

入口 `python -m experiments.pamf_phase2_practical.group2_mopc_synthetic_diagnostic`，`CUDA_VISIBLE_DEVICES=7`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`；启动前检查 GPU7 无其他 owner 且空余至少 6000 MiB。代码以本报告所属提交固定。比较和数值结果只能说明该固定合成样本的行为，实际保存 B 故障的因果仍需单独验证。

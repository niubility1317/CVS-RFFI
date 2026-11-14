# r15 论文路径原 FP32 检查诊断

为定位真实数值故障，读取 r15 的 `protonet_cda-s2026092702-c0195` 任务、r11 同次 B-K10 和原 P02/MIX05/8-8 K10/N20 received support，沿用实际 C seed、paper 损失及 r15 FP32 梯度范数检查。GPU5 无裁剪；GPU6 使用仅供诊断的全局梯度裁剪 1.0，明确更改优化器更新。两支各自加载同一 B，从源 checkpoint no-query smoke 开始，不读取 query/truth，不生成预测、不覆盖旧输出。

r15 源码 `_mopc_register` 的 paper 损失与当前相同，调用前替换为历史 FP32 观测/guard；同 GPU 2 原故障无法复用，因为现有健康队列占用 GPU 2。本次是 GPU5/6、增加逐层仪表的独立诊断，不能称逐位重放；未复现不证明根因消除。若出现故障保留原日志及最早观察边界，不能绕过非有限保护；裁剪效果只用于判断候选，不能冒充作者仓库原版。

入口 `python -m experiments.pamf_phase2_practical.group2_savedb_candidate_diagnostic --runtime <r15 runtime> --job <r15 jobs/00002.json> --k 10 --n 20 --device cuda:0 --guard historical_fp32 --variant <original|clip1>`。每卡 1 进程，启动前确认至少 6000 MiB 空余且不占他人任务；运行代码为 `ff961a807d5638033ac795ebd1195ef51c46cead`。

VERIFIED：GPU5 PID 3579148 无裁剪与 GPU6 PID 3579149 的裁剪诊断均完成并退出；两支各自完成 320 次更新，未在所检查边界观察到非有限。按上述条件仍未重现原故障，不能据此判定修复有效；裁剪仅为更改优化器的诊断变体，不自动作为仓库版。无 query/truth/预测，日志保存在本 run 的 `original.log` 与 `clip1.log`。

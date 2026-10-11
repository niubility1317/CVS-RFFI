# 第二阶段 clean 测试技术恢复

12 行身份训练均完成 E40/8,880 更新，原自动测试在首个批次调用 `torch.from_numpy` 时失败，未生成 prediction 或连接 truth。N607 使用 PyTorch 2.1.0+cu121 / NumPy 2.2.5，已复现相同 TypeError；替换为拥有独立可写存储的 float32 缓冲区复制，与原始数值完全相同。

本次仅恢复固定矩阵的 clean 测试。原权重、训练配置、样本、seed、预测规则、指标均不变；旧失败目录和日志保留。新恢复 run 单独保存调度记录与测试日志，最终 prediction/score 写回原 row 的既定位置供完整矩阵汇总。每卡总进程不超过 4，逐行先固定 prediction，再由独立 CPU scorer 连接 truth。

本地检查通过：1/64/256 批次完全数值一致、禁用旧 NumPy 桥时仍通过、错误 dtype 拒绝、复制后不修改输入。远端只读复现与缓冲区核验通过。恢复发布状态待独立读回。

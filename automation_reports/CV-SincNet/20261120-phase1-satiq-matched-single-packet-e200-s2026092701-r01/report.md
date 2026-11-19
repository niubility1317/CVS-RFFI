# 同报头 SatIQ 和 PAMF 的全波束单包独立评分

[登记](experiment.json) · [事件](events.jsonl) · [同报头训练](../20261120-phase1-satiq-ccs2023-header-matched-sup200-s2026092701-r01/report.md)

同一批 50,000 个独立测试包逐包预测。每包的候选集合固定为全部 3,103 个训练合格波束。PAMF 保存自身分类头的第一名；两组另用同一固定源 L 注册包集合建余弦原型，保留每个合格波束。预测产物只含不透明 ID 和预测的类别标号。独立评分进程先检查两组预测覆盖同一批包，再连接测试真值，报告已知类 Top-1、全测试包 Top-1 和覆盖率。源 V 的指标可用于训练期间诊断，独立 T 不参与模型选择、原型估计或分数调节。

## 启动验证

**VERIFIED 已上线等待。** 2026-11-20T02:55:21+08:00 读回调度 PID 2254014 和上游训练 PID 2248959，当前还未创建任何独立预测。准确检查点和训练退出记录都齐备才进入推理；预测结束后才启动独立 CPU 评分。

本地合成数据核对了检查点缺失时在访问测试前退出、两组不透明 ID 不一致时在读真值前退出、已知和全部测试的分母，以及正确的 Top-1。N607 在真实同报头第 5,000 步检查点上验证了两组的源 L 编码、3,103 个原型候选和 PAMF 分类头 argmax 一致性。该核对只读源 L。

- 调度日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261120-phase1-satiq-matched-single-packet-e200-s2026092701-r01/dispatch.log`
- 冻结预测 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-matched-single-packet-e200-s2026092701-r01/frozen_predictions`
- 最终指标 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-phase1-satiq-matched-single-packet-e200-s2026092701-r01/single_packet_scores.json`

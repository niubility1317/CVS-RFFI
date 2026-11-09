# 归一化独立时频恢复：预登记

- 固定代码提交 4bc1aa7209c1ad7ac305f2802ebc6dee195c207a，独立从头训练 15,558,434 参数的直接恢复网络，只读取单条 received IQ；初始输出为 received。
- 依据：原直接时频模型源域抽样显示解码层存在大量负饱和但并未完全失活。新版本在时频 Transformer 后使用逐 token LayerNorm，在两个解码卷积后使用 GroupNorm 与 GELU；该假设不保证识别收益。
- 数据与目标：合法源 32,768 物理记录、每条三个视图，源 IQ 误差加 0.02 输出与 received 的偏差约束，batch 32，AdamW 0.0003，20 轮固定最终轮。训练不读取 DURA、F 权重或输出、TX 标签、V 或目标数据；训练结束后才加载 DURA 作源域与正式评测对照。
- 测试：冻结后接同一 Tri-SIG-TIME SSL 原默认头，Phase1 Main split clean 和六种 practical，每景 168,000 条固定地面模拟记录；全部预测固定后独立 truth-last，报告 DURA 配对、TX/RX/day 及两种 3% 口径。目标结果不回流调参或选模。
- 资源：N607 GPU0；登记时空闲 18,105 MiB，但已有 3 个其他计算进程，因此只排队，直到至少 4,096 MiB 空闲且总计算进程少于 3 才启动；不影响其他任务。命令、路径及技术停止规则见 experiment.json。
- 10 项针对性测试通过；合法源 IQ 六步 GPU 更新和固定原 DURA checkpoint 的无目标 smoke 通过；独立 P0/P1 审查未发现阻断问题。
- 状态：等待 GPU0 可用容量，训练完成并冻结后自动测试。该版本尚无正式身份识别结果。

## 七景最终评分

七景各 168000 条预测固定后，已完成独立 truth-last 与 DURA 配对评分。六个受损场景均低于 DURA，均未达到相对下降 3% 门槛。逐场景、配对纠错及 TX/RX/day 分组见 [结果分析](analysis.md)。测试为已曝光的 Phase1 地面模拟结果。

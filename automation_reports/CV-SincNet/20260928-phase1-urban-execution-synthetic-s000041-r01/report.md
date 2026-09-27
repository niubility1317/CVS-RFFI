# Urban合成诊断与执行加速基准

状态：ANALYZED。两条预登记row均已完成并读回产物。本轮没有真实数据训练、checkpoint加载或target评分，没有占用N607训练资源。

## 结果

- CPU信道：四配置×六场景、7次配对重复，IQ/完整元数据/随机状态精确相等。lazy RNG在24项中23项更慢，因此最终FAST默认不启用。
- GPU教师：随机初始化真实CVS模型，RTX5070Ti，B512，FP32/FP16各15次配对重复。仅计算RC4使用的身份输出，比完整教师前向分别快2.814倍/2.995倍；输出、参数、buffer与RNG一致。
- 物理诊断：六场景各1024条合成单位功率IQ；low_urban遮挡54.59%、失锁43.16%。这些是模拟器质量统计，不是真实TX准确率或接收总SNR。

## 证据与重现

完整命令、解释器、PID、开始/结束时间、输出目录和seed角色见experiment.json。每row输出独占目录，两个benchmark串行完成。用于执行的代码提交为1d22591581d57cb3928e70475c7c7f930d7bbce9；测量后将lazy_rng从FAST默认移除，并在benchmark中显式开启实验对照。没有为了制造正结果重复筛选计时。

- [综合分析](../../../docs/performance/urban_channel_20260928/README.md)
- [全部测量](../../../docs/performance/urban_channel_20260928/MEASUREMENTS.md)
- [信道原始统计](../../../docs/performance/urban_channel_20260928/evidence/channel_benchmark.json)
- [教师原始统计](../../../docs/performance/urban_channel_20260928/evidence/teacher_benchmark.json)
- [验证记录](../../../docs/performance/urban_channel_20260928/validation.json)

根工作区的上述Git相对文档位于code/snapshots/daot_game_analysis_20260914_wt/。大体积真实数据、checkpoint和原训练输出保持原位置。

## 结论边界

已验证组件加速与执行等价；尚未验证N607整轮提速、真实source困难视图教师可靠性或新的urban训练准确率。完整训练预算仍待用户选择。当前运行中的五seed没有修改或重启。

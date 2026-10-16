# 当前 F＋跨 RX 默认头的冻结 practical 诊断

当前训练底座固定为 F（SIG_TIME_X3、CENTER_25K、CENTER_UNITS）＋跨 RX 同 TX 浅中层对比学习。默认 joint160／cosine30 分类头与全部原网络参数在正常训练中联合更新。两父模型均来自 `20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01`，从零 E20＝13L＋7U、4440 更新，C0 为四前向匹配 ordinary，C1 为固定 .5 ordinary＋.5 practical；沿用旧 U 门控、own EMA、RDrop、cos 与 NoExplicitCFO。父训练实际代码为 `8235c71cc4a1619eade9def1371e1bdfc31b9eae`，本诊断严格加载各自 final20 默认头，不继承历史 E50／E200。

RX1 首折源 V5400 的 C1 相对 C0：clean ＋2.7222 pp、urban 三场景均值＋17.7469 pp、low urban ＋12.7222 pp，low 仍为 38.4074%。LOS／locked 的改善大于 BLOCKED／unlocked；这是分层关联，尚不能归因于某个物理损伤。五源折确认正在另一不可变 release 内运行，本诊断不影响其进程、配置、选择或预算。

旧 F 诊断绑定 A0／W30，不能直接说明当前 F＋跨 RX 的机制。因此复用已核实的原 residual 控制实现，换用当前 C0／C1，不重做旧路线训练。唯一问题是：paired practical 训练后，移除噪声、频率／相位、传播核和注入 RX IQ 的效应，以及实际默认头／分支／梯度变化，是否与未增强参照不同。这个诊断只解释冻结模型，不构成新 Ag 有效性、因果训练归因或目标性能结论。

两个父模型 fit RX3/4/6/8 的原 L5040／隐藏 U45360／V21600；仅诊断其全部角色之外的原源 RX1 L1260（6 TX×3 day×70）。同一组物理样本、同一命名随机流及同一个生成 IQ tensor 送入两个模型。新增诊断 draw20261016621、hardware20261016622 与训练／源评价流分离，hardware session 仅 RX/day/eq，不含 TX。现有合法 source cache 反序列化可能带入未用容器，但仅使用和前向这 1260 个 L，U/V/target 均为 0。不改变原外层 7/63/30 或正式测试配置。

矩阵为 clean＋六种原 practical，每个 practical 有 full、no_noise／flat_H／no_carrier／no_RX_IQ 各 fixed 和 total、actual_power_quality、edge_boundary，共 11 条件。fixed 保留 full 的接收状态、补偿、EQ 和 AGC；total 移除对应损伤后重算。所有分层均按 full 状态定义，不因干预后状态变化重新挑选样本。噪声 SNR 仅指新增噪声；原录音噪声、TX／地面 RX 硬件仍未知。flat_H 保留数值延迟及共同复相位，不能解释成同时去掉延迟和相位。no_carrier 同时去掉所定义的频率及相位项，不能单称 Doppler 效应。控制效应不相加。

实际导入路径由 worker 打印／保存：native SSDG 与 `leo_practical` 来自既有 original release 的 `experiments/adv3b02_xuc/code/`，不是根 `code/`。fs25 MHz、IQ256、residual/post_sync、EQ=False（G=1）、reference_snr_db30、quality adaptation=True（reference20）、phase tracking=True、IQ compensation=True、reflect 边界。practical 前使用原 `normalize_input`，传播后原 AGC；clean 使用原 native carrier RMS，不追加新 packet RMS。FIR 延迟、有效观察区、边界误差、质量代理与 actual-power 重算一起记录。

沿真实前向采集既有 time／frequency／PA／reference／三个 SIG 描述、内部门控和 joint160；重建原默认头 logits 核对图路径，cosine logits 统一除以 30。clean 正确集合按各模型、固定 C0 和共同正确分别报告。CE（原 LS=.01）的 `autograd.grad` 仅读取 14 个既有参数张量，eval 模式、无 optimizer、无 `.grad` 累积；全部参数和 buffer 前后逐值相等。每条实际 traced packet 另有一次只读 classify 重建，不额外算作 backbone 前向。模型不另拟合任何分类头。

预算：0 新模型、0 epoch、0 更新。每模型 84422 网络包（含真实 checkpoint 的 2 个零 IQ no-query smoke）、1340 次导数；合计 168844 网络包、2680 次导数、1260 个唯一物理样本。父成本分别 E20／4440 更新，已完成，不计作本次新增训练。一个 worker 同时持有两冻结模型，CPU2，所有 owner 含 preCUDA 每 GPU 最多 3 进程，free≥12000 MiB。

预登记命令、环境、CWD、路径、owner、GPU 和逐行配置以本 run 的 experiment.json 为准。输出 root 独占且不可覆盖。仅本 run 的输入／来源／协议／配置／非有限数／预算／冻结状态／输出或执行技术错误触发停止并保留产物，不自动重试、不终止其他健康任务。没有按分数停止、选 checkpoint 或自动晋级新训练／目标测试的规则。

源 predictions 完整后由独立 harvester 重算默认头 accuracy／macro-F1、分类间隔、clean 正确保持率、repair／break、状态／质量／CFO／AGC／TX／day 分层、实际分支与梯度统计。结果及图表只写真实产物，不将单 seed／单源折当跨 seed 或正式目标结果。联合 clean 正收益、low urban>60% 及 suburban 接近 clean 的目标仍未达成。

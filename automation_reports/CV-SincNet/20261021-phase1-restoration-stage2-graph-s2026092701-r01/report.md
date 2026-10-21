# 两阶段蒸馏现有权重联合推理优化

状态：VERIFIED / RUNNING。GPU4实际进程、CWD、命令、CUDA设置、日志增长与真实checkpoint/F50首步smoke均已核实。七场景结果待完成。

保留用户指定 R69 `stage2` 源 E20/4940 现有权重，完整参数 266951；新增训练、optimizer、TX 标签读取和 F 参数更新均为 0。原始来源为匹配 original7/63/30 契约的合法源 IQ 与原 IQ 教师，不使用目标或 F 输出优化恢复器。原 F50 seed2026092701/defaulthead 固定。

单行 `stage2_residual_graph` 同时启用 CUDA Graph、device 有效性累计后一次主机检查、inference_mode、残差复用。保留两 prior、四 CG，FFT/IFFT 从 43 次减到 27 次；ConvLinearMAC 为 25068072/IQ，不含物理求解、归一化和 F。数学等价存在 FP32 舍入差异，实际分类一致性单独测量。

固定 source1800×7 和正式168000×7，场景为 clean、三 suburban、三 urban；B128、源尾批8/正式尾批64。每组全部七视图预测固定后独立 CPU truth-last；逐 TX/RX/day、净纠错、clean 损伤全部保留。原 stage2 已完成预测仅用于数值/argmax 对照，不重复原测试。正式结果不回流调参、排序或重跑。

同一 GPU4 上测原 stage2 native 与联合优化，完整 G 和 G+F，B1/8/32/128、三轮交替，P50/P90/P95、吞吐、显存及图捕获开销分列。真实预测与计时共用 InferenceEngine，包括输入 copy、输出 clone、有效性检查；输入预加载 GPU，不含 host 传输/loader/排队。正式全产物墙钟另报。

物理 GPU4 每卡/本run 总计一 GPU worker，启动占卡前 shared 容量锁检查 allowner<1 且 free≥18GB；CPU线程2，独立scorer最多1。不可覆盖的新 run/output/release，其他 owner 健康任务不变。真实2Ls/F50首步 smoke 覆盖B1/8/32/64/128、输入替换、输出所有权、原权重不变，通过立即继续。技术失败保留，不自动重试或切回 native。

14 项相关检查通过；唯一一次独立 P0/P1 审查在发布前完成。代码 47c264286ca689f060a3f36aafb763fe209a6f06。精确命令、环境/CWD、路径、六类seed、资源、停止规则和预期产物见 [experiment.json](experiment.json)。逐景报告原 stage2 差异、原 R17 的1pp容忍以及lowurban70%和其他五景rawclean5pp差距。本轮是已暴露、用户指定的单seed工程验证，不称新盲测。

发布验证：单一归档本地/远端SHA一致，远端编译及CLI通过；唯一一次独立P0/P1审查PASS。方法commit固定，后续登记修正仅记录实际CWD，不改变运行。

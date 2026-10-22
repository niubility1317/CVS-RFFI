# DURA机制：7设置×3学生种子

- run_id：`20261021-phase1-T1-DURA-G-ablation-m21-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Final T1 correction: 20(13L+7LU)+own10LU+own20LU, AsCE/KL0, sixL, nativeEMA/U, defaulthead. Existing F0 final T1 three seeds reused. Old wrong33L17U unstarted179 withdrawn; all healthy old workers/results preserved as historical. 150mainF+63mechanismF;35same-contract fullG+21Gstudents;249prediction/scoring rows;2exclusive timing panels.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

Final version correction and exact matrix: see docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md. Final T1 is13L37U, not old33L17U. Current state PLANNED until independent PID/log/CUDA evidence.

**VERIFIED / RUNNING**：8张卡均有本矩阵真实CUDA worker，F源训练/完整G谱系/G消融实际步数增长；不接管旧错误配方健康worker。当前首段F监督约85至90秒/轮，U和G后续阶段仍待校准；整批暂估3至7天。统一时延r01兼容失败已独立r02恢复等待独占卡，主队列不变。

## G机制准确率整组完成（2026-10-22）

**VERIFIED / ACCURACY ANALYZED**：7设置×3学生种子全部完成源训练E20/4940、raw/G七景预测和独立truth-last。取回21份完整评分与420轮训练元数据，重算16,128份CM和8,064份stratum净纠错；所有raw参考与同seed最终T1一致。汇总、逐seed、净纠错、参数/训练成本及初始化/KD匹配配对见G_ablation_*.csv，详细结果见[G_ablation_results.md](G_ablation_results.md)。

相较固定默认G，去除CG/镜像项/残差输入的六景差分别为−0.7073±0.2566、−0.3241±0.2849、−0.8665±0.3629pp；coarse-only为−5.5356±0.1551pp。全部正负场景保留，目标结果不替换默认。

新G学生与同号F seed按对角配对；默认仅一个原G配三个F，不能写成三独立G或完整复制2×2。无KD下复制效应为+5.1352±0.9465pp；随机初始化下KD效应为+2.0769±1.1184pp。缺少复制+KD三G的交互不估计。

IEEE表和两份矢量图已更新；本组仅准确率及学生增量成本闭合，机制专用时延仍归原latency队列。21学生累计增量elapsed5.100小时不含共享原R65谱系成本，不称独占GPU时长或免费教师。其余PAMF矩阵继续，后续阶段包r05保留先前r04并加入本组完整可复算输入。

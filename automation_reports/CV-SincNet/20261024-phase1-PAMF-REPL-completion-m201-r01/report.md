# PAMF-REPL 全部当前论文实验与匹配对照

- run_id：`20261024-phase1-PAMF-REPL-completion-m201-r01`
- group_id：`phase1-pamf-repl-complete-20261024`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（733 有效阶段和新包解包复算已验证）

## 目的与对照

以选定 PAMF-REPL 替换当前半监督方法，重做匹配主比较、三核心、四内部机制、相关 L 细项、两层结构及效率。189 新 F、12 精确复用；DURA 不训练，仅补已有 G 的同 F 读出。

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

## 发布实际状态（2026-10-24）

VERIFIED：实际执行版本 `dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f`，修复发布 `20261024-phase1-PAMF-REPL-control-r02`。源码已push并独立核对remoteOID；单发布包传输SHA一致、远端编译和隔离导入PASS。

真实同契约E20 checkpoint源侧smoke PASS：768 L前向、256隐藏标签U、6学生视图、全部REPL机制有效梯度；target读取0/U真值读取0，无正式更新。

独立核查八卡共24个GPU worker，各卡不超过3，计入全机其他进程；16个同PID日志跨137.1秒增长。实际快照见launch_verification.json，不把快照当固定当前数。

所有当前F/读出/统计/时延及论文图表交付仍在执行；不能把发布或smoke当性能完成。DURA新增训练0，IQ恢复缓存新增0。189新F通过84唯一E20前段共享减少重复，目标分数不回流训练、参数、队列或选模。

## CPU 评分入口恢复（2026-10-25）

VERIFIED：恢复版本 `a79bdf385489cb3822b37434a07e3c43e5ce839a`、`20261025-phase1-PAMF-REPL-control-r03`。只有CPU入口和owner更换；GPU科学版本仍为 `dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f`。

两失败CPU score在连接预测/真值前误调用模型初始化；本地复现，4项聚焦测试和一次限定P0/P1问题复核PASS。仅严格匹配完整dispatch argv/CWD的停派owner被替换，所有子进程保持或自然结束。旧失败/日志/预测保留，两同预测新评分均SCORED_COMPLETE。

初查13条健康训练中11条已自然完成、2条仍同PID。恢复后17个同PID日志跨244.6秒增长，本run每卡≤3，未解决故障0。

资源补充：核查期间GPU4在本run三进程之后新增其他账号进程，形成暂时四进程共驻留；非本run第四槽。保持全部健康进程，不使用管理员或结束其他owner；既有每卡/全机容量guard阻止受影响卡新派发。实际分布见verification，不声称全机始终每卡≤3。

历史735阶段含2个被替代评分；有效仍733，228预测/228评分，201逻辑F/189新F/12复用不变，DURA训练与新IQ缓存均0。当前计数是读回快照，不是全组完成声明。见score_entry_recovery_verification.json。

## 最终论文与导出（2026-10-25）

# PAMF-REPL：论文结果与使用边界

当前结果来自完整201逻辑F及27额外G读出；四外部Full沿用已核验practical residual四组合。本次未重训DURA，旧T1不改名。

|方法|跨日 S6|跨接收机 S6|双跨 S6|
|---|---:|---:|---:|
|RIEI|57.8713±1.3979%|56.7258±0.9063%|57.1098±0.7170%|
|DRIFT-v2|70.5404±0.5227%|60.7526±4.4161%|59.7844±3.5349%|
|IB-RFF|90.5050±0.1634%|62.2090±2.8589%|62.8447±0.9827%|
|SDG|91.4467±0.2120%|67.1214±0.9714%|65.6581±0.7571%|
|PAMF-Sup|90.9907±0.2264%|77.3597±0.5548%|75.3356±0.7096%|
|PAMF-REPL|92.3070±0.2195%|78.2863±0.9701%|76.7319±0.8771%|

F0跨RX：REPL raw clean 86.5640±0.6529%；raw S6 70.7741±0.4455%；G后S6 78.2863±0.9701%，clean 86.6169±0.6921%。

相对固定SDG D的seed内配对差为11.1649±1.0788个百分点，3/3正。完整系统相对Sup/G-L差0.9266±0.9743个百分点；固定G-LU后相对Sup为-0.0876±1.2754、相对普通PL为-0.2298±0.9544个百分点，不能混称为同一增益。

11契约跨RX S6下，固定G-LU的REPL−Sup均值有6/11非正，REPL−普通PL有8/11非正；保留全部负值，不宣称半监督普遍有效。

核心A=L CRIC，B=L配对KL，C=整个REPL U。完整条件移除与边际效应区分，SD不作为显著性检验。严格CE384 L前向与匹配核心768前向分层，不能孤立归因。

PAMF-REPL曾按已公开F0 clean标准选定；当前是如实披露的选择后比较，不宣称独立盲选。训练/权重冻结保持源侧，所有新目标结果不回流调参或选模。

新的G+REPL B1 median=4.524556ms，P95=4.563896ms；F eager和GF优化图执行分列。五轮不是五seed；共驻留显存不是单模型峰值。

新增8670唯一模型轮/1924740更新、265.956426共享worker小时；DURA新训练0、新恢复缓存0，祖先历史成本并非0；原F0 G-LU部分祖先计时缺失明确保留。

三类泛化互斥：源RX D0、目标RX D1-D3、目标RX D0。D0是更早未见日。ManyRx十类，其他六类。源数量仅1/2/3/4/6，固定六目标，不插F0五源七目标。

输入包保留379239份序列化CM、247863净纠错、所有seed/分层、111×720 L源诊断及1175case-round。原始IQ/权重/逐样本目标真值/私有runtime不进入导出。

文件入口：IEEE_experiments/main.pdf 为实验部分；figure_captions_en.md 为逐图英文图注；tables 为来源表；source_data 为完整汇总与成本；复算命令见README。

## 已验证交付

- 201 逻辑 F、228 预测和独立评分、4 个时延面板：733 有效阶段完整结束。
- 新不可覆盖包：`exports/phase1_external_dg/20261025-PAMF-REPL-complete-r17.zip`。279 个成员 CRC 全通过；解包后使用 Python `-S`（禁用第三方 site-packages）纯标准库复算通过。
- 9 张完整统计表共 1611783 行、4710684 个数值，最大数值差小于 1e-10。另 8 张成本/曝光/源诊断/计时补充表跨 Python 版本一致。
- 27 张独立 PDF/SVG、4800 dpi PNG/TIFF，5 页 IEEE 实验部分及 28 页中文逐图说明。
- 本次没有重训 DURA，没有新增恢复 IQ 缓存，没有再清理已隔离错误数据。旧 r14/r15 包完整保留。

详见 [completion_verification.json](completion_verification.json)。

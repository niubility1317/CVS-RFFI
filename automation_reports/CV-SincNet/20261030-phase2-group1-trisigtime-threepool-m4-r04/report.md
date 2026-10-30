# Phase2 第一组：PAMF 更正为 Tri-SIG-TIME SSL

Run：`20261030-phase2-group1-trisigtime-threepool-m4-r04`。当前完整 PAMF 必须是三视图 SIG-TIME＋原版 REPL 全部机制＋固定局部排列 `[0,5,3,6,2,1,4,7]`，定义引用 `configs/pamf_ssl_trisigtime.json`。上一轮错误中期普通 PL 不属于当前 PAMF；其任务已精确停止，原权重、输出与开销保留，不改标、不继续继承。

PAMF A 直接复用已完成 `20261027-phase1-pamf-repl-shuffled-views-m16-r02` 的三视图完整模型。八个合法候选均已核实与本轮源合同完全一致；依既定4-seed预算使用2026092701/02/03/04，按序指定，未依据目标成绩选seed。各模型E50/11100更新，20＋10＋20自身轨迹、232993参数；E1–13仅L，E14–20普通PL前缀，E21–50完整REPL（U权重6、LS.01、style/gradient guard/joint contrast/receiver extrapolation全部开启），无EMA、原默认头联合训练，固定重排训练/推理一致。

复用原PAMF源配置，不额外更改其增强：`satellite_concat=false`，原完整L中的As真实标签槽保留。外部方法仍按用户指定拼接星地增强0.68＋中期普通PL0.95/权重1运行。因此完整方法比较的源标签角色、物理ID与数据权限一致，但不宣称所有方法采用同一个SSL目标或完全相同的输入增强。若比较普通PL消融，需明确标PAMF-PL，不能代替主方法。

源权重加载前核对原run/seed/完整REPL参数、实际调用计数、三段自身祖先、源数据合同；加载后检查实际3视图、232993参数、窗口排列、默认头与两个真实L样本的无query前向。四个原模型全部复用，不重训F/G，不读目标或query来挑模型。

目标实验范围保持：三物理池、7RX、4seed、MIX8主表及LS/MU/LU各5组、MIX01的K1/5/10/20与N2/5/10/20曲线。B只适应旧类；C直接从A注册，PAMF另列B→C过程；MILKR终态全量重拟合，AC/ABC相同条件不重复宣称历史收益。DURA v1固定四格S0Q0/S0Q1/S1Q0/S1Q1，每stage只两次support拟合。query补集、最小176、received及truth-last保持。

10个外部方法及其40个源seed行继续在原r01代码/runtime/output运行，健康worker保留PID/CWD/GPU，不重启；已完成行按原completion复用。Tweak M10、CSIL作者Fisher和MoPC-HR原编码器输入绑定仍明确缺项。不会用任意扩展填补。

调度接管仅发生于已经停止并独立核实退出的错误PAMF控制器；原外部CPU/GPU资源在同一控制器下继续排队。总计每GPU最多两个本任务训练/预测worker，其他owner另计；实际free≥6000MiB，CPU scorer总计1，既有源增强服务共享。入口 `python -u -m experiments.pamf_phase2_practical.group1_recovery --runtime <release>/runtime.json`。

资源：PAMF A使用原三段实际 `phase1_resource_summary.json`、实际11100更新及源训练时间；本轮加载/无query smoke开销单列，新增训练更新0，缺失历史字段标不可用。B/C继续逐拟合实测耗时、CPU RSS、CUDA allocated/reserved、更新参数/步数和持久状态；G/F/核求解与query推理分开。有争用测量如实标记，失败旧配方开销不隐藏，也不计为当前有效源训练。

有效总矩阵仍为11家族、44个源模型（PAMF4复用＋外部40既有新训练行），35616次目标拟合＋6次固定BC增量资源测量，21263预测/21263独立评分任务。r04不重复任何外部工作。数据VALIDATED_ONCE复用，不新做数据哈希/重验。配置为 `configs/pamf_phase2_group1_trisigtime_20261030.json`，Git固定代码；一次P0/P1审查、聚焦负测和真实checkpoint smoke后直接启动。

停止规则：源方法错误或协议/输出冲突阻止对应后继，源服务系统失败停止新接纳；低分不停止、不重排、不调参。当前为PREREGISTERED，发布后独立PID/CWD/argv/GPU/log读回才能声明运行，测试全部固定后独立truth-last。

## 实际发布读回（2026-10-29T18:38:13.108313+00:00）

状态：**VERIFIED / RUNNING**。代码提交 `951bd957b8cc2cf6da96f48ad0f5aad34f7dfcb9`。四个原完整Tri-SIG-TIME SSL模型均通过真实权重/配方/源祖先/实际视图与重排检查；F＋DURA固定CUDA执行路径的source-only smoke PASS，query/target读取0，新增源训练更新0。CPU图执行尝试被GPU路径要求拦截，其日志保留；随后同一固定代码在合法GPU上完成检查，模型及精度未变。

统一controller PID `3149875` 的CWD/argv已独立核实；收养9个仍在运行的外部worker，其他完成行按原completion复用。每卡自有worker `{"0": 2, "1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2}`，总16；当前计数 `{"COMPLETE": 43, "FAILED": 0, "BLOCKED": 0, "RUNNING": 16, "QUEUED": 42511}`（工作单元计数，不是全部实验完成）。原4个错误PL训练worker与其CPUcontroller均独立确认退出，未停止其他owner。

源训练成本引用原模型三段实际resource summary和训练时间，本轮加载/检查单列；B/C资源继续由实际拟合记录。后续固定预测与独立truth-last评分已串联，当前不宣称完整性能结论。

## 2026-10-31 性能核查

已按后续r08权威jobs.json连接r04/r05的PAMF和r06的外部评分，复核完整主矩阵。主要发现为单边恢复的表示/核尺度失配；source完整训练、主矩阵保存状态及评分算术未发现相应故障。详细证据和边界见 [性能核查报告](../../../docs/analysis/PAMF_PHASE2_DIAGNOSIS_20261031.md)。本次仅分析冻结结果，不改参数/权重，不启动新实验。

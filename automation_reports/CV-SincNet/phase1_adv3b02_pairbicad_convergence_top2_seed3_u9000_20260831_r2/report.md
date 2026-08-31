# PairBiCAD前两名source-LORO收敛确认r2

## 状态

- 当前状态：`LOCAL_VERIFIED`。
- Run ID：`phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r2`。
- 目的：仅用source-only证据比较已冻结前两名P2/P1，并冻结后续完整多种子实验的训练预算。
- r1边界：r1因receiver索引/字符串标签解析错误在训练前12/12行技术停止，状态为`NO_PERFORMANCE_RESULT`，不参与选择；partial artifact保留。

## 固定代码与配置

- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 修复提交：`362c634326674a88cb2aacb3ce9cc76f00db9aa3`，远端OID已独立核对一致。
- 修复仅将ManySig held-out receiver按payload整数索引解析；真实`rx_list`标签可为`18-2`等字符串。候选、fold、seed、预算、协议和选择规则不变。
- 完整PairBiCAD测试通过；只有3个既有AMP弃用警告。trainer、launcher和分析器编译通过。
- Luna定点复审：`PASS，可以发布r2`。

## 冻结矩阵

- 候选：`P2,P1`，顺序固定。
- fold：`1,8`；对应held-out source receiver索引分别为1、8。
- seed：`392001,392002,392003`。
- 共12行，每GPU最多2个本run训练进程；GPU0–3各2行、GPU4–7各1行。
- 每行最大`9000 updates`；从U4000开始每500 updates在同row held-out source receiver上评估clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- source-LORO主分数：`H(clean_accuracy,min(three LEO accuracies))`；严格改善阈值`1e-12`，patience=5。
- 仅使用ManySig source receiver索引`[1,3,4,6,8]`和day1/2/3；禁止target、Phase2、support、query或truth。

## 路径与命令

- N607普通账户：`szu2070436088`，禁止管理员账户。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_convergence_20260831_r2`。
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r2`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r2.dispatcher.log`。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。

```text
PAIRBICAD_CONVERGENCE_CANDIDATES=P2,P1 bash code/scripts/launch_phase1_pairbicad_convergence_n607_20260831.sh
```

## 预期artifact与停止规则

每行必须保留`source_loro_curve.jsonl`、`source_loro/checkpoint_u<update>.pth`、`source_loro_selection.json`、final checkpoint、strict runtime、clean和三种LEO评估，以及`ARTIFACTS_COMPLETE.json`或`TECHNICAL_FAILURE.json`。Run级保留`plan.json`、`final_status.json`、dispatcher日志和PID文件。

只有错误candidate/fold/receiver/day/seed/update、source-only越权、输出冲突、错误release/CWD、命令无法运行、无artifact闭合、同一确定性pre-prediction异常至少2行或进程归属不清时才允许精确停止并保留partial artifact。不得因中间或最终低性能停止、重启、热补丁或选择性重跑；不得影响无关进程。

12行全部闭合后，只按source-only曲线冻结胜出候选。胜出候选6个row的最佳update取中位数并量化到500 updates，冻结为fold1–5×seed392001/392002/392003共15行最终确认预算；不得使用任何target或Phase2反馈。

## r2发布与启动证据

- 发布提交：`00c9ea7471d25d2e86ac98306010dd249c9c20cd`，远端OID与本地`HEAD`一致，工作树干净。
- release归档：本地`E:\type10-7\release_archives\phase1_pairbicad_convergence_00c9ea74.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_convergence_00c9ea74.tar.gz`。
- 唯一归档SHA256：`caf0c8214aa608a906319b28072e40e970533e67e479c7c8d125466c749b3f02`，本地/远端一致。
- 远端release解包后编译：`PASS`。
- 启动时间：2026-08-31约17:42 CST；dispatcher PID：`2946124`。
- 启动绑定：dispatcher的CWD/命令行绑定到`phase1_pairbicad_convergence_20260831_r2`release和精确r2 run根；12个直属训练worker全部存在。
- GPU映射：plan.json确认GPU0–3各2行、GPU4–7各1行，不超过每GPU2行；GPU0–7启动后均出现计算负载。
- 矩阵读回：12/12行均为P2/P1、fold1/8、seed392001/392002/392003、U9000、day1/2/3和`source_only=true`。
- 初始计数：`ARTIFACTS_COMPLETE=0`、`TECHNICAL_FAILURE=0`；trainer日志0字节属于训练中结构化指标尚未写入的预期状态，GPU计算和进程绑定健康时不得据此干预。
- 当前状态：`RUNNING`。

## 结果冻结算法

1. 每个row只使用其`source_loro_selection.json`记录的最佳source-only分数与最佳update；不得读取目标域或Phase2证据。
2. 候选分数为该候选6个row最佳主分数的算术平均值，较高者胜出。
3. 若候选均值在`1e-12`内相同，依次比较6行最小值，仍相同则沿用U4000冻结顺序`P2>P1`；禁止根据目标测试反馈破同分。
4. 胜出候选的6个最佳update取中位数；若中位数落在两个500网格中点，向上取整，再限制到`[4000,9000]`，得到唯一固定最终预算。
5. 预算与候选冻结后，最终矩阵固定为胜出候选×fold1–5×seed392001/392002/392003共15行；不允许在线早停、目标反馈调参、重训或选择性重跑。

## 收敛确认完整结果与冻结

- 终态：12/12行均为`ARTIFACTS_COMPLETE`，`TECHNICAL_FAILURE=0`，dispatcher及直属worker均已退出。
- 完整读取范围：12份`source_loro_curve.jsonl`共103个评估点、12份selection、12份完整JSONL/CSV训练遥测、12份checkpoint runtime、12份completion artifact、48份场景JSON和48份场景log；未使用tail或抽样替代全量解析。
- 严格性：12行final checkpoint重建的missing/unexpected/shape mismatch均为空；所有selection与103个曲线点均为source-only，target/Phase2/support/query/truth访问均为false。
- 分析器已扩展为支持声明矩阵规模和LORO早停实际update；对应测试9/9通过，分析提交`20060ccec3be33449ed10434d2f8177c7bebd29e`已推送。

|候选|6行最佳H均值|标准差|最小值|final clean均值|final LEO均值|final LEO场景下限|
|---|---:|---:|---:|---:|---:|---:|
|P1|42.1297%|1.8202pp|39.8983%|57.4398%|30.7407%|25.1444%|
|P2|40.7525%|1.9917pp|38.3515%|52.5500%|29.0281%|24.6056%|

|row|stop update|best update|最佳source-LORO H|
|---|---:|---:|---:|
|P1-F1-S392001|6500|4000|42.4391%|
|P1-F1-S392002|8000|5500|39.8983%|
|P1-F1-S392003|9000|9000|45.4025%|
|P1-F8-S392001|9000|9000|40.8644%|
|P1-F8-S392002|7000|4500|40.9717%|
|P1-F8-S392003|9000|7000|43.2023%|
|P2-F1-S392001|9000|7000|43.1062%|
|P2-F1-S392002|8000|5500|39.2765%|
|P2-F1-S392003|7000|4500|38.3515%|
|P2-F8-S392001|6500|4000|42.0621%|
|P2-F8-S392002|6500|4000|38.7882%|
|P2-F8-S392003|8000|5500|42.9307%|

冻结结论：

- P1的6行最佳H均值比P2高`1.3772pp`，且最差行高`1.5468pp`；按预登记规则冻结最终候选`P1`。
- P1最佳update排序为`[4000,4500,5500,7000,9000,9000]`，偶数中位数为6250；500网格中点按预登记规则向上取整，冻结最终训练预算`U6500`。
- 后续完整确认固定为`P1×fold1–5×seed392001/392002/392003`共15行、每行U6500、禁用在线LORO早停；不得使用目标域或Phase2反馈。
- 当前run状态：`ANALYZED`。

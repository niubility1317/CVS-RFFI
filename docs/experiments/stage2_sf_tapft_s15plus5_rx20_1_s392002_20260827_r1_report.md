# SF-TAPFT S15+首发实验报告

## 当前结论与状态

- run ID：`stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1`。
- 当前状态：`ANALYZED_SUPPORT_OOF`；5/5行selection与clean-single bundle闭合，未读取独立query或query truth。
- 实现提交：`2eb30bdaa8ebdc4eff0bfbc10f395b9d1568bd4a`。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`、K=10×6=60条support；适配、选择和bundle构建不读取query、query truth、clean/source样本。
- Phase1基础：ADV3B02 CORE90 checkpoint及其既有Phase1 bundle。
- capsule：`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`。
- split：`stage2b-rx20-1-seed713101-before-support-prefix`。

## 历史F0锚点

F0直接复用同一bundle、同一support和同一4折OOF的既有S15结果，不重复启动：

|候选|步数/时钟|BA|fold floor|NLL|full-support步数|wall-clock|最大RSS|GPU峰值|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|F0/S15|300/ref4500,warmup0.05|86.1111%|77.7778%|0.514836|203|0:39:17.91|1,829,304KiB|690MiB|

长程S00同run锚点为BA=86.1111%、fold floor=77.7778%、NLL=0.436715、full-support选择327步、wall-clock=3:46:23。

## 首发矩阵

|row|候选|单一科学目的|训练配置|GPU|
|---|---|---|---|---:|
|F1|S15-SCHED300|验证短程时钟校准|300/ref300,warmup0.10|0|
|F2|F1+TEMP|验证OOF正标量温度能否修复NLL且保持argmax|F1+OOF temperature|1|
|F3|HEAD60+JOINT240+TEMP|验证冻结embedding的head预热|前60步head-only缓存，后240步head+norm|2|
|Q2A|S16-A mixed norm|结构轴：t3 weight+bias+t2 weight|4500/ref4500,warmup0.05|3|
|Q2B|S16-B mixed norm|结构轴：t3 weight+bias+早层norm weight|4500/ref4500,warmup0.05|4|

F1–F3使用预登记稀疏验证步：1、5、10、20、30、50、75、100、150、200、250、300。Q2A/Q2B保持S00逐步验证和4500步长程条件，使其只改变Norm集合。

## 已落地实现

- 300步独立cosine时钟，30步warmup并在第300步附近衰减至零。
- 仅在cross-fitted OOF logits上拟合单一正温度T；记录NLL前后、argmax不变性，并把T持久化到strict clean-single bundle。
- F3前60步复用冻结support embedding，不执行昂贵backbone训练forward/backward；后240步联合head+norm。
- 支持S16-A/B分层Norm规则，不再要求所有Norm共享相同weight/bias策略。
- LOO prototype从样本×类别Python双循环改为向量化等价计算。
- KD权重为0时不复制完整teacher。
- checkpoint snapshot仅保存head和许可参数，最终以完整不可变anchor重建，非许可state必须严格相等。
- 新增fold floor、ECE、per-class NLL、温度、验证步、backbone forward、snapshot字节、可训练/实际变化元素等artifact字段。

## 验证与独立审查

- 聚焦测试：76项通过。
- Python编译与row launcher CLI回读通过。
- 独立P0/P1审查发现温度bundle丢失和短refit派生两项P1；均已定点修复并复核闭合，无残留P0/P1。
- 全仓测试收集受既有Python3.10缺少`tomllib`以及`tests/`与`code/tests/`同名模块冲突阻断；该问题不属于本次改动，未作为额外发布门。
- 真实checkpoint无query smoke作为远端launcher第一步；PASS后立即进入正式row。

## 发布命令和路径

- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/<release-checkout>`。
- 矩阵：`configs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827.json`。
- row命令模板：`CUDA_VISIBLE_DEVICES=<gpu> /usr/bin/time -v python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix <matrix> --row-id <row> --output-dir <run-root>/<row> --device cuda:0 --folds 4`。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1`。
- 预期每行artifact：`selection.json`、`sf_tapft_clean_single_bundle.pt`、stdout/stderr、GNU time记录；矩阵级保留launch receipt和GPU采样。

## 停止规则

只允许因协议/query泄漏、错误checkpoint/capsule/split/K、错误checkout、输出碰撞、无法产生selection/bundle、同一确定性预prediction异常至少两行或进程归属不清而定点停止本run进程。不得因低性能停止；不得触碰无关任务。

## 晋级门槛

support OOF相对S00同run锚点：

- BA≥85.6111%；
- fold floor≥77.7778%；
- NLL≤0.466715；
- 理想NLL<0.50且优先≤0.4667。

F4暂缓：必须先取得严格绑定的长M02 cross-fitted teacher logits，不能用普通source teacher冒充OOFKD。Q1必须使用新的独立query；Q3仅在Q1通过后启动。
## 远端发布与部分结果（2026-08-27）

- release commit：`ea189192f55953f076c472e6fb1f71625131a2ea`，GitHub远端OID独立回读一致。
- release archive：`sf_tapft_s15plus_ea189192.zip`；本地/远端SHA256均为`1219bdc6472b93478c9140e2b802993653bb15402a337470017e15cd642e7cbe`。
- N607 preflight：8张RTX3090均空闲，目标run root启动前不存在。
- 远端编译PASS；真实checkpoint无query smoke为PASS，60条support、1步、3.65秒、exit 0、`query_opened=false`。
- 首次smoke的Python`-c`因本地/远端引号剥离触发`SyntaxError`，checkpoint未加载、训练未开始；失败日志保留。改用UTF-8标准输入后通过。
- launch脚本主体已启动5行，末尾CR字符导致外层命令非零；未重复发布。独立对账确认5条receipt、PID/CWD/cmdline/GPU映射正确，状态由`UNKNOWN`恢复为可验证的`RUNNING`。
- F1/F2/F3已`ARTIFACTS_COMPLETE`；Q2A/Q2B继续`RUNNING`。

### 短程3行support OOF结果

|row|BA|最低fold BA|最低类别召回|NLL|ECE|T|full-support步数|backbone训练forward|wall-clock|最大RSS|bundle|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|F1|84.7222%|72.2222%|50.0000%|0.546213|0.192620|1.000000|150|150|0:48.94|1,838,816KiB|4,292,702B|FAIL：BA/fold floor/NLL|
|F2|84.7222%|72.2222%|50.0000%|0.542836|0.187467|1.071440|150|150|0:50.13|1,825,924KiB|4,292,702B|FAIL：BA/fold floor/NLL|
|F3|82.6389%|69.4445%|50.0000%|0.529651|0.206704|1.014850|100|40|0:40.17|1,824,268KiB|4,292,702B|FAIL：BA/fold floor/NLL|

说明：

- 表中“最低fold BA”是与历史S00/S15门槛可比的4折BA最小值；“最低类别召回”是本次新增的更细粒度诊断，不替代旧fold floor。
- F2相对F1保持所有argmax和BA不变，聚合NLL改善0.003376、ECE改善0.005153；校准方向正确但幅度太小。
- F3的OOF pooled温度拟合NLL从0.512131降至0.512044，但按fold等权聚合NLL为0.529651；两者加权口径不同，不应拼接。F3的60步head缓存使最终full-support仅执行40次backbone训练forward。
- F1–F3训练期snapshot均为6,336B，可训练/实际变化元素均为1,584；clean-single bundle仍约4.29MB，因为部署delta-only FP16属于尚未完成的独立瘦身项。
- 相对历史F0/S15的39分17.91秒，F1、F2、F3墙钟分别缩短约97.92%、97.87%、98.30%，但三者均未满足科学门槛，不能晋级。
## 全矩阵闭合与最终support OOF结论（2026-08-27 18:53 CST）

Q2A/Q2B均已生成`selection.json`、strict clean-single bundle和GNU time，stderr为0B、exit status=0、GPU进程已退出。两个bundle均完成本地`torch.load(weights_only=True)`审计：schema、60条support、Norm规则、选中步数、query/source/target_eval关闭和`nonpermitted_changed_names=[]`全部一致。

### Q2长程结构结果

|row|结构|BA|最低fold BA|最低类别召回|NLL|ECE|full-support步数|可训练/变化元素|backbone训练forward|snapshot|wall-clock|最大RSS|bundle|采样显存|门槛|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Q2A|t3 weight+bias+t2 weight|87.5000%|77.7778%|50.0000%|0.440073|0.184047|503|1248/1248|503|4,992B|16:16.49|1,820,864KiB|4,291,806B|≥668MiB|PASS|
|Q2B|t3 weight+bias+t2/t1/time_fuse weight|87.5000%|77.7778%|50.0000%|0.420259|0.175369|231|1368/1368|231|5,472B|15:15.60|1,823,176KiB|4,292,254B|≥670MiB|PASS|

“采样显存”来自启动后一次GPU采样，不是连续采样峰值；NVIDIA accounting未返回历史max，因此不能把该值声称为完整运行峰值。

相对S00：

- Q2A：BA+1.3889pp、最低fold BA持平、NLL+0.003358，可训练元素减少336（21.21%）。
- Q2B：BA+1.3889pp、最低fold BA持平、NLL-0.016456，可训练元素减少216（13.64%）。
- Q2A/Q2B墙钟相对S00的3:46:23分别缩短约92.81%和93.26%。该收益来自向量化LOO、取消每步全模型CPU距离、许可参数距离和紧凑snapshot共同作用，不能归因于单一优化。
- 两行都满足BA≥85.6111%、最低fold BA≥77.7778%、NLL≤0.466715。按“满足门槛的最小候选”规则，当前矩阵选择Q2A；按NLL/ECE和更少选中步数，Q2B是校准优先工作点。

### per-class NLL诊断

下表是4折内各类别NLL均值再做fold等权平均，类别顺序0–5；它不是样本数加权的pooled NLL。

|row|c0|c1|c2|c3|c4|c5|
|---|---:|---:|---:|---:|---:|---:|
|F1|0.609635|0.477730|0.043834|1.066165|0.206824|0.765887|
|F2|0.620776|0.487975|0.059191|1.040366|0.215461|0.731525|
|F3|0.947466|0.438040|0.042919|0.725999|0.184108|0.737495|
|Q2A|0.450965|0.510991|0.043757|0.800885|0.086046|0.675757|
|Q2B|0.545463|0.337898|0.042400|0.771942|0.095293|0.637930|

短程失败主要集中在c0、c3、c5；Q2B显著修复c1/c3/c5，但c0略差于Q2A。c2在全部候选都非常低，说明总体NLL并非由所有类别均匀贡献。

### 全路线决策

1. F1–F3均淘汰。不到1分钟的墙钟是有效工程结果，但当前稀疏选择工作点无法维持BA、最低fold BA和NLL，不能建立S15-FAST默认档。
2. Q2A/Q2B均通过support OOF门槛。Q2A是本轮最小合格结构，Q2B是本轮NLL/ECE最优结构。
3. 与既有S02比较，S02仍是参数/BA Pareto点：1152元素、BA=89.5833%、最低fold BA=77.7778%、NLL=0.424413。Q2A被S02在参数、BA和NLL上支配；Q2B仅在NLL上优于S02约0.004154，但多216元素且BA低2.0833pp。
4. 因此不改变既定优先级：下一科学动作仍是S02的新独立query闭合。Q2B保留为校准优先备选，不因support OOF结果直接提升为默认。
5. 本run从实现、发布到5/5 support selection和clean-single bundle已达到`ANALYZED_SUPPORT_OOF`；没有读取query或query truth，不能宣称独立query或最终部署晋级完成。

## F2真实query后续闭合（2026-08-27）

F2 strict bundle随后在独立run`stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`进入旧6类真实holdout评估。该holdout与历史M02同row，但truth已在16行闭合后揭示，因此证据等级为`REUSED_VALIDATED_HOLDOUT_NOT_NEW_PROSPECTIVE_QUERY`。

- `DA0_REG0`：BA=71.6667%、floor=10%、NLL=0.936688。
- `DA1_REG0`：BA=85.0000%、floor=60%、NLL=0.471599。
- 域适应效应：BA+13.3333pp、floor+50pp、NLL-0.465089。
- 相对M02：BA-1.6667pp（少1个类1正确query）、floor持平、NLL-0.037839。
- OOF温度归因：同一DA1 logits去温度重构NLL=0.462502；加入`T=1.071440445141165`后NLL=0.471599，恶化0.009097且argmax不变。

因此F2快速域适应本身有效，但未通过BA晋级门槛；当前OOF全局温度在真实query上方向反转，不能作为默认校准。完整证据见[独立F2 query报告](stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1_report.md)。

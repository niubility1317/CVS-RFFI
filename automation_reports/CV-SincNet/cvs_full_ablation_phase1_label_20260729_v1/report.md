# CVS全量消融Phase1标签率v1

## 实验登记

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_label_20260729_v1`|
|时间|2026-07-29|
|操作者|Codex主代理；N607发布将由独立唯一runner负责|
|目标|完成设计报告`P1-LABEL`的0.005、0.01、0.02、0.05标签率训练，并复用Phase1 T1中0.10的`P1-FULL`结果形成五点曲线|
|比较对象|同一`P1-FULL`方法配置；仅改变`f_L/f_U`，固定`f_V=0.30`|
|环境|本地`ssr-gpu`；远端`CVS-RFFI`|
|状态|`LOCAL_REPAIRED / INDEPENDENT_RE_REVIEW_PENDING`|

## 设计追踪

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P1L-01|4.1、6.7|标签率覆盖0.005、0.01、0.02、0.05、0.10|factory、spec、runner、本报告|verified|85个定向测试和14行dry-run通过|0.10复用当前T1的`P1-FULL`五seed完整结果|
|P1L-02|4.1|严格使用`f_L=0.70rho`、`f_U=0.70(1-rho)`、`f_V=0.30`|factory、训练P0检查|verified|请求比例精确；source split receipt同时记录实际比例并要求绝对偏差不超过0.002|不要求不同启动批次数据字节一致|
|P1L-03|6.7|每点至少3个筛选seed；0.01低标签关键点使用5个确认seed|spec、sealed plan、runner|verified|14行矩阵及seed计数测试通过|0.005/0.02/0.05各3行，0.01为5行|
|P1L-04|4.1|checkpoint只由source validation选择|factory、训练P0检查、artifact validator|verified|所有label arm终端契约与既有completion validator通过|禁止target指标选择|
|P1L-05|6.7|报告strict UDU、receiver floor、pseudo precision/coverage、对P1-SUP增益、绝对样本数|训练artifact、完成报告|pending|待N607完成产物|同一行指标一起保存|
|P1L-06|9.1、11|正式入口可达、16槽调度、独占输出、完整checkpoint/telemetry/prototype/receipt|runner、测试|implemented|label stage不再强制T1 reuse manifest且拒绝误传；独占输出和artifact故障注入回归通过，待重新独立复审|14行映射GPU0–5各2行、GPU6–7各1行|
|P1L-07|14.2|最小闭环至少覆盖0.01、0.05、0.10|N607运行与完成表|pending|待实验闭合|本轮同时覆盖完整五点网格|

当前计数：verified=4、implemented=1、pending=2、deferred=0、rejected=0、blocked=0。当前尚未通过独立复审，不发布。

## 预登记矩阵

|rho|f_L|f_U|f_V|新训练seed数|复用|
|---:|---:|---:|---:|---:|---|
|0.005|0.0035|0.6965|0.30|3|否|
|0.010|0.0070|0.6930|0.30|5|否|
|0.020|0.0140|0.6860|0.30|3|否|
|0.050|0.0350|0.6650|0.30|3|否|
|0.100|0.0700|0.6300|0.30|0|复用Phase1 T1`P1-FULL`五seed|

共14个新训练行。训练seed从未用于本设计报告正式结果的fresh Phase1 seed登记表中选取；0.005、0.02、0.05使用同一组三seed，0.01增加两个确认seed。每行只改变标签率，不改变方法、总source pool、validation比例、训练预算或checkpoint选择规则。

## 发布前必须通过

- 本地`ssr-gpu`下定向单测、`py_compile`、14行dry-run。
- 正式训练入口对每个rho应用正确比例，错误比例、错误row key、错误候选ID和输出已存在均fail closed。
- 独立复审P0=0、P1=0。
- Git提交、不可覆盖run ID、远端release独立目录。
- N607发布前记录活跃GPU进程；每GPU最多2个训练进程。

## 本地实现与验证

|项目|结果|
|---|---|
|实现文件|`code/cvsrffi/phase1_ablation_factory.py`、`code/cvsrffi/full_ablation_spec.py`、`code/SSDG/train_ssdg.py`、`code/scripts/build_full_ablation_plan.py`、`code/scripts/run_full_ablation_phase1_t1.py`、`code/scripts/seal_full_ablation_phase1_plan.py`、`configs/full_ablation_20260728/phase1_label_rho100_reference_v1.json`|
|测试文件|`tests/test_phase1_ablation_factory.py`、`tests/test_full_ablation_spec.py`、`tests/test_build_full_ablation_plan.py`、`tests/test_run_full_ablation_phase1_t1.py`、`tests/test_seal_full_ablation_phase1_plan.py`|
|编译|6个发布路径`py_compile`通过|
|定向回归|首轮85个测试通过；复审修复后`ssr-gpu`下89个测试全部通过|
|训练路径|四个label arm均通过真实训练parser的`train(...,dry_run)`，终端P0/P1契约全部为真|
|矩阵dry-run|14行、14个dispatch、14个new train、0复用dispatch、0重导出；GPU行数为2/2/2/2/2/2/1/1|
|数据处理|复用已登记WiSig数据标识，不重新读取全量数据计算hash，不做数据集审计|

## 首轮独立复审与定向修复

提交`b2e868e8`首轮独立复审结论为`FAIL`，P0=2、P1=1、P2=0，未seal、未同步、未启动。复审artifact为`independent_review_b2e868e8.json`。

|发现|修复|
|---|---|
|label execute被T1 reuse manifest强制阻断|runner仅对`stage=t1`要求该manifest；label stage不要求且显式拒绝误传，新增两个execute负/正路径测试|
|0.005请求在真实ManySig分组后实际为`rho=0.0042918455`、`f_V=0.301`，终端原先只校验请求参数|source split receipt新增请求/实际比例、绝对样本数和固定离散化容差；正式终端要求实际`rho`和`f_V`偏差均不超过0.002，超限fail closed|
|0.10复用仅为报告文字|新增Git绑定的`phase1_label_rho100_reference_v1.json`，固定T1 v5源run、五个同seed`P1-FULL`行及所需artifact；T1完成前状态保持`RUNNING_PENDING_ARTIFACTS_COMPLETE`，不得形成五点曲线结论|

论文表格必须同时显示请求标签率、实际标签率和L/U/V绝对样本数；统计横轴使用实际标签率，避免把离散化后的0.0042918误写成精确0.005观测。

## N607登记

尚未同步或启动。Phase1 T1 v5仍在运行；本轮修复等待重新独立复审，P0/P1未归零前不seal。

## 完成后结果表

|rho|seed|labeled数|unlabeled数|source validation数|best epoch|strict UDU|receiver floor|pseudo precision|pseudo coverage|P1-SUP同seed增益|checkpoint|状态|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

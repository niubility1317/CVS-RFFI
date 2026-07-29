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
|状态|`LOCAL_VERIFIED / INDEPENDENT_REVIEW_PENDING`|

## 设计追踪

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P1L-01|4.1、6.7|标签率覆盖0.005、0.01、0.02、0.05、0.10|factory、spec、runner、本报告|verified|85个定向测试和14行dry-run通过|0.10复用当前T1的`P1-FULL`五seed完整结果|
|P1L-02|4.1|严格使用`f_L=0.70rho`、`f_U=0.70(1-rho)`、`f_V=0.30`|factory、训练P0检查|verified|四arm精确比例、仅两字段diff、终端P0覆盖通过|不要求不同启动批次数据字节一致|
|P1L-03|6.7|每点至少3个筛选seed；0.01低标签关键点使用5个确认seed|spec、sealed plan、runner|verified|14行矩阵及seed计数测试通过|0.005/0.02/0.05各3行，0.01为5行|
|P1L-04|4.1|checkpoint只由source validation选择|factory、训练P0检查、artifact validator|verified|所有label arm终端契约与既有completion validator通过|禁止target指标选择|
|P1L-05|6.7|报告strict UDU、receiver floor、pseudo precision/coverage、对P1-SUP增益、绝对样本数|训练artifact、完成报告|pending|待N607完成产物|同一行指标一起保存|
|P1L-06|9.1、11|正式入口可达、16槽调度、独占输出、完整checkpoint/telemetry/prototype/receipt|runner、测试|implemented|现有严格runner已支持label stage；独占输出和artifact故障注入回归通过，待独立复审|14行映射GPU0–5各2行、GPU6–7各1行|
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
|实现文件|`code/cvsrffi/phase1_ablation_factory.py`、`code/cvsrffi/full_ablation_spec.py`、`code/SSDG/train_ssdg.py`、`code/scripts/build_full_ablation_plan.py`、`code/scripts/run_full_ablation_phase1_t1.py`|
|测试文件|`tests/test_phase1_ablation_factory.py`、`tests/test_full_ablation_spec.py`、`tests/test_build_full_ablation_plan.py`、`tests/test_run_full_ablation_phase1_t1.py`、`tests/test_seal_full_ablation_phase1_plan.py`|
|编译|6个发布路径`py_compile`通过|
|定向回归|`ssr-gpu`下85个测试全部通过|
|训练路径|四个label arm均通过真实训练parser的`train(...,dry_run)`，终端P0/P1契约全部为真|
|矩阵dry-run|14行、14个dispatch、14个new train、0复用dispatch、0重导出；GPU行数为2/2/2/2/2/2/1/1|
|数据处理|复用已登记WiSig数据标识，不重新读取全量数据计算hash，不做数据集审计|

## N607登记

尚未同步或启动。Phase1 T1 v5仍在运行，本轮先完成本地实现与发布准备，不干预当前任务。

## 完成后结果表

|rho|seed|labeled数|unlabeled数|source validation数|best epoch|strict UDU|receiver floor|pseudo precision|pseudo coverage|P1-SUP同seed增益|checkpoint|状态|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

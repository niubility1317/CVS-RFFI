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
|状态|`RUNNING / STARTUP_HEALTH_PASS`|

## 设计追踪

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P1L-01|4.1、6.7|标签率覆盖0.005、0.01、0.02、0.05、0.10|factory、spec、runner、本报告|verified|85个定向测试和14行dry-run通过|0.10复用当前T1的`P1-FULL`五seed完整结果|
|P1L-02|4.1|严格使用`f_L=0.70rho`、`f_U=0.70(1-rho)`、`f_V=0.30`|factory、训练P0检查|verified|请求比例精确；source split receipt同时记录实际比例并要求绝对偏差不超过0.002|不要求不同启动批次数据字节一致|
|P1L-03|6.7|每点至少3个筛选seed；0.01低标签关键点使用5个确认seed|spec、sealed plan、runner|verified|14行矩阵及seed计数测试通过|0.005/0.02/0.05各3行，0.01为5行|
|P1L-04|4.1|checkpoint只由source validation选择|factory、训练P0检查、artifact validator|verified|所有label arm终端契约与既有completion validator通过|禁止target指标选择|
|P1L-05|6.7|报告strict UDU、receiver floor、pseudo precision/coverage、对P1-SUP增益、绝对样本数|训练artifact、完成报告|pending|待N607完成产物|同一行指标一起保存|
|P1L-06|9.1、11|正式入口可达、16槽调度、独占输出、完整checkpoint/telemetry/prototype/receipt|runner、测试|verified|最终独立复审P0/P1/P2均为0；execute、reference篡改、sealer hash漂移和输出故障注入均覆盖|14行映射GPU0–5各2行、GPU6–7各1行|
|P1L-07|14.2|最小闭环至少覆盖0.01、0.05、0.10|N607运行与完成表|pending|待实验闭合|本轮同时覆盖完整五点网格|

当前计数：verified=5、pending=2、deferred=0、rejected=0、blocked=0。实现与发布门已闭合，性能结果仍待N607完成。

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

### 最终seal

|字段|值|
|---|---|
|发布提交|`2730f2302d96462b07fa835bb191d76ad86b7788`|
|独立复审|`PASS`；P0=0、P1=0、P2=0；`independent_review_2730f230.json`|
|sealed plan|`runner_release/phase1_label_sealed_2730f230.json`；SHA256=`f11363d588d063916d1cc1006f4d8cb442a90a4ccfd6e30c23e1bb6ec65f5620`|
|release bundle|`runner_release/cvs_full_ablation_phase1_label_2730f230.bundle`；SHA256=`4561347453c2c214dfabc9733e9ee07499c8e020e11016d507571d71cf693a9f`|
|runner handoff|`runner_release/runner_handoff_2730f230.json`|
|远端release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230`|
|远端run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_label_20260729_v1`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v1`|
|环境/CWD|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release的`code`目录|
|GPU策略|容量感知；8张GPU、每卡上限2个训练进程；当前T1占用的槽位只等待、不干预|
|成功条件|14/14新训练行完整闭合；0.10五行引用必须等T1 v5相应`P1-FULL`行完成后才进入五点曲线|

正式启动命令固定为：

```bash
set -o noclobber; : > /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v1.launch.out || exit 91; cd /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230/code && nohup setsid env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230/code:/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230/code/scripts/run_full_ablation_phase1_t1.py --plan /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230.sealed.json --repo-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_label_20260729_v1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v1 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --train-script /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v1_2730f230/code/SSDG/train_ssdg.py --poll-seconds 30 --execute >> /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v1.launch.out 2>&1 < /dev/null & printf '%s\n' "$!" > /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v1.launch.pid
```

### N607发布与启动证据

|字段|证据|
|---|---|
|直连预检|2026-07-29 19:32 CST通过；普通账号、项目根和8张RTX 3090可见|
|启动前占用|T1 v5有15个GPU计算进程；GPU0、1、3、4、5、6、7各2个，GPU2为1个；未干预T1|
|不可变目标|release、bundle、sealed、review、run root、log root、launch log和launch PID在发布前均不存在|
|远端发布物|bundle、sealed plan、review的SHA256分别为`4561347453c2c214dfabc9733e9ee07499c8e020e11016d507571d71cf693a9f`、`f11363d588d063916d1cc1006f4d8cb442a90a4ccfd6e30c23e1bb6ec65f5620`、`af8473dfad842d9632f89dc4ca438257cc4a195f3d576df58d88e8406d9f74e3`|
|release核验|detached HEAD=`2730f2302d96462b07fa835bb191d76ad86b7788`；tracked clean|
|环境与编译|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；环境ID=`CVS-RFFI`；6个发布入口`py_compile`通过|
|远端sealed dry-run|14 rows、14 dispatch、14 new train、14 slots、0 reuse、0 reexport；GPU分布`2/2/2/2/2/2/1/1`；未创建run/log输出|
|启动时间|2026-07-29 19:36:45 CST|
|进程绑定|launch PID=`745606`；实际runner PID=`745607`；CWD=`.../cvs_full_ablation_phase1_label_20260729_v1_2730f230/code`；cmdline精确绑定sealed plan、run root和log root|
|首个训练|PID=`745732`；`P1-LABEL-RHO005__train_seed_7281103`；占用GPU2第二槽；其余13行容量等待|
|SSH清理|启动命令的后台subshell一度保留本地SSH通道；仅终止本地`ssh.exe` PID 12704后只读验证远端已landed，未重复启动；后续`ssh.exe=0`、TCP22 established=0|

### 启动后首轮健康检查

检查时间为2026-07-29 19:41:25 CST，距启动约4分40秒。

|检查项|结果|
|---|---|
|runner|PID`745607`存活，父PID、CWD、release、run root和log root绑定正确|
|调度计数|launched=1、completed=0、succeeded=0、failed=0；13行等待容量|
|首行进度|日志从29158字节增长到63477字节；已推进到`E010/200`|
|输出面|14个独占row目录和14个row日志已创建；当前1个PID文件；completion/terminal/runner summary尚未生成，符合训练未完成状态|
|GPU占用|GPU0–7均为2个计算进程；总显存约6.0–6.4GiB/卡；无GPU超过2个训练进程|
|硬错误|Traceback、RuntimeError、OOM、AssertionError、P0和SystemExit指纹计数均为0|
|结论|`RUNNING / STARTUP_HEALTH_PASS`；禁止因中间性能停机，不重启、不补跑、不修改调度|

`launch.out`当前为0字节，因为runner将训练输出写入各行独占`.out`且自身尚无摘要输出；首行日志持续增长并含完整配置、telemetry和epoch标记，因此不属于日志受损或训练无输出。

### 运行中健康检查：2026-07-29 19:47:47 CST

|检查项|结果|
|---|---|
|直连预检|通过；普通账号、项目根和8张RTX 3090正常可见|
|runner绑定|launch PID=`745606`、runner PID=`745607`均存活；runner CWD、release、sealed plan、run root和log root绑定正确|
|调度计数|launched=1、active=1、waiting=13、completed=0、succeeded=0、failed=0|
|活跃行|`P1-LABEL-RHO005__train_seed_7281103`，PID=`745732`，GPU2|
|健康进度|从上次`E010/200`推进至`E025/200`；row日志从63477字节增长到149694字节|
|GPU容量|GPU0–7各有2个GPU计算进程；本run仅在GPU2占1个槽；总利用率92%–99%，显存约6.0–6.4GiB/卡；无卡超过2个训练进程|
|技术异常|硬错误0、P0=0、非零status=0、异常指纹0、重复异常指纹0|
|完成输出|completion receipt=0、terminal=0、heldout=0、prototype PT/JSON=0、runner summary不存在；首行尚未完成，因此本轮不执行完成artifact闭合验证|
|操作结论|健康规则未触发；只监控，不重启、不调参、不干预T1、不做数据hash或数据集审计|

## 完成后结果表

|rho|seed|labeled数|unlabeled数|source validation数|best epoch|strict UDU|receiver floor|pseudo precision|pseudo coverage|P1-SUP同seed增益|checkpoint|状态|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

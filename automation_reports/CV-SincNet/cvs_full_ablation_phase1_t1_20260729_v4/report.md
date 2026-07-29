# CVS-RFFI Phase1第一层全量消融v4预登记

## 身份与状态

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260729_v4`|
|创建时间|`2026-07-29T17:29:42+08:00`|
|operator|Codex主代理；N607发布仅由唯一runner执行|
|状态|`RELEASE_SEALED / NOT_AUTHORIZED_TO_LAUNCH`|
|设计来源|`CVS-RFFI_全部消融实验设计_Phase1_Phase2_20260728.md`第4.1、5.1、6.1、9.1、9.2、11、12节|
|协议|Phase1 source-only；`0.07/0.63/0.30`|
|Git分支|`codex/full-ablation-20260728`|
|技术修复提交|`a7ec013b500bcdb753b8698dcdc4e79f80bcc4b7`|
|v3证据闭合提交|`c074e764faa3f6e2e682fd5cbaab0ceaedad4198`|
|正式release提交|`34c3e723a426fabed18c0ffc547ce839d628b572`|
|独立审查|`P0=0、P1=0、P2=0 / RELEASE_READY_FOR_SEAL_ONLY`|
|前序run|v1、v2、v3均为系统性技术失败，全部`NO_PERFORMANCE_RESULT`且不可恢复、覆盖或拼接|
|性能结论|无；v4未启动|

v3的启动授权已随v3技术封口消耗。v4是新的不可覆盖run ID，正式启动需要用户对v4的新的明确授权；当前本地验证、审查、报告与封版准备均不构成启动。

## 目标、假设与对照

本run只执行设计报告T1中的Phase1第一层主消融：`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`六个arm，每个arm使用5个paired seed，共30次完整训练。所有arm共享split、label mask、初始化与数据顺序规则；checkpoint只由source validation选择。

v3在`P1-B0/s7281101`完成E200后的source-only endpoint导出阶段发现1条零方向`z_id`。v4只修复该无定义角方向的导出与运行时失败关闭行为，不改变训练方法、arm、seed、split、epoch、checkpoint选择、半径保护或性能评估。

## 设计追踪

|ID|设计要求|实现或artifact|状态|验证|
|---|---|---|---|---|
|T1-P1-01|六个第一层arm|`full_ablation_spec.py`、sealed plan|verified|30-row factory/runner测试|
|T1-P1-02|每arm 5个paired seed|`seed_registry.json`、sealed plan|verified|`7281101–7281105`|
|T1-P1-03|`0.07/0.63/0.30`重新训练|row config、runner validator|verified|plan负测试|
|T1-P1-04|200轮、source-validation-only选择|row config、completion validator|verified|target指标不得参与选择|
|T1-P1-05|`P1-A0`参数量匹配|`phase1_ablation_factory.py`及测试|verified|参数量匹配测试|
|T1-P1-06|单因素diff与协议负测试|聚焦pytest集合|verified|主代理回归123项通过、2项条件跳过|
|T1-P1-07|真实checkpoint与artifact闭合|B0真实smoke|verified|input=25200、directional=25199、prototype与manifest闭合|
|T1-P1-08|独立审查`P0=0、P1=0`|独立review receipt|verified|`P0=0、P1=0、P2=0`|
|T1-P1-09|唯一run ID与16个固定slot|v4 sealed plan、runner|verified|30 rows；GPU0–7各2 slots|
|T1-P1-10|逐row checkpoint、prototype、指标、资源与exit|v4 run/log root|pending|失败row不得静默删除|
|T1-P1-11|30行同row结果与paired统计|本报告完成段|pending|矩阵完整后才分析|

当前追踪计数：verified=9、pending=2、deferred=0、rejected=0。当前范围只对应Phase1 T1。

## v3根因与v4技术修复

|项目|证据或约束|
|---|---|
|v3可靠指纹|`00213641361353e6c0d82e72b2c09aa3b7985e2df414cd62eb2789c3aab8b876`|
|触发checkpoint|SHA256=`be2dd3ca4616a859b6740bbb66774f474eb456a398920a0eb91e272d0e19fe41`|
|根因|25,200条source-validation中仅1条`id_feat_joint/z_id`被末端ReLU映射为精确零；输入、其他中间特征、logits与source-train原型均有限且非零|
|运行时|零方向强制`REJECT_INVALID_FEATURE`，不得被归一化为可接受方向|
|校准|零方向不进入角距离校准，但按总量与每类完整审计|
|硬上限|overall和per-class均不得超过0.001；阈值配置自身不得超过0.01|
|失败关闭|非有限feature/logit、超限、字段缺失、计数不一致或manifest篡改均失败|
|未改变|`radius-to-inter ratio≤0.5`、source-only权限、训练目标、矩阵和checkpoint规则|

## 本地验证

本地环境为`ssr-gpu`。

|验证|结果|
|---|---|
|零方向聚焦测试|18项通过|
|受影响入口与封版链回归|123项通过、2项条件跳过|
|`py_compile`|通过|
|`git diff --check`|通过|
|真实B0 checkpoint smoke|1项通过|

真实smoke固定输入：

- B0 checkpoint：SHA256=`be2dd3ca4616a859b6740bbb66774f474eb456a398920a0eb91e272d0e19fe41`；
- ManySig：SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；
- 结果：source-train prototype计数全部非零；source-validation input=25200、directional=25199；只排除class 3的一条零方向，比例`1/4200`；PT/JSON、endpoint manifest、审计字段与边界哈希闭合。

独立审查绑定完整提交`34c3e723`。6个受影响入口共收集85项，83项通过，2项真实checkpoint门禁按预期跳过；指定B0真实smoke 1项通过。独立复核得到overall排除比例`1/25200`、class 3比例`1/4200`、24个启用组件、最大`r_accept/inter=0.499999`，并确认无target/query新增读取。

## 本地不可变发布证据

|artifact|SHA256或绑定值|
|---|---|
|Git bundle|`242e49dd3464213711a6b87268a856a8eb0971bf709c88d834d4e01f9802640d`|
|未授权plan|`07de2d350f892a62bba244d85156298b65a70eb894fd5ce7f6b4bf88bee366ff`|
|review receipt|`235ebf804589e8871ba47430dc079c396408c032a28af5785152a5041dca4c06`|
|sealed plan文件|`6a3a1ca35351ab063407e63a0a4509e802e2f134a54b30261b3a8a079f8a165f`|
|sealed content|`33b88447cbca0a1589b33ea7955a7c5a679800a4bbd68cf6a51bc81809932b97`|
|精确release|commit=`34c3e723a426fabed18c0ffc547ce839d628b572`；bundle完整历史校验通过|
|矩阵|30 rows；16 slots；GPU0–7各2 slots；10个release file hashes|
|关键导出模块|`code/cvsrffi/phase2_prototypes.py` Git blob SHA256=`cf293066a6daa19f9568ea5e089c902c270136482705c467916ec68fc41faf46`|

## 冻结矩阵与资源

- Phase1 seeds：`7281101–7281105`。
- 六个arm×5个paired seed=30 rows。
- 16个固定slot：GPU0–GPU7，每卡slot0/slot1。
- 每卡本run训练进程最多2个；若存在外部训练进程，则外部与本run合计不得超过2个。
- epochs=`200`；checkpoint selection=`source_validation_only`。
- 任何release、sealed plan、run、log、row output路径碰撞均失败关闭。

## N607预登记

|字段|预登记值|
|---|---|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|WiSig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|
|WiSig SHA256|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|
|release checkout|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723`|
|sealed plan|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723.sealed.json`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v4`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v4`|
|远端环境|必须为`CVS-RFFI`，记录Python、torch、CUDA和8卡可见性receipt|

正式发布前，唯一runner必须执行直连N607只读preflight，记录GPU与现有训练进程占用，验证v1–v3只读保留、v4精确目标路径不存在，核对WiSig、Git bundle、tracked-clean checkout、release file hashes、review receipt和30-row sealed plan。用户授权之前不得SCP、创建远端目录或启动。

## 正式命令模板

唯一runner只能执行以下精确命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723/code && nohup setsid env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723/code/scripts/run_full_ablation_phase1_t1.py --plan /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723.sealed.json --repo-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v4 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v4 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --train-script /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v4_34c3e723/code/SSDG/train_ssdg.py --poll-seconds 30 --execute > /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v4.launch.out 2>&1 < /dev/null &
```

## 健康停止与成功标准

- 只因P0协议/安全违规、launcher级确定性故障、输出覆盖风险、缺失artifact闭环，或至少两个不同row在prototype前出现同一归一化异常指纹而停止。
- 不因accuracy、H、BA、floor或其他中间性能值停止。
- 停止前必须证明main/child PID、CWD、cmdline、run root归属；仅终止已证明属于本run的精确进程树。
- 技术停止保留全部部分artifact并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不得恢复、覆盖或改名。
- 成功要求30/30 rows完成，逐row checkpoint、prototype、terminal/completion、资源和同row指标hash闭合，才进入`ARTIFACTS_COMPLETE`与`ANALYZED`。

## 完成后检查

完成后在本报告追加30行同row结果表，至少包含candidate、机制、split、seed、source-only身份指标、receiver floor、min-class、角几何、伪标签、LEO stress、峰值VRAM、训练时间、checkpoint、prototype、exit状态与最终判定。paired差值、置信区间和任何边际极值必须绑定完整row，不拼接不同run的最佳指标。

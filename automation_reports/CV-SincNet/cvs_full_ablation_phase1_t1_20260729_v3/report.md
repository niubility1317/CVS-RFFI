# CVS-RFFI Phase1第一层全量消融v3预登记

## 身份与状态

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260729_v3`|
|日期|2026-07-29|
|operator|Codex主代理；N607发布仅由`/root/phase1_t1_n607_runner`执行|
|状态|`RUNNING / LANDED / 16_WORKERS_HEALTHY`|
|设计来源|`CVS-RFFI_全部消融实验设计_Phase1_Phase2_20260728.md`第4.1、5.1、6.1、9.1、9.2、11、12节|
|协议|Phase1 source-only；`0.07/0.63/0.30`|
|Git分支|`codex/full-ablation-20260728`|
|代码审查基线|`5d6df9d609e8e9fe160ce87ce7fceef628163b38`|
|release提交|`a2c292481160a0707805b06401c776361053bd5a`|
|独立审查|`P0=0、P1=0、P2=0 / RELEASE_READY`|
|前序run|v1与v2均为系统性技术失败、`NO_PERFORMANCE_RESULT`、不可恢复或覆盖|
|性能结论|无；v3尚未启动|

用户于`2026-07-29T12:55:15+08:00`明确授权“启动v3”。该授权适用于本报告绑定的唯一run ID，不授权修改方法、矩阵、seed、split或干预其他服务器任务。

## 目标、假设与对照

本run只执行设计报告T1中的Phase1第一层主消融：`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`六个arm，每个arm使用5个paired seed，共30次完整训练。所有arm共享split、label mask、初始化与数据顺序规则；checkpoint仅由source validation选择。

v2在首个完成行的source-only prototype导出阶段发现内部安全契约冲突，未产生任何合法prototype或性能结果。v3只修复导出半径与既有`ratio≤0.5`安全校验之间的闭合关系，并将关键导出模块加入release哈希清单；不改变训练方法、矩阵、seed、split、epoch或checkpoint规则。

## 设计追踪

|ID|来源章节|要求|目标文件或artifact|状态|验证|备注|
|---|---|---|---|---|---|---|
|T1-P1-01|6.1、9.2|六个第一层arm|`full_ablation_spec.py`、sealed plan|verified|30-row factory/runner测试|严格对应设计表|
|T1-P1-02|4.1、6.1|每arm 5个paired seed|`seed_registry.json`、sealed plan|verified|seed registry与plan一致性测试|`7281101–7281105`|
|T1-P1-03|3.1、4.1|当前`0.07/0.63/0.30`重新训练|row config、runner validator|verified|配置diff与plan验证|不复用历史checkpoint|
|T1-P1-04|3.1、4.1|200轮、source-validation-only选择|row config、completion validator|verified|runner测试|target指标不得参与选择|
|T1-P1-05|6.1|`P1-A0`参数量匹配|`phase1_ablation_factory.py`及测试|verified|参数量匹配测试|不是简单删分支|
|T1-P1-06|9.1|单因素diff与协议负测试|聚焦pytest集合|verified|主代理测试及独立63项安全负测试|T0不构成性能证据|
|T1-P1-07|9.1、11|真实checkpoint、artifact与hash闭合|`phase2_prototypes.py`、真实smoke|verified|v2触发checkpoint source-only导出通过|checkpoint与ManySig SHA固定|
|T1-P1-08|9.1、13|独立审查`P0=0、P1=0`并Git提交|独立review receipt|verified|`P0=0、P1=0、P2=0`|代码基线`5d6df9d6`|
|T1-P1-09|1、9.2|唯一run ID与16个固定slot|v3 sealed plan、runner|verified|30 rows、16 slots、GPU0–7各2 slots|远端落地仍待授权|
|T1-P1-10|11|逐row保存checkpoint、prototype、指标、资源与exit证据|v3 run/log root|pending|待正式运行|失败row不得静默删除|
|T1-P1-11|5.1、5.3|同row指标、paired统计与资源汇总|本报告完成段|pending|待30行完整artifact|不得拼接单项极值|

当前追踪计数：verified=9、pending=2、deferred=0、rejected=0、blocked=0。当前范围对Phase1 T1为严格设计对应，不代表设计报告其余T2–T4或Phase2已完成。

## v2根因与v3修复

|项目|证据|
|---|---|
|v2可靠指纹|`cca18df5dd028d2f112cafce436620c3a3e29ee1402a8a35b9bc9658dcb83a1e`|
|v2错误|`endpoint_accept_v1 component radius-to-inter ratio unsafe: class=0 component=1 ratio=0.863507`|
|根因|source-val原始95%接受半径可超过最近异类组件距离的50%，与未放宽的verifier契约冲突|
|修复|`r_accept=min(raw_r_accept,(0.5-1e-6)×nearest_other_component_deg)`；`r_core≤r_accept`；原始分位数保留审计|
|权限|仅使用source train组件中心和source validation校准，不读取target/query|
|安全门|verifier继续硬拒绝`ratio>0.5`，没有放宽|
|加载兼容|PyTorch weights-only安全allowlist支持`SatViewStage`与NumPy RNG state；未知global继续拒绝|
|release完整性|`code/cvsrffi/phase2_prototypes.py`进入sealed release file hashes|

## 本地验证

本地环境为`ssr-gpu`。主代理完成聚焦测试、真实检查点smoke、`py_compile`和`git diff --check`；独立审查者完成28项发布测试、1项真实检查点smoke及63项安全负测试。

真实smoke输入：

- checkpoint：`best_source_validation_ssdg.pth`，SHA256=`41dea67c8cdc17a01b0bb8d1b198f703ed52ef2320f471832f6cfeff3b77d5aa`；
- ManySig：SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；
- 结果：source-only prototype PT/JSON生成、至少一个组件被安全收缩、endpoint manifest闭合、无target/query读取。

独立审查仅剩非阻断弃用告警：pin-memory API与旧AMP API。

## 本地不可变发布证据

|artifact|SHA256或绑定值|
|---|---|
|Git bundle|`9e37f1544f901ab3da50d10402235293b51acff7832c11896d41ff195d0e3e53`|
|未授权plan|`f0992481e1499b701b7f77197ba7f2916786b84fa6a7a39d02e2b5879937d1b7`|
|review receipt|`2960fafd17029725d934a72da9cbccaa11d07c76120fd02353ef46128b1860c7`|
|sealed plan文件|`0cf7fabf89595949feffcfe0c9860ecd4739a2015b9dd7a0fc3161b7cd5b0d44`|
|sealed content|`4b62658f070dd231b5c412e8c0da4ae55ff1490b5d2e45057efd5916073cb7a9`|
|精确release|commit=`a2c292481160a0707805b06401c776361053bd5a`；bundle完整历史校验通过|
|矩阵|30 rows；16 slots；GPU0–7各2 slots；10个release file hashes|
|关键导出模块|`code/cvsrffi/phase2_prototypes.py` Git blob SHA256=`6dc5c9066bdc0c4f221036516a8d470c27089b258b51e8891c12cb7607f8a07d`|

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
|release checkout|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248`|
|sealed plan|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248.sealed.json`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v3`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v3`|
|远端环境|必须为`CVS-RFFI`，并记录Python、torch、CUDA和8卡可见性receipt|

正式发布前，唯一runner必须执行本地N607直连只读preflight，记录GPU与现有训练进程占用，验证v1/v2只读保留、v3目标路径全部不存在，核对WiSig SHA、Git bundle、tracked-clean checkout、release file hashes、review receipt和30-row sealed plan。

## N607只读预检

唯一runner于`2026-07-29T05:06:41+08:00`完成只读预检，结论为`PRECHECK_PASS`。未执行SCP、mkdir、写文件、同步、落地、启动、kill或其他远端修改。

|项目|实时证据|
|---|---|
|主机|`szu2070436088@dell-DSS8440`；项目根存在|
|环境|`CVS-RFFI`；Python=`3.10.19`；torch=`2.1.0+cu121`；CUDA=`12.1`；device_count=8|
|GPU|GPU0–7均为RTX3090、0%利用率、1/24576MiB、compute PID=0|
|容量|每卡可新增2个训练slot；相关训练进程总数=0|
|WiSig|2,359,341,461字节；SHA256与预登记完全一致|
|旧run|v1/v2相关进程均为0；核心release/sealed/run/log证据保留|
|v3碰撞|release、sealed、run、log、launch.out五个精确目标均不存在|
|本地release|commit、bundle、sealed、review及30 rows/16 slots证据全部匹配|
|连接闭合|最终`ssh.exe=0`；N607与bridge的ESTABLISHED TCP22=0|

## 正式命令模板

唯一runner于`2026-07-29T12:59:41+08:00`执行以下正式命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248/code && nohup setsid env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248/code/scripts/run_full_ablation_phase1_t1.py --plan /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248.sealed.json --repo-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v3 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v3 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --train-script /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v3_a2c29248/code/SSDG/train_ssdg.py --poll-seconds 30 --execute > /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v3.launch.out 2>&1 < /dev/null &
```

## 启动与即时健康证据

|项目|证据|
|---|---|
|main|PID/PGID/SID=`545770`；PPID=`1`；CWD与cmdline精确绑定v3 release/run/log|
|workers|16个main直接子进程、16个不同row；逐PID row/CUDA/CWD/cmdline校验`BINDING_BAD=0`|
|环境receipt|`CVS-RFFI`；CUDA available；device_count=8；WiSig与sealed content匹配|
|即时计数|launched=16、completed=0、succeeded=0、failed=0|
|artifact|checkpoint=16、prototype=0、terminal=0、completion=0|
|异常|真实异常文件=0；runner summary尚未生成；无停机条件|
|GPU|GPU0–7每卡恰好2个run-owned compute PID；external=0；利用率78%–97%；显存3.18–6.48GiB/卡|
|连接闭合|最终`ssh.exe=0`、ESTABLISHED TCP22=0|

|GPU|slot0 row/PID|slot1 row/PID|
|---:|---|---|
|0|`P1-FULL/s7281101/545956`|`P1-FULL/s7281102/545996`|
|1|`P1-FULL/s7281103/545959`|`P1-FULL/s7281104/545997`|
|2|`P1-FULL/s7281105/545976`|`P1-SUP/s7281101/546318`|
|3|`P1-SUP/s7281102/546124`|`P1-SUP/s7281103/545964`|
|4|`P1-SUP/s7281104/545962`|`P1-SUP/s7281105/546125`|
|5|`P1-A0/s7281101/545971`|`P1-A0/s7281102/546191`|
|6|`P1-A0/s7281103/545968`|`P1-A0/s7281104/546126`|
|7|`P1-A0/s7281105/545974`|`P1-B0/s7281101/546127`|

## 第一worker wave技术证据

唯一runner于`2026-07-29T14:42:40+08:00`完成首个终态与第一worker wave复核。实际首波为5个`P1-SUP` paired seed，全部技术成功：

|arm|seed|return code|terminal|prototype export|completion receipt|P0|
|---|---:|---:|---|---|---|---|
|`P1-SUP`|7281101|0|`COMPLETE`|`COMPLETE`|valid|false|
|`P1-SUP`|7281102|0|`COMPLETE`|`COMPLETE`|valid|false|
|`P1-SUP`|7281103|0|`COMPLETE`|`COMPLETE`|valid|false|
|`P1-SUP`|7281104|0|`COMPLETE`|`COMPLETE`|valid|false|
|`P1-SUP`|7281105|0|`COMPLETE`|`COMPLETE`|valid|false|

每行2个prototype hash均非空，completion中的terminal SHA与实文件一致，全部completion checks为true。该证据证明执行与artifact闭环，不代表性能结论。

|项目|首波后证据|
|---|---|
|矩阵计数|launched=21、completed=5、succeeded=5、failed=0、active=16|
|artifact计数|prototype files=10、terminal=5、completion=5|
|异常|归一化异常指纹=0；无P0；未触发停止规则|
|续排|释放的5个slot按冻结顺序启动`P1-C0`四行和`P1-D0`一行|
|绑定|16个active worker全部通过row/CUDA/CWD/cmdline绑定，`BAD=0`|
|GPU|GPU0–7每卡仍恰好2个run-owned PID；external=0；利用率92%–99%；显存4.19–6.41GiB|
|连接闭合|最终`ssh.exe=0`、ESTABLISHED TCP22=0|

|GPU/slot|新续排行|PID|
|---|---|---:|
|GPU2/slot1|`P1-C0/s7281102`|598051|
|GPU3/slot0|`P1-C0/s7281103`|598553|
|GPU3/slot1|`P1-C0/s7281104`|598158|
|GPU4/slot0|`P1-C0/s7281105`|597240|
|GPU4/slot1|`P1-D0/s7281101`|597235|

当前日志仍在增长。本节仅报告截至该时间点完整封口的5行和活跃进程状态；其余行需在完成后读取完整日志与artifact，不能据此作收敛或性能判断。

### 首波完整日志审计与15:40快照

`2026-07-29T15:38:00+08:00`，runner完整读取5个`P1-SUP`日志，每行9020行、约1.12MB。Traceback、OOM、Killed、Runtime error、Assertion、协议错误和Inf计数均为0。每行发现2000个NaN，仅分布于`LOSS-SAT-RAW/DM-ACCEPT/TRAIN/TEST/JOINT-METRIC`五类训练期禁用指标占位字段，各类每epoch一条、共200条；terminal最后非空行为`COMPLETE`，因此不构成执行异常。checkpoint、prototype、terminal、completion、resource、heldout及其hash全部闭合，`ALL_ARTIFACTS_OK=true`。

`2026-07-29T15:40:16+08:00`活跃日志完整解析至：

|arm|最新epoch范围|
|---|---|
|`P1-A0`|E165–172|
|`P1-FULL`|E131–135|
|`P1-B0`|E137|
|`P1-C0`|E65–66|
|`P1-D0`|E70|

该时点仍为launched=21、completed=5、succeeded=5、failed=0、active=16；active hard error=0。16个worker绑定未漂移，GPU0–7每卡2个run-owned PID、无外部PID，利用率92%–99%，显存4.20–6.44GiB；最终SSH/TCP22均为0。

## 健康停止与成功标准

- 只因P0协议/安全违规、launcher级确定性故障、输出覆盖风险、缺失prediction闭环，或至少两个不同row在prediction前出现同一归一化异常指纹而停止。
- 不因accuracy、H、BA、floor或其他中间性能值停止。
- 停止前必须证明main/child PID、CWD、cmdline、run root归属；先温和终止，仅对仍存活且已绑定本run的PID升级。
- 技术停止保留全部部分artifact并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不得恢复、覆盖或改名。
- 成功要求30/30 rows完成，逐row checkpoint、prototype、terminal/completion、资源和同row指标hash闭合；随后才进入`ARTIFACTS_COMPLETE`与`ANALYZED`。

## 完成后检查

完成后在本报告追加30行同row结果表，至少包含candidate、机制、split、seed、source-only身份指标、receiver floor、min-class、角几何、伪标签、LEO stress、峰值VRAM、训练时间、checkpoint、prototype、exit状态与最终判定。paired差值、置信区间和任何边际极值必须绑定完整row，不拼接不同run的最佳指标。

当前最高风险是服务器重新运行仍可能发现第二个此前被v2首个P0遮蔽的技术闭合问题；预登记停止规则会在不读取性能的前提下失败关闭。

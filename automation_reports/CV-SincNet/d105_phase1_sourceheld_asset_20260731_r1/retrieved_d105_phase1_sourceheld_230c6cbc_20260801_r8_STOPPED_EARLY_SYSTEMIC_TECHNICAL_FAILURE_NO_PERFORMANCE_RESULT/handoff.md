# D105 Phase1 R8 N607终态交接

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / ARTIFACTS_INCOMPLETE / SOURCE_HELD_GATE_REJECT_OBSERVED / NO_PERFORMANCE_RESULT / NO_TARGET_PERFORMANCE_RESULT`

运行ID：`d105_phase1_sourceheld_230c6cbc_20260801_r8`

唯一runner：`gpt-5.6-terra max`

fresh-run retry：`NO`

Target25/authority/formal seal：未执行

## 冻结身份

|字段|值|
|---|---|
|source commit|`230c6cbc9149250ca0303ca240945d0e0992360e`|
|实现证据commit|`a4e14e83235dc33c8287500cb0540234d6201ea6`|
|R8预登记commit|`582aa634bd7943eaa567242b2ac6c133527e2356`|
|预登记绑定HEAD|`a7f3cf1b742fb815981ff9e0db5875830b7645ec`|
|source archive|`source_230c6cbc.tar`；243005440B；`16d57519cfa15d9929a38282217b0a2e2908e5c92e8b42672dae1537386855c7`|
|runtime/method|`5de5926bbb2e9fd78b2f3315ec6e109964ddd6216ebe4f75e428b6b9f6bf11bc`/`7345f81e88588c46ad453eb315786306f28291478a5eaddce618ef7ee6998ecd`|
|launcher|`run_d105_phase1_stage1_230c6cbc.sh`；`db0757789fa4b3a4155e793c28a2d7c76926248b59cd51c3758cf93364a3cdc9`；5624B；LF123/CRLF0|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_230c6cbc_20260801_r8`|

## 预启动证据

- direct普通N607账户preflight通过；R8 root初始ABSENT；8张RTX3090均0%/1MiB且无compute process；磁盘余量7.4T。
- 4个冻结输入的SHA/大小精确；launcher的LF123/CRLF0与`bash -n`通过。
- archive共4775成员：4206个文件、569个目录；绝对路径、反斜线、`..`逃逸、错根、重复、软链、硬链和特殊成员均为0。
- 54/54 manifest=archive=extract、LF54/54、隔离pyc54/54、source pycache/pyc=0；与本地独立handoff的54/54 Git blob attestation合并闭合四方身份。
- cache/salt/checkpoint/reference及两个preflight helper的SHA均通过；9/9 CLI帮助面返回0。
- production bridge fixed256真实checkpoint smoke通过：torch=`2.1.0+cu121`；loader=`legacy_pickle_exact_frozen_sha_only`且checkpoint精确SHA受约束；195个state tensor；eval=true；state digest前后相同；GRB before/after=false。
- 1/208/256行均固定`[256,2,256]`前向一次，末批实/补零为1/255、208/48、256/0；pre_relu/z_dom/z_id均float32、finite，eager hook、strict pre-ReLU路径和ReLU parity通过，三路reference max_abs均为0。Target/query/performance均为false。

主要preflight SHA256：

|证据|SHA256|
|---|---|
|archive/runtime integrity|`806aafefa4f9641c45c58ae93769f55344d0ab597076a33778fe8ccefc094ae9`|
|9/9 help receipt|`d6a5fc23f26df78ae9cd30192468ba502d33e84d2077628d0c0f9510b78e6c7e`|
|production bridge smoke|`d9537e878783e728875fc5342020976cc5aaf1dba569913b0b8de436997dea72`|
|loader branch|`a5847cad918974b45bf8f807a1ee7dd3b745da591ec991bc020a0d2e2238fc0e`|
|shape/dtype/finite|`33ae7d248ff5016cc4e994c71ed83a21fa3378d81f767dc6e11b0cefd2390207`|

## 唯一detach与流水线终态

main PID=`2894396`。launch receipt绑定实际CWD、cmdline、run root、`CUDA_VISIBLE_DEVICES=0`、PYTHONPATH、launcher/runtime/method SHA与日志路径。唯一detach后没有第二次启动、远端修补、重试、调参、阈值修改或方法修改。

|阶段|数量|状态|
|---|---:|---|
|strict tap|1|8400行、33次forward，末批208实+48补零，reference parity通过|
|prediction|1|21 receiver-held+42 class-LOCO结构；不作性能结论|
|truth-open|1|prediction后开启；truth标签未持久化|
|score|1|63个scored结构+7个TX probe结构；不作性能结论|
|derived gate|1|`SOURCE_HELD_GATE_REJECT_OBSERVED`|
|component|0|diagnostic序列化生命周期失败|
|authority/formal seal/Target25|0/0/0|未执行|

gate中的`tx_probe_gate_pass`是原生`bool=false`。formal missing token为：

- `receiver_held_all_noninferior`
- `class_loco_complete_and_noninferior`
- `tx_probe_max_balanced_accuracy_at_most_0_25`

交接不披露任何BA、floor或其它未闭合性能数值。

## 技术失败根因

归一化错误指纹：`D105 aggregate bundle cannot be serialized after quantization closure`。

底层精确原因：`failed dual TX probe bundle is diagnostic-only and cannot be serialized`。

控制流证据：

1.第一轮provisional `build_rxid_metabias4_bundle`已成功，否则错误会在`D105 aggregate bundle compilation failed`处结束，无法进入quantization summary。
2.第二轮build继续使用完全相同的U/B/bank_g/bank_t/precision/sigma、count数组、TX gate字段和receipt绑定；唯一变化是把provisional quantization receipt SHA替换为正式quantization summary SHA。
3.第二轮bundle因此仍满足20个payload成员的dtype、shape、finite、无INT8 -128、正scale、固定scalar和int16计数闭包；随后serializer看到`tx_probe_gate_pass=false`，按D103 ABI拒绝生成wire。
4.该行为与R8预期“合法负向gate也生成不可签名的DIAGNOSTIC component”冲突，导致component为0并使pipeline exit=2。

量化结构未损坏，计划中的20成员为：

|组|编码结构|
|---|---|
|U|INT8 `[32,160]`+FP16 scale `[32]`|
|B|INT8 `[160,4]`+FP16 scale `[160]`|
|bank_g|INT8 `[C,32]`+FP16 scale `[C]`|
|bank_t|INT8 `[C,4]`+FP16 scale `[C]`|
|precision|INT8 `[C,4]`+FP16 log offset/scale scalar|
|sigma|INT8 `[C]`+FP16 log offset/scale scalar|
|fixed state|FP16 temperature、lambda0 `[4]`、amax `[4]`、radius|
|counts|int16 min physical/class count各`[C]`|

没有bundle wire、component manifest或DIAGNOSTIC component落盘。

## 回收与完整性

本地回收目录：`retrieved_d105_phase1_sourceheld_230c6cbc_20260801_r8_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT/artifacts`。

|证据|SHA256|
|---|---|
|launch receipt|`5e10871e6a810e22e298f3b6b4c8dd8af58fc15f8857209e56b9f2a51ccbc350`|
|pipeline log|`aca2458f906d6961080fe7577a75cf063ceb3a1de17f1d241541b8cbec440a57`|
|pipeline exit|`53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3`|
|strict tap archive|`6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578`|
|strict tap receipt|`d8090c50fa8aa2fd2496d3883cb821dc91e87132e21db909fa0890af29a567b2`|
|prediction|`f1e9a4e9c9e009c0ffbc77b819960651f46c0cfde1c19dd04424420b78acb50a`|
|truth-open|`e447d29deab94362b0b4c58d50ac7d8911ec5cd1b12dd07b5b8de201255185f2`|
|score|`ac1c4a021afa8640cbf1612115f8a960c7f3b76ca24b069f9c9fb4d5c15116ac`|
|derived gate|`209629115298f9ac5f537d36d7ae01841673739b101599d33a3cd3a8035b9424`|
|64行retrieval manifest|`568ac89b0d20c10161af45f9ab48019a99c87f78bc684c75f4fbc4b59f66103a`|

64个选定远端文件与本地文件逐项SHA一致。终态main/child均不存在，GPU0为0%/1MiB且无compute process；本地`ssh.exe`/`scp.exe`与N607/bridge TCP22均无残留。

## 后续门

R8永久关闭，不得恢复、覆盖、重试或把gate/score重新解释为性能。下一轮必须先在本地修复diagnostic component持久化与serializer生命周期，补充真实gate-reject端到端回归，完成独立审查和新Git提交，再使用新的不可覆盖run ID。仅当新的Phase1 component、独立审查、离线authority和formal seal全部闭合时，Target25才可重新评估启动资格。

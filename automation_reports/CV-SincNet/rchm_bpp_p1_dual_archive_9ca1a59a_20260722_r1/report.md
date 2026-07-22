# RCHM-BPP Phase1双表征归档发布报告

## 1.预注册状态

- run_id:`rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`
- 预注册时间:`2026-07-22T21:58:13+08:00`
- 主agent:`/root`；唯一实验runner:`/root/rchm_dual_archive_runner`（`gpt-5.6-terra high`，已完成）
- 当前状态:`TECHNICAL_FAILURE`
- 候选:`JOINT-RCHM-BPP/r1f`
- 方法提交:`9ca1a59a7522393c43ee09c7f95dde6588cd8f4a`
- retry授权:`NO`
- 结果语义:`NO_PERFORMANCE_RESULT`

本run只生成冻结候选所需的Phase1同观测双表征证据输入和coverage receipt；不运行held分类，不访问target/query，不运行125，不产生promotion结论。

## 2.假设与即时证伪

假设：既有Phase1 checkpoint能够对同一条已选received IQ在一次dual-runtime调用中同时导出`z_id`、`z_dom`与原始`tx_logits`；每行`observation_id`逐字复制既有v1 source-validation cache的已验证`overlay_id`，不改变received IQ、物理ID、receiver/TX集合、场景、K、support/query划分或`p2_min_v1`。

任一情形立即标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`并停止进入held runner复审：

- 任一冻结输入、代码、receipt或产物SHA不匹配；
- cache不是既有v1谱系或需要修改allowlist/authority；
- 双表征不是同一IQ、同一次runtime调用产生；
- `observation_id`与所选cache行的`overlay_id`不一致；
- NPZ成员、顺序、维度、有限值、行序、唯一性或内部验证失败；
- 实际总行数不是8400，唯一physical/observation不是8400，或类别/receiver/day/scenario计数不是6/7/4/3；
- 168个receiver×day×class cell中出现零coverage，或任一cell不足11行而无法在K10 support后保留至少1条query。

coverage只用于判断冻结held split能否被数据支持，不是模型性能，也不得用于选择target/query结果、阈值、rank或fallback。

## 3.本地冻结与验证

- `E:/type10-7`根目录不是Git仓库；Git承载面为`E:/type10-7/code/snapshots/ground_proto_da_rd_wt`。
- 双归档实现和测试已在`ssr-gpu`完成独立审查，结论`P0=0,P1=0,P2=0→MERGE`。
- 新旧归档、dual exporter和parity verifier共44项测试通过。
- Git归档:`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/source_9ca1a59a.zip`
- Git归档SHA256:`95127701d2c9f9989fcc6409b1e069e232f2d3e3654611d92e9ac2abe26937a0`；大小32990146字节；4433个成员。
- 归档安全检查:`absolute=0,dotdot=0,backslash=0,duplicates=0,symlinks=0`。
- 精确解包树:`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/source_9ca1a59a`；同一44项测试通过。
- run wrapper:`run_pipeline.sh`；SHA256:`eb4e591f875434bb7e7f4c90b6a020435f3d7f356b4e05a33091231438210ffd`；WSL `bash -n`通过。

|发布树文件|SHA256|
|---|---|
|`code/scripts/export_phase1_singleobs_dual_feature_archive.py`|`44ceff9d1afb0c6a1832ef0d09bfb19f24ce1190481387ddf984b3ef7bdc8b4b`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|
|`code/scripts/export_adv3b02_dual_feature_torchscript.py`|`6c637520ca0e5877740a6b9a45dafb7d52ad0d881da4282538e32524c865ba7a`|
|`code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py`|`606d0e27a826f917e4e28171775e2cb0b8f8edfd68b50e7ab5ba554be175d069`|
|`code/cvsrffi/dual_feature_forward.py`|`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`|
|`code/cvsrffi/leo_weak_cache.py`|`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`|

## 4.冻结N607输入

|输入|N607绝对路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|adapter|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/effective8_adapter_fp16.pt`|`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`|
|v1 source-validation cache|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|selection salt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|

- 显式class registry:`14-10,14-7,20-15,20-19,6-15,8-20`。
- `tx_logits`只保留checkpoint原始列索引审计语义，不绑定class ID；后续held runner禁止使用它作分类输入或标签映射。
- adapter仅因既有dual runtime exporter接口要求而导出base/candidate配对runtime；正式归档固定使用base role。

## 5.N607落地与启动约束

- 项目根:`/home/szu2070436088/2510044040/CV-SincNet`
- run根:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source目录:`<run-root>/source_9ca1a59a`
- 日志:`<run-root>/logs/pipeline.log`
- PID:`<run-root>/logs/pipeline.pid`
- 退出码:`<run-root>/logs/pipeline.exit`

唯一runner必须先执行本地只读direct preflight；仅当direct TCP/SSH路径失败且identity/config无歧义时才使用已验证lab bridge。所有SSH/SCP均为短连接，结束后验证本地无遗留`ssh.exe`或到N607/lab的ESTABLISHED连接。

落地严格分为两个阶段：

1. 只读阶段证明精确run根不存在，并记录GPU/process/CWD/cmdline占用；若run根已存在立即停止，不覆盖、不重试。
2. 唯一状态变更阶段先以`umask 077`创建一次run根及空`input/`、`logs/`，随即确认`input/`与`logs/`为空，且`output/`、`pipeline.log`、`pipeline.pid`、`pipeline.exit`均不存在；之后才同步两个冻结输入。任何部分落地均标记技术失败，不复用该run ID。

同步映射：

- `source_9ca1a59a.zip`→`<run-root>/input/source_9ca1a59a.zip`
- `E:/type10-7/automation_reports/CV-SincNet/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/run_pipeline.sh`→`<run-root>/input/run_pipeline.sh`

同步后核验远端ZIP/wrapper SHA，安全解包到唯一`source_9ca1a59a/`，再核验全部冻结源文件SHA和`py_compile`。根据已记录的实时占用分配GPU，任何GPU不得超过两个compute/training进程。

唯一允许的启动模板：

```bash
nohup env CUDA_VISIBLE_DEVICES=<live-assigned-gpu> bash /home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/input/run_pipeline.sh > /home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/logs/pipeline.log 2>&1 < /dev/null &
```

wrapper在通过不可覆盖前置检查后，用自身`BASHPID`写入临时文件并`mv`为`pipeline.pid`；runner必须在首个短连接探针中核验PID、GPU和完整cmdline。PID未原子落地或进程不匹配即为启动失败。runner使用短连接监控，禁止修改方法、调参、访问target/query、kill/restart、干预无关作业或未经授权重试。

## 6.预期artifact

- `output/runtime/base_dual_runtime.pt`与`output/runtime/candidate_dual_runtime.pt`
- `output/runtime/dual_export_receipt.json`
- `output/runtime/base_parity_receipt.json`与`output/runtime/base_parity_vector.json`
- `output/archive/phase1_singleobs_dual_feature_archive.npz`与对应manifest
- `output/coverage_receipt.json`
- `output/sha256sums.txt`
- `logs/pipeline.log`、`logs/pipeline.pid`、`logs/pipeline.exit`

## 7.成功条件与结果边界

wrapper在写coverage receipt、SHA清单和完成marker前硬断言：总行数=8400，唯一physical=8400，唯一observation=8400，class/receiver/day/scenario计数=6/7/4/3，class与scene registry精确匹配，receiver×day×class cell=168、zero cell=0且最小cell>10。技术完成还要求child exit=0、`PIPELINE_ARTIFACTS_COMPLETE` marker存在、export/parity receipts均为PASS且archive内部verify通过。即使全部满足，本run仍只能标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；它不能证明JOINT-RCHM-BPP有效或1+1>2。

## 8.完成后回填

- runner handoff:`/root/rchm_dual_archive_runner`；唯一启动次数=`1`；retry=`NO`；方法commit=`9ca1a59a7522393c43ee09c7f95dde6588cd8f4a`；release-control commit=`7bec37f4aa7f0e99cc46b30a3c1be7e464e97848`。
- preflight/route:2026-07-22T22:28:54+08:00本地direct只读preflight通过：`N607`直连身份、项目根、服务器时间与8张GPU可见；未使用bridge。每次SSH/SCP后均核验本机无`ssh.exe`残留、无到`172.31.111.215:22`或`172.31.105.18:22`的ESTABLISHED连接。
- GPU/PID/process:只读落地门确认run根不存在、8张GPU无compute进程；GPU0分配前仍无compute进程。精确run根:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`。按冻结模板启动一次，launcher PID=`316120`，原子`pipeline.pid=316120`已落地；后续自然退出，无残留匹配进程或GPU compute进程。
- 远端SHA/compile:本地与远端ZIP SHA256均为`95127701d2c9f9989fcc6409b1e069e232f2d3e3654611d92e9ac2abe26937a0`；本地与远端wrapper SHA256均为`eb4e591f875434bb7e7f4c90b6a020435f3d7f356b4e05a33091231438210ffd`。安全解包后6个冻结源文件SHA均匹配，远端`py_compile`与`bash -n`通过；checkpoint、adapter、v1 cache、selection salt SHA均与第4节匹配。
- child exit/marker:child exit=`1`；`PIPELINE_ARTIFACTS_COMPLETE`不存在。dual export receipt=`PASS`，但base runtime/checkpoint parity在256行超过冻结`1e-5`容差：`z_id=0.0001963973045349121`、`z_dom=0.0005043148994445801`、`tx_logits=0.0032591819763183594`，因此在archive前硬停止。
- artifact清单与本地回收路径:已回收至`E:/type10-7/automation_reports/CV-SincNet/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/retrieved/`：`logs/pipeline.log` SHA256=`547ed302716f021c5c9880fbb26144b0a97778f377be7fd6d75dceaafd0bc026`；`logs/pipeline.pid` SHA256=`4044e5344876db3dd589f50cdf2ace132f91a5d3a054a6d1385eb9092066efab`；`logs/pipeline.exit` SHA256=`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`；`output/runtime/base_dual_runtime.pt` SHA256=`b13f78b2da617279603e4b29a96b84baa6dd7361c245acaa4daf403e8744b364`；`output/runtime/candidate_dual_runtime.pt` SHA256=`e0f6706275e1af6f77a48248c97662170ee71aff86ff19176e9a0a0dbe9f4ed0`；`output/runtime/dual_export_receipt.json` SHA256=`a25d678fedc715a2b72e6e99fead549af3f438974f717d9efe4b7078e4bbf1d7`。
- coverage同row结果:未生成。`coverage_receipt.json`、archive NPZ/manifest、`base_parity_receipt.json`/vector、`sha256sums.txt`均因parity硬门失败而不存在；因此总行、unique physical/observation、classes、receivers、days、scenes、168cell、zero/min/max与K1/K5/K10 remaining均为`NOT_GENERATED`，不得推断或补填。
- 异常:冻结wrapper的parity硬门失败；未访问target/query、未运行held/125、未调参、未kill/restart、未重试。此run根已含不可变输入和部分output，禁止复用；若要修复或再次执行，必须经授权创建新run ID并重新预注册。
- 最终状态:`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 下一步:主agent已完成冻结parity差异的本地诊断；本run不进入held runner可行性监督，也不产生任何算法性能或promotion结论。

## 9.发布审查闭环

- 首轮独立Terra发布审查:`P0=1,P1=4,P2=0→REVISE`；阻断项为coverage未硬失败、4个文件路径漂移、PID未原子落地、run目录步骤不唯一和`set -u`下GPU变量错误路径。
- 最小修订:在完成marker前硬断言全部预注册coverage常量；校正发布树路径；wrapper以`BASHPID`临时文件加`mv`原子写PID；冻结两阶段目录顺序；使用带默认空值的`CUDA_VISIBLE_DEVICES`展开。
- 本地验证:root/Git镜像逐字节一致，`bash -n`通过，embedded Python compile通过；synthetic coverage正例1个、row-count/zero-cell/K10-min负例3个均按预期；unset CUDA负例退出码70。
- 复审裁决:`P0=0,P1=0,P2=0→MERGE`。该MERGE只授权按本报告发布Phase1归档run，不是算法性能或promotion认证。

## 10.技术失败诊断与后续边界

- 失败形状:独立verifier按batch 1/8/256依次调用时，maxabs从全0增长到`z_id=1.9640e-4`、`z_dom=5.0431e-4`、`tx_logits=3.2592e-3`；它在`1e-5`硬门处停止，未写parity receipt/vector、archive或coverage。
- 根因诊断:SHA绑定base runtime在本地CUDA相同256行fresh第1→2次出现`5.90e-6/1.96e-5/1.47e-4`漂移，第2→3次全0；首图含843个`prim::profile`节点，热图为0；CPU全0；仅关闭graph executor optimization后CUDA连续三次全0。现有证据支持TorchScript CUDA冷/热执行计划切换，不支持checkpoint、dtype、device或dual-feature语义错位。
- 结果边界:现有artifact没有逐行checkpoint/runtime parity vector，不能推断top1、距离排序或真实语义等价；本run永久保持`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 技术revision:`P1-DUAL-ARCHIVE-GEOFF/r2`经独立监督`MERGE`并进入`DESIGN_FROZEN`。唯一delta是在export、verify和archive consumer首次JIT边界前fail-closed封存并回读`graph_executor_optimize=false`，升级v2 execution contract并绑定精确Torch/CUDA版本和contract SHA。
- 门限不变:三输出在fresh batch1/8/256的checkpoint-vs-runtime及runtime第1/2/3次均须`maxabs≤1e-5`；不得warm-up取热结果、放宽容差、恢复优化路径或接受旧v1 receipt。
- 后续:只允许本地实现、测试、独立review和新Git提交；若再发布，必须使用全新不可覆盖run ID。本run不进入held复审。

## 11.GEOFF/r2本地实现闭环与旧解包树边界

- 实现范围:正式Git工作树中的dual runtime export、独立checkpoint parity、dual archive consumer及三份对应测试；schema升级到v2，封存并严格回读`graph_executor_optimize=false`，绑定Torch/CUDA/device、`max_abs=1e-5`和canonical contract SHA；fresh batch1/8/256均执行runtime第1/2/3次，不warm-up、不放宽门限。
- 本地验证:`ssr-gpu`下`py_compile`通过，35项GEOFF专项及相邻dual-forward/joint core合计`48 passed`、exit0；r1回收candidate runtime的无数据CUDA探针在batch1/8/256上第1↔2、第1↔3三输出最大差均为0。弃用/trace警告及pytest临时目录清理权限提示不改变exit0。
- 独立审查:首轮`P0=0,P1=1,P2=0→REVISE`，唯一P1是receipt实际调用3次却同时声明1次和3次；verifier、archive consumer、fixture及正式断言经4行修复统一为3，复审`P0=0,P1=0,P2=0→MERGE`。该MERGE只授权新Git提交和后续新run预注册，不改变本run的技术失败状态。
- 本地源边界:旧解包树`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/source_9ca1a59a/`中的6个GEOFF源码/测试文件已与不可变ZIP成员SHA不同，标记为`CONTAMINATED_LOCAL_EXTRACTION / DO_NOT_RELEASE`。原ZIP SHA256仍为`95127701d2c9f9989fcc6409b1e069e232f2d3e3654611d92e9ac2abe26937a0`，远端r1源未变；后续只能从新Git提交生成新archive并使用新run ID。
- 时间:`2026-07-22T23:28:56+08:00`；prediction、archive、coverage仍为0/未生成，本run永久保持`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

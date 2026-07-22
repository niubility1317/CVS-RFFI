# RCHM-BPP Phase1双表征归档发布报告

## 1.预注册状态

- run_id:`rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`
- 预注册时间:`2026-07-22T21:58:13+08:00`
- 主agent:`/root`；唯一实验runner:`gpt-5.6-terra high`，待委派
- 当前状态:`LOCAL_VERIFIED`
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

- runner handoff:待回填
- preflight/route:待回填
- GPU/PID/process:待回填
- 远端SHA/compile:待回填
- child exit/marker:待回填
- artifact清单与本地回收路径:待回填
- coverage同row结果:待回填
- 异常:待回填
- 最终状态:`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
- 下一步:仅在全部技术与coverage条件满足后，重新提交held runner可行性监督。

## 9.发布审查闭环

- 首轮独立Terra发布审查:`P0=1,P1=4,P2=0→REVISE`；阻断项为coverage未硬失败、4个文件路径漂移、PID未原子落地、run目录步骤不唯一和`set -u`下GPU变量错误路径。
- 最小修订:在完成marker前硬断言全部预注册coverage常量；校正发布树路径；wrapper以`BASHPID`临时文件加`mv`原子写PID；冻结两阶段目录顺序；使用带默认空值的`CUDA_VISIBLE_DEVICES`展开。
- 本地验证:root/Git镜像逐字节一致，`bash -n`通过，embedded Python compile通过；synthetic coverage正例1个、row-count/zero-cell/K10-min负例3个均按预期；unset CUDA负例退出码70。
- 复审裁决:`P0=0,P1=0,P2=0→MERGE`。该MERGE只授权按本报告发布Phase1归档run，不是算法性能或promotion认证。

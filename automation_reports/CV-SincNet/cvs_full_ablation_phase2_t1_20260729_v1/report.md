# CVS-RFFI Phase2全消融T1发布报告

## 基本信息

|字段|值|
|---|---|
|experiment ID|`cvs_full_ablation_phase2_t1_20260729_v1`|
|时间|2026-07-29|
|operator|Codex主代理；独立复审员`phase1_t1_v4_independent_review`|
|当前状态|`LOCAL_VERIFIED / INDEPENDENT_IMPLEMENTATION_REVIEW_P0_0_P1_0 / WAITING_PHASE1_INPUTS`|
|目标|完成Stage2-A、Stage2-B及Stage2-C T1筛选，复用既有合法输入和完整预测，不重复数据集审计|
|协议|`p2_min_v1`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|

## 假设与比较目标

Stage2-A验证冻结Phase1 bundle的零标签跨接收机能力；Stage2-B比较`P2-S2B-PROTO/P2-S2B-DIAGOFF/P2-S2B-FULL`的旧类适配；Stage2-C比较7个同权限基线、`P2-FULL`及A/B/C/D/E/F核心消融。所有预测均逐样本在全部已注册类上argmax，fit侧不接收query truth。

`P2-F3`与`P2-FULL`共享同一物理执行，分别生成logical score记录，`P2-F3`不计独立观测。前批次完整预测允许通过`reuse_prediction`复用；不同启动批次不要求绑定相同数据缓存，但每个logical row的预测与truth-side评分必须绑定同一封存包。

## 冻结矩阵

|部分|logical row|physical execution|说明|
|---|---:|---:|---|
|Stage2-A|25|25|5receiver×5confirmation method/query seed；零support|
|Stage2-B|300|300|3arm×5receiver×4K×5seed|
|Stage2-C screening|1425|1350|19logical arm×75row/arm；`P2-F3`复用`P2-FULL`|
|T1合计|1750|1675|每row内含3个LEO场景|

Stage2-C中的75row/arm来自`5receiver×5个预登记(K,Cn)slice×3development seed×1class draw`，不是全矩阵合计75row。

## 本地实现与验证

|文件或模块|作用|状态|
|---|---|---|
|`stage2_ablation_executors.py`|23臂真实support-only数值执行；真实低秩adapter基线|已实现|
|`stage2_ablation_quantization.py`|F0/F1/F2/F3编译、解码、误差和资源|已实现|
|`stage2_ablation_feature_builder.py`|从封存包一次提取288维特征，不打开truth/raw dataset|已实现|
|`stage2_ablation_feature_cache.py`|不可覆盖、truth-free、跨arm复用缓存|已实现|
|`stage2_ablation_row_executor.py`|单个physical row预测、behavior/quant/resource receipt|已实现|
|`stage2_ablation_release.py`|缓存绑定、既有预测复用、物理别名去重、冻结计划|已实现|
|`seal_full_ablation_stage2_plan.py`|生成不可覆盖predict/score请求|已实现|
|`run_full_ablation_stage2.py`|8GPU×2槽调度、外部占用等待、止损、terminal/summary|已实现|
|`score_full_ablation_stage2_row.py`|预测封存后才打开truth-side评分|已实现|

独立复审在`ssr-gpu`环境完成关键模块编译和18文件跨链回归：244项通过、2项因需要真实大型checkpoint而跳过、0项失败。实现内容P0=0、P1=0；审查时唯一发布P0是工作树尚未Git封存，本报告所在发布提交闭合后归零，无需修改算法或重审数据。当前Phase2正式执行仍等待运行中的Phase1 `P1-FULL`完整部署输入，不以跨批次数据一致性为前置条件。

## 服务器发布位置与命令

预留位置如下，最终Git commit和实际文件映射将在独立复审通过后写回，不覆盖既有目录：

|用途|路径|
|---|---|
|release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_<commit8>`|
|request root|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2_t1_20260729_v1`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2_t1_20260729_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_t1_20260729_v1`|

正式子命令固定由`run_full_ablation_stage2.py --execute`调用`run_full_ablation_stage2_row.py`和`score_full_ablation_stage2_row.py`；确切plan、Python、release root及请求路径在seal后补入。

## 资源与调度

- GPU0–GPU7，每卡最多2个训练/适配进程，共16槽。
- 发布器先读取`nvidia-smi`计算进程占用；已有外部进程时等待，不超过每卡2进程。
- 当前N607槽位由Phase1动态占用；Phase2在`P1-FULL`完整部署输入出现后按实际空槽自动接续，不与Phase1争抢，也不超过每卡2个进程。
- predictor结束并验证不可变预测后才运行truth-side scorer；调度器不读取准确率、H、BA、floor等性能值。
- P0协议/安全错误立即停止本run后续派发并仅终止本run已验证进程树；两个不同零预测row产生相同确定性异常指纹时执行同样止损。

## 完整性与成功标准

每个physical execution必须有：

1. 独占log与launch PID/CWD/cmdline记录；
2. `predictions.cvspred`及其seal；
3. `row_execution_receipt.json`；
4. behavior、quantization和resource receipt；
5. 每个logical row的same-row score artifact；
6. physical terminal status；
7. 全矩阵`runner_summary.json`。

只有`completed_physical_count=physical_execution_count`、`completed_logical_score_count=logical_row_count`、`failed_physical_count=0`且无系统性止损时进入`ARTIFACTS_COMPLETE`。启动、日志存在或局部checkpoint均不算完成。

## 风险、假设与完成后检查

|风险|控制|
|---|---|
|复用不完整旧输出|只接受带完整immutable prediction receipt的`reuse_prediction`；仍重新生成当前logical score|
|跨批次数据不一致|允许；不做跨批次一致性阻塞，报告保留每row绑定来源|
|重复数据审计浪费时间|复用`VALIDATED_ONCE`输入；仅验证运行所需的封存artifact，不重扫原始数据|
|别名被误计独立样本|`P2-F3.alias_of=P2-FULL logical row`，summary单列alias数|
|输出目录残留导致混合|run/log/request均不可覆盖；存在即拒绝启动|
|首波系统性故障继续扩散|第一失败row与第一worker wave核对指纹、prediction/score数；达到预登记条件即止损|
|失败row只留日志而无结构化证据|为每个未评分logical row写入无性能值的immutable failure record，并在physical terminal记录数量|

完成后重点核对per-receiver、per-class、per-scenario同row指标，K1/K2 fallback计数，Fisher accept/rollback，量化误差/flip/state bytes，以及每个失败row的非性能failure closure。

## 2026-07-29 22:25–22:40启动前闭环

- N607只读盘点确认已完整闭合的`P1-FULL__train_seed_7281105`checkpoint、prototype PT/JSON、terminal和completion receipt可复用；不重跑Phase1、不重审数据、不要求其他启动批次使用相同cache。
- D18的5receiver×6seed LEO_weak cache可作为底层输入；它们将直接包装成当前row自己的predictor package、truth sidecar和`VALIDATED_ONCE`句柄，不做跨批次数据或数据hash对齐。
- 真实checkpoint本地重建成功，checkpoint→TorchScript在batch`1/8/64/256`上的`z_id[*,160]`和logits`[*,6]`逐项一致，全部有限，最大绝对误差为0。
- 原训练prototype的tensor与JSON内容一致，但旧`endpoint_accept_v1`边界摘要不能通过当前正式读取器。已实现只重建该摘要的确定性规范化链：非endpoint tensor/字段必须逐项不变，另存新PT/JSON，不覆盖训练原件。
- prototype链改为：同row completion receipt绑定原始PT/JSON和checkpoint→规范化PT/JSON哈希→generation config→组件manifest的`generation_config_sha256`→外层签名→正式deployment binding→Stage2 feature builder复核。私钥仅在本地`sign`子命令读取，绝不上传N607。
- predictor package构建器已拆分support seed、query seed和new-class draw seed；support/query物理样本仍强制不交叠，新类标签必须与预登记draw seed从冻结pool得到的顺序一致。

定向回归：46项通过、0项失败；真实P1-FULL unsigned prepare smoke完成，package共9个正式成员，状态`AWAITING_EXTERNAL_SIGNATURE`。最终独立发布复审确认P0=0、P1=0，允许Git封存并在精确commit、N607干净发布目录、常规preflight和签名往返闭合后正式发布。

Stage2-C的新类候选池不由调用方预选：构建器从当前已验证cache中按receiver导出每个LEO_weak场景的全部`target_new`TX，要求三个场景全集一致，并要求命令行候选池与canonical sorted全集逐项一致，随后才按显式`new_class_draw_seed`抽取。负测确认即使部分pool数量足够完成抽取也会被拒绝。该检查只约束当前启动内部的完整候选池，不要求不同启动使用相同数据或相同cache。

22:25全机训练进程占用为`2/2/2/2/2/2/1/1`，未超过每卡2个进程；Phase1 T1主矩阵`launched/completed/succeeded/failed/active/waiting=16/8/8/0/8/4`，label v2为6行活动、8行排队。两条运行链均无P0、非零退出或重复确定性异常指纹，SSH与TCP22连接已清零。

22:54 Phase1 T1主矩阵更新为`launched/completed/succeeded/failed/nonzero/active/waiting=19/16/16/0/0/3/1`，10个历史复用行加16个新完成行均已闭合；剩3个D0活动、1个D0排队。Label v2为11行活动、3行排队，尚无完成或失败。整机GPU进程占用仍为`2/2/2/2/2/2/1/1`，两条运行链均无P0、非零退出、异常指纹或输出损坏证据。

## 2026-07-29 23:00发布增量

- Phase1正式部署bundle实现、完整候选池检查、同row prototype签名链和启动前报告已封存为Git commit`fff5cad186d40ed25335d2095ed7b4007a6651be`；该提交的独立发布审查为P0=0、P1=0。
- 新增`build_full_ablation_stage2_binding_registry.py`：用`stage_scope/receiver/method_seed/support_seed/query_seed/new_class_draw_seed/K/new_class_count`精确匹配当前启动的唯一输入identity，核对feature payload/manifest、predictor package/seal、candidate lock、truth-side scoring manifest、`VALIDATED_ONCE`、capsule/split和Phase1 bundle/prototype哈希。
- registry要求当前计划全部logical row恰好覆盖；缺行、额外identity或重复identity均拒绝。不同arm可共享同一输入identity，`P2-F3`与`P2-FULL`因此绑定相同feature和scoring artifact，但保留两个独立logical score。不同启动不要求缓存或数据hash相同。
- 新增工具与既有release链的定向回归为49项通过、0项失败；当前等待该增量的独立P0/P1复审后再封存第二个精确commit。

## 2026-07-29 23:01–23:12 N607 Phase1 deployment bundle发布证据

本次由唯一服务器发布代理`stage2_t1_n607_release`执行，仅落地Phase1正式部署输入、构建source-labeled域×类P90组件并尝试unsigned prepare；未启动Stage2矩阵，未读取性能指标，未重审数据，未跨启动对齐数据hash，未干预Phase1或其他进程。

|项目|证据|
|---|---|
|Git提交|`fff5cad186d40ed25335d2095ed7b4007a6651be`|
|远端release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1`，启动前确认不存在并以不可覆盖方式创建|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|代码落地|commit归档SHA256=`3db0e8bed9e90d7faee7cf5ffe9e29dbb3d21f58ff417deacc7085b7d21c14e5`；Git归档规范化换行后，5个发布关键文件重新以本地已审查字节精确同步，远端SHA依次为`9083d34d...c23e8`、`04b21568...57ee`、`0dff3dec...edba`、`25ea1b36...af6`、`28be12ac...7f3`，全部匹配交接值；远端`py_compile`和入口`--help`通过|
|Phase1输入|checkpoint=`1eb6d07b...307d7`，原prototype PT=`a2cd82b7...dfd2`、JSON=`3c7c1183...77a5`，completion receipt=`829d83b3...6727`；receipt为`phase1_training_complete=true`、`terminal_status=COMPLETE`、`exit_code=0`并绑定`P1-FULL__train_seed_7281105`|
|WiSig同文件验证|当前`ManySig.pkl`SHA256=`2b0a7a74...694f`，只用于本次组件命令验证所读同一文件；未把它用作不同启动批次一致性门禁|
|class binding|复用源文件SHA256=`4f701ac9...b5b7`；只读取6个TX及稳定class handle，当前派生binding SHA256=`a90931dd...22d0`，未采用其历史checkpoint字段|
|normalize|`COMPLETE`；另存PT=`e0e10b67...88f0`、JSON=`89c1f21a...c527`，generation config=`59d8acf5...0364`；当前读取器判定`prototype_normalization_status=UNCHANGED_VALID`，原训练文件未覆盖|
|P90组件|PID`864644`，CWD绑定本release的`code`，`CUDA_VISIBLE_DEVICES=6`且进程内部`cuda:0`；启动前GPU6已有1个进程，构建期间为2个，未超限；完成后回落为1个|
|组件闭合|状态`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`；3个成员齐全：NPZ SHA256=`6b651fb5...6aa8`、manifest=`03b5761d...448a`、manifest.sha256=`87683202...686a`；未签名、未提升|
|unsigned prepare|`PHASE1_DEPLOYMENT_PREPARE_FAILED_CLOSED / NO_STAGE2_RUN / NO_PERFORMANCE_RESULT`|
|失败证据|正式N607 CUDA parity在生成runtime后拒绝：`runtime/checkpoint parity failed: max_abs=0.0009131431579589844`；partial目录只含`work/runtime/p1_full.torchscript.pt`，大小4,684,341字节|
|签名状态|`signing_request.json=ABSENT`，`package=ABSENT`；未生成、未取回、未签名，私钥未读取、未上传|
|资源与连接清理|prepare退出后GPU6只剩原PID`823219`；所有SSH/SCP任务结束后本地`ssh.exe=0`，N607与bridge的ESTABLISHED TCP22连接=0|

实际normalize命令：

```text
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_full_ablation_phase1_deployment_bundle.py normalize --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/best_source_validation_ssdg.pth --prototype-pt /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase2_zid_prototypes.pt --prototype-json /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase2_zid_prototypes.json --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase1_training_completion_receipt.json --class-binding-source /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/input/class_binding_source.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_normalized --component-batch-size 512 --component-num-workers 0 --min-samples-per-cell 2 --radius-histogram-bins 4096
```

实际组件命令：

```text
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/export_adv3b02_center_lowrank_radius_component.py --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/best_source_validation_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --output /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_component --device cuda:0 --expected-checkpoint-sha256 1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7 --expected-wisig-sha256 2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f --expected-class-handle-binding-sha256 a90931dd0266cbd42b1163a61d015d5bfe955d2ab287733d8674b9da92d722d0 --class-binding-source /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/input/class_binding_source.json --generation-config /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_normalized/generation_config.json --expected-generation-config-sha256 59d8acf54778ae181ccfa3762b90b2fa861ca81fb8c3da9d4c88cc2d0e030364 --expected-generation-code-sha256 04b21568a6444bce50095d052f384692cb56cf7519fbcd6125caffdd855857ee --batch-size 512 --num-workers 0 --min-samples-per-cell 2 --radius-histogram-bins 4096
```

实际prepare命令与失败点：

```text
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_full_ablation_phase1_deployment_bundle.py prepare --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/best_source_validation_ssdg.pth --prototype-pt /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.pt --prototype-json /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.json --component-dir /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_component --class-binding-source /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/input/class_binding_source.json --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase1_training_completion_receipt.json --generation-config /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_normalized/generation_config.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_unsigned --device cuda:0 --parity-seed 7281105 --parity-rows 8
FullAblationDeploymentError: runtime/checkpoint parity failed: max_abs=0.0009131431579589844
```

已取回的必要小证据位于`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_fff5cad1\`。原partial远端目录保持不变、不可重用。本发布没有获授权对该失败目录重跑；后续须先本地修复、独立复审、Git封存，并使用新的release/output ID。

## 2026-07-29 23:13–23:25 fail-closed修复验证

失败release的既有checkpoint与partial TorchScript runtime仅做只读CUDA数值诊断，没有重跑prepare、修改旧目录、打开truth或启动Stage2。batch`1/8/64/256`的feature最大绝对差依次为`0/9.6411e-06/4.0054e-05/7.5340e-05`，logit最大绝对差依次为`0/7.9632e-05/2.6608e-04/9.1314e-04`；相对eager输出标度的最高比例分别为`3.9867e-05`和`7.1414e-05`。全部batch的feature/logit在`atol=1e-3,rtol=1e-4`下均allclose，329个probe的6类logit argmax mismatch合计为0。诊断JSON为`release_evidence/n607_fff5cad1/cuda_parity_diagnostic_v1.json`，SHA256=`9d231424f450997ea9d5c76ff50ffa8e92a18362c3e20e821ce25439dba67624`。

据此将正式门禁改为固定单精度CUDA策略：formal prepare必须实际运行在可用CUDA设备上；禁用matmul/cuDNN TF32，关闭cuDNN benchmark并启用cuDNN deterministic和PyTorch deterministic algorithms；最大绝对差固定不超过`1e-3`，且batch`1/8/64/256`全部输出有限、全部probe的6类logit argmax必须完全一致。parity receipt显式绑定device type/index/capability、Torch/CUDA/cuDNN版本、五个实际后端开关、容差和decision equivalence；CPU、自声明策略漂移、`0.0011`超差、decision false或容差字段漂移均由负测拒绝。没有按本次观测值动态调容差。

binding registry同步修复独立复审的两个P1：

1. Stage2-A的`support_seed/k_shot/new_class_draw_seed/new_class_count`在index与feature manifest两侧都必须严格为0；Stage2-B要求draw/new count为0；Stage2-C要求正support/draw/new count和冻结K集合。错误字段不再被identity规范化掩盖。
2. registry改用正式scorer的严格sidecar loader，在发布前验证truth exact schema、完整rows、stage和receiver；Stage2-A误绑Stage2-B truth或错误receiver会立即拒绝，不再推迟到正式评分时失败。

最新7文件定向回归为59项通过、0项失败。此前本地CPU prepare只属于pre-CUDA-binding技术smoke，当前正式构建器会在checkpoint加载前拒绝CPU，不能作为正式bundle或替代N607新commit/new release root的fresh CUDA闭合。

最终独立增量复审确认上述registry与CUDA parity修复为P0=0、P1=0，允许Git封存；封存后仅允许唯一release runner在fresh N607新root生成全新的runtime、receipt、bundle和seal，旧`fff5cad1`partial/root不得复用或覆盖。

修复实现已封存为Git commit`122d7a72038bb6a9eb49f80af9722d3a6a1f922a`。基于该实现提交重新生成两份不可启动的源计划：

|计划|logical row|SHA256|
|---|---:|---|
|`stage2_states_plan_122d7a72.json`|325|`c84580d510254e6d3cefaf17dca40125384958628221a682780d353dda06e075`|
|`stage2c_screening_plan_122d7a72.json`|1425|`338910847dbbebf3dab269ff6757597bb82d00b442436c142075af2e9c249200`|

两份计划的`git_commit`字段均精确绑定`122d7a72038bb6a9eb49f80af9722d3a6a1f922a`；计划与release链定向回归12项通过、0项失败。它们当前仍为`formal_launch_authority=false`，只有在fresh Phase1 deployment bundle、全部cache binding registry和seal闭合后才能启动。

## 2026-07-29 23:17–23:20 partial runtime只读CUDA数值诊断

本诊断没有重跑prepare、没有写入失败release、没有签名、没有启动Stage2，也没有读取数据truth或任何准确率。诊断脚本先在本地`ssr-gpu`环境完成`py_compile`与CLI加载，root与Git镜像SHA256均为`b515370862dea3348a1ddf00486da355121c4a745316e23bb704a637b15058a2`；随后同步到独立且启动前不存在的目录：

```text
/home/szu2070436088/2510044040/CV-SincNet/diagnostics/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1_cuda_parity_diag_v1
```

执行前GPU6只有原PID`823219`一个计算进程，满足全机每卡少于2进程的启动条件。诊断读取同一checkpoint、既有partial runtime，使用`parity_seed=7281105`和batch`1/8/64/256`：

```text
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python diagnose_n607_cuda_parity.py --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/best_source_validation_ssdg.pth --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1/artifacts/phase1_unsigned/work/runtime/p1_full.torchscript.pt --output /home/szu2070436088/2510044040/CV-SincNet/diagnostics/cvs_full_ablation_phase2_t1_20260729_v1_fff5cad1_cuda_parity_diag_v1/output/parity_diagnostic.json --device cuda:0 --parity-seed 7281105
```

checkpoint SHA256仍为`1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7`；partial runtime SHA256为`5b834846bb7df553a8c6d2cf54d2eee4a9999845239b3a46887cfcc2ffbe922a`。

|batch|feature max_abs delta|logit max_abs delta|eager feature max_abs|eager logit max_abs|feature相对标度|logit相对标度|argmax mismatch|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0|0|1.1311485767364502|2.0624685287475586|0|0|0|
|8|9.641051292419434e-06|7.963180541992188e-05|1.2359713315963745|7.303960800170898|7.800384237041403e-06|1.0902551040260053e-05|0|
|64|4.00543212890625e-05|0.00026607513427734375|1.8897825479507446|12.668201446533203|2.119520117936155e-05|2.100338673965105e-05|0|
|256|7.534027099609375e-05|0.0009131431579589844|1.8897825479507446|12.786611557006836|3.986716412308482e-05|7.141400627428916e-05|0|

四个batch的feature与logit在`atol=1e-3,rtol=1e-4`及`atol=2e-3,rtol=2e-4`两组条件下全部`torch.allclose=true`，输出全部有限；总logit argmax mismatch为0。诊断只表明既有partial runtime的数值差随batch增大且最大项位于batch256的logit，不改变原`1e-5`正式门禁失败结论，也不自动授权放宽门禁。

运行时后端记录：

|设置|值|
|---|---|
|设备|NVIDIA GeForce RTX 3090，capability 8.6|
|PyTorch/CUDA/cuDNN|`2.1.0+cu121`/`12.1`/`8902`|
|`torch.backends.cuda.matmul.allow_tf32`|false|
|`torch.backends.cudnn.allow_tf32`|true|
|`torch.backends.cudnn.benchmark`|false|
|`torch.backends.cudnn.deterministic`|false|
|deterministic algorithms|false|

远端与本地JSON SHA256均为`9d231424f450997ea9d5c76ff50ffa8e92a18362c3e20e821ce25439dba67624`，本地证据为`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_fff5cad1\cuda_parity_diagnostic_v1.json`。诊断退出后GPU6恢复为原PID`823219`一个进程；本地`ssh.exe=0`且N607/bridge的ESTABLISHED TCP22连接=0。

## 2026-07-29 fresh Phase1 deployment bundle v2预登记

- release提交：`122d7a72038bb6a9eb49f80af9722d3a6a1f922a`；独立复审P0=0、P1=0；7文件定向回归59项通过。
- fresh release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72`。必须启动前不存在；旧`fff5cad1`release、partial runtime、normalize和P90组件均不得复用、复制或覆盖。
- 重新从原始`P1-FULL__train_seed_7281105`完成输入和稳定class binding source执行normalize、source-labeled域×类P90组件、formal CUDA prepare；不读取性能，不重审数据，不做跨启动数据hash对齐。
- formal prepare固定使用batch`1/8/64/256`；receipt必须记录`device=cuda`、输出全有限、`max_abs<=1e-3`、六类logit argmax mismatch总数为0，并实际回读`matmul TF32=false`、`cudnn benchmark=false`、`cudnn deterministic=true`、`deterministic algorithms=true`。任一条件失败即fail-closed。
- 本轮只到unsigned package、detached seal与signing request闭合并取回最小签名证据；不读取或上传私钥，不自行sign，不启动Stage2矩阵。

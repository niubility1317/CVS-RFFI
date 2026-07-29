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

## 2026-07-29 23:32–23:37 fresh v2 N607发布证据

本次由唯一release runner从原始Phase1完成输入重新构建，没有读取、复制或覆盖旧`fff5cad1`release及其partial。Stage2矩阵未启动，性能和truth未读取，数据未重审，也未做跨启动数据hash对齐。

|项目|证据|
|---|---|
|Git提交|`122d7a72038bb6a9eb49f80af9722d3a6a1f922a`；独立复审P0=0、P1=0；7文件59项通过|
|fresh release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72`，启动前不存在并以不可覆盖方式创建|
|commit归档|`122d7a72_code.tar.gz`，SHA256=`830fdaadb5adfb405dffa9a5baabbe992ebd1f7dbe2ed429238a163d3d93ea1d`|
|强制远端SHA|deployment builder=`230ff92de4899a66d0dc37e560c75b7ab5d4cd26a443b13deaf547f40bc28d15`；binding registry=`d0e9700cb1b3451d1ddd8516df9fe1d87e2fc20c003e49bba91f3a2b800ba039`；deployment bundle=`f81dfc8b4782f0148e1bc5f7d3ad48124d294bac3622f1c64c6b17ba9199be21`；P90 exporter=`04b21568a6444bce50095d052f384692cb56cf7519fbcd6125caffdd855857ee`|
|远端代码验证|7个发布文件`py_compile`通过，deployment builder入口加载通过|
|normalize|从原始checkpoint/prototype/completion receipt重新执行，`COMPLETE/UNCHANGED_VALID`；generation config SHA256=`59d8acf54778ae181ccfa3762b90b2fa861ca81fb8c3da9d4c88cc2d0e030364`|
|fresh P90组件|PID`879937`，GPU6；启动前GPU6为0个进程，执行中为1个，结束后为0；CWD和命令绑定fresh release|
|组件闭合|状态`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`；NPZ=`6b651fb5f00318cd073f0329e146c5e3522ab22e244e7ba279babe0028676aa8`、manifest=`03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a`、manifest.sha256=`87683202866765897c0098ed2933e7279c0a80f529b1550e495c74c94896886a`|
|formal prepare|`FRESH_V2_PREPARE_FAILED_CLOSED / NO_STAGE2_RUN / NO_PERFORMANCE_RESULT`|
|失败点|脚本已请求`deterministic algorithms=true`，但N607 CUDA/cuBLAS在首次trace时拒绝：Python进程启动前未设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`或`:16:8`，因此没有进入parity统计和receipt发布|
|partial范围|`artifacts/phase1_unsigned`目录存在但其中无文件；`package=ABSENT`、`signing_request.json=ABSENT`、parity receipt=ABSENT、prepare receipt=ABSENT|
|签名与资源|未读取或上传私钥，未自行sign；失败后GPU6为0个计算进程；本地`ssh.exe=0`且N607/bridge ESTABLISHED TCP22=0|

实际formal prepare命令没有设置`CUBLAS_WORKSPACE_CONFIG`：

```text
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_full_ablation_phase1_deployment_bundle.py prepare --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/best_source_validation_ssdg.pth --prototype-pt /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.pt --prototype-json /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.json --component-dir /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/artifacts/phase1_component --class-binding-source /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/input/class_binding_source.json --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase1_training_completion_receipt.json --generation-config /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/artifacts/phase1_normalized/generation_config.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v2_122d7a72/artifacts/phase1_unsigned --device cuda:0 --parity-seed 7281105 --parity-rows 8
RuntimeError: Deterministic behavior was enabled ... set CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8
```

fresh必要小证据已取回`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v2_122d7a72\`。由于formal prepare未产生签名请求，本轮在fail-closed安全边界结束；同一release/output不得补环境变量重跑。后续需本地补齐launcher环境前置条件、复审、Git封存，并使用新release ID。

## 2026-07-29 fresh Phase1 deployment bundle v3预登记

- 实现提交固定为`6fd77c22e1edb5eb710fb1f152e25214fb27e437`；当前报告HEAD`384f00d72fddf4c04917fe1eb10e1289bdffe3f4`仅含其后的文档变化，4个发布关键实现文件与`6fd77c22`逐项一致。
- fresh root固定为`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22`；只读预检确认启动前不存在。v1/v2 release及partial均不得读取、复制、续跑或覆盖。
- 23:50全机计算进程占用为GPU0–7=`2/2/1/2/2/1/0/0`，本轮选择GPU6；P90和formal prepare各自运行时仍须每卡不超过2个进程。
- 从原始合格`P1-FULL__train_seed_7281105`checkpoint、completion receipt、prototype和稳定class binding source重新执行normalize、fresh P90和formal prepare；不重审数据，不要求跨启动数据或hash一致。
- formal prepare父进程必须显式前缀`CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=<fresh_v3>/code`且CWD=`<fresh_v3>/code`。parity固定batch`1/8/64/256`，要求全有限、`max_abs<=1e-3`、六类argmax mismatch总数0，并在receipt中精确绑定CUDA设备、确定性后端及CUBLAS配置。
- 本轮只闭合unsigned 9成员package、prepare receipt、detached seal和signing request并取回最小证据；不读/传私钥，不自行sign，不启动Stage2，不读truth或性能。

## 2026-07-29 23:50–23:56 fresh v3 unsigned bundle闭合

本次由唯一release owner完成至`AWAITING_EXTERNAL_SIGNATURE`并停在本地签名安全边界。未读取/上传私钥，未自行sign，未启动Stage2，未读取truth或性能；v1/v2 release及partial未读取、复制、续跑或覆盖。

|项目|证据|
|---|---|
|实现提交|`6fd77c22e1edb5eb710fb1f152e25214fb27e437`；代码归档SHA256=`fd98388ed291647ef7c9cc1668bbbd04b112787a2164aab53d5e1b393aa6a139`|
|fresh root|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22`|
|关键远端SHA|builder=`b3d516a6a8464e0fc3fdd23a1c19e908674222af069b95eda467684e124f699c`；bundle loader=`79b361811a4ed9a91f0159248d024c8dd7d2319c830e9a3785313c671d065074`；binding registry=`d0e9700cb1b3451d1ddd8516df9fe1d87e2fc20c003e49bba91f3a2b800ba039`；P90 exporter=`04b21568a6444bce50095d052f384692cb56cf7519fbcd6125caffdd855857ee`|
|远端代码验证|7个发布文件`py_compile`通过，builder入口加载通过|
|normalize|从原始合格Phase1输入重新完成，`COMPLETE/UNCHANGED_VALID`；generation config=`59d8acf54778ae181ccfa3762b90b2fa861ca81fb8c3da9d4c88cc2d0e030364`|
|fresh P90|PID`891355`，GPU6；启动前/结束后GPU6为0个计算进程，执行中为1个；3成员重新生成并闭合，component root=`123283d72da728e833cf40ad41a407f08f5905554da7f56500110ad256b8263a`|
|formal prepare|父进程CWD=`<fresh_v3>/code`，显式前缀`CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=<fresh_v3>/code`；状态`AWAITING_EXTERNAL_SIGNATURE`|
|runtime|SHA256=`240692e2b89a0b422978cdbc8aa6de0857540d61c49738d35c4a627e543ad467`|
|parity|CUDA；batch=`[1,8,64,256]`；finite=true；`max_abs=1.4066696166992188e-05<=0.001`；`decision_equivalence_verified=true`，因此四个batch的六类logit argmax mismatch总数为0|
|后端字段|`CUBLAS_WORKSPACE_CONFIG=:4096:8`；CUDA matmul TF32=false；cuDNN TF32=false；benchmark=false；cuDNN deterministic=true；deterministic algorithms=true；numeric policy=`fp32_cuda_tf32_disabled_cudnn_deterministic_cublas4096_v1`|
|package|远端9个文件；manifest列8个成员，所有relative path、size和SHA逐项验证；manifest SHA256=`9e5710a801bf0507fd2785ab59e94a469f484808e922aa3c48fa5f780684af66`|
|prepare receipt|SHA256=`88d148e9774769d76e35a7ae22bf12f7e32d9f4b0084a9057849bdc3cb992efe`|
|parity receipt|SHA256=`64f57ad33d19e656b2bc8fb4236a4abf3750dcce0feeabf9b0d81df9636d54dc`|
|detached seal|SHA256=`0c1da89df6863e60e63bd03c9236921959d7adc5fc405e1371ae0404d6e61133`|
|signing request|SHA256=`d28babfa01c7a7da46dee1baf0cf141c268cdd48dbe6a96ba9a8a57d3ecc8b0c`；签名消息、unsigned envelope、seal和outer root严格闭合|
|清理|结束后GPU6=0个计算进程；本地`ssh.exe=0`，N607/bridge ESTABLISHED TCP22=0|

实际formal prepare命令：

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_full_ablation_phase1_deployment_bundle.py prepare --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/best_source_validation_ssdg.pth --prototype-pt /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.pt --prototype-json /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.json --component-dir /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_component --class-binding-source /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/input/class_binding_source.json --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase1_training_completion_receipt.json --generation-config /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/generation_config.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned --device cuda:0 --parity-seed 7281105 --parity-rows 8
```

最小本地签名证据：

|文件|本地路径|SHA256|
|---|---|---|
|signing request|`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v3_6fd77c22\signing_request.json`|`d28babfa01c7a7da46dee1baf0cf141c268cdd48dbe6a96ba9a8a57d3ecc8b0c`|
|detached seal|`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v3_6fd77c22\deployment.seal.json`|`0c1da89df6863e60e63bd03c9236921959d7adc5fc405e1371ae0404d6e61133`|
|prepare receipt|`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v3_6fd77c22\prepare_receipt.json`|`88d148e9774769d76e35a7ae22bf12f7e32d9f4b0084a9057849bdc3cb992efe`|
|parity receipt|`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v3_6fd77c22\runtime_checkpoint_parity_receipt.json`|`64f57ad33d19e656b2bc8fb4236a4abf3750dcce0feeabf9b0d81df9636d54dc`|

本地签名完成后只同步signature envelope到当前确认不存在的：

```text
/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned/external/signature_envelope.json
```

随后在fresh v3`code`目录使用CVS-RFFI Python执行：

```text
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_full_ablation_phase1_deployment_bundle.py finalize --package-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned/package --detached-seal /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned/external/deployment.seal.json --signature-envelope /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned/external/signature_envelope.json --deployment-binding /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_final/deployment_binding.json --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse/P1-FULL__train_seed_7281105/phase1_training_completion_receipt.json --generation-config /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/generation_config.json --prototype-pt /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.pt --prototype-json /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.json
```

当前远端signature envelope与deployment binding均不存在；必须由本地主代理完成签名并显式接续后才能finalize。

## 2026-07-30 00:00–00:03离线签名接续与formal finalize

本地主代理完成离线签名后，唯一release owner只同步`signature_envelope.json`到预登记的ABSENT路径。私钥和sign receipt均未上传；同步前再次确认远端unsigned manifest、seal、request、prepare receipt及parity receipt SHA未漂移。

|项目|结果|
|---|---|
|本地signature envelope|`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v3_6fd77c22\signature_envelope.json`；SHA256=`aba1215362debd1e5c44fac6402b8d4c89c1806d3b308d31dcfdb00469b0629b`|
|本地sign receipt|SHA256=`d4665fe264bbd3e080e66976ac9813187e2e1ae57f287922f6b083b85afe8014`；仅本地保留|
|远端envelope|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned/external/signature_envelope.json`；同步后SHA与本地一致|
|formal finalize|`FORMAL_PHASE2_ELIGIBLE`；`class_count=6`|
|deployment binding|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_final/deployment_binding.json`|
|binding SHA256|`1deec70778965f41010fe155335a30db3ec172cb3788c7074ffbadfe6236dee7`|
|runtime/outer root|runtime=`240692e2b89a0b422978cdbc8aa6de0857540d61c49738d35c4a627e543ad467`；outer root=`276afd459ab8cee97e86837faf541fc576190f6da773f7cfb7b22dd4edfb1702`|
|独立复核|再次调用正式bundle loader，返回`formal_phase2_eligible=true`；binding内package、seal、envelope、completion、generation config、prototype路径与SHA逐项匹配；6个class handle闭合|
|本地binding证据|`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_phase2_t1_20260729_v1\release_evidence\n607_v3_6fd77c22\deployment_binding.json`；SHA与远端一致|
|资源/连接|finalize不占GPU；GPU6为0个计算进程；本地`ssh.exe=0`，N607/bridge ESTABLISHED TCP22=0|

结论：该deployment binding已完成正式签名和加载器验证，可作为当前Stage2启动的Phase1 bundle绑定输入。该结论只表示技术与协议输入闭合，不是任何性能结果；本轮没有启动Stage2、没有读取truth或性能。

## 2026-07-29 fresh v3启动前CuBLAS修复与校验

v2失败已按本次启动自身的运行时完整性问题修复，不引入数据集重审，也不要求不同启动使用相同数据或进行跨启动数据hash对齐。已有合格Phase1完成行、checkpoint及稳定类绑定继续复用；v1/v2失败release只保留为证据，不作为v3输出根。

- formal CUDA prepare要求Python进程启动前已有`CUBLAS_WORKSPACE_CONFIG=:4096:8`；正式`prepare()`入口在解析任何输入路径、计算hash、读取JSON或加载checkpoint前执行CUDA/CuBLAS preflight，`_runtime_and_parity()`保留同一门禁作为二次防线。
- parity数值策略更新为`fp32_cuda_tf32_disabled_cudnn_deterministic_cublas4096_v1`；receipt显式记录`cublas_workspace_config=:4096:8`，正式bundle loader要求精确一致。
- 负向测试覆盖CPU设备、缺失CuBLAS环境、`:16:8`漂移，以及设备、后端、容差和决策字段漂移。
- 首轮独立复审发现门禁原先只在`_runtime_and_parity()`生效，晚于正式入口的checkpoint与输入加载，结论为P0=0、P1=1，未提交、未发布。修复后新增3个正式`prepare()`入口负向测试，分别证明CPU、缺失环境变量和`:16:8`漂移均在任何checkpoint hash/load前拒绝。
- 修复后2文件定向回归28项通过；7文件定向回归共64项通过、0项失败；`git diff --check`通过。
- v3必须使用新的、启动前不存在的release/output路径；正式命令必须以`CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=<gpu>`启动。只有fresh N607 CUDA parity满足固定batch`1/8/64/256`、全部输出有限、`max_abs<=1e-3`且六类argmax mismatch总数为0，才允许继续生成签名请求。
- 最终独立增量复审为P0=0、P1=0，确认正式入口零输入访问门禁、runtime二次门禁、receipt/loader精确绑定和fresh/non-overwrite边界均闭合，允许Git封存及fresh v3发布。
- 修复已封存为Git commit`6fd77c22e1edb5eb710fb1f152e25214fb27e437`。关键文件SHA256：deployment builder=`b3d516a6a8464e0fc3fdd23a1c19e908674222af069b95eda467684e124f699c`；deployment bundle loader=`79b361811a4ed9a91f0159248d024c8dd7d2319c830e9a3785313c671d065074`；binding registry=`d0e9700cb1b3451d1ddd8516df9fe1d87e2fc20c003e49bba91f3a2b800ba039`；P90 exporter=`04b21568a6444bce50095d052f384692cb56cf7519fbcd6125caffdd855857ee`。
- fresh v3 release root预登记为`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22`，启动前必须不存在，且不得复制、续跑或覆盖v1/v2 partial。
- formal prepare固定使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，工作目录为fresh v3 release root下的`code`，命令前缀必须为`CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=<free_gpu> PYTHONPATH=<fresh_v3>/code`；输入继续复用已闭合的`P1-FULL__train_seed_7281105`checkpoint、completion receipt和稳定class binding，不重审数据，也不要求跨启动数据一致。
- 预期输出为fresh runtime、固定batch parity receipt、9成员unsigned package、prepare receipt、detached seal和signing request；本轮发布执行者只能运行到签名请求闭合并回传，私钥始终只在本地离线签名。
- 当前状态为`FORMAL_PHASE2_ELIGIBLE / STAGE2_PLAN_REBINDING`；v3已完成离线签名与formal finalize，Stage2尚未启动，未读取性能。

## 2026-07-30 Stage2 T1源计划重绑定

deployment实现提交更新为`6fd77c22e1edb5eb710fb1f152e25214fb27e437`后，重新生成不可直接启动的源计划；仅更新实现提交绑定，不要求跨启动使用相同数据或缓存。

|计划|逻辑行数|Git绑定|SHA256|启动权限|
|---|---:|---|---|---|
|`stage2_states_plan_6fd77c22.json`|325|`6fd77c22e1edb5eb710fb1f152e25214fb27e437`|`8014268910937b0f4b8cda6de5a787147aabc7179996046d5b9913ea6994ac0e`|false|
|`stage2c_screening_plan_6fd77c22.json`|1425|`6fd77c22e1edb5eb710fb1f152e25214fb27e437`|`a6addc27802c492658c2e46a3ff959c62e61b609fd8437fbce7624ae0d902f53`|false|

下一步只对本次实际复用的合法feature cache、predictor package与truth-sidecar生成current-launch binding registry并封存计划；不重新审计数据。封存后先启动325行Stage2-A/B states矩阵，runner按8张GPU×2槽预分配并根据外部Phase1占用自动等待，确保任意时刻每卡总进程不超过2；其闭合后启动1425逻辑行的Stage2-C矩阵。

## 2026-07-30 Stage2 T1正式发布预登记

- 唯一执行者：`stage2_t1_n607_release`；不得由主代理或第二个runner重复启动同一run ID。
- 正式实现checkout：`6fd77c22e1edb5eb710fb1f152e25214fb27e437`，必须是clean Git checkout；报告与源计划封存提交为`d674dbc212901c418e18a81680c69ebad3403e0d`，只作为外部计划/证据承载，不改变执行checkout绑定。
- Phase1 deployment binding：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_final/deployment_binding.json`，SHA256=`1deec70778965f41010fe155335a30db3ec172cb3788c7074ffbadfe6236dee7`。
- states run ID：`cvs_full_ablation_phase2_states_t1_20260730_v1_6fd77c22`；run root=`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2_states_t1_20260730_v1_6fd77c22`；log root=`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_states_t1_20260730_v1_6fd77c22`。
- Stage2-C run ID：`cvs_full_ablation_phase2c_t1_20260730_v1_6fd77c22`；run root=`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v1_6fd77c22`；log root=`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v1_6fd77c22`。只在states达到`ARTIFACTS_COMPLETE`后启动，避免两个独立capacity runner竞态超配。
- Python固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；runner使用`run_full_ablation_stage2.py --execute`，8张GPU各2个固定slot，发现外部Phase1进程时自动等待。
- current-launch artifact策略：复用已存在且本row内部schema/身份/完整性闭合的合法cache或prediction；其余只构建缺失项。不同启动批次不要求数据或cache hash相同，不做原始数据重审。
- 启动前必须闭合当前实际采用的feature cache、predictor package、严格truth-sidecar、binding registry和sealed plan；run/log root必须启动前不存在，所有输出不可覆盖。
- 健康停止仅限P0协议/安全问题、输出覆盖风险、checkout/binding错误，或至少两个不同row在产生prediction前出现相同确定性异常指纹；不得因准确率或其他性能值停止。
- 首行及首个worker wave后必须报告launched/completed/succeeded/failed、prediction/score数、活动PID、GPU进程占用和归一化异常指纹；runner不读取性能值。

## 2026-07-30 predictor→feature旧类handle启动前拦截

current-launch覆盖率盘点确认states源计划325个logical row对应125个唯一input identity；D18母缓存30/30 cell可直接复用，但当前deployment/candidate-lock/seed链下没有可直接复用的完整feature+package+scoring identity，因此需要从既有IQ cache补建125个identity，不需要重建或重审IQ数据。

在任何正式request/run/log目录创建前发现确定性P0断口：旧predictor builder用每次随机secret生成旧6类handle，而full-ablation feature builder强制before/after旧类handle与formal Phase1 deployment的固定6类handle逐项同序，因此旧`6fd77c22`链会在首个feature cache稳定fail-closed。states和Stage2-C均未启动，正式run/request/log路径仍不存在；旧预登记run ID与两份`6fd77c22`源计划撤销启动权限，只保留设计证据。

本地修复：

- predictor builder新增成对的`--phase1-deployment-binding`与`--phase1-class-label-binding`输入；只提供其中一个立即拒绝。
- 正式模式完整验证外部签名Phase1 bundle，要求`formal_phase2_eligible=true`、outer signature通过，并把candidate lock、runtime、component、prototype和generation config限制为binding锁定路径。
- candidate lock、runtime、prototype和generation config SHA必须分别与formal binding一致；class-label source的TX顺序和handle必须与正式class binding逐项一致。
- Stage2-B与Stage2-C封包的旧6类handle直接复用formal handles；仅新增类继续生成当前package内opaque handle。predictor manifest既有`candidate_lock_sha256`绑定signed method lock，而method lock绑定`class_handle_binding_sha256`；scorer audit另记录formal class binding与class-label source SHA。
- 新增before/after正式handle→feature builder registered-handle跨链正测、class-label顺序漂移负测和formal双参数原子性负测。单文件16项通过；predictor、feature、registry三文件定向回归24项通过。

当前状态：`P0_FIX_LOCAL_VERIFIED / INDEPENDENT_REVIEW_PENDING / NO_STAGE2_RUN / NO_PERFORMANCE_RESULT`。独立复审P0=0、P1=0并Git封存前不得生成新计划或启动。

## 2026-07-30 predictor正式绑定首轮复审修复

首轮独立复审为P0=0、P1=2，未提交、未生成新计划、未启动。两个P1分别是：历史class-label source中的旧checkpoint SHA未与当前formal deployment lineage区分；artifact path/digest、formal authority和完整loader参数传播缺少cache-open前负测。

修复不改变数据复用策略：旧TX→class handle映射继续复用，不要求旧source checkpoint等于当前checkpoint，也不要求不同启动使用相同数据。新增current-launch attestation：

`release_evidence/n607_v3_6fd77c22/phase1_class_label_binding.json`

其SHA256为`f8abb25522b8b6d30f657be5de19e4922317bc271dbc9ff95dcd8de5c89dbb06`，原子记录当前checkpoint lineage=`1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7`、current semantic handle binding=`a90931dd0266cbd42b1163a61d015d5bfe955d2ab287733d8674b9da92d722d0`和formal deployment binding SHA=`1deec70778965f41010fe155335a30db3ec172cb3788c7074ffbadfe6236dee7`；历史mapping SHA与旧checkpoint SHA仅保留为复用来源证据，不冒充当前lineage。builder要求attestation顶层键、6个entry键/数量/顺序、当前三项绑定及复用语义精确一致。

新增负测证明以下错误均在打开LEO_weak cache之前失败：candidate/runtime/component/prototype/generation config路径漂移；candidate/runtime/prototype/generation config内容漂移；当前checkpoint lineage、semantic handle或formal deployment digest漂移；entry额外键或缺行；formal authority为false；formal loader抛错。另一个正测逐项断言deployment binding的全部路径和期望SHA参数均传入正式loader，component内容根由该loader闭合。

验证结果：

|验证面|结果|
|---|---:|
|predictor builder单文件|33 passed|
|predictor+feature builder+binding registry|独立复审42 passed|
|7文件发布链回归|79 passed|
|`git diff --check`|通过|
|root/Git报告镜像|SHA一致|
|root/Git attestation镜像|SHA一致|

第二轮独立复审结论为P0=0、P1=0，确认current-launch attestation与当前deployment及历史mapping证据一致、formal loader先于cache open、五类artifact路径锁定、显式digest与component content root闭合、authority/异常/参数传播负测完整，legacy无formal参数路径兼容。允许Git封存并生成新commit绑定计划；提交必须强制纳入受`.gitignore`影响的attestation，旧`6fd77c22`计划继续禁止启动。

当前状态：`P0_P1_REVIEW_PASS / READY_FOR_GIT_SEAL / NO_STAGE2_RUN / NO_PERFORMANCE_RESULT`。

## 2026-07-30新commit绑定计划与不可覆盖发布

正式实现已封存为Git commit`25c725c43b0420348ad89186f27e0c8ad7aa6d4e`，current-launch attestation已强制纳入该提交。旧`6fd77c22`计划和run ID继续撤销启动权限。用同一plan builder仅替换Git绑定后生成：

|计划|逻辑行数|Git绑定|SHA256|启动权限|
|---|---:|---|---|---|
|`stage2_states_plan_25c725c4.json`|325|`25c725c43b0420348ad89186f27e0c8ad7aa6d4e`|`a48f468c16c3c764b3fc235b360559f73059dc37a95aa2eb81de52f840ff07f6`|仅在current-launch artifact registry与sealed plan闭合后|
|`stage2c_screening_plan_25c725c4.json`|1425|`25c725c43b0420348ad89186f27e0c8ad7aa6d4e`|`7d0d48a4cb4ab95441b6e9ba0c29bf18c409d4440413de4e4cab4509f22f93c7`|仅在states达到ARTIFACTS_COMPLETE后|

plan builder回归5项通过。新发布绑定：

- 唯一N607执行者：`stage2_t1_n607_release`。
- input root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v2_25c725c4`。
- states run ID：`cvs_full_ablation_phase2_states_t1_20260730_v2_25c725c4`；run/log root使用同名且启动前必须不存在。
- Stage2-C run ID：`cvs_full_ablation_phase2c_t1_20260730_v2_25c725c4`；只在states完成后创建。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- Phase1 deployment binding保持`1deec70778965f41010fe155335a30db3ec172cb3788c7074ffbadfe6236dee7`；class-label attestation SHA为`f8abb25522b8b6d30f657be5de19e4922317bc271dbc9ff95dcd8de5c89dbb06`。
- 复用D18的30/30完整LEO_weak母缓存，只补建当前125个缺失feature/package/scoring identity；不重建或重审数据，也不要求跨启动数据一致。
- 8张GPU各2个总槽；先计入现有Phase1占用，再填剩余槽，任意GPU总训练/adapter进程不超过2。

当前状态：`IMPLEMENTATION_COMMITTED / SOURCE_PLANS_GENERATED / N607_RELEASE_HANDOFF_READY / NO_STAGE2_RUN / NO_PERFORMANCE_RESULT`。

## 2026-07-30 00:54–01:02 N607 v4发布失败封口

本轮唯一N607执行者`stage2_t1_n607_release`完成直连preflight后，只复用D18的30/30完整LEO_weak母缓存并尝试补建当前states缺失的predictor package；没有重建或重审数据，也没有做跨启动数据/hash对齐。

### 落地与启动前状态

|检查项|结果|
|---|---|
|fresh release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v4_25c725c4`|
|input root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v2_25c725c4`|
|执行checkout|HEAD=`25c725c43b0420348ad89186f27e0c8ad7aa6d4e`，tracked/untracked dirty count=`0`|
|脚本验收|8个发布脚本`py_compile`通过；predictor CLI存在两项正式Phase1 binding参数|
|states plan|SHA256=`a48f468c16c3c764b3fc235b360559f73059dc37a95aa2eb81de52f840ff07f6`|
|Stage2-C plan|SHA256=`7d0d48a4cb4ab95441b6e9ba0c29bf18c409d4440413de4e4cab4509f22f93c7`|
|class-label attestation|SHA256=`f8abb25522b8b6d30f657be5de19e4922317bc271dbc9ff95dcd8de5c89dbb06`|
|Phase1 deployment binding|SHA256=`1deec70778965f41010fe155335a30db3ec172cb3788c7074ffbadfe6236dee7`|
|启动前GPU外部进程数|GPU0–7分别为`2/2/0/1/1/0/0/0`|
|states request/run/log root|均保持不存在|

代码只来自tracked-clean的`25c725c4`checkout。50条命令由短连接内存orchestrator调用该checkout内已提交脚本，完整argv已保存在summary；没有remote-only代码或config修改。

### 系统性技术失败

package orchestrator PID=`924709`，最终状态为已退出。50条命令覆盖25个receiver×method pair的before/new20两类package，结果为`completed=50,succeeded=0,failed=50`。至少三个不同receiver/stage的完整日志具有相同normalized exception fingerprint：

`722a6762f89999f663682361fd75e8ac58d635d3efcfb122b86cc83a600c44de`

共同异常为：

```text
ValueError: LEO cache manifest contract failed: ['overlay_role_policy']
```

错误发生于`load_verified_leo_weak_cache_set`，早于predictor/scorer输出创建。实际D18`cache_set.json`和三个场景NPZ的embedded manifest均不存在`overlay_role_policy`键；只读工具仅把缺省显示为`null`，并非显式JSON`null`。当前加载器要求非external模式为`all_roles`。只读交叉证据显示三个场景均为：

|字段|实际值|
|---|---|
|`channel_views`唯一值|`['rx_base']`|
|`overlay_applied`唯一值|`[true]`，且all true|
|embedded manifest `output_roles`|`['target_old','target_new']`|
|NPZ `dataset_role`唯一值|`['target_new','target_old']`|
|embedded manifest `overlay_role_policy`|键不存在；只读工具缺省显示`null`|

这是一项当前加载器与已一次准入D18资产之间的manifest合同缺口。本失败批次不得通过远端修改manifest、放宽代码或原地重启处理。

### 产物与清理

|产物/状态|数量或结果|
|---|---:|
|package build log|50|
|package根下已创建目录|80|
|package根下文件|50，全部为build log|
|predictor manifest/seal|0/0|
|scoring manifest|0|
|feature manifest|0|
|prediction|0|
|orchestrator PID存活|否|
|本run GPU进程|0|
|release dirty count|0|
|本地`ssh.exe`|0|
|到N607/lab bridge的ESTABLISHED TCP22|0/0|

远端input root、50份日志和空目录全部保留；没有删除、覆盖或复用。fresh release保留且仍为tracked-clean；states request/run/log root从未创建。该批次不可原地恢复、不可覆盖、不可自动fresh retry。

回收证据位于`release_evidence/n607_v4_25c725c4/`：

|文件|SHA256|
|---|---|
|`package_build_summary.json`|`db4fbf825309c3bce123069ec0137654ff91e4efd14e5a8b1586fc6cb2ea514a`|
|`runner_handoff.json`|`1cd54125349bf1d68712c6fcb67187ef0e42dc8a756eba9d443cab55a7768560`|
|`rx_20_1_m7283101_before_build.log`|`f2943efafb386c952da88664d6bf713a716e7e080fbb7713002dadd465160a62`|
|`rx_20_1_m7283101_new20_build.log`|`f2943efafb386c952da88664d6bf713a716e7e080fbb7713002dadd465160a62`|
|`rx_3_19_m7283102_before_build.log`|`f2943efafb386c952da88664d6bf713a716e7e080fbb7713002dadd465160a62`|
|`local_connection_cleanup.json`|`64995bf1c672e40f8edcc868639d69afe4b6cd2e046973b04c10ef3813c5a308`|

当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT / NO_STAGE2_RUN`。

## 2026-07-30 D18旧manifest最小兼容修复

失败证据证明D18数据本身满足当前`all_roles`语义：三个场景每行`channel_views=rx_base`且`overlay_applied=true`，role集合为`target_old/target_new`；唯一缺口是旧builder生成时尚无`overlay_role_policy`字段。修复只调整加载器对该后来新增声明的兼容方式，不重建、不重审、不修改远端数据或manifest：

- embedded manifest显式声明`overlay_role_policy`时，仍必须与当前期望完全一致；错误值继续fail-closed。
- 字段缺失时，不直接信任或补写manifest；加载器继续完成现有逐行IQ、view、overlay、物理ID、IQ SHA、overlay ID和跨场景单观测验证，只有非external模式所有行均为`rx_base`且`overlay_applied=true`时才推断`all_roles`。
- audit记录`overlay_role_policy=all_roles`和`overlay_role_policy_evidence=legacy_missing_manifest_field_verified_from_rows`，保留声明来源边界。
- 新负测证明显式`target_new_only`漂移仍被拒绝、缺字段不能掩盖任一`overlay_applied=false`行。

本地验证：

|验证面|结果|
|---|---:|
|LEO_weak lineage与新增兼容正负测|16 passed|
|lineage authority+predictor+feature+registry五文件跨链|75 passed|
|性能读取|0|

首轮独立复审发现P0=0、P1=1：显式JSON`null`会被旧实现误当缺键，且external target-new-only模式不应使用该兼容。修复后只有`overlay_role_policy`键真正缺失且`allow_target_new_only_overlay=false`时进入旧资产逐行推断；任何显式值（含`null`和空串）必须精确等于期望，external缺键继续拒绝。新增显式null与external缺键两项负测，五文件75项通过。

当前状态：`P1_REVIEW_FIX_VERIFIED / SECOND_INDEPENDENT_REVIEW_PENDING / FAILED_BATCH_PRESERVED / NO_PERFORMANCE_RESULT`。第二轮独立复审与Git封存后使用new release/input/run ID做单个D18真实cache no-query smoke；只有package与feature完整闭合才重发全批次，禁止原地重启旧25c输入根。

## 2026-07-30 D18兼容修复封存与fresh smoke预登记

第二轮独立复审结论P0=0、P1=0；五文件独立实测75/75通过，允许commit与单个真实D18 cache no-query smoke。修复、测试、失败报告和小型回收证据已封存为Git commit`5097a33d0af76fe29c260557af55a9d083f34773`。

新source plan：

|计划|逻辑行数|Git绑定|SHA256|
|---|---:|---|---|
|`stage2_states_plan_5097a33d.json`|325|`5097a33d0af76fe29c260557af55a9d083f34773`|`14c6c89c65592ef70bbda686e448aa44234c25281296e6a4bc6e5eb25e88fa29`|
|`stage2c_screening_plan_5097a33d.json`|1425|`5097a33d0af76fe29c260557af55a9d083f34773`|`3747ee94d5eb0bf5e9d44bcb4af7f6515e6758e3818dce446ccff6efefdf4e81`|

fresh发布边界：

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v5_5097a33d`。
- input：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d`。
- states run ID：`cvs_full_ablation_phase2_states_t1_20260730_v3_5097a33d`。
- Stage2-C run ID：`cvs_full_ablation_phase2c_t1_20260730_v3_5097a33d`。
- smoke只选择receiver=`20-1`、method seed=`7283101`的before package及K=10 Stage2-B feature，使用真实D18 cache和正式Phase1 binding/attestation；只检查package、feature、audit、handle和无query-truth泄漏，不读取性能。
- smoke的所有输出使用fresh smoke子目录且不可覆盖。只有package与feature均完整加载、audit明确`legacy_missing_manifest_field_verified_from_rows`且0异常指纹，才允许在同一fresh input root继续构建其余49个package和124个identity；任何错误立即停止，不创建正式request/run/log root。

当前状态：`COMPATIBILITY_FIX_COMMITTED / FRESH_REAL_CACHE_SMOKE_READY / FAILED_25C_BATCH_PRESERVED / NO_PERFORMANCE_RESULT`。

## 2026-07-30 fresh 5097真实cache smoke失败封口

fresh release/input已按预登记完成单个receiver=`20-1`、method seed=`7283101`的真实D18 cache no-query smoke。package阶段成功构建同一对before(Stage2-B)与after(new20, Stage2-C)封存包；随后K=10 feature构建在首次真实加载正式Phase1 component时发生确定性技术失败。该smoke未进入预测、评分或任何性能读取。

### 实际启动与失败

|字段|证据|
|---|---|
|release HEAD|`5097a33d0af76fe29c260557af55a9d083f34773`，tracked dirty count=`0`|
|remote Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|feature PID|`938779`，检查时已退出|
|GPU绑定|物理GPU2；`CUDA_VISIBLE_DEVICES=2`，进程内`--device cuda:0`|
|feature输出根|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/feature_k10`|
|完整日志|同级`feature_k10.log`，SHA256=`ef572cf02b82a94eb4224d1ab9d4ff0518a967cf9c3f662158f7e4e9ca4738de`|
|异常|`d80_d66_loader_scaffold.D66ProbeError: D66 ground component member missing`|
|normalized fingerprint|`690846f32177fad9342ceb52ea41d05052d53b4252a2755c3126aee6f10b9f0e`|
|feature产物|`0`个|

实际feature子命令绑定如下；外层以`CUDA_VISIBLE_DEVICES=2`分离启动并将stdout/stderr写入上述日志：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v5_5097a33d/code/scripts/build_full_ablation_stage2_feature_cache.py --before-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/before/predictor --before-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/before/predictor.seal.json --before-seal-sha256 6089f40b0e3a9412609661ab1d159019ce7e8fbc3ff2e9d9ace2548895709274 --after-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/new20/predictor --after-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/new20/predictor.seal.json --after-seal-sha256 220231dea15378e7fdbf000b9188cc0f293b30d142273818512507dd7882689d --phase1-deployment-binding-path /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_final/deployment_binding.json --ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_component --ground-manifest-sha256 03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a --phase1-prototype-path /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.pt --phase1-prototype-manifest-path /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.json --expected-phase1-prototype-sha256 e0e10b671dec5088bcb6e59b475dc3a99060b0ccbc03581e345a5e953b6088f0 --expected-phase1-prototype-manifest-sha256 89c1f21a5476e8d6b6a27264af6505d9ac4ab6eb66b32ea1e54c1d21405fc527 --expected-phase1-bundle-sha256 276afd459ab8cee97e86837faf541fc576190f6da773f7cfb7b22dd4edfb1702 --cache-output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/feature_k10 --phase2-data-status VALIDATED_ONCE --capsule-id d18-rx20-1-cache713101-m7283101-k10-new20 --split-id p2_min_v1-rx20-1-m7283101-s7283201-q7283301-d7282401-k10 --k-shot 10 --method-seed 7283101 --support-seed 7283201 --query-seed 7283301 --new-class-draw-seed 7282401 --device cuda:0
```

异常栈完整链为`build_full_ablation_stage2_feature_cache.py`→`stage2_ablation_feature_builder.py:592`→`probe_d81_ground_nuisance_cauchy_center.load_ground_basis`→`probe_d80_ground_commonmode_covariance_denoiser.load_ground_covariance`→`probe_d66_ground_domain_reliability_residual.load_ground_domain_reliability`。5097实现的D66加载器要求成员`int8_domain_class_prototypes.npz`；只读检查确认正式Phase1 component目录只有：

|成员|大小|SHA256|
|---|---:|---|
|`int8_domain_class_center_lowrank_residual_radius_v2.npz`|6323|`6b651fb5f00318cd073f0329e146c5e3522ab22e244e7ba279babe0028676aa8`|
|`manifest.json`|5739|`03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a`|
|`manifest.sha256`|80|`87683202866765897c0098ed2933e7279c0a80f529b1550e495c74c94896886a`|

manifest实际schema=`int8_domain_class_center_lowrank_residual_radius_v2`，member allowlist也仅包含同名NPZ。因此本次失败是D81→D80→D66加载链与正式Phase1 component成员/schema的合同不一致，不是D18数据损坏，也不是package兼容修复回归。

### package成功闭包

|包|stage|注册类|support/query|package root SHA256|seal SHA256|audit|
|---|---|---:|---:|---|---|---|
|before|Stage2-B|6|60/520|`5d5293ecfe1c70b34fbce6cf7dbb0a95e03671216399fb3f186e3a2d1b893e68`|`6089f40b0e3a9412609661ab1d159019ce7e8fbc3ff2e9d9ace2548895709274`|PASS|
|after new20|Stage2-C|26|260/520|`6d2028465577d742356e0bc2359e89d10731832694634022b7f11dacec405a26`|`220231dea15378e7fdbf000b9188cc0f293b30d142273818512507dd7882689d`|PASS|

两份audit均确认`phase2_single_observation_compliant=true`，三个场景的`overlay_role_policy_evidence=legacy_missing_manifest_field_verified_from_rows`，且support/query同场景物理ID互斥、跨场景物理ID互斥均PASS。

### 输出、资源与保留状态

|检查项|结果|
|---|---:|
|smoke总文件|35|
|before/new20 package子树文件|32|
|package manifest/seal/offline audit/scoring manifest|2/2/2/2|
|support/query NPZ|6/6|
|feature log/PID文件|1/1|
|feature cache文件|0|
|prediction/score|0/0|
|PID`938779`存活|否|
|本run GPU进程|0|
|检查时GPU进程数0…7|`2/2/0/0/0/0/0/0`；GPU0/1均为既有外部任务|
|正式states request/run/log root|`ABSENT/ABSENT/ABSENT`|
|本地`ssh.exe`|0|
|到N607/lab bridge的ESTABLISHED TCP22|0/0|

远端v5 release与v3 input全部保留；没有删除、覆盖、远端改代码/数据或重跑。剩余48个package、124个identity、正式325逻辑行和后续Stage2-C均未启动。

回收证据位于`release_evidence/n607_v5_5097a33d/`：

|证据|SHA256|
|---|---|
|`runner_handoff.json`|`dab7b8d99ab24814a44496ec6551c22470d31b8bf26e1f3ea3c40bf3c3625758`|
|`remote_status_snapshot.json`|`02a28917cdbfab3fd00e75b826d45cb7231b33bb1ff9043daec19b6c160990ec`|
|`local_connection_cleanup.json`|`3cc2af13312c2a61155ce5635cd10a09141398ab12fda05b56128eb4c9b98e62`|
|`feature_k10.log`|`ef572cf02b82a94eb4224d1ab9d4ff0518a967cf9c3f662158f7e4e9ca4738de`|
|`package_summary.json`|`aae5b1da24f71906f7f8337d346ec8eb3b462a205eea27720ebb9542b5bcb308`|
|`phase1_component/directory_listing.tsv`|`114451ae1e74a1b108385c8e1512ce989c232b25ea21279d2ae461f027ed8ee7`|
|`phase1_component/manifest.json`|`03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a`|

当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT / NO_FORMAL_STAGE2_RUN / REMOTE_OUTPUTS_PRESERVED`。

## 2026-07-30正式v2地面组件兼容修复与fresh发布前检查

v5失败已经证明predictor package与复用D18缓存闭合，唯一系统性故障是feature builder仍通过D81→D80→D66旧链读取v1 dense组件，而正式Phase1 deployment bundle只封存v2 center-lowrank-residual-radius组件。修复严格限定在该兼容面：

- `_load_formal_runtime`继续先验证完整外层deployment bundle，并直接返回其中已经联合封存验证的v2 component对象；不新增任何数据、clean/source或query-truth输入。
- feature builder只接受`deployment_binding.package_root/component`这一外层封存目录；新smoke必须把`--ground-component-dir`从原始导出目录改为`artifacts/phase1_unsigned/package/component`。其他目录即使文件内容相同也fail-closed。
- `manifest.json`必须是普通文件，SHA256必须与命令行绑定一致，解析内容必须与外层loader实际验证的component manifest完全一致；目录、hash、schema或160维特征语义漂移均在写feature前拒绝。
- 从封存v2 component按14个domain×6个旧类重建84个聚合中心并读取聚合Q90半径，调用既有D89固定公式生成半径可靠性加权、类无关D81扰动谱；量化噪声底仍为manifest reconstruction RMSE平方，不扫描rank或超参数。
- Stage2-A仍不携带ground state；Stage2-B/C复用同一次正式component谱。D18 package、support/query划分、数据manifest和旧v5远端输出均不修改、不重验、不删除。

本地`ssr-gpu`定向验证：

|验证面|结果|
|---|---:|
|feature builder+v2谱+cache+executor首轮|48 passed|
|Phase1 outer bundle+v2 component+D89 probe+feature cache/executor/row executor跨链|83 passed|
|独立复审8文件定向回归|86 passed，P0=0，P1=0|
|`git diff --check`|PASS|
|性能读取|0|

新增负测覆盖错误component目录与manifest hash漂移；正测确认160维谱、84个component input、D81 basis/hash字段和outer joint seal证据完整。发布仍使用fresh不可覆盖路径；真实D18 smoke必须先产生并重新加载Stage2-A/B/C三份feature cache及manifest，确认文件数、SHA、ground audit、无query-truth字段和0异常指纹后，才允许继续构建剩余package与正式registry/seal。旧v5 release/input仅保留为失败证据，不原地续跑。

独立复审确认：outer loader已闭合signature/runtime parity/class registry；新入口只能消费该verified v2 component，D89输出满足D81/D92的`[160,r]`正权谱接口，K1/K2 exact fallback不变，且不存在truth/raw dataset输入面。结论P0=0、P1=0，允许Git封存，并要求commit后只能使用新run ID执行fresh N607 real-checkpoint/no-query smoke。

当前状态：`LOCAL_V2_COMPATIBILITY_FIX_REVIEWED_P0_0_P1_0 / COMMIT_PENDING / FRESH_RUN_NOT_YET_RELEASED / NO_PERFORMANCE_RESULT`。

## 2026-07-30 v2兼容修复封存与fresh v6预登记

正式v2地面组件兼容修复、负测、v5失败证据、追踪表和报告已封存为Git commit`00b7ad3597e3958ea1fc9c152d2119be36dd55a2`。基于该实现提交生成新的未授权source plan：

|计划|逻辑行数|Git绑定|SHA256|
|---|---:|---|---|
|`stage2_states_plan_00b7ad35.json`|325|`00b7ad3597e3958ea1fc9c152d2119be36dd55a2`|`e60573d89fb11ac5f5ee8dbe2963f296e76efc808bd3e30e8238e2d88cb1cfc4`|
|`stage2c_screening_plan_00b7ad35.json`|1425|`00b7ad3597e3958ea1fc9c152d2119be36dd55a2`|`0f14aecead471b4935816cda1e8246fdc9c682d9b5c9cfabc79155fc1e29a0cc`|

两份计划均保持`formal_launch_authority=false`，旧commit出现次数为0；计划与release定向回归12/12通过。fresh发布边界：

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v6_00b7ad35`。
- input：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v4_00b7ad35`。
- states run ID：`cvs_full_ablation_phase2_states_t1_20260730_v4_00b7ad35`。
- Stage2-C run ID：`cvs_full_ablation_phase2c_t1_20260730_v4_00b7ad35`。
- 继续复用现有D18 `VALIDATED_ONCE`缓存和正式Phase1 deployment bundle；不要求与旧v4/v5输入数据一致，不做跨批次数据hash对齐或数据重验。
- 首个真实smoke仍使用receiver=`20-1`、method seed=`7283101`、K=10，但`--ground-component-dir`必须指向正式外层封存目录`artifacts/phase1_unsigned/package/component`，不能再指向`artifacts/phase1_component`。
- 只有before/after package、Stage2-A/B/C三份feature cache、manifest重载、ground audit、文件SHA和0异常指纹全部闭合，才允许构建剩余package、feature、registry和seal；在此之前不得创建正式request/run/log根。
- smoke或批次发现两个不同row出现相同prediction前确定性异常时，立即停止本run新增dispatch并保留证据；不得续跑、覆盖或删除旧v5。

当前状态：`COMMIT_00B7AD35_LOCAL_VERIFIED / SOURCE_PLANS_READY / FRESH_V6_REAL_SMOKE_AUTHORIZED / NO_PERFORMANCE_RESULT`。

## 2026-07-30 fresh v6真实smoke失败封口

direct preflight、磁盘、GPU进程和fresh root缺席检查均PASS；v6 release已检出commit`00b7ad3597e3958ea1fc9c152d2119be36dd55a2`且tracked clean，6个关键文件compile PASS。smoke复用v5已封存before/after package，在物理GPU2启动feature PID`950875`，CWD/cmdline/output/log绑定均正确。

进程随后确定性退出，feature输出`0`个；完整异常为`Stage2AblationFeatureBuilderError: formal Phase1 deployment binding drift`，normalized fingerprint=`7a13cd0cc4e6778ce7fd7ccf3efa6dd90e43166d8d24d9f7ece890743ea084b3`，日志SHA256=`5219d1aaa40dfe16827a246def6a5accbdfb2a6a29c29f7667918c6f9dcca255`。

根因是启动参数身份字段误配，不是代码或数据损坏：

|字段|实际值|
|---|---|
|CLI `--expected-phase1-bundle-sha256`|`276afd459ab8cee97e86837faf541fc576190f6da773f7cfb7b22dd4edfb1702`|
|binding `outer_content_root_sha256`|`276afd459ab8cee97e86837faf541fc576190f6da773f7cfb7b22dd4edfb1702`|
|binding `checkpoint_lineage_sha256`|`1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7`|
|当前loader比较字段|`checkpoint_lineage_sha256`|

binding schema为`cvs.full_ablation.phase1.deployment_binding.v1`，23键集合与loader常量一致。v5→v6实际feature参数只改变release/CWD、正式ground component目录和fresh output/log/PID根；其余参数相同，包括上述误传的outer content root。v5 feature在更早的D81→D80→D66加载处失败，未曾验证该后续formal runtime参数。

失败后PID已退出，本run GPU进程为0，GPU2已释放；GPU0/1各2个既有外部进程，其余GPU为空。states与Stage2-C的request/run/log共6个正式root全部`ABSENT`，未构建剩余输入、未启动正式矩阵、未读取性能、未修改或重试代码/数据。v6 release与v4 input全部保留；本地`ssh.exe=0`，N607/lab bridge ESTABLISHED TCP22=`0/0`。

证据位于`release_evidence/n607_v6_00b7ad35/`，包括完整`feature_k10.log`、`launch_command.txt`、`deployment_binding.json`、`binding_drift_evidence.json`、`v5_v6_parameter_diff.json`、`remote_status_snapshot.json`、`local_connection_cleanup.json`和`runner_handoff.json`。25.8MB Git bundle只保留在root证据目录，未复制到Git证据目录。

当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / ZERO_FEATURE_OUTPUT / NO_FORMAL_STAGE2_ROOT / NO_PERFORMANCE_RESULT / REMOTE_OUTPUTS_PRESERVED`。

## 2026-07-30 fresh v7启动参数预检与补跑预登记

v6已经把参数歧义隔离为单一启动错误：CLI名称中的“bundle”实际由feature builder解释为Phase1 checkpoint lineage，而v6误传了outer content root。代码、正式deployment binding、v2 component、v5 predictor package和D18缓存均不需要修改。v7只补跑失败的feature smoke，继续复用前序完整产物，不重新运行Phase1、predictor package或数据构建。

启动前新增一项不写输出、不打开D18 package/query、CPU-only的真实formal-input预检，直接调用当前commit中的三个正式函数：

1. `_load_formal_runtime`使用`expected_phase1_bundle_sha256=1eb6d07b...307d7`，必须返回6个固定handle、formal eligible、outer signature verified和runtime parity verified。
2. `_verified_deployment_prototypes`使用同一checkpoint lineage及已封存prototype PT/JSON哈希，必须返回`[6,288]`有限单位范数prototype，且inference identity的source checkpoint匹配。
3. `_ground_spectrum_from_formal_v2_component`必须使用`phase1_unsigned/package/component`和manifest SHA`03b576...448a`，返回`[160,r]`basis、正且和为1的weights、84个component input与outer joint seal verified。

只有上述三段全部PASS并把实际CLI参数逐项写入launch artifact，才允许占用GPU启动feature。fresh不可覆盖边界：

- 复用release：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v6_00b7ad35`，代码仍为`00b7ad3597e3958ea1fc9c152d2119be36dd55a2`。
- 新input：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v5_00b7ad35`。
- feature smoke ID：`cvs_full_ablation_phase2_feature_smoke_t1_20260730_v7_00b7ad35`。
- states run ID：`cvs_full_ablation_phase2_states_t1_20260730_v5_00b7ad35`。
- Stage2-C run ID：`cvs_full_ablation_phase2c_t1_20260730_v5_00b7ad35`。
- CLI必须显式传入`--expected-phase1-bundle-sha256 1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7`；launch artifact必须同时记录该值与outer content root，禁止再次混用。
- v7 smoke失败时仍不自动重试；成功时只补剩余缺失输入，然后按原325/1425计划启动，不重跑已完整闭合的旧package/cache。

当前状态：`V6_CLOSED_NO_RESULT / V7_CPU_FORMAL_PREFLIGHT_REQUIRED / REUSE_COMPLETED_ARTIFACTS / NO_PERFORMANCE_RESULT`。

## 2026-07-30 fresh v7 CPU-only formal预检失败封口

direct preflight PASS；release v6仍为HEAD=`00b7ad3597e3958ea1fc9c152d2119be36dd55a2`、tracked clean。fresh v5 input以及smoke/states/Stage2-C全部request/run/log根在预检前后均`ABSENT`。CPU-only检查未打开D18 package/query、未写远端项目输出、未占用GPU。

三段只读分段结果：

|检查|结果|关键证据|
|---|---|---|
|`_load_formal_runtime`|PASS|checkpoint lineage=`1eb6d07b...307d7`；6个class handle；sealed component目录正确|
|`_verified_deployment_prototypes`|FAIL|打印`before_verified_prototypes`后发生SIGSEGV；未返回`[6,288]`或identity|
|`_ground_spectrum_from_formal_v2_component`|PASS|basis=`[160,8]`；weights正、sum=`1.0`；input count=`84`；outer joint seal=`true`；manifest SHA匹配|

合并预检与独立prototype探针均在`_verified_deployment_prototypes`内确定性崩溃。服务器core limit=`0`，未生成core文件。因CPU门未闭合，没有生成feature launch artifact，没有启动GPU smoke，也没有构建输入全集或正式矩阵。

封口时v7 GPU进程为0；GPU0/1各有2个既有外部进程，GPU2–7为空。本地`ssh.exe=0`，N607/lab bridge ESTABLISHED TCP22=`0/0`。证据位于`release_evidence/n607_v7_00b7ad35/`：`cpu_formal_preflight_stdout.jsonl`、`cpu_prototype_probe_stderr.txt`、`remote_status_snapshot.json`、`local_connection_cleanup.json`和`runner_handoff.json`。

当前状态：`CPU_FORMAL_PREFLIGHT_FAILED / GPU_SMOKE_NOT_LAUNCHED / NO_V7_INPUT_ROOT / NO_FORMAL_STAGE2_ROOT / NO_PERFORMANCE_RESULT`。

## 2026-07-30 N607 Torch/NumPy prototype转换兼容修复

v7把SIGSEGV精确定位到`_deployment_prototypes`中的`tensor.detach().float().cpu().numpy()`。该路径只处理正式prototype artifact内的6×160小张量；同一feature backbone forward路径已经使用`cpu().tolist()`，没有第二个Torch→NumPy ABI bridge。修复将该处改为：

```text
np.asarray(tensor.detach().float().cpu().tolist(), dtype=np.float32)
```

该修改不改变prototype值、归一化、padding、class order、Phase1 binding或任何数据输入，只绕开N607 Torch2.1.0+cu121与NumPy2.2.5组合中的不安全`.numpy()`边界。新增静态负测确保正式函数内不再出现`.numpy(`，并保留原有`[6,288]`、单位范数和辅助128维全0数值测试。

本地`ssr-gpu`验证：

|验证面|结果|
|---|---:|
|feature builder+predictor+v2谱+outer bundle+cache/executor/row executor九文件|117 passed|
|独立复审九文件+多dtype/非连续tensor等价检查|100 passed，P0=0，P1=0|
|`git diff --check`|PASS|
|数据重验/性能读取|0/0|

独立复审确认FP32值、shape、finite、归一化和padding语义保持；float16/32/64、int64、bfloat16及非连续tensor的新旧桥接逐元素一致，6×160列表转换资源开销可忽略，同路径无`.numpy()`或`torch.from_numpy()`残留。结论P0=0、P1=0，允许commit及fresh N607 CPU-only formal preflight；预检通过前仍不得启动D18/GPU。

当前状态：`LOCAL_TORCH_NUMPY_ABI_FIX_REVIEWED_P0_0_P1_0 / COMMIT_PENDING / FRESH_V8_NOT_RELEASED / NO_PERFORMANCE_RESULT`。

## 2026-07-30 d1f5e45c封存与fresh v8预登记

Torch/NumPy ABI修复、测试、v7失败证据和追踪表已封存为Git commit`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`。新的source plan：

|计划|逻辑行数|Git绑定|SHA256|
|---|---:|---|---|
|`stage2_states_plan_d1f5e45c.json`|325|`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`|`f8d4687b05ffeada700d9e1b76da30b5fbc4e8657e50e7e2b08cf97a767d77f2`|
|`stage2c_screening_plan_d1f5e45c.json`|1425|`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`|`546fe838730f6c22d1bb3197213e9ae246c85874e63dbda810421898a186dcdf`|

两份计划旧commit出现次数为0，计划/release定向回归12/12通过。fresh v8使用独立报告`automation_reports/CV-SincNet/cvs_full_ablation_phase2_t1_20260730_v8_d1f5e45c/report.md`，新release/input/run根均不可覆盖。v8只重做CPU formal预检和此前从未成功的feature cache；复用v5完整predictor package及所有既有合法D18缓存，不要求不同批次数据一致。

当前状态：`COMMIT_D1F5E45C_LOCAL_VERIFIED / SOURCE_PLANS_READY / FRESH_V8_CPU_PREFLIGHT_AUTHORIZED / NO_PERFORMANCE_RESULT`。

## 2026-07-30 v8至v11输入闭合进展

v8在N607真实环境完成formal runtime、prototype和ground spectrum三段CPU预检，随后GPU2 feature smoke产生Stage2-A/B/C共6个文件并由当前loader全部重载通过。v8 package补齐因class-label binding路径被错误替换而48/48在产物前失败，0 package seal；失败根按`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`封口。

v9用Git承载的原50行命令、只改三个输出参数的分波控制器重启package补齐。第一波8/8成功，最终新增48/48 package成功；加v5复用2份达到50/50。v9 feature首波因子进程缺少`PYTHONPATH=<release>/code`而13/13在产物前同指纹失败，其余86项未启动，0 feature产物。

v10分离package输入根和fresh feature输出根，使用单文件自包含控制器并显式传递release code的`PYTHONPATH`。最终99/99 feature extraction成功，新增297个scope caches/594个物理文件；加v8 smoke复用3个scope caches后达到300个，即Stage2-A/B/C各100个，300/300由当前loader重载通过，异常指纹为空。Stage2-A sidecar首项随后因正式package truth schema为v2而被仅接受v3的publisher拒绝，其余24项未启动；feature全集不受影响。

本地兼容修复只允许精确顶层的v2/v3 Stage2-B truth，先验证rows，再发布为v3 Stage2-A truth；未知schema、额外顶层键或非法rows均在写出前拒绝。相关23项与独立组合36项通过，代码复审P0=0、P1=0。fresh v11固定25个K=10 canonical Stage2-A sidecar；必须逐项由正式loader验证25组/50文件后才允许生成binding registry和plan seal。

当前状态：`PACKAGE_50_OF_50 / FEATURE_SCOPE_300_OF_300 / STAGE2A_SIDECAR_FRESH_V11_PENDING / STATES_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`。

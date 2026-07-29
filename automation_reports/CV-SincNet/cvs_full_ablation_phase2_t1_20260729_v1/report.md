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

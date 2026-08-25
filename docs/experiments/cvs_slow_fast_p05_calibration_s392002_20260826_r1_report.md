# Slow-Fast P0.5地面校准与独立目标审计报告

- run ID：`cvs_slow_fast_p05_calibration_s392002_20260826_r1`
- 当前状态：`ANALYZED`（地面校准完成；独立目标确认因无新capsule未启动）
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- Git提交：`ec51ac8c0ffacd78471a1304dbfedae2dace829f`
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`

## 候选与最小矩阵

- 主候选仅为`FAST_FILM_R8`；LOWRANK和COMMON不进入P0.5地面校准。
- source receiver-held-out episode固定`K=10`，覆盖`clean`、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`；support/query物理ID互斥，样本不足的receiver/scene明确跳过且不降低K。
- 比较四个预登记门控：`P05_Q90_HARD`、`P05_RELATIVE_K12`、`P05_RELATIVE_K08`和`P05_RELATIVE_K08_FOLD_LCB`。
- 本run只冻结纯deployment参数。完整source校准JSON不得被Phase2 runner打开；正式Phase2 config只抄入`p05_*`数值/布尔参数。

## 本地验证与独立审查

- 首轮Slow-Fast聚焦回归：`51 passed`；响应面阈值修正后的定点回归：`33 passed`。
- 语法编译：selection、scorer、diagnostics、calibration、runner和校准CLI全部通过。
- `git diff --check`通过。
- 唯一独立P0/P1审查最初发现Phase2打开完整source校准JSON；已删除`calibration_path`输入并改成纯row config参数。原问题定点复审结论：`FIXED`。

## N607预登记

- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/slow_fast_p05_ec51ac8c/checkout`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，沿用已验证的r3运行环境；N607不存在独立`ssr-gpu`可执行路径。
- GPU：`0`。预检时8张RTX3090均为0%利用率、1MiB显存占用，当前用户无活跃训练进程。
- 输入cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt`
- 输入FILM bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt`
- 输出root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_p05_calibration_s392002_20260826_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_slow_fast_p05_calibration_s392002_20260826_r1.out`
- 预期artifact：`calibration.json`和完整stdout日志。
- release归档：本地`local_artifacts/slow_fast_p05_ec51ac8c/release.tar`→远端`releases/slow_fast_p05_ec51ac8c/release.tar`；本地SHA256=`7049af4c069b24405bce6575472ce4ab622068bc43a9b4ea85c7a830f9af74fe`。只进行这一次本地到远端归档SHA比较。

## 精确命令

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/calibrate_slow_fast_p05.py --ground-cache /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt --film-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_p05_calibration_s392002_20260826_r1/calibration.json --k-shot 10 --seed 392002
```

## 技术停止规则

- 仅因协议/source/query越权、错误checkout或输入、输出覆盖、进程归属不清、无法启动、确定性重复异常、无校准artifact闭合而停止。
- 不因低性能或门控全部回退停止；低性能只进入分析。
- target验证只允许新的receiver／seed capsule。旧receiver20-1、旧方法seed392002对应的truth仅可用于回溯诊断，不得用于调参后重跑并声称独立收益。

## Capsule审计

- 现有V2使用capsule=`d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`，方法seed=`392002`，不满足独立目标确认要求。
- 新receiver／新数据划分且未消费旧truth的`p2_min_v1`、`VALIDATED_ONCE` capsule状态：`MISSING_INDEPENDENT_TARGET_CAPSULE`。
- 服务器现有可运行Target5族全部绑定receiver=`20-1`，且同一capsule已被P1/P2/P3/P4与后续融合诊断使用；旧特征包也绑定同一receiver/split，并非P0.5原始support/query输入。因此本轮没有复用它来声称独立收益。

## N607执行闭合

- release归档本地与远端SHA256一致：`7049af4c069b24405bce6575472ce4ab622068bc43a9b4ea85c7a830f9af74fe`；远端解包后语法编译通过。
- 真实checkpoint无query smoke为`SMOKE_PASS`：support计数60，`query_input_capability=false`、`query_opened=false`、`query_truth_opened=false`、`source_opened=false`。
- 校准进程PID=`3192009`自然退出；日志33965字节，无`Traceback/ERROR/Exception`指纹。
- `calibration.json`独立回读通过：schema=`cvs.slow_fast.p05.calibration.v1`，status=`CALIBRATED_SOURCE_ONLY`，28个episode、7个source receiver、无跳过项，全部数值有限，`target_support_used=false`、`target_query_used=false`。

## 地面校准结果

|配置|最差receiver平均变化(pp)|最差episode旧类floor变化(pp)|最大置信侵入代理|平均梯度更新数|平均cross-fit次数|
|---|---:|---:|---:|---:|---:|
|`P05_Q90_HARD`|-0.128205|-0.769228|0.00027455|2.035714|6|
|`P05_RELATIVE_K12`|-0.128205|-0.769228|0.00027455|2.035714|6|
|`P05_RELATIVE_K08`|-0.128205|-0.769228|0.00027455|2.035714|6|
|`P05_RELATIVE_K08_FOLD_LCB`|-0.128205|-0.769228|0|1.285714|6|

最终冻结`P05_RELATIVE_K08_FOLD_LCB`。四种策略的最差receiver均值和最差episode floor相同，但LCB版本把平均梯度更新从2.035714降至1.285714（下降36.84%），同时把旧类置信侵入代理从0.00027455降为0。它因此是当前证据下更保守、计算更少的deployment gate，而不是性能更高的gate。

冻结的纯Phase2部署参数为：`steps=3`、`step_size=0.02`、`crossfit_repeats=3`、`lambda_grid=[0,0.125,0.25,0.5,0.75,1]`；`hard_move=0.15`、`q90_move=0.1125`、`q90_relative_move=0.8`、`minimum_positive_folds=5`、`lcb_z=1.2815515655446004`、`require_fold_lcb=true`。Phase2只接收这些数值和布尔量，不读取地面校准JSON。

## 设计优化与代码结论

- 将“trust是否通过”从单个最大移动量扩展为Q90、hard max、相对决策边界移动、逐fold正增益数和LCB的联合条件。
- lambda不再是裸缩放系数，而按support统计归一化；重复2-fold按物理ID分层、去重，fit/calibration/evaluation角色在receipt中分开记录。
- scorer在truth最后打开后才统计old positive/negative flip、new flip、raw-cosine margin、score L2和新类侵入；这些诊断不回流选择。
- 修复Phase2读取完整source校准JSON的P0问题：runner现在只接受纯`p05_*`参数，并在打开输入前拒绝`calibration_path`。
- 按指导报告把support/query Spearman的P0停止阈值从`≤0`修正为`<0.2`，并新增`rho=0.1`的先红后绿回归测试。

## 实验结论与推进决定

1. 本轮确实完成了地面端慢快适配校准，但没有使用Phase2目标域样本进行新的快速域适应实验。此前旧V2诊断使用过receiver=`20-1`目标域support/query；P0.5本轮为避免truth复用偏差，未再次把它当作独立验证集。
2. 当前结果不能证明目标域性能提升。source留一接收机的最差平均变化为-0.128205pp，最差episode floor变化为-0.769228pp；这说明门控主要减少不必要更新，尚未获得稳定正收益。
3. P0目标确认保持待执行：获得新的receiver或新的合法独立split后，先跑单seed Target5的`DA0_REG0`与`DA1_REG0`同row验证，再truth-last评分。只有平均旧类增益≥1pp、旧类floor增益≥0.5pp、support/query Spearman≥0.2，且无预冻结的新类侵入恶化，才进入P1因子化慢基；否则判为`SCIENTIFIC_SIGNAL_NO_PROMOTION`并转向P1机制修改。
4. 新类侵入目前只有旧类置信代理，没有source class-heldout真新类阈值，因此P05-22不能宣称完全满足。后续应在地面端增加class-heldout校准后冻结阈值，再用于目标域truth-last晋级，不允许在目标query上调阈值。

最高交付状态：地面校准与实现为`ANALYZED`；目标域独立性能验证为`UNKNOWN/MISSING_INDEPENDENT_TARGET_CAPSULE`，不得表述为已验证提升。

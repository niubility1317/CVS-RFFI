# SF-TAPFT→ERBT-IDR M29级联实验记录

## 实验身份

- run ID：`stage2_sf_tapft_erbt_idr_m29_rx20_1_s392002_20260826_r1`
- 请求：使用已产生的SF-TAPFT域适应checkpoint运行ERBT-IDR。
- 拟用方法：`M29-FFT96-A4`，即当前ERBT-IDR单seed筛选中优于TASR48的保留臂。
- 拟用目标接收机：`rx20-1`。
- SF适配checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1/sf_tapft_bundle.pt`。
- 最终状态：`STOPPED_EARLY_PROTOCOL_MISMATCH / NO_PERFORMANCE_RESULT`。

## 直接停止事实

SF bundle绑定`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id=d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`、`split_id=stage2b-rx20-1-seed713101-before-support-prefix`。其适配输入是旧类6类各10条，共60条K10 support，且该数据是`support_only_no_query_smoke`导出。

现有ERBT-IDR Stage2-C注册包来自另一数据胶囊：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_20_1/method_7282101`。逐数组只读核对得到：

|对象|形状|标签摘要|received IQ摘要|
|---|---:|---|---|
|SF适配support|`[60,2,256]`|`40c57b341cc22934`|`f5591fa081b197c9`|
|ERBT before旧类support|`[60,2,256]`|`40c57b341cc22934`|`db5d6f8c88a22560`|
|ERBT new5 support pool|`[110,2,256]`|不适用|整体`bf227216428a6e81`|

标签排列相同不能替代物理样本和received IQ一致性。两个旧类support的received IQ摘要不同，因此不能把SF bundle当作该ERBT row的同row适配状态。

若强行级联，ERBT row除自身support外还会隐含消费另一胶囊的60条目标域旧类样本，形成跨胶囊target-derived state。此时不能再报告为ERBT的K10/new5或K10/new10同row结果，也无法根据现有bundle字段证明SF support与ERBT query物理ID互斥。这属于错误split/K和潜在query边界污染风险，是项目允许的直接科学停止条件。

## 已执行与未执行

- 已执行N607直连preflight；GPU0–3、6、7空闲，GPU4/5存在既有任务，未作干预。
- 已独立读取SF support、SF导出审计、ERBT before/new5 support和package manifest。
- 已确认SF support为6类各10条，即域适配实际使用`K=10`。
- 未加载ERBT query，未读取query truth/role，未产生prediction，未连接scorer。
- 未启动训练或推理进程，未占用GPU，未覆盖任何run output。

## 合法重入条件

只有满足以下任一条件才可使用新run ID继续：

1. 由一次性Phase2 builder提供与SF bundle完全相同`capsule_id/split_id`、相同60条旧类received IQ和物理ID绑定的new-class support/query包；或
2. 在现有ERBT Stage2-C胶囊的旧类support上重新执行SF-TAPFT，再将该同row适配checkpoint接入`M29-FFT96-A4`。这将是新的checkpoint，不是本记录中的既有SF bundle。

在此之前，本次最高交付状态仅为协议阻断记录，不存在ERBT-IDR性能数据，也不得把先前SF target-inner OOF指标当作级联query性能。

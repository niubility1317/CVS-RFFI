# ERBT-IDR M2.4／F1-SafeResidual实验报告

## 启动前登记

|字段|值|
|---|---|
|run ID|`erbt_idr_m24_safe_residual_diagnostic_20260820_v1`|
|当前状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|实现提交|`e85146f27ce9ecc0863bc7f3496b60718f673ec3`，分支`work/m24-safe-residual`|
|候选／矩阵|receiver=`3-19`，method seed=`7282101`；`K1/new20`与`K10/new5`两条row；每条按D0→D10固定顺序运行11臂及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`三场景|
|科学停止规则|D1必须与历史F1逐query预测完全一致；若任一row出现差异，当前suite在D1后停止，不运行D2–D10，修正几何后使用新run ID|
|协议|`p2_min_v1`；复用匹配的`VALIDATED_ONCE` capsule/split和固定received IQ；不因M2.4方法变化重验数据；query仅用于独立全注册类argmax，不更新状态|
|K1输入|base feature cache：`.../stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new20/k_1/stage2c/features.npz`；M2.3 overlay：`.../runs/erbt_idr_m23_rfguard_targetscreen_20260820_v1/overlays/k1_new20/overlay.npz`|
|K10输入|base feature cache：`.../stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new5/k_10/stage2c/features.npz`；M2.3 overlay：`.../runs/erbt_idr_m23_rfguard_targetscreen_20260820_v1/overlays/k10_new5/overlay.npz`|
|truth-last输入|K1：`.../stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars/stage2c/rx_3_19/method_7282101/new20/scoring_manifest.json`；K10：同根目录`new5/scoring_manifest.json`|
|环境／CWD|本地`ssr-gpu`；N607使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`与本run不可变release CWD|
|资源|闭式head在CPU执行，不新增GPU计算进程；发布前记录8卡占用，既有GPU任务不干预|
|输出|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_safe_residual_diagnostic_20260820_v1`；所有prediction、score与log目录不可覆盖|
|技术停止规则|仅协议／query越界、错误row或checkout、输出覆盖、非PSD、无prediction闭合、scorer接线错误或至少两个执行单元的相同确定性pre-prediction故障；不因低性能停止|
|预期artifact|2个suite index、22个arm execution receipt、22个不可变prediction artifact、22个same-row score、D1 parity收据、配对诊断、资源与量化收据|

## 本地实现与验证

M2.4把M2.3的共享质量、中心、先验、不确定性和RF路径拆成D0–D10逐臂因果链。D1固定物理256维F1；D2使用relative trace jitter；D3、D4分别改变center与covariance权重；D5加入IF残差可靠性；D6仅在support-LOO no-harm证据下启用低权重旧类prior且K1关闭；D7只引入nuisance covariance；D8加入归一化封顶uncertainty；D9加入低幅RF-lite对角残差；D10执行全局no-harm门并仅在K10允许层级收缩类门。任一完整候选不能通过support侧安全判定时整体回退到精确D1。

本地`compileall`通过；M2.4聚焦与M2.3相邻回归共48项通过。全仓`pytest`在收集阶段受既有环境限制阻断：当前Python缺少`tomllib`，且`tests/`与`code/tests/`存在同名模块导入冲突；未修改这些无关基础设施。首次独立P0/P1审查发现4个P1：非D0的before状态、D1 mismatch继续执行、RF门无实际support证据、prior门未做真实LOO；四项均已定点修复并由聚焦回归覆盖。唯一一次定点复审进程已结束但终端输出未保留，结论记为`UNKNOWN`，不冒充PASS。

## 扩展矩阵预设

只有两条真实D1 parity都为零差异后，才从D2–D10中最多选择2个无明确同row伤害候选，并与D1共同进入扩展筛选。目标是现有合法cache交集上的5个receiver×至少3个seed/draw×4条件（`K1/new20`、`K5/new20`、`K10/new20`、`K10/new5`）×3场景。实际规模以已有`VALIDATED_ONCE` cache交集为准，不为扩大矩阵重建或重验数据；若method seed不足，则明确降格为support/query seed与draw重复证据，不伪称多method-seed确认。

## 证据边界

本轮为研发筛选证据，不是独立fresh confirmation、完整125结论、Phase3开放世界能力或星载部署证明。最终结果必须保持receiver、seed、draw、K、新类数、场景和候选同row，不拼接跨row最优值。

## v1技术停止

release归档本地与N607 SHA256一致，远端编译通过；但正式预检发现既有合法v3 base manifest不含顶层`protocol_schema`，协议由其`split_id=p2_min_v1-*`与匹配overlay的`protocol_schema=p2_min_v1`共同绑定。预检实现错误地要求base顶层字段，导致两条row均在输出目录创建前安全失败。该问题属于launcher级确定性pre-prediction故障；v1没有prediction或性能结果。修复后使用新run ID发布，不覆盖本run。

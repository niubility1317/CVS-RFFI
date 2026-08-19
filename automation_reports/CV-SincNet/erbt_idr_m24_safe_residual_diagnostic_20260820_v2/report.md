# ERBT-IDR M2.4／F1-SafeResidual实验报告

## 启动前登记

|字段|值|
|---|---|
|run ID|`erbt_idr_m24_safe_residual_diagnostic_20260820_v2`|
|当前状态|`LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED`|
|代码提交|`4c8e1d2cff5f593f5592ec82543e5bb660bdd65c`，分支`work/m24-safe-residual`；包含M2.4实现和v3 protocol binding预检修复|
|前序|`..._v1`在prediction前因预检错误安全停止，无性能结果；本run使用全新release与输出根|
|候选／矩阵|receiver=`3-19`，method seed=`7282101`；`K1/new20`与`K10/new5`两条row；每条按D0→D10固定顺序运行11臂及三个`leo_*_weak`场景|
|科学停止规则|D1必须与历史F1逐query预测完全一致；若任一row出现差异，suite在D1后停止，不运行D2–D10，修正几何后使用新run ID|
|协议|`p2_min_v1`；复用匹配的`VALIDATED_ONCE` capsule/split和固定received IQ；不因方法变化重验数据；query仅用于独立全注册类argmax，不更新状态|
|K1输入|base：`.../stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new20/k_1/stage2c/features.npz`；overlay：`.../runs/erbt_idr_m23_rfguard_targetscreen_20260820_v1/overlays/k1_new20/overlay.npz`；scoring：`.../sidecars/stage2c/rx_3_19/method_7282101/new20/scoring_manifest.json`|
|K10输入|base：同feature根`rx_3_19/method_7282101/new5/k_10/stage2c/features.npz`；overlay：`.../overlays/k10_new5/overlay.npz`；scoring：同sidecar根`rx_3_19/method_7282101/new5/scoring_manifest.json`|
|环境／CWD|本地`ssr-gpu`；N607使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`与本run不可变release CWD|
|资源|闭式head在CPU执行，不新增GPU计算进程；既有GPU任务保持不动|
|输出|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_safe_residual_diagnostic_20260820_v2`；prediction、score和log目录不可覆盖|
|技术停止规则|仅协议／query越界、错误row或checkout、输出覆盖、非PSD、无prediction闭合、scorer接线错误或至少两个执行单元的相同确定性pre-prediction故障；不因低性能停止|
|预期artifact|2个suite index、22个arm execution receipt、22个不可变prediction artifact、22个same-row score、D1 parity、配对诊断、资源与量化收据|

## 本地实现与验证

M2.4把M2.3的共享质量、中心、先验、不确定性和RF路径拆为D0–D10逐臂因果链。D1固定物理256维F1；D2使用relative trace jitter；D3、D4分别改变center与covariance权重；D5加入IF残差可靠性；D6仅在support-LOO no-harm证据下启用低权重旧类prior且K1关闭；D7只引入nuisance covariance；D8加入归一化封顶uncertainty；D9加入低幅RF-lite对角残差；D10执行全局no-harm门并仅在K10允许层级收缩类门。任一完整候选不能通过support侧安全判定时整体回退到精确D1。

本地`compileall`通过；M2.4聚焦与M2.3相邻回归共49项通过。全仓`pytest`在收集阶段受既有`tomllib`缺失与两套同名测试模块冲突阻断，未修改无关基础设施。首次独立审查的4个P1均已定点修复；唯一一次定点复审输出不可恢复，记为`UNKNOWN`。v1实际预检另发现base manifest协议字段兼容P1，已按红→绿测试修复。

## 扩展矩阵预设

只有两条真实D1 parity都为零差异后，才从D2–D10中最多选择2个无明确同row伤害候选，并与D1共同进入扩展筛选。目标为现有合法cache交集上的5个receiver×至少3个seed/draw×`K1/new20`、`K5/new20`、`K10/new20`、`K10/new5`四条件×3场景。实际规模以已有`VALIDATED_ONCE` cache为准；若method seed不足，将明确标为support/query seed与draw重复证据，不伪称多method-seed确认。

## 证据边界

本轮为研发筛选证据，不是独立fresh confirmation、完整125结论、Phase3开放世界能力或星载部署证明。最终结果保持receiver、seed、draw、K、新类数、场景和候选同row，不拼接跨row最优值。

## 执行结果与技术停止

最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / D1_PARITY_GATE_FAILED / NO_PERFORMANCE_RESULT`。

- K1/new20完成D0–D10，D1与历史F1逐query一致；这些prediction保持truth未打开。
- K10/new5在D1发现14/660条prediction不一致，一致率97.8788%；suite按预登记规则立即停止，未运行D2–D10。
- 本run没有调用truth scorer，因此不得产生或引用性能结论；K1的已完成prediction仅作为故障定位证据。
- 根因为D1把历史F1的冻结对角度量与逐样本归一化错误近似为固定support中位数尺度。修复版显式保留256维冻结log-diag预处理，并复用历史量化头；真实K10缓存回归达到0/660差异，紧凑推理态7677B。
- v2输出和日志完整保留，不覆盖、不补跑；修复后的正式验证使用新run ID。

定点复审工具因本机旧CLI模型／配置兼容问题未形成有效结果，记为`UNKNOWN`。修复后本地`compileall`通过，M2.4聚焦及M2.3相邻回归42项通过；真实K10 regression在truth-unopened条件下通过。

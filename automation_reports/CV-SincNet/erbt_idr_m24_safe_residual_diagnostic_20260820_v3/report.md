# ERBT-IDR M2.4／F1-SafeResidual正式诊断报告

日期：2026-08-20

run ID：`erbt_idr_m24_safe_residual_diagnostic_20260820_v3`

最终状态：`ANALYZED / SMALL_MATRIX_DEVELOPMENT_EVIDENCE / D1_PARITY_PASS / M2.4_MODULES_NOT_PROMOTED`

实现分支：`work/m24-safe-residual`

## 一、结论

设计报告要求的M2.4 D0–D10模块化实现已经完成并进入真实N607诊断。D1在`3-19/K1/new20`的1560个query和`3-19/K10/new5`的660个query上均与历史F1逐query完全一致，证明物理256维输入、冻结log-diag、逐样本归一化和紧凑量化头的重构正确。

但D2–D10没有获得晋级证据：K1条件下它们与D1输出相同；K10条件下每个候选在三个场景均触发support-harm整体回退，最终query输出仍等于D1。因此本轮的正确结论是“实现完成、D1等价性通过、附加模块未晋级”，而不是把安全回退后的重复结果解释成收益。

## 二、实现范围

|臂|机制|
|---|---|
|D0|历史F1参考路径|
|D1|物理256维F1：ID与FFT分块归一化、冻结对角度量、逐样本归一化、量化头|
|D2|相对trace PSD jitter|
|D3|RF quality作用于support center|
|D4|RF quality作用于covariance center|
|D5|IF残差可靠性|
|D6|support-LOO门控ground prior；K1强制关闭|
|D7|nuisance covariance|
|D8|归一化且封顶的uncertainty|
|D9|RF-lite对角后置残差|
|D10|RF-lite安全门与K10类级收缩|

实现同时完成了support／decision／covariance三中心分离、整候选原子回退、K1/K2/K5/K10保守分段、量化margin误差统计、持久态与瞬时注册态分账、D0–D10同row执行器、truth-last scorer和协议预检。

## 三、诊断矩阵与完整性

|条件|臂数|场景|prediction|score|D1 parity|
|---|---:|---:|---:|---:|---:|
|3-19/K1/new20|11|3|11/11|11/11|1560/1560一致|
|3-19/K10/new5|11|3|11/11|11/11|660/660一致|

共22个不可变prediction artifact和22个same-row score。全部prediction在truth接入前闭合，独立scorer随后连接truth，`truth_last_scoring=true`。

## 四、关键same-row结果

下表为三个场景的query数加权结果。

|条件|臂|`A_o_pre`|`A_o_post`|`A_n`|`H`|`F`|`min_old`|`min_new`|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|K1/new20|D0历史F1|0.5417|0.3556|0.1450|0.2058|0.1861|0.0000|0.0000|
|K1/new20|D1物理256 F1|0.5167|0.3556|0.1450|0.2058|0.1611|0.0000|0.0000|
|K10/new5|D0历史F1|0.7500|0.6778|0.6433|0.6600|0.0722|0.3500|0.4500|
|K10/new5|D1物理256 F1|0.6250|0.6778|0.6433|0.6600|-0.0528|0.3500|0.4500|

D1与D0的注册后`A_o_post`、`A_n`、`H`、`min_old`和`min_new`相同，因为注册后prediction逐query等价。`A_o_pre`与`F`不同源于D1的注册前紧凑路径不是历史D0注册前head，不能把这一差异表述为四状态完全等价。

## 五、安全回退诊断

- K1：D2–D10均未形成相对D1的有效输出差异。
- K10：D2–D10在`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`三个场景全部触发整候选support-harm回退。
- 各候选support-harm计数均为非零；例如D2三个场景分别为24、40、35，D7分别为22、24、23。
- 回退机制按设计阻止了support侧明确伤害进入query预测，但也意味着这些模块尚未提供可评分的独立增益。

因此D2–D10被标记为`rejected_by_evidence`，没有进入扩展矩阵；扩展只保留D1以验证跨receiver、seed和K条件的稳定性。

## 六、资源与量化

- D1真实K10回归推理态为7677B；K1/new20为16527B。
- `persistent_update_state_bytes=0`，不保留query驱动状态。
- D1显式持久化冻结256维log-diag和量化仿射头，不保留FP32旁路workspace。
- 量化receipt包含margin归一化误差P50/P95/P99/max及阈值比例。

## 七、问题修复与版本轨迹

1. v1在prediction前因base manifest顶层协议字段兼容问题停止，无性能结果。
2. v2的K1 D1通过，但K10出现14/660个预测差异，按预登记规则在truth接入前停止。
3. 根因是旧实现用固定support中位数bias近似冻结对角度量后的逐样本归一化。
4. v3改为持久化冻结log-diag，并从历史F1量化状态解码、按相同分块语义重编译紧凑head；真实K10回归和N607正式运行均达到660/660一致。
5. scorer对M2.4扩展resource／quantization receipt做了只用于旧评分函数的兼容适配；原始完整receipt仍保留。

## 八、验证与证据

- M2.4聚焦及M2.3相邻测试最终47项通过。
- 机器可读结果：`results_summary.json`，包含22个同row结果及回退计数。
- 完整原始证据：`evidence/remote_run/`，共120个文件、约41MB。
- 扩展矩阵另见`erbt_idr_m24_d1_expanded_20260820_v1`。

## 九、证据边界

本诊断支持代码实现正确、D1注册后决策等价、安全回退真实触发和附加模块未晋级。它不支持D2–D10有效、M2.4优于F1、fresh confirmation、完整125结论、Phase3开放世界能力或星载部署结论。

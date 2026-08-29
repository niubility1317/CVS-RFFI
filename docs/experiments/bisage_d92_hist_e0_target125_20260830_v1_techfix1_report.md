# BiSAGE-D92历史D92 E0 Target125技术修复1实验报告

- run ID：`bisage_d92_hist_e0_target125_20260830_v1_techfix1`
- 状态：`ANALYZED`（pilot科学未晋级；Stage B与完整Target125未启动）
- 代码提交：`32998243133e6ec34af263bab2e84435e47b988b`
- 唯一修复：零方差维度在`sqrt`前选择安全方差，保持D92前向等价且保证梯度有限。
- 验证：32项聚焦测试与`py_compile`通过，Git远端OID一致。
- 历史矩阵：5个receiver×5个seed（713102–713106）×5个切片×3个场景，共125个outer、375个场景单元。
- pilot：`rx_3_19__seed_713102__k_10__new_5`；三个场景阶段A全部通过才进入阶段B，pilot通过才进入完整125。
- GPU：`cuda:1`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1`。
- query边界：`p2_min_v1`、`VALIDATED_ONCE`、query只读、prediction完成后独立truth-last评分。
- 四状态：`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。
- 技术停止仅限协议泄漏、错误矩阵绑定、输出碰撞、启动/闭合/scorer技术失败；低性能不停止。

## 运行闭合

- 技术修复后的真实ADV3B02 checkpoint无query smoke通过。
- 正式pilot PID=`1875196`，固定物理GPU1；PID、CWD、完整cmdline和GPU UUID与预登记一致。只读采样显示GPU利用率约46%–61%、显存约834MiB，不声明为峰值资源。
- pilot正常退出，无确定性异常指纹。三个场景各形成220条query的四状态prediction，均为`PREDICTIONS_COMPLETE`；prediction前support状态已冻结，query不参与更新、选择或拟合。
- 独立scorer PID=`1895778`在prediction闭合后启动并正常退出；三个场景均为`ANALYZED`，且`truth_join_after_prediction_only=true`、`truth_handle_alignment_verified=true`。

## Stage A门槛结果

三个场景只在`nonaffine_energy>=0.1`检查失败，其余旧类伪指标、遗忘、floor和prediction变化检查均通过。

|场景|`nonaffine_energy`|`delta_lcb_h_pseudo`|prediction变化数|伪旧类准确率|伪新类准确率|伪旧类floor|伪遗忘|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|`leo_clear_weak`|0.016440|0.573257|12|1.0000|1.0000|1.0000|-0.208333|`STOPPED_SCIENTIFIC_GATE`|
|`leo_low_elev_weak`|0.022024|0.347814|9|1.0000|1.0000|1.0000|-0.208333|`STOPPED_SCIENTIFIC_GATE`|
|`leo_rain_weak`|0.010210|0.297807|4|1.0000|1.0000|1.0000|-0.041667|`STOPPED_SCIENTIFIC_GATE`|

三场景均有`query_rows_used=0`、`actual_new_class_rows_used=0`。联合结果为`stage_a_all_scenarios_passed=false`、`full_target125_authorized=false`和`STOPPED_SCIENTIFIC_GATE_STAGE_B_NOT_RUN`。Stage B未运行，runner使用冻结`S0`模式形成评估prediction。

## 独立truth-last四状态评分

|场景|状态|旧类准确率|旧类floor|新类准确率|新类floor|H值|遗忘|
|---|---|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|`DA0_REG0`|0.783333|0.5500|`N/A`|`N/A`|`N/A`|`N/A`|
|`leo_clear_weak`|`DA1_REG0`|0.783333|0.5000|`N/A`|`N/A`|`N/A`|`N/A`|
|`leo_clear_weak`|`DA0_REG1`|0.641667|0.2500|0.5700|0.3000|0.603714|0.141667|
|`leo_clear_weak`|`DA1_REG1`|0.641667|0.2500|0.5700|0.3000|0.603714|0.141667|
|`leo_low_elev_weak`|`DA0_REG0`|0.700000|0.3500|`N/A`|`N/A`|`N/A`|`N/A`|
|`leo_low_elev_weak`|`DA1_REG0`|0.675000|0.3500|`N/A`|`N/A`|`N/A`|`N/A`|
|`leo_low_elev_weak`|`DA0_REG1`|0.550000|0.3000|0.4500|0.3000|0.495000|0.150000|
|`leo_low_elev_weak`|`DA1_REG1`|0.550000|0.3000|0.4500|0.3000|0.495000|0.125000|
|`leo_rain_weak`|`DA0_REG0`|0.716667|0.4000|`N/A`|`N/A`|`N/A`|`N/A`|
|`leo_rain_weak`|`DA1_REG0`|0.758333|0.5000|`N/A`|`N/A`|`N/A`|`N/A`|
|`leo_rain_weak`|`DA0_REG1`|0.566667|0.2500|0.5200|0.3000|0.542331|0.150000|
|`leo_rain_weak`|`DA1_REG1`|0.566667|0.2500|0.5200|0.3000|0.542331|0.191667|

旧类准确率的DA效应（注册前/注册后）、注册效应（无DA/有DA）和交互项：

|场景|DA注册前|DA注册后|注册效应无DA|注册效应有DA|交互项|
|---|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.000000|0.000000|-0.141667|-0.141667|0.000000|
|`leo_low_elev_weak`|-0.025000|0.000000|-0.150000|-0.125000|0.025000|
|`leo_rain_weak`|0.041667|0.000000|-0.150000|-0.191667|-0.041667|

## 结论

技术修复有效，pilot实现了prediction闭合和独立truth-last评分；但SAGE-D在三个LEO场景均未达到预登记非仿射能量门槛。该负结果不是系统技术失败。阶段A未晋级，Stage B及历史D92 E0完整125 outer/375 scene验证均不启动，因此没有125结果可与历史D92 E0完整矩阵比较。

当前候选不晋级。后续候选应直接改进非仿射形变能力，同时保留旧类伪指标、floor和遗忘约束；不得根据本次已打开的pilot truth调参或选择性重跑。

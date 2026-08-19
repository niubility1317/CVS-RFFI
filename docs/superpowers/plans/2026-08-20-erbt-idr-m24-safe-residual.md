# ERBT-IDR M2.4 F1-SafeResidual实施计划

> 执行方式：在隔离分支`work/m24-safe-residual`内按测试驱动开发逐项完成；同一run ID只有一个发布者。

**目标：**把M2.3中已证伪的共享质量／中心／先验／不确定性／协方差基座拆开，以精确F1为硬基准，实现可逐臂归因、可完整回退、持久态受控的M2.4，并取得足量真实同row证据。

**结构：**复用既有base feature cache和M2.3 overlay。新增八个小组件、一个编译／状态模块、一个D0–D10执行器及suite脚本；预测先封闭，truth只由独立scorer连接。D1只做物理256维F1，D2–D8逐项打开单一机制，D9/D10最后加入RF-lite残差和安全门。

**技术栈：**Python、NumPy、既有CVS预测artifact／量化／scorer接口、pytest、N607 CPU闭式头执行。

### 任务1：锁定物理F1与安全权重基础

- 新增`tests/test_stage2_m24_features.py`和`tests/test_stage2_m24_quality.py`。
- 先写D1几何、ESS下界、K边界和残差可靠性失败测试并运行确认失败。
- 新增`stage2_m24_features.py`、`stage2_m24_quality.py`，最小实现后转绿。

### 任务2：拆分中心、协方差、先验与不确定性

- 新增`tests/test_stage2_m24_statistics.py`，覆盖三类中心、相对jitter PSD、K1 prior关闭、D8归一化封顶。
- 先运行确认失败。
- 新增`stage2_m24_center.py`、`stage2_m24_covariance.py`、`stage2_m24_prior_transport.py`、`stage2_m24_uncertainty.py`并转绿。

### 任务3：实现RF残差、整候选安全选择和紧凑持久态

- 新增`tests/test_stage2_m24_compiler.py`，覆盖`alpha<=0.1`、K门控、完整F1回退、量化margin误差和状态字节。
- 先运行确认失败。
- 新增`stage2_m24_rf_residual.py`、`stage2_m24_compiler.py`、`stage2_m24_safe_residual.py`并转绿。

### 任务4：实现D0–D10同row执行与truth-last评分

- 新增`tests/test_stage2_m24_integration.py`，覆盖11臂、同seed、四状态、不可覆盖输出和truth-unopened。
- 先运行确认失败。
- 新增`stage2_m24_row_executor.py`以及`run_m24_safe_residual_suite.py`、`score_m24_safe_residual_suite.py`、`preflight_m24_safe_residual.py`。
- 复用M2.3 overlay loader、预测artifact和配对诊断，不改变旧入口。

### 任务5：本地闭合与独立正确性审查

- 在`ssr-gpu`中运行M2.4聚焦测试、M2.3相邻回归和一次真实checkpoint无query smoke。
- 只进行一次独立P0/P1审查；若有直接P0/P1，只修复原问题并定点复审一次。
- 更新追踪表状态和最小预登记报告。

### 任务6：版本发布与N607诊断矩阵

- 精确stage本轮文件，提交并自动push，核对远端分支OID。
- 运行N607只读预检；创建单一release归档，做一次本地／远端SHA比较和一次远端编译。
- 以不可覆盖run ID执行两条row×D0–D10×3场景；启动后检查一次PID／CWD／cmdline／GPU／日志增长。
- prediction完整后由独立scorer连接truth，产出同row结果与配对诊断。

### 任务7：证据驱动扩展矩阵

- 先核对D1等价性。失败时仅修D1并新run；通过后按冻结规则选择最多2–3候选。
- 从现有合法cache中冻结5 receiver×3 seed×3 draw×4条件的最大可用交集；不得因方法变化重验数据。
- 发布、监控、评分扩展矩阵；低性能不作为技术停止。

### 任务8：正式报告与最终发布

- 在根目录正式报告和Git镜像中记录实现、命令、矩阵、资源、同row指标、异常、结论和证据边界。
- 生成`results_summary.json`与完整证据目录索引。
- 精确stage报告，提交、push并核对远端OID。

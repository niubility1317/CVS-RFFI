# WISER-RF v2/P3-Primary设计追踪

设计来源：用户提供的WISER-RF最新失败机制与P3-Primary重构报告，以及`docs/superpowers/specs/2026-08-31-wiser-rf-p3-primary-design.md`。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P3-01|报告4.1/设计4|实现5-fold cross-fitted old-only D92主损失|`stage2_wiser_p3.py`、对应测试|verified|`tests/test_stage2_wiser_p3.py`|K10每类8fit/2held-out；本地165测试绿色|
|P3-02|报告4.2/设计4.2|可微D92与精确D92五类输入误差`<1e-4`|`stage2_wiser_p3.py`、`test_stage2_wiser_p3.py`|verified|`tests/test_stage2_wiser_p3.py`、`tests/test_stage2_binova_d92.py`|禁止近似路径进入正式arm；本地165测试绿色|
|RISK-01|报告5/设计5|每类P3风险增广拉格朗日与soft floor|`stage2_wiser_p3.py`、runner|verified|`tests/test_stage2_wiser_p3.py`、`tests/test_stage2_wiser_runner.py`|只读support OOF风险|
|MANI-01|报告6/设计6|以所有类别共享的目标域权重替代classwise VSW|`stage2_wiser_p3.py`、runner|verified|`tests/test_wiser_source_summary.py`、`tests/test_stage2_wiser_p3.py`|复用现有26×6量化中心|
|SUM-01|报告7/设计2|允许低秩域内协方差等摘要升级|后续Phase1 summary builder|deferred|本轮不验证|不阻塞现有中心摘要验证共享域流形|
|GRAD-01|报告8/设计7|所有辅助梯度冲突时投影到不损害P3方向|`stage2_wiser_p3.py`、runner|verified|`tests/test_stage2_wiser_p3.py`、`tests/test_stage2_wiser_runner.py`|full-support梯度|
|COMP-01|报告9/设计8|identity–FFT交叉协方差冗余约束|`stage2_wiser_p3.py`、runner|verified|`tests/test_stage2_wiser_p3.py`、`tests/test_stage2_wiser_runner.py`|相对冻结基线只惩罚增加量|
|ENERGY-01|报告10/设计8|zero-id硬门与激活能量下界|`stage2_wiser_p3.py`、scorer|verified|`tests/test_stage2_wiser_p3.py`、`tests/test_stage2_wiser_scoring.py`|D92零安全不等于候选可晋级|
|UNFREEZE-01|报告11/设计9|P3驱动time-first渐进解冻|`stage2_wiser_rf.py`、runner|verified|`tests/test_stage2_wiser_rf.py`、`tests/test_stage2_wiser_runner.py`|Sinc保持冻结|
|ROLLBACK-01|报告10/设计9|support-only基础模型插值回滚|runner|verified|`tests/test_stage2_wiser_runner.py`|最大可行`alpha`，最差`alpha=0`|
|REMOVE-01|报告12/设计3|C/ABC和旧VSW退出主矩阵|pilot/config|verified|`tests/test_stage2_wiser_pilot.py`、`tests/test_run_stage2_wiser_pilot.py`|旧A仅保留对照|
|MATRIX-01|报告13/设计3|N0～N6因果矩阵|pilot/config|verified|`tests/test_stage2_wiser_pilot.py`、`tests/test_run_stage2_wiser_pilot.py`|每个arm独立fresh checkpoint|
|GATE-01|报告14/设计10.1|P3主导三场景pilot门槛|pilot/scorer|verified|`tests/test_stage2_wiser_pilot.py`、`tests/test_stage2_wiser_scoring.py`|P1/P2降为辅助guardrail|
|DIAG-01|报告15/设计8、9|阶段P3轨迹、梯度夹角、VSW错配、互补性和zero-id诊断|runner/scorer|verified|`tests/test_stage2_wiser_runner.py`、`tests/test_stage2_wiser_scoring.py`|不使用query选择训练状态|
|DATA-01|项目协议/设计2|复用`p2_min_v1/VALIDATED_ONCE`且query只读|现有package loader、launcher|verified|`tests/test_stage2_wiser_runner.py`、`tests/test_stage2_wiser_pilot.py`、`tests/test_run_stage2_wiser_pilot.py`|新实现继续通过本地165测试；真实数据证据待Task10|
|PILOT-01|设计10.1|历史pilot覆盖1outer×3scene×N0～N6|launcher、run report|pending|待真实实验|prediction完整后独立truth-last|
|T25-01|用户要求/设计10.2|5receiver×5seed×3scene Target25大query确认|`stage2_wiser_target25.py`、launcher|verified（软件）|`tests/test_stage2_wiser_target25.py`、`tests/test_run_stage2_wiser_target25.py`、`tests/test_stage2_wiser_scoring.py`|真实Target25未运行；仅pilot通过后启动|
|K10-01|设计10.3|Target25通过后扩展三个K10切片共225scene unit|Target25 runner|pending|待真实实验|K1/K5不冒充K10|
|STAGEB-01|两阶段边界/设计10.3|阶段A确认通过后冻结`phi_D`并进入注册适应|后续Stage B runner|blocked|等待阶段A科学门槛|当前不得启动|
|SCORE-01|用户要求/设计11|报告绝对query Accuracy/BA/floor/NLL及适应增量|`stage2_wiser_scoring.py`、报告|verified|`tests/test_stage2_wiser_scoring.py`|Accuracy类变化使用百分点|
|SCORE-02|用户要求/设计11|报告per-class、help/harm、receiver/seed/scene及跨单元统计|scorer、报告|verified|`tests/test_stage2_wiser_scoring.py`、`tests/test_stage2_wiser_target25.py`|不拼接不同row最优|
|RESOURCE-01|设计11|报告训练/预测时延、峰值VRAM/RSS和状态大小|runner/scorer|pending|待真实实验|不作为pilot前额外gate|
|RELEASE-01|项目工作流/设计13|本地验证、真实无query smoke、一次P0/P1审查、release和N607绑定|run report|pending|待执行|严格八项最小流程|

当前统计：`verified=17`（其中`T25-01`仅软件验证）、`pending=4`、`deferred=1`、`blocked=1`、`rejected=0`、`implemented=0`。本地聚焦套件为165项通过，约284秒；`py_compile`、两个CLI`--help`和`git diff --check`均通过。真实pilot、Target25、资源、release及Stage B均不由该软件验证完成。

当前最高风险是`P3-02`：若可微D92不能在零模态和高条件数输入上与正式D92数值同构，所有后续P3梯度结论均不可信。因此实现顺序固定为先完成`P3-01/P3-02`，再接入风险、域流形、梯度投影和互补约束。

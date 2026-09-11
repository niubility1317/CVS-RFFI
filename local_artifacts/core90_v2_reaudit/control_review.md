# CORE90 V2审计/控制独立复核

基线：`2a80889a`。范围为`source_audit.py`、`audit_evidence.py`、`runtime_control.py`、`controller.py`及runtime审计/控制消费；另外读取`runtime_audit_v2.py`、实际labeled loss和capability接口核对估计对象。重新对照设计§5/§6/§10及实施计划§4/§5，未将既往验收结论当作证据。

本协作者只读项目代码；仅在本目录保存复现脚本/JSON和本报告。复现使用Windows原生ssr-gpu Python、CPU小型头/标量控制逻辑，没有正式训练、目标读取/推理或提交。审查期间主Agent已经根据反馈开始修改代码，所以下列初始缺陷证据与修后结果分别列出。

## A01：[P1]高gap的“已发现改进”可被定标阈值重新解释成低gap校正依据

**基线位置：**`controller.py:141-161`，特别是160—161的CORRECT分支；关联`runtime_control.py:102-105`和`source_audit.py:412-415`。

`_fit_v2`允许未达到stationary/plateau的固定预算恢复，凭明确CE改进输出`RELIABLE_HIGH_GAP`；这只证明已有可恢复空间，不证明剩余gap小。Controller把HIGH/LOW合并为一个`reliable`布尔量，随后仅比较数值和source自适应`lag_exit`。当历史gap较大时，`lag_exit`可能大于探针high_gap_threshold，新HIGH状态就会进入CORRECT分支，绕过低gap必须有收敛支撑的区分。

**精确复现：**3个合法历史gap=1.0定标得到lag_enter=1.0、lag_exit=0.5。随后两个新鲜、独立、身份保护通过的观测都为HIGH、gap=.2、stationary=false、plateau=false、方向差=.8。第一次NORMAL确认，第二次实际返回`CORRECT/trusted_correct`。保留初始输出：`repro_high_gap_correction.json`，`defect_reproduced=true`。

**设计要求：**计划§4要求只有固定预算/收敛支持才能报告可靠低gap；§5表格规定低gap可信＋恢复方向可信才申请EG。当前路径使本来不充分的恢复结果重新成为EG授权依据，影响机制结论。

**修复建议：**HIGH/LOW不仅作为共同数值有效性；CORRECT必须具有独立低gap状态/收敛资格，且仍满足定标阈值、梯度及身份保护。不能仅调大或调小阈值遮盖类型矛盾。

**审查期间修后观察：**主Agent已修改typed门控。本目录脚本随后改为调用真实`fit_empirical_lag`生成小型CPU两类头：40步、lr=.002、scale=.35，CE=.693147→.486686、normalized_gap=.297860、HIGH、stationary=false、plateau=false。修后返回NORMAL。完整输出为`repro_high_gap_correction_actual_probe.json`，`defect_reproduced=false`；这是修后回归证据，不覆盖或改写初次缺陷证据。

## A02：[P2]仅关闭head步数会静默关闭明确启用的correction模式

**基线位置：**`runtime_control.py:99`的`args.game_max_extra_head<1`整体返回None条件；关联`ControllerConfig`对catchup_steps必须正数的校验。

**触发配置：**`--game_control correction --game_max_extra_head 0`通过正式parse，correction_fraction仍为.2；即使传入3个有效lag及3个有效capability观测，`calibrate_game_v2`也始终返回None。runtime只在controller存在时进入correction逻辑，所以用户已开启的仅校正机制永远不会运行。

**复现证据：**`repro_zero_head_disables_correction.py`及同名JSON，输出accepted_configuration=true、control=correction、head_budget=0、correction_fraction=.2、controller_is_none=true。

**影响：**非默认但合法的消融配置，把“禁止额外head步数”耦合成“禁止全部控制器”，日志容易被解释为可信测量下天然无动作，而实际是配置接线关闭。计划§5/§9要求追赶与校正可分别比较、未定标和未执行原因明确。

**修复建议：**允许catchup_steps=0、只禁止CATCHUP分支，仍建立correction控制器并保留独立EG预算；或明确拒绝不支持的组合。不能偷偷将0改成1。主Agent已接受前一种修复方案；本报告不代替其最终测试结果。

## 未发现额外实质缺陷的覆盖范围

- 当前lag在owned-copy训练模式main+sat完整head batch上拟合、主视图权重非零/satellite直接head权重0，保留head随机态/模式/尺度，并显式标有界经验对象；与实际domain CE（无label smoothing）吻合。跨TX读出单列，不进入lag替代路径。
- 非有限/优化失稳/预算不足未被裁成健康0；原头与恢复头使用固定同目标、参数副本、固定终点，monitor不反传、不择优。恢复方向另外检查代表性与reference目标未恶化。
- controller要求v2 capability schema、身份有效、非塌缩及freshness；不接受旧capability简单包装。动作outcome消费observation_id，消费/请求集合进入checkpoint。
- 两clock独立推进、capability独立定标；runtime先冻结ctx，再在副本审计。阶段/有效policy改变失效game证据，capability链未被错误清除；v1控制状态不能以v2外层正常恢复。
- H1跨步缓存猜测已排除：`runtime_control.py:125`每step observe先清self.sets；runtime仅使用当前返回值，非新game审计步没有旧sets可重放。当前H1还要求lag质量/control_ready及response_fit/monitor。未把这个猜测列作缺陷。

本次未发现P0。上面两项为可复现功能缺陷；没有把“目前无动作”“source阈值尚无性能收益证明”或“实际大规模运行尚未执行”当成代码缺陷，也未扩展检查B8数值实现或目标评分。

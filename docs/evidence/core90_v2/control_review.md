# V2控制与runtime交叉审查

范围：另一作者实现的`runtime_control.py`/`controller.py`与本协作者负责的runtime接线、曝光消费；不重复审核B8数学实现。runtime部分属于实现者自查，并有真实合成source端到端状态比较；不冒称全部代码由独立第三人审核。采用code-review-excellence的可定位正确性检查，不添加实验审批链。

## 发现与处理

1. **P1，head追赶目标不一致。**旧路径从eval-clean提取ctx.x，而V2探针描述训练模式main+sat目标。已由主Agent新增`head_catchup_v2`：owned模型按当前训练模式和完整拼接提取固定特征，同全batch head前向只监督main行；重用原head调用处RNG，只有head持久更新。主Agent新增测试已通过；本次检查实现确认没有修改v1路径。此动作是当前batch有界追赶，不宣称整个probe池被精确优化。
2. **P1，replay绕过质量保护。**已修复默认V2`state_checked`：冻结请求必须获得recipient当前controller的同类型合法授权；未定标、坏/陈旧证据、动作错配记拒绝。已发行但被拒绝的观测也消费，不能重复授权。原始请求、预算裁减前head k与实际决定/提交分别保留。v2 metadata显式execution_mode；旧v1语义保留。纯mechanistic无状态策略未擅自扩展。
3. **P1，V2在线fixed/random混入严格对照。**配置现在拒绝这两个旧在线模式并指向独立donor预冻结replay；旧v1仍可读。矩阵旧行显式version1，新行显式version2。
4. **P1，H1质量与覆盖未接线。**已改为`response_fit`/`response_monitor`，每源容器1条，拒绝缺参考或lag质量/control_ready不通过的响应；拒绝计数和原因进入日志。没有提高隐式算法预算。
5. **P1，曝光物理配置未验证。**schedule新增实际三场景SatSimConfig内容，builder和runtime独立与recipient当前配置对比；改变fc/fs或物理参数不能仍称相同曝光。真实mask、scenario、channel_seed与ordered sample IDs逐batch验证。count模式明确只匹配实际计数；exact无法非平凡置换时nontrivial=false。
6. **P1，阶段变更后的旧博弈证据仍可fresh。**已在目标权重或E41/E80/E91/E131等阶段变化时失效game证据，reason为`objective_or_stage_changed`；不改变独立capability观测/时钟。新增合成E2阶段边界测试验证旧game证据UNAVAILABLE、clock仍250、capability原时间戳保留。
7. **P1，capability schema混用。**主Agent已令V2 controller拒绝嵌套v1 capability，即使外层写v2；真实旧测量不能被重新标记成v2授权。

## 已检查的闭合点

- 两独立clock/观测列表/定标对象保存到`game_coordinator_v2`，epoch checkpoint使用外层v2 schema，禁止从旧v1直接恢复控制状态。
- capability拟合仅source L_s fit，monitor只评估，lag失败不阻止其独立采样/定标；低于能力质量条件不能借capability给lag补有效性。
- controller校准要求可靠lag及独立身份保护，方向阈值0.5/0.3标为fixed。缺足够source观测时保持普通更新，不制造动作。
- 一次动作请求后必须record_outcome，已接受/已拒绝均消费观测；resume保存消费集合。
- V2先构造ctx冻结U teacher伪标签/base mask/strong输入，审计在隔离副本预览训练强mask。连续有审计与无审计的模型、优化器、BN、EMA、原型、RNG及卫星generator逐值一致。
- 实际曝光回放已在完整一step合成source runtime测试中执行：日志sample IDs/scenario/mask/channel seed与文件完全一致，真实选中9条，不只验证builder输出。
- B8配置仍仅planned；长轨迹验收待解决，manifest明确`PENDING_B8_LONG_TRAJECTORY_ACCEPTANCE`。单步等价测试不能替代此项。

## 验证记录与边界

`test_game_tracking_runtime_v2.py`新增6项在阶段失效补丁前全部通过，包括V2审计/无审计/epoch续跑端到端一致性、replay拒绝与曝光文件实际消费。补丁后单独执行`-k objective_stage_boundary`，1项通过（2026-09-11，真实合成source两step）。此前`test_game_tracking_resume.py`、`test_game_tracking_integration.py`与当时2项runtime_v2共12项通过；旧fixtures显式version1。

矩阵/独立分析31项通过；replay/分析38项通过。已生成18行核心配置和3行强普通source候选，21份持久化JSON已独立parse并核对scratch/E200/fixed/no-audit。真实source donor、正式新增seed运行、目标结果与B8长程验收均未由本报告证明。

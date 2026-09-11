# B8与能力课程只读复核

复核基线：`2a80889a274734483452c5a651345179f8d0c9be`。范围为计划§7/§8、设计§10.4/§10.5对应的runtime→solver→head_lookahead和capability→实际policy/状态失效接线。未修改生产代码、未启动正式训练、未接触target。本报告不重复审查source_audit。

## 发现

1. **P1：中间带观测被计为连续进入确认，尚未通过进入门槛就升级。** `code/cvsrffi/game_tracking/curriculum.py:157-163`首次enter即置ready，随后只要未跌破exit就累加streak；升级判断也不再要求enter。计划E5要求下一难度直接能力通过进入门槛并连续3个新观测确认。默认配置下step0输入identity=.95/margin=.3/next_identity=.95，随后step250和500均输入.6/-.02/.55（低于对应enter=.7/0/.65，但高于exit），第三次观测仍将level从0提升至.1。next_margin=.1、next_worst_tx=.5保持合格，且所有观测新鲜、policy一致，故不是缺失/陈旧观测造成。现有`test_v2_hysteresis_band_retains_readiness_but_cannot_enter`将类似升级写成预期，测试通过不能消除与计划的偏差。应将滞回ready状态与连续enter确认分开：中间带可保留ready但不能增加连续enter计数，升级仍应满足当前进入门槛。

2. **P2：capability课程在E91没有实际问题变化时仍重置历史并作废博弈证据。** `code/cvsrffi/game_tracking/runtime.py:425-428`仅按epoch属于(41,91,...)设置stage_changed。最小复现比较默认capability配置E90/E91：loss weights、增强配置和MixStyle配置完全相同，实际satellite policy均为level0、p=.3、w=(1,0,0)，但stage_changed仍为true，触发solver.reset_history与coordinator.invalidate_game_evidence。该固定课程里程碑不应无条件应用到能力课程；应比较实际目标/有效policy变化。E40/E41的loss weights确实变化，因此不将E41列为缺陷。本发现是无变化事件额外重置，不否认真正loss/阶段变化的合法重置。

3. **P2：固定课程的“当前难度”审计始终使用level0。** `code/cvsrffi/game_tracking/runtime_control.py:131-132`在curriculum不存在时向capability_audit_v2传policy_level=0；未使用observe收到的epoch来恢复固定课程实际policy。复现固定课程E131时，`satellite_stage(131)`给出三场景/p=.8，而能力审计收到level0，即p=.3、只抽clear。固定课程不会因此主动升级，但所报current能力并非训练当前难度，不能用于计划E4对应的当前/下一难度解释或跨课程比较。应显式传实际policy，而不是以0代表所有固定课程阶段。

## B8验证与边界

runtime将b8_impl及telemetry_interval传入GameSolver，测量步为真实完整目标构造可复用对象。D1第一遍梯度范围为head_only；graph路径保持该范围并复用表示图。使用现有真实CORE90合成fixture在E131做reference/head_grad_only/graph_reuse各一个CPU FP32单步，冻结同一初态与RNG，全部accepted，所有模型状态相对reference最大绝对差均为0。实际forward计数依次5/5/3；三者telemetry均AVAILABLE，first scope依次all/head_only/head_only。该结果验证此上下文的接线和单步等价，不是长期训练、GPU性能或收敛结论。代码检查未发现该范围新的P0/P1 B8问题。

## 可复现证据

- 脚本：`local_artifacts/core90_v2_reaudit/solver_curriculum_probe.py`。
- 完整输入、课程事件、配置比较、B8数值与telemetry：`local_artifacts/core90_v2_reaudit/solver_curriculum_probe.json`。
- 原生命令：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 local_artifacts/core90_v2_reaudit/solver_curriculum_probe.py`；退出码0。
- 固定课程审计接线使用mock捕获传参，未伪装为真实source能力数值；E91通过调用实际配置函数比较，并对应静态runtime分支。

结论：发现1项P1和2项P2；交主Agent修复，不在本只读复核中改动。

# 独立P0/P1审查：V2能力与协调器

范围：`code/cvsrffi/game_tracking/runtime_control.py`、`capability.py`；仅为追踪输入/恢复调用读取runtime、source_audit、coordinator测试，不重复B8测试。审查结论：当前未发现未关闭P0/P1。

核对结果：

- 独立时钟：V2Coordinator分别保存next_game_audit_step与next_capability_step；各自按独立interval推进。capability先独立观测/定标，不要求lag有效；game失败不会延长capability时钟。预算不足生成各自invalid证据，不伪造成功或新鲜旧值。
- 固定fit-only读出：capability_audit_v2对在线模型deepcopy().eval()，在isolated_rng内提取合法L_s fit/monitor组；evaluate_capability只把fit_z/fit_tx传给fit_probe，features.detach()隔离编码器。validate_groups拒绝fit/monitor容器相交。
- 同head三视图：只创建/拟合一个线性head，然后no_grad评价clean/current/next；没有monitor反传或逐视图重拟合。三视图均记录CE、TX准确率、logit margin、per-TX和per-RX×TX；cross-RX几何margin另列，未将其误称为同head的logit margin。
- 信道/随机状态：当前/下一policy共享selection、mixture及三个固定物理信道的随机实现，差别仅为policy概率/混合；owned模型的BN变化与训练模型隔离，Python/NumPy/Torch RNG通过isolated_rng恢复。
- 恢复：coordinator保存两个clock、两类观测、最新证据和两套校准组件；恢复各自配置及状态。runtime另比较完整训练args、source_info和checkpoint schema，不允许任意改interval后继承旧日程。当前课程ready已纳入恢复状态。

发现并关闭的低优先级边界缺陷：`evaluate_capability_v2`原来对current特征形状错误先标invalid，之后仍调用cosine_similarity并抛RuntimeError。已向主任务报告，主任务补负测后修为仅在`'current' in result['views']`时计算consistency，11项相关测试通过（主任务证据）。审查方随后用独立CPU probe确认返回`valid=false, reason=missing_or_invalid_difficulty_features`，没有异常。状态CLOSED；不作为新阻断条件。

独立检查命令：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 local_artifacts/core90_v2/capability_review_probe.py
# {"returned": true, "valid": false, "reason": "missing_or_invalid_difficulty_features"}
```

限制：本审查不将静态接线/有界单测等同于正式源域能力有效或真实课程晋级。时间预算的实际耗时受共享负载影响；独立clock/恢复持久化的代码正确性不保证不同物理执行的墙钟预算结果完全相同。

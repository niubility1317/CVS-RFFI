# 三项修复的定向复核

范围仅限先前`solver_curriculum_review.md`提出的3项问题。只读检查工作树差异与相关新增测试；未修改代码、未覆盖原复现JSON、未重复运行主Agent已报告通过的25项聚焦测试、未启动benchmark或训练任务。

1. **原P1关闭。** `CapabilityCurriculumV2.update`现仅在`enter and not exit_band`时增加streak，否则清零。中间带保留ready但不再累积确认；升级所需streak因此意味着连续3个新enter观测。修改后的测试明确覆盖中间带清零、同观测去重、resume事件相等和之后连续3次enter才升级。原来的1次enter＋2次中间带触发路径已被直接切断。

2. **原P2 E91无变化重置关闭。** runtime的E41/E91固定课程边界仅适用于旧版本或v2 fixed，v2 capability不再触发该边界。`weights_changed`仍独立触发reset，label/一致性真实阶段边界仍保留。新增参数化测试调用runtime.train执行E90/E91两个真实更新，捕获reset调用并验证optimistic_history_used：capability保留历史、fixed重置。该测试有意跳过前89个epoch，未将两步测试包装成完整E91训练。

3. **原P2固定课程当前难度错误关闭。** coordinator将epoch传给真实capability_audit_v2；fixed_satellite_policy通过训练同源satellite_stage得到场景和概率并映射等权。E41为p=.6、w=(0,.5,.5)，E131为p=.8、三场景等权。能力函数实际按这些policy生成current视图；固定课程next=current，且标记`fixed_policy_no_adaptive_candidate`，不再暗示存在自适应升级候选。新增测试调用真实V2Coordinator.observe及真实能力函数核对E41/E131，两层接线均覆盖。

在以上三项修复范围未发现新增P0/P1/P2。此结论不构成全树复审，也不替代正式实验的运行证据。另一个`calibrate_game_v2`差异属于本次定向核验范围之外，未在此作结论。

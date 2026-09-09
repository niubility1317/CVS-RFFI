# 16组机制实验周期测试更新

用户要求全部启动，并沿用中后期每10轮测试；此前用户已明确采用target clean及三种LEO，仅作探索性观察。本轮固定E80、E90至E200，共13次。训练仍为原16组、同seed、同数据角色、scratch-only、E200，训练数据builder仍不构造目标测试集。

周期评估在训练进程内部暂停更新后执行：保存该组本轮产生的固定epoch状态，以exact loader创建独立eval模型，读取既有opaque IQ包产生四场景预测；预测完整落盘后，独立CPU scorer连接truth。Python/NumPy/PyTorch RNG恢复，训练model/EMA/optimizer不传入预测器，不读取准确率或排名作任何决策。此方式不要求额外GPU进程名额，也不等待满卡训练结束。评估耗时独立记录。

这是用户授权的探索性目标观察，不是新的独立确认集。机制保留及组合仍只依据source V证据，目标结果不得回流调参、选模或选择性重跑。

更新必须通过新release和新run执行，现有run不热改参数、不覆盖产物。因用户要求新增周期测试而替换本轮早期训练时，仅停止经过PID/CWD/argv验证的本轮进程；原run、日志和checkpoint保留。3个旧任务的停止单独等待用户答复；未获授权时保留它们并遵守每卡最多两个实验进程，剩余候选排队。

本地验证：3项周期边界测试通过；16组FP32/AMP及actual predictor exact-loader路径通过；E2响应模型驻GPU时，真实四场景预测与独立CPU scorer对48条合成记录闭合通过，原训练模型所有state tensor保持不变。独立审查未发现P0/P1阻断项。证据为a1_periodic_models_exact_v1.json、a1_periodic_actual_pipeline_v1.json；合成检查不是正式性能结果。

预期产物：每组epoch_080_ssdg.pth至epoch_200_ssdg.pth，以及target_epochs/E080至E200下predictions.json、score.json、evaluation_scope.json；缺任一期或四场景覆盖不足时不能标为完整评分完成。新周期配置不启用预算压缩，不改变既有训练日程。

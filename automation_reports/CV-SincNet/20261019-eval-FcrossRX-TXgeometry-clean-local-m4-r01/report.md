# F 四组冻结模型：仅 clean 测试

按用户最新要求只测 clean，practical 队列已取消并保留中途产物。本 run 不重新训练，复用 r04 四个固定20轮 final.pt 和 checkpoint 内默认头；每组 clean168000 物理样本，预测固定后独立 CPU truth-last 评分。最多3个 GPU 预测 worker，另1个 CPU scorer。

需要明确：r04 实际训练使用 clean＋satellite 拼接，卫星CE权重0.68；每组记录568320个卫星分支样本，其中371968个实际叠加信道。这里只修改测试范围，不追溯改写训练配方。

四组源域扩种子决定已在目标评分前固定，WEIGHT、GEOMETRY、BOTH均未通过，因此初始四组各保留1个模型seed，不根据目标分数扩种子。

旧程序把clean预测数组暂存内存，七个视图全结束才落npz，取消时没有完整预测文件。因此本run单独重新预测clean，保留旧目录，不做模型更新、头拟合或选择性重训。实际命令、环境、输出、指标与停止规则见experiment.json。

结果已完成：168000 clean物理样本，四组模型seed均为2026092701。BASE 84.317262%、WEIGHT 83.914286%、GEOMETRY 84.209524%、BOTH 84.038690%。三个变体相对BASE分别降低0.402976、0.107738、0.278571个百分点；源域固定弱类14-7/20-19平均recall分别降低0.982143、0.750000、0.783929个百分点。本轮三种变体未证明弱类增益，不扩种子，不把单种子超过84%视为稳定达标。完整逐类、最差RX/day、Macro-F1和成本见results.md/results_summary.json。

下一条普通PL路线在读取本轮目标评分前已根据四组源域结果固定（next_source_route_fixed.json）：当前student、阈值0.95、warm5/ramp10、最大权重0.5、U128、20轮，默认头联合训练，无EMA/强一致性/额外头。实现仍在准备，尚未发布或启动；本轮目标分数不改变这组参数。

用户随后进一步指定以原85.1139%的F＋跨RX完整50轮配方为后续基底，训练不使用星地增强，测试只clean。上述未启动的E20 plainPL准备已在发布前取消；其记录保留并标记SUPERSEDED_BEFORE_LAUNCH。后续基底约束见experiments/cvs_f_crossrx50_local/baseline_anchor.json。

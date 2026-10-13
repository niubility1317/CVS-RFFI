# 固定集成的零训练汇总恢复

原run六组训练、1008000固定clean预测与独立评分完整，总表因score.finish未导入preflight失败。只补正确导入，增加exclusive新输出路径；原模型/数据/预测/评分/日志保持原样。本次训练/预测/评分均0，复用原truth-last行级结果生成完整三primaryseed均值/SD/配对与84%判定。一次实际汇总操作包含六组input_rows，只有一个真实output，不虚构新模型rows。

局部合成fixture在修复前复现同NameError，修复后总表/4个配对比较/新输出和拒绝覆盖PASS；合成满分不是真实成绩。本阶段不执行模型/checkpoint，原noquerysmoke复用，不新增smoke token。仅该导入与新输出问题定点P0/P1、单archiveSHA/compile/普通账户/独立poststate。原root不重启或覆盖，只有helper自身输入/输出/执行错误停止。

命令、commit、环境、原输入及新输出见experiment.json。不读queryIQ/truth，不选seed/arm，不把技术恢复当新盲测或性能改善。

缺导入/新输出问题的唯一独立定点P0/P1复核PASS；恢复实际preflight与dispatch均指向recover_aggregate，依赖完整。原模型/训练/预测代码未重审或改配方。运行完成状态待独立artifact读回。

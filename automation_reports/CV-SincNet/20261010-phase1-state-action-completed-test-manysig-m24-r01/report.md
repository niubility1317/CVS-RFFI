# 已完成24个多解耦模型即时测试

按用户要求测试已完成的native、real_views、L、LT、L_EG、LT_EG各4个seed。源配置、E200权重、物理样本、10个测试视图及评分规则保持原登记。所有240份预测完成后独立truth-last评分。

独立输出与唯一launch owner，不修改健康训练及原控制器；原完整矩阵后续测试保留，不依据本次结果调参、选模或重跑。当前尚无测试结果。

本地聚焦检查PASS：快照变更拒绝、缺预测truth关闭、六组配对及独立路径。独立P0/P1审查PASS，无阻断；登记validate --launch-ready为VALID。

首次发布FAILED于本地REMOTE字符串编译：命中compileall的check=True。独立读回确认远端release/run不存在，仅归档已传。修复为精确换行check赋值位置；新release r02保留旧归档，未启动的run ID不变；增加REMOTE实际编译回归PASS。

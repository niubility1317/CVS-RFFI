# MILKR 类内散度度量 K/N 验证

预登记，未启动。完整矩阵、seed、命令、路径、owner、stop rule在experiment.json。固定Tri-SIG-TIME SSL完整三视图REPL与DURA v1.0，F/G零更新。

## 候选与源端规则

在固定支持核空间估计每类均值及合并类内scatter W；A=(I+W/ridge)^(-1)，按2<q,A mean_c>-<mean_c,A mean_c>统一判别所有注册类。部署合并成kernel(q,S)乘n×C系数加截距，无query拟合。目标端不读取source IQ或源样本记忆。该改变是独立新候选，不称旧MILKR的数值修复。

固定网格沿用R6：历史交互/平衡交互/joint160；nearest/all旧支持异类带宽；global比例0/0.5；single/multi尺度，共24核；ridge0.001/0.01/0.1/1/10。K1规则仅按五源RX×两episode、合法L_s K1/held32源代理的伪旧新宏H选定。K>1在当前support选择LOO旧macro（B）或旧新H（C），再Brier损失、再较大ridge，固定次序打破余平局。全120规则事前固定。

与只条件于完整scatter的近似不同，LOO利用Sherman-Morrison精确从scatter和本类centroid移除该物理样本；核带宽本身仍条件于完整support，不能称无偏nestedCV。已用显式特征的完整求解和逐样本重建scatter/mean验证解析公式。真实query仅在所有state固定后逐样本推理。

## 配对验证与判定

P03/MIX01、model seed2026092701、七RX，K1/5/10/20×N2/5/10/20，全补集query、四严格S/Q格，预计560行指标；原MILKR同物理/同split冻结预测复用。主目标R7 S0Q0均值至少+5pp；仅全部16格均达到才宣称各K/N都+5pp。保留旧、新、H、F1、floor和逐RX负差。

此前P01/P02两候选未达主门槛，完整结果保留；新方案以源与当前支持标签选择参数，不根据query重新挑规则。该研发已暴露历史基准，P03也不是独立盲测。全部七RX预测固定才由独立scorer接truth，不重复评分来选模型。

GPU0/2/4/6，每卡最多2新worker，预启动检查剩余显存；不改其他owner。新不可变run/output root，失败保留、技术失败不盲重启，低性能不触发技术停止。按既有最小发布流程完成一个commit、聚焦负测/真实checkpoint无query smoke、唯一一次独立P0/P1审查和归档比较；无额外gate。

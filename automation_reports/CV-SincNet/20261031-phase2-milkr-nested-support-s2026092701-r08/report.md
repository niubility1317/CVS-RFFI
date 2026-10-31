# MILKR 物理支持留出选择先导

预登记，未启动。完整Tri-SIG-TIME SSL和DURA冻结，F/G新训练与更新0。唯一矩阵、seed、命令、环境、owner、路径、stop rule在experiment.json。

## 固定候选

R12不使用源已见六类的gain决定目标权重。只继承原R9 source-only K1核规则；K1保持原R9。K>1在当前合法support中，按每类physical id排序及位置mod min(3,K)形成最多3外层fold，每个物理支持样本恰好held一次，各类训练K一致。对0/0.25/0.5/0.75强度，每fold仅训练数据估计对角between/within权重、带宽、核与系数，held标签仅计选择指标。内层核/ridge条件LOO完全限于外层train。按全部OOF旧新H（B为旧macro）选强度，精确平局取小值，然后全K重拟合。query不参与任何统计或选择。

这区别于R11全局源强度，也区别于将held物理样本放在权重估计里的条件LOO。源L12个真实物理样本仅作执行数值smoke，K2外层训练K1时四强度必须精确并列并选0；不把源六已见类得分作为真新类质量门槛。没有新source模型或额外标签。

## 小规模先导与晋级

固定首个RX1-1（按原数字/列表顺序，不按历史性能），P01/MIX01、seed2026092701、K1/5/10/20×N2/5/10/20，全补集query、四严格S/Q、R12/R9C两arm共160行。R9C为force_uniform，精确原R9，F/G特征共享，标签额度不增加。原MILKR对应80行冻结B/C结果复用。全160行prediction固定后独立truth-last评分。

pilot S0Q0平均H相对原MILKR>=+5pp，旧/新宏各不低于原版超过1pp，且16格中至少12格非负，才按完全相同策略/参数在新不可变run验证其余6RX；不因pilot分数调参或选择重跑。pilot通过不是goal完成，最终仍需合适的多RX配对证据。失败则不扩该候选；保留完整负差。P01已曝光，不声称盲测。

两项测试及本候选唯一P0/P1审查PASS：physical folds互斥/全覆盖/各类训练K相同，K1/控制精确R9、K2强度平局和query批次独立。源真实checkpoint smoke仍验证336与原输出不变。源GPU4、目标GPU0一个worker，启动前查显存，不改其他owner；系统技术错误保留产物不原地重启。

# MILKR 源配对特征映射四路先导

预登记；方法 R13，完整 Tri-SIG-TIME SSL、DURA v1.0 冻结。命令、环境/CWD、源模型、物理数据句柄、GPU、owner、不可变路径和停止规则在 experiment.json。

源阶段只使用合法 L_s 6300 个物理记录及既有第91/92轮模拟缓存，共12600配对观察；F(raw) 到 F(DURA(raw)) 的8个特征块分别拟合 rank8、ridge fraction0.01 残差映射。映射只包含均值和低秩矩阵参数，冻结成npz；目标阶段不接触源样本/特征库。五次留一源RX验证只诊断配对误差，不作目标调参。F/G及默认分类头更新0，源数据/权重继承与现有F0契约匹配；不读取U/V真值。

严格S0只使用raw support并经过冻结映射；严格S1只使用真实DURA support，不混入raw。Q0经F和冻结映射，不运行DURA；Q1经真实DURA和F，不映射。明确披露Q0的离线映射曾用源DURA配对学习，不冒称完全无DURA知识。R12 nested支持物理留出选择和全部K标签预算不变，R12C对照使用同head但没有映射。

固定首RX1-1、P01/MIX01、seed2026092701、K1/5/10/20 × N2/5/10/20，4严格S/Q，R13/R12C两arm，共160评分行（含B16行/arm、C64行/arm）。每格全部注册类query补集。全部预测固定后独立truth-last评分；复用原版同case80行。源GPU4、目标GPU0一个worker，启动前实测余量，不改其他owner。

新主验收：对S0和S1分别计算16格等权C宏H，Q0较匹配原版至少+5个百分点，且当前Q1高于当前Q0；Q0旧/新macro均不低于原版超过1个百分点。逐K/N同时列出每格差异及更严格的每格达标情况。只在这两S条件全部通过时，用同冻结方法在新run验证其余6RX；先导不能代表完整goal达成。用户若要求每格而非均值+5pp，按更严格口径验收，现有原始矩阵足够核算。失败则不扩展，不修改本run参数。技术失败只停止所属后续，保留产物，不盲重试。

独立P0/P1审查PASS；新runtime明确include_R9_control=false，严格保持R13/R12C两arm。源映射query批次独立、保存重载一致、错误checkpoint/未登记超参拒绝以及原nested/Fisher定点测试PASS。真实checkpoint source smoke为launcher第一步，通过即继续，不设额外批准。

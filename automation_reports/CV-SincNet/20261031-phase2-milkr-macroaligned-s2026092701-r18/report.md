# MILKR 源宏 H 目标对齐先导

预登记R19，原R18停止记录/原root完整保留，不改写其门槛。此前source每K1新类分量也须+1pp比用户的宏H目标更强：有源方案各K/S宏H均正、整体+3.851pp、new overall+2.474pp，但一项K1new −1.5625pp而停止。本版另建run，按宏H和DURA收益重算已冻结源网格，不降低任何目标端每格要求。

只读取R18 source-only1440候选验证摘要，核对源数据/F/G/seed/augmentation/版本句柄、sourceFROZEN/原targeteligiblefalse、target/query/Utruth/F/Gupdates0及完整网格；不继承目标适配状态，不重跑源CV，不修改parent文件。新源门槛：四路H≥control+0.5pp、new macro overall≥+1pp、每K/S Q0H≥+0.25pp、每K/S Q1−Q0≥0且两S平均>0、raw old/new两S均值各≥control−1pp。额外K1new分量+1只作诊断；H/newgain/较大ridge/较小rank/较小metric/固定rule序tie。完全只用源数据选择唯一配置。

新配置选定后，才用全source L6300物理记录及既有ep91/92共12600raw/G配对重拟合最终encoder一次，F/G/神经默认头不动，U/V真值0。只有普通PairedCCA投影参数进入target，无cov/类均值/源样本上传。保留adapter-TX/RX-held源验证限制：F仍见过六sourceTX/RX，源old4/new2、三景不等于目标old6/new20、六景，不称新盲测。K1 mean-loss lambda=源6类参考×当前support注册类数/6；K>1原R14路径。完整代码/命令/环境/CWD、来源、路径、owner、GPU、stoprule见experiment.json。

仅源门槛通过才首RX1-1/P01/MIX01/seed2026092701，全K1/5/10/20×N2/5/10/20、四严格S/Q80行，query全补集。S0/Q0只raw，S1/Q1只实际DURA输出，同一support状态共享两query路，全类竞争，不知query真角色/配额。全部80预测固定后独立truth-last，原版/R14/R17同case历史复用。用户目标不变：每16KN两S Q0较原≥5pp且Q1>Q0，保留原均值旧新≥原版−1pp guard，逐格报告四路原版差；全部格通过才同参数六RX确认，mean不晋级。

新增source-reuse路径唯一P0/P1审查PASS；旧aux分量门槛可fail但宏H主目标可pass的逻辑、sourceparent反馈/契约拒绝、K1与subset隔离测试PASS。真实F/G no-query smoke为launcher第一步，之后source选择/拟合；源GPU4/targetGPU0，actualfree检查，不动其他owner；技术失败保留产物不盲重试。

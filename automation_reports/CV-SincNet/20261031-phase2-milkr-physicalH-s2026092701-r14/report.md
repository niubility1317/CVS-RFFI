# MILKR 物理支持留出 H 校准

预登记。R15沿用R14原源rank16 CCA、完整Tri-SIG-TIME SSL和DURA v1.0，不重训基座/恢复器/默认头；当前support物理OOF校准旧新组判决。命令/环境/CWD、来源、路径、owner、GPU、stoprule见experiment.json。

K>1时，每个Fisher strength0/.25/.5/.75使用原物理fold：train独占特征权重/核/ridge/系数，held产生OOF分数。只按每条分数的类别轴中心/std，不跨query样本。扫描oldmax/newmax的全部阈值使OOF H最大，平局近0，再用K/(K+kappa)收缩；按收缩后OOF H选strength，全K重拟合。query只将固定beta加到旧类组分数再全类argmax，不知道该query真实old/new角色，不使用配额/联合分配。K1/B完全原R14，none收缩控制全K精确R14。不是旧R8条件LOO：本版物理held不进入特征权重/核估计，并以source不对称组数验证收缩。

源代理：五heldRX×两cached模拟epoch91/92×K1/5/10/20×old_count2/3/4/5（六类随机次序）×四路。shadowCCA16训练排除heldRX全部观察，每类52，support前K≤20，held固定后32，无physical overlap。source old/new比例覆盖new/old0.2至2，目标N20/6=3.33属于比例外推，不能隐去。源六类都被F见过，不能作为真novel确认。训练用合法sourceL6300物理/12600配对，不读U/V真值；最终CCA参数直接复用r12。原审查发现old_count1无法计算旧类异类带宽，已在发布前删除该源代理并同步40个均值单元，不改目标old6后端；边界K1/K2测试和原P1定点复审PASS，无其余P0/P1。

全局kappa0/5/20/none由源held选择。相对none，四路平均H至少+0.5pp、两S Q0不低超过0.5pp、每K均值不低超过0.5pp，且两S Q1>Q0。通过者取源H最高，平局较大kappa；没有通过则source-only科学停止、target0。

通过才启动P01/MIX01/首RX1-1、seed2026092701、K1/5/10/20×N2/5/10/20四严格S/Q，共80预测/评分行，全补集query。匹配原版/R14历史80行复用。全部预测固定后独立truth-last。用户目标仍两S分别Q0比原版≥5pp、Q1>Q0，Q0旧新各≥原版−1pp，逐KN完整列出；不以先导替代七RX确认。源GPU4、目标GPU0一worker，实际free检查，不动其他owner，技术失败保留产物不盲重试。

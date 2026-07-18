# D61 identity-primary共享Fisher残差追溯与预注册

## 1.动机与单一机制

D58的逐类score校准在outer-held崩溃，D59/D60表明继续选择full与block之间的协方差位置不能突破D46。历史D11、D21-M2/M6与D6b还表明：用低秩表示替代identity、训练低秩metric或扫描低秩残差权重，会在support内过拟合或回退identity。D61因此不选择rank、不替换identity、不训练参数，也不做逐类校准；它只在D46每个full/block组件拟合前施加一个类共享、恒等主导的有界Fisher残差。

对当前拟合support的类均值`mu_c`、全局类均值`mu`和类内残差，令`U`为中心化类均值矩阵的非零右奇异向量，rank仅由机器精度矩阵秩确定且不扫描。对每个模态：

`b_j=mean_c[((mu_c-mu)^T u_j)^2]`，`w_j=mean_i[((x_i-mu_yi)^T u_j)^2]`，`g_j=b_j/(b_j+w_j)`。

最终共享变换为`A=I+U diag(g)U^T`。由`0≤g_j≤1`，`A`在Fisher子空间特征值属于`[1,2]`，正交补严格为1，因此identity几何永不被替换。组件在`x'=xA`上拟合，随后把系数编译回原空间：`W=W'A^T`，截距不变；query仍只读取一套编译后的仿射状态，额外MAC和状态均为0。

## 2.与历史路线的非重复边界

- 不同于D33：不球面归一、不使用半径评分、不做旧类对角Fisher参数更新。
- 不同于D11/D21-M2/M6：无梯度、无epoch、无固定rank、无可训练低秩投影，且不以低秩分支替换原表示。
- 不同于D6b：不混合identity cosine与低秩cosine分数，不扫描`alpha/rank/shrinkage`；残差由闭式Fisher能量唯一决定并作用于统一LDA头。
- 不同于D58：所有类共享同一`A`，无逐类scale/intercept/bias。
- 不同于D59/D60：不再构造full↔block协方差插值或跨块谱收缩；D46的full/block机制保持，D61只改变其输入度量。

## 3.协议与折内闭包

- 固定receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer physical-rank held折；实际outer fit K8。
- 复用匹配`VALIDATED_ONCE/p2_min_v1`的received-IQ capsule；方法变化不触发数据重验。
- D46每个inner-LOO组件fit都只用该折train support独立重估`U,b,w,g,A`；inner-held、outer-held和query不参与变换拟合。outer full-support组件同样只读outer train support。
- 不读取clean/source、receiver、场景handle、old/new角色、query truth、真实batch类数、quota或global assignment。
- 类标签仅用于support内统一Fisher散度；公式、rank规则和增益对类ID置换等变，对old/new完全同构。
- K1因无类内能量证据，精确回退D46；K≥2使用同一闭式公式，不做K特例调参。

## 4.预注册判门

基准为当前最强D46：before92.22%、after81.67%、new84.67%、H82.33%、forgetting10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%、混淆25/8/15。

D61必须同时满足：105/105行、query0、所有outer/inner变换audit闭包；量化before/final argmax变化和margin翻转均为0；before≥92.22%、after≥81.67%、new≥84.67%、H≥82.33%、forgetting≤10.56pp；joint≥23.33%、三项floor不低于D46且至少一项严格提高；三场景联合不退化；混淆不超过D46；至少1/15个prediction变化。即使全通过也只进入后续独立开发验证，不直接运行125。

若失败，停止固定rank、残差倍数、增益指数、逐类增益、场景增益和support门扫描。结果报告必须覆盖总体、3场景、6旧类、5新类、15fold、Fisher rank/gain、混淆、量化、资源和artifact。

## 5.资源与实现

- D46 before/final各2个outer组件和各组件K个inner-LOO fit，D61对每次组件fit独立计算共享Fisher变换；实际计算次数和稠密代数MAC必须完整入账。
- 持久态仍只有编译后的int8/FP16 affine head；query额外MAC/state/optimizer step均0。
- 实现：`code/scripts/probe_d61_identity_primary_fisher_residual.py`。
- 测试：`tests/test_probe_d61_identity_primary_fisher_residual.py`。
- 输出：`automation_reports/CV-SincNet/d61_identity_primary_fisher_residual_probe_20260719/identity_primary_fisher_residual`。
- 本地`ssr-gpu`串行验证并使用detached clean worktree；本轮不访问N607。

## 6.首次执行失败与R1修复预注册

首次锁定实现`759be372`在首个真实block组件上由D43 fail closed：全局`A`先旋转288维特征，再把auto-shrinkage协方差强制截成z160/FFT96/RF32三块，所得数值矩阵触发非正定检查。未完成105行、未产生可评分结果，因此不得报告性能或据此调参；失败说明“先全局旋转、后强制分块”与D46 block组件的结构假设不兼容。

R1在不增加任何超参数的前提下改变运算顺序：D46 full/block组件先在原始合法support上按既有机制拟合`W0,b0`，再从同一fit可见support独立计算上述`A`，并编译`W=W0A^T,b=b0`。等价地，D61只对每个组件的判别系数施加identity-primary共享Fisher残差，不再改变其协方差估计坐标；因此full/block原有SPD与结构保持不变。每个inner折仍只用该折train support重新计算`A`，K1仍精确D46回退，rank/gain/阈值/权重扫描仍为0，query仍为单一仿射state且额外MAC/state为0。

R1沿用第4节全部性能门和停止条件；不得对残差倍数、左右乘顺序或block专用增益做第二次修补。新输出使用`identity_primary_fisher_residual_r1`，首次失败目录原样保留。

# D73新旧任务冲突投影联合度量追溯与预注册

## 机制定位

D62是当前协议合法的同row联合最强开发基线：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`，但注册后旧类与新类仍分别距K10目标9.78pp和7.33pp。D70–D72回顾表明，support内的生命周期行、top-2 pair门和完整头bagging不能稳定预测outer-held的新旧竞争方向。D73因此转向表示层，但不重复D36的12步软损失加权或D21-M6的低秩多臂选择：它从D42/D62强状态出发，只执行一次确定性的对角log-metric更新。

## 锁定公式

对Stage2-C全部已注册support，使用当前D42 log-diagonal度量和leave-one类中心构造原型softmax。旧类support与新类support分别得到任务损失`L_old`和`L_new`；两者都以全部已注册类为竞争集合。分别计算梯度`g_old`、`g_new`并单位化。若余弦为负，则对两个任务做对称PCGrad投影，去除相互冲突分量；否则直接等权相加。合成方向去除全维共同缩放分量后单位化，并执行唯一一次更新：

`delta=-sqrt(K/(K+D))*g_joint/||g_joint||_2`。

其中`K`是当前每类合法物理support数，`D=288`。公式不扫描学习率、rank、损失权重、阈值或场景分支。Stage2-B的before状态保持D62逐位不变；Stage2-C用新metric在全部已注册support上重新拟合一次同一D62匿名联合头，再编译为单一residual-int8/FP16状态。K1因leave-one不足而精确回退D62。

## 追溯矩阵

|要求|D73实现约束|验证证据|状态|
|---|---|---|---|
|LEO_weak-only与K语义|只读取固定单观测support特征；无新增view、增强或物理样本|support multiplicity与Runner审计|PREREGISTERED|
|Stage2-B/C同等|before完整保留最强D62；Stage2-C把旧类保持与新类注册作为两个等权任务|梯度、损失与before hash审计|PREREGISTERED|
|无query泄漏|梯度、中心、温度、metric和D62 refit全由outer-fit support生成|API与source audit|PREREGISTERED|
|类与场景边界|组内所有类同一公式；无类ID、receiver、scene、fold或结果门|置换等变测试与geometry字段|PREREGISTERED|
|冲突处理|仅当任务梯度余弦为负时做无权重的对称正交投影|first-order task audit|PREREGISTERED|
|正式量化态|query仅见一个all-registered residual-int8/FP16 affine state|量化误差、argmax与margin翻转审计|PREREGISTERED|
|地面组件|D22当前eligible=false；D73输入、更新和状态均为0|geometry/resource字段|PREREGISTERED|
|资源|新增288维闭式梯度、1个stage2c step和1次D62 refit；query额外MAC/state为0|resource verifier|PREREGISTERED|
|完整报告|总体、3场景、11类、15fold、梯度、训练、量化、混淆、资源、artifact、缺陷和比较|report完成门|PREREGISTERED|

## 与历史路线的非重复性

- D21-M6学习rank-2/4投影并以support-fold arm选择、软pair-preserve和多步优化为核心；D73没有rank、arm、选择门或多步训练。
- D31冻结共享旧metric，只更新新类suffix；D73更新一个最终对全部注册类同构的共享metric并重新拟合统一头。
- D36从较弱B3几何执行Stage2-B/C各6步、用固定权重相加CE/CVaR/preserve/proximal并可带rank-2；D73从D42/D62强状态执行一个等权冲突投影步，不使用软权重折中。
- D61是无训练的共享Fisher残差，已表现为旧类保护但新类崩塌；D73显式要求旧类与新类两个一阶方向均不受冲突分量伤害。
- D67–D72都在已拟合score/head层混合、替换、门控或聚合；D73改变support阶段的共享表示metric，query图仍不增加。

## 开发门与停止边界

固定开发单元仍为receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer physical-rank held折，outer-fit实际K8。必须完成105/105行和30个目标量化行。相对D62，只有`A`、`N`、`H`、min-A、min-N不退化且至少一项严格改善，同时`B`、`F`、三场景和混淆不出现联合交换伤害，才允许第二seed；否则标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不运行125，不扫描步长、温度、任务权重、投影顺序、按场景/类/角色门或多步版本。

即使开发单元通过，也只进入独立第二seed和后续矩阵，不能直接声称满足目标。

## 完成结果

R3真实105/105行完成且生命周期/资源闭包通过。D73总体`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`，min-B/A/N=`80.00/53.33/73.33%`，混淆`23/8/15`；15/15 outer prediction与D62完全相同。机制确实改变15个metric state并同时降低旧/新support leave-one CE，但任务梯度余弦全为正、PCGrad 0/15激活，后续D62 refit吸收了共享metric重参数化。D73相对D62性能0增益且适配MAC增加85.39%，负向关闭，不运行第二seed或125；当前最强仍为D62。

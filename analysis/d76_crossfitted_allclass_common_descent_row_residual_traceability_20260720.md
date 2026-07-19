# D76全类共同下降row residual追溯

## 需求到实现追溯表

|ID|来源章节|可验收要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D76-R1|数学机制1—3|按物理rank执行8折K−1 equal-prior automatic-shrinkage LDA，并从88个held support形成11个类CE梯度|`code/cvsrffi/stage2_d76_allclass_common_descent.py`|pending|待单元测试与probe审计|不得读取query或类角色|
|D76-R2|minimum-norm组合|以20次固定Frank-Wolfe、解析线搜索和并列vertex平均求class-simplex最小范数组合|`code/cvsrffi/stage2_d76_allclass_common_descent.py`|pending|待解析性质与置换等变测试|不得扫描迭代数或类权重|
|D76-R3|解析步长与trust cap|按逐类共同下降内积及Lipschitz上界求步长，并施加类无关Frobenius cap|`code/cvsrffi/stage2_d76_allclass_common_descent.py`|pending|待逐类OOF CE非增测试|仅退化minimum-norm点允许identity fallback|
|D76-R4|final-row集成|只在D62 final rows上编译`W'=W_D62+ΔW`，intercept不变，更新后不refit|`code/scripts/probe_d76_crossfitted_allclass_common_descent.py`|pending|待D62同行集成测试|before、D42 metric与其余链保持匹配|
|D76-R5|量化与协议闭包|target-old/new正式int8；query独立全类argmax；clean/source/query truth/role/quota/ground访问均为0|probe、测试与run artifact|pending|待协议审计与INT8/FP32审计|复用D18 `VALIDATED_ONCE/p2_min_v1`|
|D76-R6|开发单元|运行receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、7个D42同行候选|运行脚本与7个artifact|pending|待105/105闭包|actual outer-fit K8|
|D76-R7|完整性能与资源报告|同row报告总体、场景、11类、15fold、混淆、量化、MAC、状态、显存、日志和artifact|`automation_reports/CV-SincNet/d76_crossfitted_allclass_common_descent_row_residual_probe_20260720/report.md`|pending|待完整日志解析|失败亦必须详细报告|
|D76-R8|晋级门|相对D62，`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格改善，混淆无交换伤害|summarizer与报告|pending|待同row差值判定|失败关闭D76，不开第二seed或125|

## 要修复的失败

D73的共享可逆metric被D62 refit吸收；D74非可逆盲删伤害旧/新类和新类floor；D75硬安全门0/15接受，只能回退D62。D76要保留D62已验证的row-local最终边界优势，但把support-held证据直接变成连续更新，而不是候选后的二元门。

## 数学机制

在D42变换后的全部已注册support上，对每个类内物理rank`r`：

1.用`S_{−r}`拟合equal-prior automatic-shrinkage LDA`(W_r,b_r)`；
2.在每类一个held样本上计算multiclass CE；
3.对每个真实注册类`c`，聚合其K个held样本相对公共仿射residual`R∈R^{C×D}`的梯度`g_c`。

在class simplex上求minimum-norm convex combination：

`g*=argmin_{α≥0,Σα=1} ||Σ_c α_c g_c||²`。

使用20次固定Frank-Wolfe解析线搜索；并列vertex按同值平均，保持类置换等变。若`g*`非退化，则`d=−g*`是所有类的一阶共同下降方向。softmax CE对仿射行的Lipschitz上界取每类held特征最大平方范数，解析步长：

`η=min_c <g_c,g*> / (L_c ||g*||²)`。

最终residual为`ΔW=ηd`，并以`||W_D62||_F/sqrt(CD)`作为类无关trust cap；编译`W'=W_D62+ΔW`到单一int8 affine state，intercept不变。无step/rank/threshold/loss-weight扫描。

## 与matched baseline的单一主要差异

相对D62只增加一个由全类OOF CE共同定义的连续final-row residual；before、D42 metric、D62 final基头、int8编译、query规则和协议输入不变。更新后不refit，因此不会被坐标重参数化吸收。

## 预期可观察结果

- 每个target row记录8个OOF LDA、88个held样本、11个类梯度、simplex权重、共同下降内积、解析步长、trust cap和更新前后逐类OOF CE/正确数；
- active时所有类OOF CE均不增加且至少一类严格下降；退化的minimum-norm点才允许数学identity fallback；
- 若D62的剩余误差可由小幅连续row correction修复，outer A/N/H与下尾应至少一项提高且不牺牲另一侧。

## 失败与停止条件

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格提高；三场景和三类混淆不得出现交换伤害。失败即关闭D76，不扫Frank-Wolfe次数、步长倍率、trust cap、class loss权重、rank、角色/场景/类门，不开第二seed或125。

## 最小验证矩阵与边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、outer-fit K8；
- 7个D42同行候选，D76只替换target INT8/FP32 final state；
- 单LEO_weak、support-only、query独立全注册类argmax；clean/source/query truth/role/quota/global assignment/dense query graph访问0，ground int8输入0；
- 报告总体、场景、11类、15fold、before/after/new/H/F、混淆、训练、量化、MAC/状态/显存和artifact。

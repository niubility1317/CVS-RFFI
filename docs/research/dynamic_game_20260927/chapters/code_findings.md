# 动力博弈代码考证：更新语义、状态与证据边界

本章于2026-09-27完成源码只读核查，随后保存分析文档；未修改算法代码、未运行训练、未重新执行GPU测试。以下历史验收记录按原时间和原范围引用，不把历史113项通过写成本次测试结果。

## 1. 核查范围与路径约定

源码基准目录为`E:/type10-7/code/snapshots/native_dr_eg_prepare_20260914_wt/experiments/adv3b02_xuc/`。下文代码、测试和验收路径均相对该目录；行号对应本次读取的快照。交付副本位于[implementation/response](../implementation/response/README.md)，下文路径均从该目录起算；不能将同名的其他分支或工作副本当作这些行号的依据。

实施计划为快照根的`analysis/response_games_plan_20260914.md`，主要验收说明为`acceptance/response_games/README.md`、`acceptance/response_games/reaudit_report.md`和`acceptance/native_dr_eg/README.md`。计划与README说明意图，下面以实际执行代码确定语义。

## 2. 任务场、域场与U梯度权限

令身份编码器参数为\(\theta\)，域对抗头为\(\phi\)，\(D_L,D_U\)为标注源集与未标注源集上的域交叉熵，\(T\)代表分类、DAOT、FastTrust及其他已启用非对抗任务。在关注身份分支与域对抗头的简化表示下，有效场为：

\[
F_\theta=\nabla_\theta T-\lambda_E\nabla_\theta D_L,\qquad
F_\phi=\alpha_L\nabla_\phi D_L+\alpha_U\nabla_\phi D_U.
\]

这是非对称博弈：U训练对抗头，但U域交叉熵不给身份骨干对抗梯度。不能由此写成“U不给身份骨干任何梯度”，因为DAOT和FastTrust的身份目标仍可作用于身份骨干。完整网络还包含域骨干及其他任务参数，上式不是对所有损失与参数块的逐项展开。

源码证据：

- [code/cvsrffi/xuc_fusion/dr_objective.py:151–172](../implementation/response/code/cvsrffi/xuc_fusion/dr_objective.py#L151)：U学生使用`grl_lambda=0`；RC4身份目标单独存在，并支持专门移除U头监督。
- [code/cvsrffi/xuc_fusion/response_runtime.py:63–65](../implementation/response/code/cvsrffi/xuc_fusion/response_runtime.py#L63)：明确记录L编码器、U编码器为0、L头、U头四类有效系数。
- [code/cvsrffi/xuc_fusion/response_dric.py:169–186](../implementation/response/code/cvsrffi/xuc_fusion/response_dric.py#L169)：局部编码器对抗目标只用L，头目标为L加U。
- [code/model_dual_cvsincnet.py:1096](../implementation/response/code/model_dual_cvsincnet.py#L1096)：`response_explicit_domain_field`让DRIC显式域场绕开GRL。不能对GRL直接求高阶导并将其解释为真实Jacobian。
- [tests/test_native_joint.py:160](../implementation/response/tests/test_native_joint.py#L160)：验证U梯度权限及编码器乘数不改变头训练强度。
- [tests/test_response_games.py:93](../implementation/response/tests/test_response_games.py#L93)：验证任务漂移反例及非对称交叉导数。

相应交互块为：

\[
B=\partial_\phi[-\lambda_E\nabla_\theta D_L],\qquad
C=\partial_\theta[\alpha_L\nabla_\phi D_L+\alpha_U\nabla_\phi D_U].
\]

一般不能设置\(C=-B^\top\)。U输入随编码器变化，虽然正式U编码器对抗梯度为零，仍会影响头响应随编码器的变化。这也是显式分别计算B/C的必要性。

[code/cvsrffi/xuc_fusion/objective.py:38–58](../implementation/response/code/cvsrffi/xuc_fusion/objective.py#L38)接入标注任务、原生clean前向以及到达规定阶段才启用的卫星分类前向。`dr_objective.py:121–137`则表明后续`pure_game`对照只保留登记的U对抗头CE，关闭DAOT、RC4身份和域/self辅助目标。不能将pure-game与原生DR关闭的早期四格对照完全混为一谈。

## 3. SIM、完整EG与RK2的真实更新

主实现位于[code/cvsrffi/game_tracking/solvers.py:172–254](../implementation/response/code/cvsrffi/game_tracking/solvers.py#L172)。

- SIM在当前位置计算一次梯度场，提交一次正式AdamW更新。
- 完整EG先按当前完整场对全部参与参数做一次虚拟AdamW预测，在预测参数上重算完整场，然后恢复原点参数与AdamW状态，再用第二场原始梯度提交一次正式更新。
- 预测与正式梯度都经过`_install`的全局裁剪，见`solvers.py:128–134`。
- 完整EG不是“预测位置上继续做第二次AdamW”，也不是只预测域头。只预测域头属于另一个显式`head_lookahead`模式。
- RK2/Heun使用两个原始场梯度的平均，见`solvers.py:232–236`；不应与完整EG混称。
- `predictor_lr_ratio`控制虚拟预测学习率，正式主路线取1；预测步的临时学习率在更新后恢复。

令\(A(q,s,g)\)表示从参数\(q\)、优化器状态\(s\)出发，使用原始梯度\(g\)，经裁剪和AdamW更新得到新参数与状态。完整EG可写为：

\[
(q_p,s_p)=A(q_t,s_t,F(q_t)),\qquad
(q_{t+1},s_{t+1})=A(q_t,s_t,F(q_p)).
\]

第二次调用重新从\((q_t,s_t)\)出发，虚拟状态\(s_p\)不被提交。AdamW矩、裁剪和权衰减使它不能被简单欧氏SGD公式完全替代。

[tests/test_native_joint.py:57](../implementation/response/tests/test_native_joint.py#L57)覆盖不同预测比例下的一次提交与独立参考；`:75`覆盖异常后参数、优化器、随机状态和buffer回滚。

## 4. CF-EG：完整EG反事实之间的梯度插值

实际实现位于[code/cvsrffi/xuc_fusion/response_solver.py:41–90](../implementation/response/code/cvsrffi/xuc_fusion/response_solver.py#L41)。

1. 从相同原点分别执行编码器对抗开启和关闭两条完整EG轨迹，得到各自的校正原始梯度\(g_1,g_0\)。
2. 构造候选：

\[
g_\beta=g_0+\beta(g_1-g_0).
\]

3. 每个候选从相同原始参数与AdamW矩状态出发，独立裁剪、权衰减和更新。
4. 默认按\(\beta\in\{1,0.5,0\}\)寻找最大可行候选。固定β与随机β是独立消融，会绕过正常可行性筛选。
5. 关闭编码器对抗的\(F_0\)仍保留L/U域头监督，因此它不是“无域损失”。

这不是对单次loss简单调整一个系数：两个端点已有不同的EG预测位置和第二场。

源L风险见[code/cvsrffi/xuc_fusion/response_context.py:43–50](../implementation/response/code/cvsrffi/xuc_fusion/response_context.py#L43)：

\[
r_{s,c}=-\operatorname{mean}_{y=c}
\left[p_y-\max_{k\ne y}p_k\right].
\]

默认约束要求候选相对β0的每个TX乘已激活场景的平均负概率margin损伤不超过0.01。它不是逐样本保证，也不是最差RX/day风险保证。监测批在`response_context.py:24–41`按6TX乘5RX乘3day覆盖90条合法L，并轮换组内样本。

H/P仅作辅助监测，见`response_runtime.py:35–45`：H监测伪类概率，P监测候选集合概率质量；它们不被当成真实标签硬约束。

验收证据：`tests/test_response_games.py:17,33,40`覆盖有历史AdamW矩和裁剪的β端点精确等价、异常回滚、最大可行候选选择；[tests/test_response_reaudit.py:45](../implementation/response/tests/test_response_reaudit.py#L45)覆盖候选风险使用的buffer与最终提交buffer一致。

## 5. TR-EG：真实域头输入上的坐标补偿

实现见[code/cvsrffi/xuc_fusion/response_solver.py:92–121](../implementation/response/code/cvsrffi/xuc_fusion/response_solver.py#L92)与`response_fields.py:39–47`。

TR先完成双方普通预测，再在原点和预测点提取相同IQ的真实域头输入，不额外归一化特征。它通过带行列式修正的SO(d) Procrustes估计旋转：

\[
\Sigma=Z_p^\top Z_0/n+\rho\bar e I,
\qquad
R=U\operatorname{diag}(1,\ldots,\det(UV^\top))V^\top,
\]

其中\(U,V\)来自\(\Sigma\)的SVD，\(\bar e\)是原点输入二阶矩的平均尺度。R停止梯度，只改变预测域头第一层：

\[
W_p\leftarrow(1-\gamma)W_p+\gamma W_pR^\top.
\]

随后计算完整第二场，恢复原点并正式提交。拟合使用当前训练批；监测L物理ID与拟合批不重合，见`response_runtime.py:22–25`。坐标估计使用隔离eval条件，两次训练场仍走训练模式。默认γ为0.5，在E21至E60渐增。

旋转解释率可能为负，不能将其天然解释为提升百分比。当前遥测另计4次坐标估计/监测前向和一次SVD，不是零额外开销。[tests/test_response_games.py:71](../implementation/response/tests/test_response_games.py#L71)覆盖已知旋转及退化情况；[tests/test_response_reaudit.py:128](../implementation/response/tests/test_response_reaudit.py#L128)验证γ为0时包括buffer在内精确退化为完整EG。

## 6. XT-DANN：头meta梯度与编码器输入梯度分离

实际梯度链见[code/cvsrffi/xuc_fusion/response_fields.py:6–37](../implementation/response/code/cvsrffi/xuc_fusion/response_fields.py#L6)。

源L按真实TX分成互斥A/B，两侧都必须覆盖全部源域。以A拟合、B监测的方向为例，一步临时头响应为：

\[
\phi'_A=\phi-\eta_h\nabla_\phi
D_A(\operatorname{stopgrad}z,\phi).
\]

头meta目标保留二阶链：

\[
\frac{\mu}{2}\left[
D_B(\operatorname{stopgrad}z,\phi'_A)
+D_A(\operatorname{stopgrad}z,\phi'_B)
\right].
\]

编码器对抗目标则停止临时头参数梯度，保留特征输入梯度：

\[
-\frac{\lambda_E}{2}\left[
D_B(z,\operatorname{stopgrad}\phi'_A)
+D_A(z,\operatorname{stopgrad}\phi'_B)
\right].
\]

临时内环头不直接提交，不读取U隐藏TX。`response_runtime.py:14`显示XT活动步关闭普通L编码器GRL，再接入跨TX对抗场，避免简单叠加两份编码器对抗。两次EG场共享划分，各自在自身位置重算临时响应。

默认每20个接受步使用额外90条合法L；内环学习率等于当前域头学习率，μ为0.1。相同数据曝光不等于相同计算预算。[tests/test_response_games.py:80](../implementation/response/tests/test_response_games.py#L80)验证梯度归属和域覆盖；`tests/test_response_reaudit.py:37,83`验证含Dropout的零内环学习率诊断变化为0，以及诊断开关不改变loss、梯度和RNG。

## 7. DRIC-Lite：固定背景上的局部风险预算

主路径位于`code/cvsrffi/xuc_fusion/response_dric.py:108–112,162–329`。它只修正`id_backbone.t_proj/f_proj`身份末端与`adv_head`，早期骨干和其他参数保持普通DANN更新，未冻结或删除其对抗。

先从相同原点分别取得普通AdamW状态与仅关闭编码器对抗的任务AdamW状态。任务位移\(b\)是真实参数位移，不是梯度的别称。旧尾部、任务尾部及最终风险，都在“其他参数已普通更新”的相同背景下比较，见`:209–225`。因此这里的任务参照不提供全骨干反事实风险保证。

任务漂移为任务尾部变化造成的域头梯度变化，见`:218–232`；它不同于被动观测记录的整步域头梯度变化。两侧方向基最多各4维，按当前学习率平方根白化，见`:234–247`。

局部矩阵为：

\[
J=\begin{bmatrix}I&B\\C&Q\end{bmatrix}.
\]

Q来自PSD投影梯度外积、正近端项与曲率阻尼，见`:248–252`。它不是完整网络Hessian，也不是Adam二阶矩预条件。局部身份半空间预算为：

\[
A u\le0.1\max(r_{\mathrm{old}}-r_{\mathrm{task}},0),
\]

同时加入与任务位移尺度相关的球约束。预算保护的是任务在该固定背景上形成的局部风险改善的一部分。

`constrained_game`位于`:30–106`：先加谱阻尼，使对称部最小特征值至少为\(10^{-4}\)；在低维空间枚举活动半空间，并对每个活动集合的球乘子最多执行5次Newton迭代。残差阈值为\(10^{-5}\)。零半径单独精确处理。无法取得满足残差和可行性的解时显式回滚报错，不悄悄提交无约束解。

正式提交直接施加参数位移，绝不将u/v当作AdamW梯度。优化器矩保留唯一一次普通AdamW状态，任务分支状态只供虚拟参照、不混合或提交，见`:272–283`。E21至E40将普通候选的实际位移渐增插值到完整局部候选，只有满强度候选声明局部线性预算；真实非线性风险违约和线性化误差单独记录，见`:305–326`。

FR与CGD使用匹配的局部参数空间、原点独立B/C以及普通AdamW位移，不使用DRIC任务反事实或身份预算，见`:119–160`。这些是局部比较实现，不是文献算法全维逐项复现。

可支持的结论是“实现了带显式局部风险预算的响应校正”。不能据此声称全骨干风险不增加、全局收敛或泛化必然提高。

测试证据包括`tests/test_response_games.py:93,105,121`的非对称解析导数、漂移反例、联合约束残差与真实模型路径，以及`tests/test_response_reaudit.py:19,68,143,160`的球约束、已知KKT解、缓存开关一致性和整个事务回滚。

## 8. 归一化、RNG、EMA与缓存事务

|证据位置|核实语义|
|--|--|
|[code/cvsrffi/xuc_fusion/joint_normalization.py:9–22](../implementation/response/code/cvsrffi/xuc_fusion/joint_normalization.py#L9)|记录原点L/U调用实际使用的除数，第二场按调用顺序重放。仅恢复EMA统计原始状态不足以保证相同除数，因为normalizer会按当前loss更新统计再做除法。|
|[code/cvsrffi/xuc_fusion/dr_objective.py:87–119](../implementation/response/code/cvsrffi/xuc_fusion/dr_objective.py#L87)|每主步固定U、增强、教师、路由、scale原点和缓存。|
|[code/cvsrffi/xuc_fusion/dr_objective.py:138–181](../implementation/response/code/cvsrffi/xuc_fusion/dr_objective.py#L138)|局部normalizer从原点初始化，后续场重放实际除数，只保留原点的待提交scale。|
|[code/cvsrffi/xuc_fusion/dr_objective.py:206–222](../implementation/response/code/cvsrffi/xuc_fusion/dr_objective.py#L206)|接受后一次提交scale与路由历史。|
|[code/cvsrffi/xuc_fusion/runtime.py:270–291](../implementation/response/code/cvsrffi/xuc_fusion/runtime.py#L270)|一次建立ctx，求解器接受后才提交DR、EMA和原型；教师不会每个虚拟场更新。|
|`code/cvsrffi/game_tracking/state.py:13–30,54–77`|快照包含Python、NumPy、Torch CPU/CUDA随机状态、参数、梯度、optimizer、buffer及可选scaler/stateful。|
|[code/cvsrffi/game_tracking/solvers.py:228–251](../implementation/response/code/cvsrffi/game_tracking/solvers.py#L228)|第二场恢复原始随机条件；最终保留第一次前向后的buffer和RNG。|
|[code/cvsrffi/xuc_fusion/response_context.py:8–22](../implementation/response/code/cvsrffi/xuc_fusion/response_context.py#L8)|被动eval观测恢复模式、buffer与RNG。|
|[code/cvsrffi/xuc_fusion/response_replay.py:30–80](../implementation/response/code/cvsrffi/xuc_fusion/response_replay.py#L30)|只缓存确定性、无buffer的指定内建叶模块；检查实际输入、参数版本和选中参数依赖。|

缓存中的autograd独立不被当作前向独立的充分条件，detach路径仍要核对实际输入。Dropout、有状态模块和任意自定义模块不缓存。叶模块缓存计数不能直接换算成完整骨干FLOPs；外层前向控制流及函数式计算仍会执行。

## 9. 新响应路线与旧C2/C*的区别

新路线以固定调度、逐步候选评估或局部校正执行，不依赖C*恢复证据门控。[code/cvsrffi/xuc_fusion/response_config.py:53–59](../implementation/response/code/cvsrffi/xuc_fusion/response_config.py#L53)给出实际时钟：DRIC默认每4个接受步，XT每20步；CF/TR/XT预热使用完整EG，DRIC预热使用SIM。DRIC与TR在E21的渐增强度恰为0，不能把配置存在写成所有阶段都在执行非零修正。

旧C2位于[code/cvsrffi/game_tracking/controller.py:54–102](../implementation/response/code/cvsrffi/game_tracking/controller.py#L54)，根据lag、readability、identity、margin与gradient imbalance决定NORMAL、CATCHUP、CORRECT或HOLD_CURRICULUM。`xuc_fusion/runtime.py:283`表明旧路线仅在CORRECT切入EG；新joint路线按固定solver模式选择。

旧C*位于[code/cvsrffi/xuc_fusion/control.py:147–190](../implementation/response/code/cvsrffi/xuc_fusion/control.py#L147)，要求几何能力连续确认、恢复证据有效、动作连续确认、观测不超过10步、250步冷却以及每次观测最多消费一次。`xuc_fusion/runtime.py:228`每250步取得观测，因此即使所有条件有效，动作密度也受约1/250，即0.4%的结构性上限限制，不能把名义20%校正预算当作实际执行比例。

若两个旧C*实验都没有修正动作但分数不同，不能将差异归因于C*修正收益；应核对模型与数据随机流、课程、其他配置以及初始状态。控制器修正未激活，也不自动证明整行实验与别的行完全等价。

## 10. 历史验收、修复证据与局限

本次读取核实：

- `acceptance/native_dr_eg/acceptance.json`记录53项通过，正式实验0、远端动作0、科学性能未评估。
- `acceptance/response_games/acceptance.json`记录113项通过，其中53项旧回归、28项响应路线、32项再审；失败、错误、跳过均为0。正式实验0、remote actions为0、性能`NOT_EVALUATED`。
- [acceptance/response_games/reaudit_report.md:11–18](../implementation/response/acceptance/response_games/reaudit_report.md#L11)保留球约束求解、错误缓存、Dropout诊断污染、CF buffer不一致、阶段时钟、低频轮换覆盖、遥测与配置边界修复。
- `acceptance/response_games/reaudit_before_fixes.xml`是修复前8个失败的历史证据，不是当前失败状态。
- [acceptance/response_games/README.md:11–13](../implementation/response/acceptance/response_games/README.md#L11)区分108份新配置、18行主候选和8个不可自动执行的依赖模板；配置覆盖不是108项全部已运行。

113项属于2026-09-14相关历史验收，不是本次重跑。本次是文档与源码考证，复用已有验收证据，不为保存分析文档重复GPU测试。历史有界合成测试与真实模型单步路径证明可达性、数学/状态语义和配置一致性，不证明完整训练每一步均成功、长期修正非零或目标泛化提高。后续真实实验是否完成、采用哪个提交和checkpoint，应另由对应run记录、产物及独立评分证据确认。

DRIC局部预算、FR/CGD全维复现缺口、计算预算匹配和U对称场禁用的边界，在`reaudit_report.md:32`有明确记录。等计算预算及CF平均有效对抗强度匹配依赖合法源域测量；相同epoch、主步数或数据条数都不能自动替代计算匹配。

## 11. 建议上传及核对的源码、配置和证据

应核对Git已有内容后上传缺失的相关集合，保留来源和原路径映射：

1. `code/cvsrffi/xuc_fusion/`完整包。
2. `code/cvsrffi/game_tracking/`完整包，包括legacy对照、state和solver。
3. `code/cvsrffi/cross_response/`、`code/model_dual_cvsincnet.py`以及原生DAOT/RC4相关依赖。只上传response文件不能独立复现。
4. `code/scripts/train_response_games.py`及相关原生、pure、resume入口。
5. `configs/native_dr_eg/`、`configs/response_games/`、`configs/separate_controls/`及recipe参考。
6. 相关`tests/`，重点`test_native_joint.py`、`test_response_games.py`、`test_response_reaudit.py`、`test_separate_controls.py`和`test_pure_resume.py`。
7. `tools/prepare_response_games.py`、`tools/accept_response_games.py`、`tools/verify_response_games.py`、原生验收及评测契约工具。
8. `acceptance/native_dr_eg/`和`acceptance/response_games/`中的设计、验收、修复、发布证据与经甄别的后续实验统计。
9. 快照根`analysis/response_games_plan_20260914.md`。

数据、checkpoint、`__pycache__`等大体积或生成内容不属于上述源码上传集合；应保留路径引用及实际来源说明。历史文件中的“尚未训练”属于当时的范围声明，不能覆盖后续实验状态；同样，后续有训练也不能把历史本地验收改写成真实性能验证。


本次上传源码位于[implementation/response](../implementation/response/README.md)。正文行号以该快照内相对路径为准；历史验收并非本次重新运行。

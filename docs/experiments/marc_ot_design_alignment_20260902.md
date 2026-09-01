# MARC-OT/BiNOVA-D92设计落实追踪

- 状态：`LOCAL_VERIFIED / P0_P1_REVIEW_FIXED`
- 性质：`NONBLOCKING`；不改变或阻塞正在运行的r6
- 设计来源：用户提供的MARC-OT与Meta-SF-RDC组合设计报告
- 当前正式实验：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r6`
- 解释边界：r6验证MARC-OT首轮可证伪子集，不等同于Meta-SF-RDC完整版

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|T01|协议总则|Phase2仅使用合法support，query只读且truth-last|pilot lifecycle/scorer|verified|r6 smoke与分片屏障测试|r6当前3/18支持态，尚未开query|
|T02|冻结分类头|Phase1旧类头永久冻结，增益进入backbone|runner/functional forward|verified|聚焦测试与真实smoke|不训练目标头|
|T03|目标判别|CE、cross-fit、LOO、SupCon和最差类保护|`stage2_marc_ot.py`、runner|verified|聚焦测试与主Agent反向审计|R1/R2及以上消费；selection已对齐exact D92 old-only|
|T04|权重空间基库|source domain expert、低秩block bank、receiver LORO|Phase1 entry/source experts|verified|r3 bundle真实回读＋新版fit/select回归测试|新版域专家由独立select CE选择step0～N|
|T05|support联合推断|输出权重系数、block gate、block LR和不确定度|support encoder/calibrator|verified|r3 bundle与r6 smoke|仅R4/R6/R8使用bank初始化|
|T06|渐进开放|按stage开放norm/projection、t3/f3、t2/f2、t1/f1并允许拒绝适配|runner|verified|runner测试与alpha=0回退测试|Sinc未进入首轮矩阵|
|T07|receiver/channel传输|仅在校准描述空间做support-to-bank OT和统计约束|`stage2_marc_ot.py`|verified|OT几何、收敛、特征分流与零权重lazy-skip测试|属于MARC-OT描述符近似，不是完整多层类条件Bures|
|T08|物理校准|共享复数FIR/CFO/SFO/IQ失衡校准器|未建立|deferred|无正式实现|属于Meta-SF-RDC-Core后续候选|
|T09|多层源统计记忆|style/subspace/W2及receiver-class交互记忆|未建立|deferred|无正式实现|r3仅有权重bank与685D任务描述符|
|T10|receiver barycenter|基于26×6原型推断跨类别共享receiver状态|未建立|deferred|无正式实现|不得把权重bank等同于barycenter|
|T11|虚拟源特征|class-conditional virtual feature OT|未建立|deferred|无正式实现|当前OT是support描述符到bank任务描述符|
|T12|信任域|Phase1 Fisher重要度约束|runner|pending|当前仅L2参数漂移项|不能称完整Fisher trust region|
|T13|冲突控制|按网络块进行主任务优先梯度投影|runner|verified|极值稳定性与分块测试|R8消费|
|T14|优化器参数|AdamW、weight decay=1e-4、clip=5.0及分层LR|runner/config|verified|真实行为测试＋新版v2配置解析|t1/f1下界为3e-6；其余block下界保持1e-5|
|T15|性能增强|EMA/SWA、fold soup、后段SAM、plateau停止|未建立|deferred|无正式实现|设计明确属于性能完整版，不是首轮gate|
|T16|旧方法对比|MRIOR-H/B/HB及既有方法同row对照|config/未来矩阵|deferred|当前仅声明权限范围|历史结果不能倒填为同row比较|
|T17|D92交互|先完成REG0域适应，再做REG1注册适应|runner/scorer|implemented|P3使用精确D92 REG0|REG1与difference-in-differences属于后续Stage2-C|
|T18|指标闭环|P1/P2/P3、BA、floor、per-class、help/harm、资源与时延|scorer/report|pending|等待r6完成|当前没有性能结论|
|T19|域专家选择|同域fit/select物理隔离、held-out选步、全旧类覆盖、clean+LEO成对一致性|source experts/entry|verified|物理ID排除、覆盖/不足fail-closed、held-out选步与配对一致性测试|新版Phase1配置权重为0.05|
|T20|functional安全|每次调用保持冻结buffer不漂移并恢复模型mode|Phase1 functional bridge|verified|连续调用、bitwise state和逐模块mode恢复测试|fast参数集合与bank BlockSpec精确绑定|

## 当前最高风险

当前最大的科学解释风险不是协议泄漏，而是名称与机制边界：r6的`R4/R6/R8`分别表示权重bank初始化、描述符OT/统计约束和分块投影，不能解释成Meta-SF-RDC矩阵中的物理校准器、多层统计记忆和完整Fisher路线。

## 本轮优化范围

1. 已修复runner优化器参数偏差，并用真实行为测试证明配置被消费。
2. 已修复Phase1 functional bridge的buffer/mode安全边界。
3. 已实现域专家独立fit/select、held-out选步、全旧类覆盖和clean＋LEO成对一致性。
4. 已明确arm语义与设计近似边界；不热补丁当前run。

## 本地主验收

- 聚焦测试：定点修复后270项通过。
- 语法检查：6个变更生产模块/脚本通过`py_compile`。
- 配置检查：两份20260902 JSON均可解析；v2配置进入runner并冻结exact D92、K=10、AdamW参数、梯度裁剪与分层学习率。
- 差异检查：`git diff --check`通过，仅有工作树LF→CRLF提示。
- 当前计数：`verified=11`、`implemented=2`、`pending=1`、`deferred=6`、`rejected=0`、`blocked=0`。

## 独立P0/P1审查与定点修复

唯一一次独立审查结论为`P0=0`、`P1=3`，三项均已定点修复：

1. Stage2 v1曾被新优化器默认值改变。现已恢复历史语义：`weight_decay=0`、不裁剪、t1/f1下界等于旧`learning_rate_min`；仅v2显式启用`1e-4/5.0/3e-6`。
2. Phase1 v1曾被无条件切到held-out选步。现已严格分流：v1继续单batch固定最终step；v2必须声明`expert.mode=stratified_select`，才启用fit/select和paired consistency。
3. v2的D92维度与类范围字段曾可被静默忽略。现已严格要求四键和值精确为`EXACT_D92_OLD_ONLY/160/96/OLD_ONLY_6`，篡改或额外键均fail closed。

## r6技术失败与替代边界

2026-09-02 00:44只读回读发现`leo_low_elev_weak/R2`因零权重OT分支仍被旧实现计算而失败，指纹为`row_error=0.00010051`。这不是性能结果：r6保留3个已完成R0和14个仍运行单元，但无法达到18/18冻结态，因此不得打开query或评分。本地新版已用零权重lazy-skip消除该根因并有回归测试；不得热补丁、覆盖或原地重启r6，只能在当前健康单元产物保留后，以新release和新run ID执行替代实验。

## 实施交付

- 实施提交：`c816eb0b4acad8c2a59f7ffa5590d90da8b07b53`。
- 分支：`codex/binova-d92-20260829`。
- 远端回读：`VERIFIED`，实施提交与`origin/codex/binova-d92-20260829`一致。

## 完整设计差距分层

### 已进入MARC-OT首轮生产路径

- 权重空间block bank、support encoder联合输出`q/u/gate/LR`、receiver LORO、全秩backbone残差、冻结旧类头。
- R1的support CE；R2增加cross-fit、LOO、类风险和SupCon；R4增加bank初始化和分块LR；R6增加support-to-bank描述符OT与统计约束；R8增加分块主任务优先投影和普通L2 trust。
- 五折support选择、逐stage接受/回退、`alpha=0`拒绝适配、18单元support屏障、query只读和独立truth-last评分。

### 当前是近似而非设计同构

- 域专家从同一`θ0`出发并冻结head，但仍是固定25步单批训练；缺独立`expert_fit/expert_select`、held-out选步、全旧类覆盖和clean＋LEO成对一致性。
- 685D任务描述符包含identity/time/frequency、CFO/SFO proxy、PSD、RF-lite、K和mask，但不是Sinc至identity的逐层类条件统计记忆。
- 当前balanced OT连接support任务描述符与冻结bank任务描述符，不是逐类32～64个虚拟source feature的unbalanced OT，也不是类条件Bures/Wasserstein桥。
- 当前R8的trust项锚定原始参数并使用普通L2，不是以`θ0+Bq`为锚点的Phase1 Fisher trust region。
- 当前P1/P2/P3分别是冻结head、target support prototype和old-only exact D92；缺独立冻结source prototype probe。

### Meta-SF-RDC完整版尚未实现

- 共享近恒等复数物理校准器、经验贝叶斯normalization融合、receiver-style/identity/interaction多层记忆。
- 显式identity/nuisance子空间、共享receiver barycenter、类间关系保持、成对身份不变性与nuisance等变损失。
- calibration warm-up、EMA/SWA、validation plateau、fold soup、SAM/ASAM和有界Sinc开放。
- 永久meta-test receiver、难receiver/low-elevation重采样，以及元学习损失权重、Fisher半径和开放阈值。

这些项目保留为后续候选，不作为r6完成gate。若r6未达到首轮科学门槛，应根据同row失败机制选择最小下一候选，不能一次叠满完整版。

## 与既有域适应方法的证据对齐

|历史方法|已证实的机制信息|对下一版的约束|
|---|---|---|
|MRIOR-SDA|高结果包含source replay和head/backbone混合贡献，权限宽于当前Phase2|只作为机制参考或宽权限上界；不能倒填同row排名|
|SF-TAPFT＋D92|仅`t3.norm`逐维仿射适配在REG1被D92重新估计均值、协方差、尺度和统一头抵消；三场景660条最终prediction与直接D92一致|优先非仿射、能改变D92实际使用identity几何和old/new margin的更新|
|WISER-P3|18/18旧类REG0 D92评分闭合，但仅clear场景局部提高，其余场景无稳定收益；support OOF方向可与query相反|support选择必须与exact D92对齐，但仍需方向安全门，不能把support OOF当query收益|
|旧Meta-adapter|activation adapter可快速适配，但收益可能停留在临时模块|MARC-OT要求最终变化进入正式backbone权重|
|当前MARC-OT r6|权重bank、全秩残差、OT和分块投影已进入old-only REG0矩阵|先取得真实r6同row证据，再决定是否扩展物理校准、多层记忆或REG1|

## D92四状态边界

r6只回答`DA0_REG0`与各候选`DA1_REG0`的旧类域适应问题。`DA0_REG1/DA1_REG1`、新类准确率、old/new harmonic、注册效应及difference-in-differences尚未实现于MARC-OT生命周期。即使r6的REG0指标提高，也不能推断注册后收益；后续必须冻结`φ_D`，再独立训练/构造`φ_R`并用同一truth-last scorer完成四状态比较。

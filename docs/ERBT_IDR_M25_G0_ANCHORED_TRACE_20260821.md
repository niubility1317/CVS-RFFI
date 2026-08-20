# ERBT-IDR M2.5 G0锚定交叉拟合残差实现追踪

设计输入：ChatGPT参考任务《对提交`db0fa41fdfe2ea5994462a7e6620df3fe0e41d90`的继续分析与优化裁决》。该答复仅作为设计输入；`项目.md`、`p2_min_v1`和既有同row证据决定最终实现边界。

主比较基线：去RF32的D92 E0/R1。既有G1–G4完整125结果保持`DO_NOT_PROMOTE`，本轮不覆盖、不重跑。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|M25-01|核心结论、十一|保留G0/R1主分数，只叠加局部残差|`stage2_m25_anchored_residual.py`|verified_local|单元与row集成测试|禁止整体替换分类头|
|M25-02|十一|只修改G0低margin query，高margin严格保持|同上|verified_local|高margin逐query分数相等|门控阈值冻结为0.10|
|M25-03|十一|残差跨类中心化并封顶|同上|verified_local|最大logit扰动≤所选强度|残差归一化后位于`[-1,1]`|
|M25-04|十二|强度只由support留一证据选择|同上|implemented_approximation|query不可达与选择器测试|固定G0锚点＋局部证据留一；未复用D92 fold held logits|
|M25-05|十二|旧/新support角色损失均不得退化，p10真类margin受保护|同上|verified_local|双角色CE、harm与p10测试|任一条件失败回退0|
|M25-06|十三|K1/K2固定回退G0|同上、row executor|verified_local|B3 K2逐query parity|不启用残差推理|
|M25-07|十四|用query依赖的收缩类半径替代常数类别负bias|同上B2|verified_local|宽类近/远query测试|类尺度只由support估计|
|M25-08|十五|双原型需最小簇、SSE下降和jackknife稳定性门控|同上B3|verified_local|稳定双簇接受、单离群点拒绝|混合权重来自簇大小|
|M25-09|十五|双原型只进入残差，不替换G0|同上B3|verified_local|高margin与lambda0测试|避免G4全头退化|
|M25-10|十九|多原型MAC按`256*sum_c M_c`统计|row executor|verified_local|资源receipt测试|强度为0时按真实跳过计0|
|M25-11|协议|query逐样本独立、truth-last|row executor、runner、scorer|verified_local|truth-unopened row测试|query不更新任何状态|
|M25-12|用户要求|B0–B3每臂完整125|full125 runner|implemented|静态500行、每臂125|N607尚未启动|
|M25-13|分析要求|输出全部分层、四状态和资源对比|summarizer、正式报告|implemented|共享full125汇总器接线|待真实结果验证|
|M25-14|十、十八|1:4/1:1×D92/cosine归因矩阵|独立后续诊断|deferred|不进入本轮性能候选|避免扩大当前实现面；不影响B0–B3 full125|

本地验证：56项M2.4/M2.5聚焦与相邻回归通过；五个生产脚本/模块编译通过；`git diff --check`通过。独立审查初审发现3项P1：B0 parity未闭锁、task/receipt身份未核对、summary可覆盖。三项均完成定点修复，定点复审结论为`NO_P0_P1`。

当前状态：`LOCAL_VERIFIED / NO_P0_P1 / N607_NOT_LAUNCHED`。

# 原 F＋跨 RX：弱发射机优化基底

用户最新指定以已完成三种子均值85.113889%的原F＋跨RX同TX对比学习为基底。历史run为`20261016-phase1-F-supcon-e50-clean-manysig-m9-r02`，执行commit为`4df7b94d0fd8ce29056bd8236364e45afa92e07f`。该值是三种子均值，样本SD约0.379881个百分点，相对同配方F增加0.843651个百分点，不是某个最佳seed。

固定骨干与训练：三路SIG_TIME共同输出joint160；统一居中Sinc物理Hz优化与clip5；RDrop1；L-only time_down/t2.pool跨RX同TX对比；原随机L128、普通增强、LS0.01；从零50轮＝33L＋17U、每轮222更新，共11100更新；原batch_neighbor门控、ownEMA和cos保留。默认头正常联合训练，冻结后直接预测，不单训或替换头。无星地信道增强，只测clean。

当前E20＋sat CE0.68四组已完成，三个变体的整体及源固定弱类平均recall差均为负。它们只作为独立错误配方研究的负结果保留，不能代替这里的基底或否定原50轮无sat条件下的新假设。刚准备但未登记、未训练的E20 plainPL分支已由用户最新要求取消，不接着发布。

后续本机变体复用原native训练执行路径，新增机制只进入f_theta，保持原50轮配方中未声明改变的所有部分。先核对历史实际resolved args、物理角色和完整祖先，合法完整匹配结果可复用；同名F、20轮或新sat配置不算匹配。若本机迁移引入实际配方/执行差异，应建立对应匹配控制并披露环境差异，不用85.11直接相减冒充严格配对。任何checkpoint复用仍须核实源契约与继承来源；不凭模型名称免查。

弱类与其权重只由既有合法源V风险统一公式确定，不根据本轮目标域分数改权重、阈值、候选或重跑。新实验先固定候选/源规则/预算/seed/输出再发布；训练冻结后只测clean168000物理样本并独立truth-last评分。已暴露ManySig代理基准，保留所有负结果，报告逐seed、弱类与其余类、最差RX/day、Macro-F1、成本和实际seed数。

代码入口`experiments/cvs_f_crossrx50_local/baseline.py`以显式检查拒绝E20、错误阶段、sat开启、EMA关闭或外部teacher初始化，防止再次偏离用户指定的底座。正式 run `20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01` 已从实现提交 `31467086c3ced30237f3ac86e91eae68645425d7` 在本机发布。首种子 BASE／WEIGHT／GEOMETRY 三行并行，BOTH 排队；原样源规则决定是否扩展剩余两种子。每行冻结后自动 clean 预测与独立 truth-last 评分；完整配置和当前状态见 `automation_reports/CV-SincNet/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01/`。旧训练与结果全部保留。

# 参考约束残差网络：全部八行clean测试

状态：LOCAL_VERIFIED，尚未发布。固定源run `20261003-phase1-cvs-reference-residual-identity-manysig-m8-r01` 的全部8份E200模型完成冻结后，直接执行已授权的测试，不依据源或目标分数淘汰候选。

源执行版本 `8a3871650c2a35ab5c8a63d9e6d0634946292542`。两种结构各四seed，测试同168000物理query、6TX、7RX的clean视图。所有预测固定后，独立scorer连接truth，并用独立NumPy从原始预测重算。报告准确率、macroF1、分RX、分TX、四seed均值/标准差及同seed逐频减标量差值。旧native/residual_fusion预测只作同row报告对照，不回流设计。

没有适应、support、新类、卫星视图、query拟合或新训练。K和适应指标N/A。每份checkpoint加载前核对完整源物理角色、从零继承链、实际E200/10000步、架构、精度和执行版本；缺项阻止该来源，不回退旧权重。

本地17项测试已通过，包含完整168000-ID矩阵、缺行/重复/错ID/错类别/truth隔离、评分产物保护、独立重算与checkpoint来源负测。真实权重无query smoke在源冻结后的发布入口执行，当前尚未执行。独立P0/P1测试管线审查PASS；17项测试独立复跑通过，发布依赖及冻结代码一致。

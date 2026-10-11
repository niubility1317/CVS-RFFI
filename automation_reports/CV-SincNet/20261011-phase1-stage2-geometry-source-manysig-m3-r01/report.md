# 三种子源域可辨识特征几何审计

状态：PLANNED，尚未启动。

完整L_s几何、公共前导物理、独立源V读出及动作核验，冻结身份网络并拟合新动作与R辅助。

本轮设计原文：[用户报告](../../../docs/CVS_STAGE2_GEOMETRY_USER_REPORT_20261011.md)；[逐项实现追踪](../../../docs/CVS_STAGE2_GEOMETRY_TRACE_20261011.md)。

严格继承既有数据角色，只用L_s拟合，源V用于验证，未读取target。原始父模型来源由 parent_contract 和实际加载核验；不继承前轮多解耦权重。

源审计全部完成并审阅后再启动12行身份训练；每行40轮结束固定final，自动clean预测与独立评分。主终点固定B3三种子均值84%，不按target重排或选择性重跑。

当前没有新测试成绩，目标未达成。逐行预算、seed角色、路径、组件及命令以 experiment.json/configs 为准。

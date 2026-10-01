# 性能优先射频行为算子 CVS：clean 确认实验

run_id：`20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01`。状态：LOCAL_VERIFIED，尚无本轮测试结果。源域 8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部文本日志已解析，物理数据角色与 scratch 来源核实。按预登记源性能规则冻结 `rf_gmp`，不测试未选候选。性能优先，参数和计算成本只在源性能完全并列后参与比较。见[源性能冻结](evidence/performance_selection.json)。

新 4 份预测只读使用原 clean capsule，原 16 个基准和上轮 4 个残差 CVS 预测只读复用，共 24 行、168000 个相同物理 query、6 类和 7 RX。预测逐包面对全部注册类 argmax，无 truth、query 拟合、类别配额或全局重排。全部 24 份预测完整且物理 ID 对齐后，独立 scorer 才读取 truth。原配置、预测、日志和权重保持原位置。仅 clean 闭集，不增加 LEO、support 适应或新类测试。

本地 49 项聚焦检查通过，源模型与确认协议各有一次独立 P0/P1 审查，均 PASS。使用已有 VALIDATED_ONCE 数据，未增加重复数据验证。见[验证](evidence/local_validation.json)、[审查](evidence/independent_review.json)、[源完整审计](evidence/source_completion_validation.json)、[逐行配置](experiment.json)及[设计分析](../../../docs/CVS_RF_OPERATOR_HYPOTHESIS_20261002.md)。发布后默认完成测试集测试，报告全部 seed、RX、TX、F1、混淆矩阵、资源成本和负收益。测试结果只用于确认报告，不回流训练、选模或重跑。

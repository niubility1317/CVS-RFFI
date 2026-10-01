# SupportMetric 独立分析入口验证

2026-10-02，状态VERIFIED_LOCAL_IMPLEMENTATION_NOT_DISPATCHED。

新工具 `tools/publish_d92_support_metric_support_analysis.py` 只使用stdlib。运行源码、spec和原始产物不变。它只处理本次预登记r01；原运行必须完成全部4 row、160 parent、1800 path，才可root显式调用。只发布原analyzer与spec两个文件，独立输出不得覆盖原运行。普通N607用户、独立远端OID、明确路径和实际完整marker属于直接输入/交付正确性检查，不重新验证数据。

root在已激活ssr-gpu环境执行 `tests/test_d92_support_metric_support_analysis_publication.py`：49项离线用例全部通过，工具调用exit0，完整原始stdout/stderr以gzip无损保留。fixture验证RUNNING拒绝派发、完整矩阵和runtime绑定、原目录保护、精确archive清单、普通用户边界、模拟transfer/dispatch超时及不重试。它们没有调用真实SSH、数据、模型或数值拟合。测试身份为Python3.10/NumPy2.2.6/SciPy1.15.3/Torch2.1.0+cpu，native stack activation，无Conda CLI或hooks。

root另以真实record做只读输入绑定检查：原状态RUNNING被拒绝，record逐值保持不变；仅内存副本中的TRAINING_COMPLETE字面值用于核对已登记argv/paths，不代表真实运行完成，不执行publish。证据位于 `automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-support-m2-r01/evidence/analysis_route_validation_20261002`。本次未派发独立分析；真实远端分析结果、性能和资源报告仍为N/A。

成功派发也只能记录DISPATCHED_AWAITING_INDEPENDENT_READBACK；返回码0不能证明分析完成。transfer/dispatch/readback的不确定结果保留UNKNOWN及原产物，由root只读核实，不能自动重发或晋级。分析完成后才独立核对compact/summary、原runtime及完整覆盖，随后采集完整表和成本。数值closure由已冻结analyzer负责；不重拟合kernel/head/JVP/basis，不读query/源数据，不选择参数。

新数学文档 [固定旧条件与统一竞争恒等式](D92_FIXED_OLD_CONDITIONAL_COMPETITION_IDENTITY_20261002.md) 仅从公开公式推导。配对旧类下降恰为B原本正确而C被新组夺走的比例；固定旧类内部排序不保证小遗忘。新类正确还要求新条件分类正确且赢得统一竞争。这一确定性分解不承诺实际泛化或星载成本，未用于修改当前实验。

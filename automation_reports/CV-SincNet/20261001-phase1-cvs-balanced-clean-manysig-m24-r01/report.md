# 性能优先的CVS结构改进：clean确认实验

run_id：`20261001-phase1-cvs-balanced-clean-manysig-m24-r01`。状态：LOCAL_VERIFIED，尚无本轮目标结果。源域8行已完成E200/10000步，完整1600轮与80000步记录解析，源数据契约与scratch无继承核实。按用户最新优先级，以源性能最高选中`balanced_fusion`。参数/计算只在性能完全并列时参考，取消原0.2个百分点成本优先容差；原source_selection仍保留，另存[performance_selection](evidence/performance_selection.json)。本轮目标访问尚未发生。

新增4个选中候选clean预测，只读复用原16基准＋上一轮4个残差CVS预测，共24行，全部168000同物理query/6类/7RX。预测器不含truth路径、无query拟合、逐包对全部类argmax；独立scorer在24份预测完整后连接truth。旧预测与旧配置/日志/权重不修改、不重跑；新release/run/log/output不可覆盖。仅clean，不追加LEO或support适应。

代码与实际权重的完整来源、物理数据角色、类别与runtime配置在query前核对；任何来源不明/跨数据/目标污染均拒绝。默认每GPU只用一个空闲卡的评估进程，健康任务保持。原source release的训练与原选择从未热改；用户性能优先是在新target访问前显式更新的独立确认选择，不依据测试反馈。

本地25项聚焦检查PASS，其中source model5项、clean协议20项，独立审查PASS。见[验证](evidence/local_validation.json)、[审查](evidence/independent_review.json)、[源完成审计](evidence/source_completion_validation.json)、[逐行配置](experiment.json)和[设计及原理](../../../docs/CVS_BALANCED_IDENTITY_RESEARCH_20261001.md)。后续完整结果包括accuracy/Macro-F1/各RX/TX/CM/paired seed、参数/MAC/状态/时间/实测内存，以及全部负收益和声明边界。

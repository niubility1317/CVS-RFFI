# 已完成：固定合成本机软件测量

状态ANALYZED/VERIFIED。8行和24组配对计时全部完成，完整分数逐位相等。PAIR中位耗时减少8.09%至11.65%；C_ONLY增加5.76%至31.01%，必需B prior没有减少。原C默认路径和正在运行的实际基准保持不变。

完整K表、口径与范围见[成本报告](../../../docs/D92_MARGIN_SINGLE_QUERY_COST_MEASUREMENT_20261001.md)，运行证据见[evidence/result_readback_20261001.json](evidence/result_readback_20261001.json)，完整小型结果位于results/measurements。峰值为全进程working set，不是星载或每row峰值；所有真实准确率与训练改善项N/A。

以下保留预登记与历史状态：

当前LOCAL_VERIFIED：固定测量CLI的8项软件正确性测试全部通过；尚未实际测量。实际AI文本文件名为measurements.jsonl/measurements.csv，配置为源码PLAN与build_states，不采用占位名称。

# Margin单样本prior复用：固定合成本机计算测量

状态PLANNED，唯一launch owner为root。固定8行：旧类6、新类20，K=1/5/10/20，PAIR与C_ONLY；8个固定评分输入、1次预热、3次计时。输入是纯内存合成评分展开，不是模型训练、真实target评估或星载测试。

真实Phase2准确率主基准仍由既有run执行，本测量不热改或重启该基准。本测量所有准确率、H与三阶段变化均为N/A；只有完整scores等价与本机软件耗时、真实工作量、进程资源可报告。

配置、seed角色、模型来源、不适用项、每row独占输出、日志与精确命令见[experiment.json](experiment.json)。启动前绑定实际已推送source OID，完成后独立读回全部8行和终态，失败保留产物。

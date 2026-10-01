当前LOCAL_VERIFIED：固定测量CLI的8项软件正确性测试全部通过；尚未实际测量。实际AI文本文件名为measurements.jsonl/measurements.csv，配置为源码PLAN与build_states，不采用占位名称。

# Margin单样本prior复用：固定合成本机计算测量

状态PLANNED，唯一launch owner为root。固定8行：旧类6、新类20，K=1/5/10/20，PAIR与C_ONLY；8个固定评分输入、1次预热、3次计时。输入是纯内存合成评分展开，不是模型训练、真实target评估或星载测试。

真实Phase2准确率主基准仍由既有run执行，本测量不热改或重启该基准。本测量所有准确率、H与三阶段变化均为N/A；只有完整scores等价与本机软件耗时、真实工作量、进程资源可报告。

配置、seed角色、模型来源、不适用项、每row独占输出、日志与精确命令见[experiment.json](experiment.json)。启动前绑定实际已推送source OID，完成后独立读回全部8行和终态，失败保留产物。

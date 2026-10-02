# CVS 加性坐标实际使用：源域冻结消融预登记

状态 PLANNED，尚未发布或读取本轮实际权重/源数据。固定8个own scratch E200模型，每个5种内部干预，共40行、每行完整V27000/90单元；无optimizer/augmentation/target/reselection。source选择已冻结，结果只用于机制诊断，不能回改本轮选择。

频率参考只置零两圆周输入，相干度参考只置零质量输入；全部坐标参考保留conditioner bias，zero_delta另去掉全部加性项及偏置。相同输入/核心/头及源权重，所有模式全部六类统一argmax；不是硬件TX/RX干预或因果身份恢复。加载前核对完整物理源角色、继承、E200、d6c29b2b实际commit与FP32，全部payload/strictstate通过后才构建源IQ。只迭代V，L/U不迭代；V不拟合。每模型所有干预后state逐tensor不变，选择前后不变；全40行/3600单元/CM/F1/日志独立分析。

一个矩阵launch owner在空闲GPU顺序执行，独占新run/output/log；技术失败保留，不自动重发或干预其他进程。本轮明确source-only，clean/LEO/support/SFT/新类/D92不适用。目标没有回流，当前总体性能及真正RFF physics aware目标尚未完成。

19项局部检查及一次独立P0/P1审查PASS。当前指标N/A。[检查](evidence/local_validation.json) · [实际源完成与输出不存在核实](evidence/source_preflight.json)。

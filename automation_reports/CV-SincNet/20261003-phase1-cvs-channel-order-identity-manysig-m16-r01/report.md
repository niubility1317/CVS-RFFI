# CVS信道补偿与非线性交互：源实验

状态LOCAL_VERIFIED，尚未发布训练。用户于2026-10-03授权依据新架构落地实现并跑实验。4种结构各4个modelseed，共16个从零训练模型；原单一CE、E200及数据/优化器/预算保持不变。

对照为channel_capacity、channel_compensated、channel_dual与channel_order。所有候选保留原始浅层RFF主干；补偿及交互位于新增的全采样率辅助路径。只用既有shallow四份源元数据作为控制，不继承权重。实际结构和参数见[设计及判定](../../../docs/CVS_CHANNEL_ORDER_EXPERIMENT_20261003.md)、[逐行配置](experiment.json)。

固定20行源记录的四seed平均0.5V+0.5最差源RX最大者胜出；容量对照也可胜出。新候选胜出后只对其4个冻结模型做clean测试，复用36份旧冻结预测形成40行矩阵；全部预测固定后独立truth-last评分。若旧控制胜出，复用它的完成测试。源报告不接收目标评分。

当前输入为equalized=1、center256、unitRMS，结果不直接证明原始多径鲁棒性或TX/RX分离。历史确认benchmark已有暴露，不称首次盲测。没有LEO、额外增强、support、SFT或新类。

本地模型/协议验证及一次独立P0/P1审查已通过，见[本地证据](evidence/local_validation.json)与[公开输入烟测](evidence/public_cpu_smoke.json)。训练、冻结、测试和评分尚未完成；低性能不触发技术停止，所有负结果保留。

## 远端启动VERIFIED

16个任务均已完成至少1轮，分布于8张RTX3090，每卡2个。进程父子关系、CWD/argv、实际配置、日志增长、单一CE与无增强均已独立读回。执行提交`8d9600a432f9dcb80e2bfcc9dc90cbb1847e1908`。训练/测试结果尚未完成，源checkpoint按固定E200选择。

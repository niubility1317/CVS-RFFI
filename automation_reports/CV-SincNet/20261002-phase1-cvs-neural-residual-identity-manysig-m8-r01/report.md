# CVS可学习复卷积残差：浅深结构纯CE实验

状态PLANNED。两候选×四seed，共8个scratch模型；固定E200/原物理数据/单一CE/无增强，保留现有有效控制。只改变可学习残差卷积深度，不增加训练策略。

[具体设计](../../../docs/CVS_NEURAL_RESIDUAL_20261002.md) · [完整诊断依据](../../../docs/CVS_NETWORK_DIAGNOSIS_20261002.md) · [无数据CPU验证](evidence/local_cpu_smoke.json)。

固定源选模后，仅选中新候选才测试clean；控制胜出则复用已有控制测试，保留所有负结果。

发布VERIFIED：N607独立读回8个健康GPU进程、实际参数与源契约，远端无数据CPU smoke通过。正式代码commit`f929f4d1116640474d324844e2f0824f97ca7dd0`。仅源训练运行，尚未访问新query或取得性能改进结论。见[启动读回](evidence/launch_readback.json)。

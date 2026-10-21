# 完整 CRIC＋DURA：跨配置对比及监督／PL控制

**PLANNED / NOT LAUNCHED**。当前交付为完整设计，尚无新增跨配置训练、预测或时延结果。

逐行矩阵唯一登记见 [experiment.json](experiment.json)。17契约、153个F实例（3复用/150新训练）；36个G来源（1复用/35条新谱系）。F0已有三Fseed/单G的七景结果由独立分析run复算。

详细方法、源/目标RX数量、标注控制、预算、权限、15项F消融、7项G新设置、时延5行、ETA及未完成项见 [设计说明](../../../docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md)。

当前缺项：新数据契约F/G适配launcher、全部G教师来源和逐阶段预算展开、聚焦验证与一次P0/P1审查、实际发布与逐行运行。原63000硬编码恢复器不能直接套用新契约。不得误标RUNNING；原360行对比任务继续由原owner运行。

## 本次交付

IEEE实验章节已编译为9页，新增七景表与机制／时延设计；无未定义引用或溢出，新增相关页面已目视核对。 [PDF](../../../paper/phase1_experiments_ieee_20261020/main.pdf)，[完整源码与数据ZIP](../../../exports/phase1_external_dg/20261021-cric-dura-experiments-design-r01.zip)。新跨配置训练与新时延尚未启动；本次不更改现有360行健康比较任务。

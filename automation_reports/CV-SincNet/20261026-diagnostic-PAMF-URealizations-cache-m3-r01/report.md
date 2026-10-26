# 三份额外源 U H/G 缓存

ANALYZED，独立产物复核VERIFIED。三份均完整覆盖原U56700×三景，新增paired H/G共2090188800字节。独立核实全部六个IQ数组shape/dtype/有限值、物理索引/IDs与原U顺序一致、U metadata无y；worker记录G参数及全部registered buffers不变。三个worker和控制器已正常退出。没有F训练、分类评分或目标访问。

实际代码`8f0c94480f8f352189531fb499794763e1006438`；共享真实G hidden-U2 eager/graph smoke PASS，3项聚焦检查、唯一P0/P1、archive传输比较及remote compile通过。此前路径格式preflight失败在创建run/release前，经独立确认未创建后修正required路径并复用同一archive，未重复启动。

三份fresh各G缓存逻辑170100包，三份共510300，另worker warmup共6、共享smoke4；graph内部warmup/capture未冒充独立物理数据。每份walltime430.47至436.72秒，包含加载/CPU信道/恢复/写盘，不是部署延迟基准。冻结原G、receiver/day hardware和场景参数，仅改变预登记信道namespace；旧fixed缓存未覆盖，fresh不评分择优。

这是三个数据实现，不是三个学生种子；不能证明半监督或原raw clean/raw六景/同G目标提升。下一独立四条F匹配训练复用全部fresh与旧r0，各五个完整Uepoch；原损失、剂量、接受集合、默认头和预算保持。见[完整缓存单元](source_cache_cells.csv)、[独立复核](cache_result_evidence.json)、[启动证据](launch_evidence.json)、[预登记](preregistration.md)。

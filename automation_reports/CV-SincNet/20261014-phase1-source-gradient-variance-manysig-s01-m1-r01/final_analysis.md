# 浅中层源梯度方差：完整只读结果

run 20261014-phase1-source-gradient-variance-manysig-s01-m1-r01；执行代码 c0f068f5463e34d0ad94aec8d59c44ce33e6a283。固定model seed2026092701，M12旧门控＋cos final50/11100来源，180个合法L_s源包（6TX×3day×5RX×2），各RX36包。CPU实际1.8813秒，180次真实源前向；无新训练、分类头拟合、optimizer更新、U/target读取或V前向。V共享cache仅物化后丢弃。

| 实际参数路径 | raw归一化方差差异 | TX/day均值去除后 |
|---|---:|---:|
|t1.pw.weight|2.455803|1.714985|
|t2.pw.weight|1.938690|1.373195|

归一化差异为各RX坐标方差与RX均值之差的均方，除以均值方差向量的坐标平方均值。各RX样本方差使用 n−1=35 分母；这是平衡分层样本的描述统计，不保证源总体无偏，也不把TX/day均值去除当作因果解耦。只在同一层的真实参数坐标内比较，不用不同参数单位的raw数值排名。

| 路径/形式 | RX1方差trace | RX3 | RX4 | RX6 | RX8 |
|---|---:|---:|---:|---:|---:|
|t1.pw.weight/raw|0.201860|2.558046|20.219302|0.014742|3.503991|
|t1.pw.weight/TX_day_centered|1.227410|2.322890|15.369083|0.781856|3.080958|
|t2.pw.weight/raw|0.216035|3.472910|24.026398|0.020363|5.170089|
|t2.pw.weight/TX_day_centered|1.490420|3.017830|18.361747|0.986388|4.391059|

本次填补了以往norm/cosine无法回答的真实跨RX梯度方差证据。它不证明Fishr训练有效，也不证明接收机信息必须删除；只有一个模型seed、每物理cell两个样本，不以180个样本冒充180个模型seed，不凭此扩大λ或追目标分。用户当前优先任务已转为SIG_TIME与统计信息综合，本诊断不自动将Fishr堆入组合。

实际求导对象仅t1.pw3456和t2.pw6912，总10368特征坐标；默认cosine6×160/scale30提供每样本CE(.01 smoothing)，不求分类W梯度、不新建辅助分类器。全部参数/buffers逐元素不变，autograd.grad不累积.grad。原始梯度、选中源physicalID和loss保存在远端gradients.pt，保存后独立重读重算全部方差与结果完全一致；小型聚合在results.json。

一次直接P0/P1及单项路径修复确认PASS、本地采样与方差手算/6负例/真实native无查询求导PASS；单release归档SHA本地远端一次一致、remotecompile和真实CP zeroIQ noquery smoke PASS。CPU进程自然退出，source-only root ANALYZED、无failure、终态artifact独立回读VERIFIED。没有停止/重启其他健康训练，未读取target分数。

完整raw产物定位：/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-gradient-variance-manysig-s01-m1-r01/SOURCE_GRADVAR-s2026092701；raw gradients.pt、checkpoint及数据不进入Git。实现、预登记、审查和本次结果/报告提交并推送，远端OID独立核对。

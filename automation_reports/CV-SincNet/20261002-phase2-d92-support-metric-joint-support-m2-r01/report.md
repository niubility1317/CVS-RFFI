# 物理 support 度量与 LocalRidge 联合诊断预登记

状态 READY。独立 core 的75项及新入口/分析器/控制的116项合成验证通过，共191项；只读输入元数据与隔离源码导入已核实，尚未发布或启动。此候选由数学结构定义，不扫描参数组合，也不读取 query 成绩选模。固定 Phase1 和地面原型，practical residual/post_sync/noeq/25 MHz。

保留 BranchLocalRidge 解析分类器、自由截距、新类 head 和 barrier gate。新 adapter 使用已存地面原型差字典的精确秩与物理基 U，名义最多 5 坐标，实际秩 r≤5；不继承现行 ProtoFrame 或历史方法的 adapter。旧类 B 从零开始，C 只继承本次同路径实际 B，并冻结该 B 的旧条件函数。全部注册类统一竞争，结构冻结不保证旧类胜率不下降。

完整导数包括训练核与交叉核两端、Ridge 系数与截距、新 head、gate 和阈值。度量 M=实际 UᵀU+类均衡预测 Fisher，保留 RMS 类 CE 的完整 GGN 曲率 G；方向解带 dᵀMd≤0.25 的二次子问题。每阶段至多 1 次更新，从 η=1 开始，最多 12 次固定折半，按同一真实 support RMSCE/Armijo 判断接受，不以二次模型下降替代真实目标。

固定 2 模型×2 cohort，共 4 row/160 parent/1800 path；旧类 6，K=1/5/10/20，新增 0/2/5/10/20。实际 receivers19-1/20-19、practical_high/practical_low_urban、support seed2026092711。现有 VALIDATED_ONCE capsule/split/物理 support 不变，producer 的更大缓存矩阵另行保留。源域样本、源逐样本特征、query 及其 view/标签/角色/配额均不进入拟合。

outer-held OOF/proxy 是 support 信息诊断，不能写成 query 性能。parent K 与实际 train K 分列，K1 held 指标 N/A。R0 对照为原 BranchLocalRidge，C 独立重拟合；新方法为真实顺序 B→C。全部四 row 与声明路径关闭后独立分析，每轮给出配对 A、B、C旧/C新、完整 K×新增类数、gain/drop/absGap/H和必要分层，不把理想目标变成每轮硬门槛。

basis 每 row 只由冻结地面 Q 构建一次，归档普通方法数据证据并真实单列构建、绑定与 Gram 使用成本。Fraction 操作 guard65536（源码循环上界≤27000）；整数位宽 guard65536 和 secular guard128 为明确有限执行上限，不是精度选模参数，不声称覆盖所有病态输入。head guard沿现行100/64/160 MiB。技术失败保留已耗工作和 partial state，健康 row继续；不自动增限或重试，不修改现行 query。

详细文本和结构化日志保留实际损失/权重/步长/梯度/计数/耗时；源验证因 Phase2禁读记 N/A。记录实际可训练参数、训练/推理耗时、RSS、factor buffer、常驻状态和传输 bytes；星载实测/native wire/未测项 N/A。浮点 KKT 与几何包络仅是有限数值证据，完整 head/JVP 严格区间认证尚未建立，不声称泛化或算力优势。

本次相关验证与预登记审查已完成，无未解决P0/P1；37模块/46文件隔离导入，四row预检VERIFIED且输出未存在。验证记录见 [implementation_validation](evidence/implementation_validation_20261002.json)。后续实际runtime commit、PID、resolved config和新方法成绩必须由产物读回，不从READY推断运行。

配置审查发现的非阻断记录已修正：固定预测产物指向fixed_predictions.jsonl，记录中的旧验证pending文字更新为本次191项通过。检查时状态与审查原文保留，不触发重复审核。

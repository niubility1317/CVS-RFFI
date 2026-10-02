当前状态：ANALYZED。完整4row/160parent/1,800路径、20文件已读回，合法support OOF含新类96任务A旧62.64%、B旧71.04%、C旧64.89%、C新50.18%、H56.00%；R0 H58.42%，未显示综合改善。详见[完整三阶段、K×新增类数与实测成本](support_diagnostic_20261002.md)。这是support诊断，不能作为query性能。以下保留历史预登记、执行及原r01失败记录。

# 物理 support 度量与 LocalRidge 联合诊断预登记

状态 RUNNING。独立 core 的75项及新入口/分析器/控制的116项合成验证通过，共191项；只读输入元数据与隔离源码导入已核实，一次发布及独立启动读回VERIFIED，完整实验尚未完成。此候选由数学结构定义，不扫描参数组合，也不读取 query 成绩选模。固定 Phase1 和地面原型，practical residual/post_sync/noeq/25 MHz。

保留 BranchLocalRidge 解析分类器、自由截距、新类 head 和 barrier gate。新 adapter 使用已存地面原型差字典的精确秩与物理基 U，名义最多 5 坐标，实际秩 r≤5；不继承现行 ProtoFrame 或历史方法的 adapter。旧类 B 从零开始，C 只继承本次同路径实际 B，并冻结该 B 的旧条件函数。全部注册类统一竞争，结构冻结不保证旧类胜率不下降。

完整导数包括训练核与交叉核两端、Ridge 系数与截距、新 head、gate 和阈值。度量 M=实际 UᵀU+类均衡预测 Fisher，保留 RMS 类 CE 的完整 GGN 曲率 G；方向解带 dᵀMd≤0.25 的二次子问题。每阶段至多 1 次更新，从 η=1 开始，最多 12 次固定折半，按同一真实 support RMSCE/Armijo 判断接受，不以二次模型下降替代真实目标。

固定 2 模型×2 cohort，共 4 row/160 parent/1800 path；旧类 6，K=1/5/10/20，新增 0/2/5/10/20。实际 receivers19-1/20-19、practical_high/practical_low_urban、support seed2026092711。现有 VALIDATED_ONCE capsule/split/物理 support 不变，producer 的更大缓存矩阵另行保留。源域样本、源逐样本特征、query 及其 view/标签/角色/配额均不进入拟合。

outer-held OOF/proxy 是 support 信息诊断，不能写成 query 性能。parent K 与实际 train K 分列，K1 held 指标 N/A。R0 对照为原 BranchLocalRidge，C 独立重拟合；新方法为真实顺序 B→C。全部四 row 与声明路径关闭后独立分析，每轮给出配对 A、B、C旧/C新、完整 K×新增类数、gain/drop/absGap/H和必要分层，不把理想目标变成每轮硬门槛。

basis 每 row 只由冻结地面 Q 构建一次，归档普通方法数据证据并真实单列构建、绑定与 Gram 使用成本。Fraction 操作 guard65536（源码循环上界≤27000）；整数位宽 guard65536 和 secular guard128 为明确有限执行上限，不是精度选模参数，不声称覆盖所有病态输入。head guard沿现行100/64/160 MiB。技术失败保留已耗工作和 partial state，健康 row继续；不自动增限或重试，不修改现行 query。

详细文本和结构化日志保留实际损失/权重/步长/梯度/计数/耗时；源验证因 Phase2禁读记 N/A。记录实际可训练参数、训练/推理耗时、RSS、factor buffer、常驻状态和传输 bytes；星载实测/native wire/未测项 N/A。浮点 KKT 与几何包络仅是有限数值证据，完整 head/JVP 严格区间认证尚未建立，不声称泛化或算力优势。

本次相关验证与预登记审查已完成，无未解决P0/P1；37模块/46文件隔离导入，四row预检VERIFIED且输出未存在。验证记录见 [implementation_validation](evidence/implementation_validation_20261002.json)。后续实际runtime commit、PID、resolved config和新方法成绩必须由产物读回，不从READY推断运行。

配置审查发现的非阻断记录已修正：固定预测产物指向fixed_predictions.jsonl，记录中的旧验证pending文字更新为本次191项通过。检查时状态与审查原文保留，不触发重复审核。

实际不可变runtime为`3c3eb6c513aaf20fe84514dcbeab3584bad1f238`，supervisor1276233/start_ticks14215164；2026-10-01T22:42:14Z只读PID/argv/cwd/environment/resolved config均匹配。rx3两row PID1276296/1276297 TRAINING_ON_SUPPORT，rx1两row PENDING。原始源码与spec冻结，不热改，不重发。完整四row诊断尚未关闭，没有新方法准确率结果。

两个已启动row的冻结地面字典均精确秩5；各构建一次，测得0.2081s/0.1986s，各26519次Fraction调用，basis数值buffer各32000B。普通研究证据JSON为442565B/396128B，是卫星本地可派生字典证据，不能当作新增星地传输或整方法训练成本；峰值RSS、全部拟合/推理与星载成本待完整产物，未测项N/A。

独立分析入口已预登记，状态PREREGISTERED_NOT_LAUNCHED。原运行保持RUNNING，结果仍为N/A。只有原不可变runtime的全部4 row、160 parent、1800 path完成，且原complete.json关闭全部声明工作后，root才可一次发布只含analyzer与原spec的两文件分析包。分析输出位于独立新目录，不改原始训练、预测、日志或数值状态，不重复拟合head、kernel、JVP或Fraction basis，不读取query或源域样本，不依据结果修改本轮方法。

分析固定报告配对A/B/C、完整K×新增类数、旧类适应提升、注册后旧类下降、新旧绝对差与H。OOF与proxy分别报告；parent K与实际train K分列，K1无held评估记N/A。各模型与receiver分层、逐阶段参数、真实耗时、峰值RSS、常驻状态、basis构建/绑定/Gram工作和传输口径独立列出。真实query泛化、星载能耗及未测量项不由support诊断推断。分析失败保留partial产物，不自动重试或部分晋级。

<!-- SUPPORT_METRIC_INDEPENDENT_ANALYSIS_PREREGISTERED_20261002 -->

独立分析发布工具的49项离线测试通过；真实RUNNING记录在任何Git/SSH或目录创建前被拒绝，分析尚未派发。2026-10-01T23:08:38ZPID/argv/cwd/environment读回VERIFIED，当前两rx3训练、两rx1等待，完整marker仍不存在。新增数学说明给出固定旧条件后的精确竞争损失恒等式，不构成性能保证，也不改变本轮参数。

<!-- SUPPORT_METRIC_ANALYSIS_ROUTE_VALIDATED_20261002 -->

2026-10-01T23:51:44Z独立metadata读回VERIFIED：全部4 row、160 parent、1800 path完成，supervisor与四worker均正常退出。原runtime3c3不变；完整marker的query_access/truth_read/source_sample_access均false。记录转为TRAINING_COMPLETE，原预登记独立analysis r01仍未发布。此处未读取support成绩或query，不声称方法改善。

<!-- SUPPORT_METRIC_FULL_TRAINING_COMPLETE_20261002 -->

独立analysis r01已发布并读回FAILED，child1324356/wrapper1324346，29.09秒后在结构性Trial arithmetic tolerance changed检查退出；无summary/完整诊断表，无query/源样本读取或重拟合，原训练全4/160/1800仍TRAINING_COMPLETE。当前只核对原保存容差与分析重建公式，禁止放宽容差或改变方法；没有自动重发或重训。

独立scalar-r02已预登记，原r01 FAILED记录保持不变。根因是verifier以NumPy eps重建tol后，比例成为np.float64，被仅接受Python int/float的_equal类型检查拒绝。三个入口数学式均为tol=128ε max(1,|f0|,|ftrial|,|rhs|)，无需扩大容差或改训练。独立clone仅对内部比值加float，字节唯一差已验证；真实非零trial analyzer12、query scorer6、发布11个必要用例通过，原始raw失败与通过均保留。新的analysis-scalar-r02输出/2文件源包未发布；必须全4/160/1800保存档案闭合后才诊断成绩。

scalar-r02已唯一发布，sourcee0eb4041；2026-10-02T00:32:12Z独立proc/startup/argv/cwd/environment读回VERIFIED_RUNNING，wrapper1341024/child1341030，原fit3c3和全部保存状态不变。此时完整分析产物仍无，未读诊断或query，不以dispatch/进程正常声称分析完成。

<!-- SUPPORT_METRIC_COMPLETE_SUPPORT_RESULT_REPORTED_20261002 -->

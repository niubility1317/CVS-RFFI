# PrototypeTransport-LocalRidge support入口

状态：TECHNICAL_RECOVERY_VALIDATED；r01为FAILED，r02尚未prepare。本文记录入口合同，不报告实验收益。正式pytest与同一160 parent pilot的登记、启动均由主Agent统一执行；本子任务未读query、历史评分或实验总索引，未拟合真实数据。

`tools/evaluate_d92_prototype_transport_probe.py`复用现有source-only冻结Phase1 support缓存绑定：capsule、checkpoint SHA256、model seed、物理split和practical residual/no-equalization协议须一致。不会加载checkpoint或更新encoder，不读取source样本。输出R0、R_transport_seq和R_transport_reset三条固定路径，旧类6个，K为1/5/10/20，新增类数为0/2/5/10/20；模型seed×cohort沿用已声明的4 row与160 parent。

物理K1仅做数值检查；proxy trainK1保持精确R0。B从零开始，C_seq继承同一row/物理旧support的B参数与最终旧原型，C_reset使用同一准备对象、零参数和零anchor。Nnew=0直接复用B，不新增C拟合。每个inner head独立估计inner-train原型与几何，合法inner-held标签属于训练目标。outer-held只在拟合结束后评分；B/C旧类记录使用同一物理ID配对，C的每个样本对全部注册类竞争。A与B−A保留N/A，B−B0不冒充地面适应提升。

方法只调用`prepare_prototype_transport_training`及`fit_prototype_transport_local_ridge`。10个存储参数、9个有效自由度；参数及trial日志与真实费用来自core。最多4次归一化投影梯度迭代，每次最多3次Armijo试探；13次目标前向、39个内头、24次伴随三角求解只是每个三折阶段的上界。汇总不要求固定满步，不添加最好步、外层反馈选择、额外诊断拟合或重跑。

实际计数逐层记录：baseline/inner/final head及分解、目标前向、cache反向、接受与拒绝trial、trial尝试、优化器迭代、原型构造/距离、准备原几何距离、transport前向、最终评分调用与物理样本数。准备共享一次收费，接受cache的反向不增加head。完整持久状态字节包含adapter、prototype、head和继承原特征/原始support状态；另外记录准备内存、耗时、线程和峰值RSS。未测部署包、传输字节或GPU显存为N/A，不根据80 B参数大小声称完整部署大小。

`trained_transport_stage_count`仅统计至少接受一次更新的阶段。初始目标与零梯度反向仍按真实调用收费，不因该字段为0而归零；完整费用来自逐阶段objective/backward计数。事件中的`trial`保留core的Armijo试探索引，外层proxy anchor索引另存为`outer_trial`，两者均独立核对。

输出保留`training.log`中的详细文本与完整`fit_trace.jsonl`、`training_events.jsonl`。紧凑JSONL/CSV删除大数组但保留beta/eta范数、梯度、损失分量、计数、状态、缓存引用和停止原因；全向量留在完整流。source validation明确N/A，原因SOURCE_ACCESS_FORBIDDEN。

`tools/summarize_d92_prototype_transport_probe.py`先验证全部4 row的完成状态及source缓存绑定，再读取outer-held轨迹。它重新计算配对准确率、转换、margin和H，核对物理fold隔离及原型继承、跨折逐类RMS风险、proximal项、投影KKT、每个Armijo条件、最后接受参数、停止原因、cache/计数和完整事件覆盖。完成marker本身不能替代这些证据。最终输出完整120行K×新增类数×诊断×路径表，以及model/cohort和receiver/scene分层；适应、注册下降和新旧差距的10/1/3个百分点仍是描述性方向。

两个测试文件仅使用确定性合成数组，覆盖K1、N0、共享B/C准备、固定三路径、proxy精确R0、失败上下文保存和日志紧凑化。summary测试另覆盖篡改风险/anchor/trial/计数/停止/原型/事件的拒绝，以及不完整pilot不得打开outer score流。

正式验证暴露并修复了两处入口问题。首轮12项中11项通过，正常summary核对失败：独立eta零和检查遗漏冻结公式中的`5*eta_bound`，原用`128*eps64`，现与core冻结的`128*eps64*5*(log(2)/2)`一致；投影KKT与篡改拒绝仍完整保留。第二轮入口与driver共27项中26项通过，正常summary事件核对发现外层`trial`覆写core的Armijo `trial`；入口现保留core字段，外层索引另存`outer_trial`，两套坐标分别核对，未跳过完整事件比较。

两次失败的完整pytest证据分别为`E:/type10-7/.codex_tmp/pytest_utf8_1790762368169490200.stdout`及`.stderr`、`E:/type10-7/.codex_tmp/pytest_utf8_1790762500600874000.stdout`及`.stderr`。最后仅重跑受影响的8项summary测试和1项日志integration测试，9项全部通过，耗时3.61 s；证据为`E:/type10-7/.codex_tmp/pytest_utf8_1790762791182069900.stdout`及`.stderr`。结合此前通过且未改变的4项evaluator测试，当时13项入口测试均有对应版本的通过证据；主Agent另确认core 27项与driver 14项通过。该次验证完成时真实160 parent pilot尚未运行。

随后r01（`20260930-phase2-d92-prototype-transport-support-m2-r01`，runtime `cdde6b366`）出现真实JSON输出技术失败。四row均在完成5个K1数值parent后，首个K5完整parent的`json.dumps(record, allow_nan=False)`报`TypeError: Object of type bool is not JSON serializable`。主管PID 93368已退出，所有row均FAILED，失败产物保留。诊断证据仅来自`E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r01/results/failure_diagnostics/*.log`，未读取历史query或评分。

代码中的来源为NumPy标量类型传播：Armijo容差曾使用NumPy eps，比较可产生`np.bool_`；core回调经安全转换，公开state audit原使用不转换NumPy标量的容器展开，故完整parent写出时暴露该类型。core已单独修正容差/比较的Python标量边界及公开audit转换，未改变公式或冻结配置。入口增加严格`json_native`转换，只将NumPy scalar/array和容器变成原生JSON类型；布尔与数值保留原类型，未知对象和非有限数值明确失败，不使用`default=str`。转换覆盖完整parent/event、紧凑日志、startup和marker写出，防止紧凑scalar过滤遗漏NumPy flag。

新增真正流式integration测试使用6旧类、2新增类、K5确定性合成数据及NumPy labels/classes/physical IDs，实际调用evaluator写出完整/紧凑JSONL、文本日志和3个CSV。测试公开audit的NumPy bool/int/float保持Python bool/int/float，完整与紧凑事件对应一致，另覆盖未知类型和NaN拒绝。正式验证由主Agent串行执行；本修复未热改r01、未自动重试、不据性能改参。

主Agent实际验证结果：真实writer单项1/1通过，耗时7.36 s，证据为`E:/type10-7/.codex_tmp/pytest_utf8_1790764079439220700.stdout`及`.stderr`；完整入口15/15通过，耗时9.96 s，证据为`E:/type10-7/.codex_tmp/pytest_utf8_1790764153519320100.stdout`及`.stderr`。修复通过后保留r01失败状态和全部产物，尚未prepare或启动r02；后续同算法技术恢复由主Agent单独登记和执行，不以r01成绩选参或重跑。

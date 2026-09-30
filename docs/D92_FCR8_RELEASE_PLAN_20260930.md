# FCR8与LocalRidge联合方法的固定support试验

状态：ARTIFACTS_COMPLETE，完整四row/160parent已结束并独立核实，性能分析尚待完成。新run为`20260930-phase2-d92-fcr8-support-m2-r01`，group为`d92-fcr8-support`；已发布；完整结果尚待收集。主Agent是唯一launch owner。实际配置及四row的完整矩阵由`configs/d92_fcr8_support_20260930.json`维护。

## 数学改动和证据边界

方法依据见[冻结设计与推导](D92_JOINT_AFTER_MC_DESIGN_20260930.md)和[核心实现](D92_FCR8_CORE_20260930.md)。LocalRidge继续作为唯一最终分类器，可微解析求解器把分类间隔风险反传给adapter。固定DCT/GELU字典避免两层因子共同更新的尺度歧义；当前合法support字典薄SVD仅建立优化坐标，不作为预测归一化器。

令当前outer-train的字典矩阵为H，H/√N=PΣRᵀ，保留数值可辨识的列，W=RΣ⁻¹。学习U=U_anchor+ZWᵀ。精确算术下，平均投影与饱和之前的残差位移平方为`||Z||²`，因此函数近端项是`0.5||Z||²`，梯度为Z，不能额外除以N。浮点白化误差与原坐标重构残差均实际记录。这是本项目的函数坐标设计，不能称为完整Fisher自然梯度，不能推导query遗忘或准确率保证。

最终完整interaction映射是`φ=(b,a,b⊗a)`，维数123616；不是单独的张量积。原距离与适配距离各占一半。原始旧类B状态以原U精确继承，C只重新建立当前support坐标并从Z=0开始；不投影丢弃旧U的零空间分量。reset只是初始化对照，教师仍来自实际B。最大活动参数为736×8=5888，固定V0和物化U各47104B；参数数目减半不能说明总计算节省。

每阶段最多4次归一化梯度更新，每次最多3次回溯试探；所有试探都计费。接受必须同时满足Armijo、总目标非增和固定物理松弛下的旧类保持风险要求。步长与函数位移正则由同一冻结结构统一定义，不进行rank、学习率或损失权重组合搜索。固定字典减少表达能力，可能造成失败；这是尚待真实support验证的结构假设。

## 数据和三阶段报告

固定practical residual、post_sync、noeq、25MHz。复用现有`p2_min_v1/VALIDATED_ONCE`接收数据与物理support划分；不重建、不重复数据验证。沿用固定source-only scratch final200基座所产的未适应support缓存，preflight核对实际身份与来源。运行不加载encoder checkpoint，也不复用以前训练过的目标adapter、head或EMA；只有当前row、当前物理旧support的B→C继承。

不读取源域样本、源域逐样本特征、query IQ/truth/score或历史目标成绩。H/W只使用当前训练权限内support的无标签字典值；每个inner头/教师的带宽、中心化和迹来自自己的inner-train。inner-held标签用于训练，不是独立验证；outer-held只在预测固定后评分。

完整试验是四个model/cohort rows，每row40parent，共160；旧类6个，新增0/2/5/10/20，K=1/5/10/20。两个固定model seeds、两个receiver、high与low_urban场景，不选择性重跑。true K1没有独立监督，仅做数值；proxy trainK1不训练。新增0时直接复用B。

报告完整K×新增类数、B0/B/注册后的旧类与新类/H、注册旧类下降、逐task绝对新旧差，以及model/cohort和receiver/scene分层。当前support试验缺少地面适应前A，A与B−A为N/A；B0不能替代A。该试验不是最终query性能或新增独立数据验证，不据此直接晋级或挑选较好对照。

## 实际计费和验证

准备3168次，阶段4576次；最大936信息阶段、3744接受更新、12168内目标前向、36504内头、936最终头、1728教师头、3168基线头，共42336头与44928伴随三角求解。实际SVD和字典物理求值按已执行次数记账，没有按stage数推算。零更新阶段计真实准备/前向/反向；initial-inner是inner总数的解释分项，不重复加。C共享prepared时只计一次。

CPU-only，两lane、每lane两BLAS线程，不干预其他GPU训练。完整CVS文本、实际参数/损失/梯度/接受与拒绝、compact JSONL/CSV、完整动态rank的H/W/Z/U与方向NPZ保留；源域验证为N/A并说明访问禁止。记录实际训练/推理时间、峰值RSS、数值state和档案字节。地面新增原型/统计传输为0B；完整部署包、星载/GPU项未测为N/A。函数坐标上的路径界不保证最终特征或query位移。

核心19/19：`.codex_tmp/pytest_utf8_1790778450928803500`。入口/汇总/调度27/27：`.codex_tmp/pytest_utf8_1790779028837391100`。唯一实验身份修正后，仅受影响调度16/16复测：`.codex_tmp/pytest_utf8_1790779078011027300`，不是额外16个不同测试。混合零带宽新增1项通过（1790779489929461400），原汇总7项在1790779456170264700均通过；合计47个不同相关检查。极端subnormal逆谱不可表示时显式报错；没有静默跳过或把数值失败算作有效学习。独立P0/P1审查见[D92_FCR8_P0_REVIEW_20260930.md](D92_FCR8_P0_REVIEW_20260930.md)。

本run不因低性能停止。技术故障只记录所属row，保留健康row和原产物；无自动重试、无热修改。完成160parent后才由独立analysis release做完整汇总。预登记、启动与结束状态维护原experiment.json/report.md/events.jsonl及索引。发布采用已推送Git版本，启动后独立读回PID/argv/cwd和产物，不凭退出码判断外部成功。

实际runtime commit为`33673fe02a857d790aa69a1ace17fdb9638099d0`，supervisor PID247876，首批worker PID247888/247889。启动证据为`evidence/readback_1790779962176104900.json`。Git受限环境凭据错误已经通过普通push恢复，远端OID独立核对一致；没有重复launch或热修改。

ARTIFACTS_COMPLETE/VERIFIED：FCR8完整四row/160parent结束，原supervisor及worker均退出；完整marker/state/实际计数独立读回一致。实际更新2673次、头拟合30708次、latent SVD648次。未读query/源样本；完成不代表性能改善。下一步单次独立support分析与完整训练机制诊断，root sole owner，目标ACTIVE。

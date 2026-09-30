# MC-Residual8联合方法的固定support试验

状态：LOCAL_VERIFIED，本地相关检查及独立P0/P1审查完成，尚未启动。run ID为`20260930-phase2-d92-mc-residual8-support-m2-r01`，group为`d92-mc-residual8-support`；主Agent是唯一launch owner。实际运行配置与逐行矩阵由`configs/d92_mc_residual8_support_20260930.json`维护，不另复制完整矩阵。

## 方法与权限

采用[冻结联合设计](D92_JOINT_AFTER_PROTOTYPE_DESIGN_20260930.md)与[数学依据](D92_JOINT_FINETUNING_MATH_FOUNDATIONS_20260930.md)。BranchLocalRidge是唯一最终分类器；合法目标support监督训练11776参数的rank8跨分支adapter，解析头完整反向。顺序主线固定为R_MC_seq，R0与R_MC_reset_init为对照，不根据结果切换主线。

Phase1保持固定。只消费已有source-only scratch final200所产生的未适应support特征缓存；缓存checkpoint身份及训练来源沿用已核实记录，preflight逐row核对实际身份。运行不加载encoder checkpoint，也不复用任何此前目标适应state。B从U零、V固定DCT开始；C只继承当前row、当前物理旧support的B状态。合法B监督继承明确记录，不能称为干净独立验证。

不读取源域样本、源域逐样本特征、query IQ/truth/score或历史目标成绩。地面原型/摘要本候选没有实际依赖；新增地面统计传输为0B，完整部署包和星载测量仍待验证。研究与实现worker保持QUERY-BLIND；训练程序不消费总实验索引或root历史分析。

信道固定为practical residual、post_sync、equalization=false、25MHz。输入继续复用原VALIDATED_ONCE数据，只核对缓存身份、文件可用性及输出防冲突，没有数据重建或重复数据验证。

## 完整矩阵与计费

四个model/cohort rows，每row40个support parent，共160。原旧类6个，新增0/2/5/10/20类，K=1/5/10/20。使用既定两个model seeds、两个receiver cohorts的已冻结support选择；不按表现挑row或选择性重跑。

K1没有独立内持出监督，仅做精确恒等数值诊断，不伪造K1收益。K>1用物理ID折隔离的support OOF与既定proxy诊断，内held标签用于训练、outerheld仅评分。A缺失为N/A；B0是原support LocalRidge，不能当作地面适应前A。

最大工作量为936信息阶段、3744接受更新、12168完整内目标前向、36504内头、936最终头、1728教师头、3168共享基线头，共42336头、44928伴随三角求解。所有拒绝试探计费；没有更新的合法信息阶段仍计准备和反向，不能用updated-stage数掩盖成本。B首批内头预付在preparation.inner_head中，initial_inner仅为解释分项；C教师另计，两条对照真实共享prepared时才能记实际复用。

四次迭代、每次至多三次试探，固定步长1/8、回溯1/2。实际接受同时满足Armijo、总目标在声明浮点容差内非增、保持风险不超过起点加物理松弛。此约束不能保证遗忘<=1个百分点；理想目标继续是逐步提升方向。

## 产物与结论边界

每row独占output/log。完整float64参数、梯度、方向和试探状态保存为不可覆盖NPZ，JSONL记录逐状态引用；完整标量事件、CVS详细文本、compact JSONL/CSV及NPZmanifest保留。source validation为N/A、理由SOURCE_ACCESS_FORBIDDEN。状态数组保留原路径，摘要完整扫描，不截断坐标或只采样训练曲线。

CPU-only，两lane、每lane两BLAS线程；不占用或干预其他GPU训练。记录实际准备/教师/训练/评分耗时、真实head/两通道adjoint数量、峰值RSS、常驻state和NPZ档案字节，GPU及星载项未测写N/A。参数量不等于计算节省。

完成四row全部160parent后，独立analysis release汇总完整K×新增类数、B0/B/注册旧类/新类/H、旧类下降与逐task绝对新旧差、分层和资源。该结果属于support诊断，不是query最终性能或新增独立数据验证。若只内部loss下降而外层结果仍退化，记录机制失败，不换参数组合挑选较好路径。

本run不因低性能停机。技术故障保存所属row与其他健康row产物，没有自动重试；新恢复run须按现有技术恢复流程说明原因，不能覆盖或热修改健康实验。

# LocalRidge与受旧类间隔约束的跨分支Residual8监督adapter联合适应及新类注册

- run_id：`20260930-phase2-d92-mc-residual8-support-m2-r01`
- group_id：`d92-mc-residual8-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

LocalRidge为唯一最终分类器；合法support训练rank8跨分支非线性切向残差U/V共11776参数，保留半份原interaction；折内旧教师间隔保持与物理松弛约束，B后C顺序继承。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 联合方法、数学来源与输入权限

固定Phase1及已核实source-only scratch final200原support缓存，运行不加载encoder/checkpoint；不继承历史目标适应状态。当前B从U零/V_DCT起步，仅当前row合法旧support训练状态传给C。来源/物理数据契约与选择规则未改变，preflight已逐row独立读回缓存身份VERIFIED，不重复验证数据。

BranchLocalRidge为唯一最终分类器，11776参数的跨分支adapter通过解析头完整伴随监督训练；R_MC_seq为固定主线。R2-D2/GEM/adapter论文映射和公式见docs/D92_JOINT_FINETUNING_MATH_FOUNDATIONS_20260930.md；rank8/κ/有限预算是设计选择，不是假称理论最优，不执行参数组合扫描。参数投影后原Armijo可能接受目标微升的独立数学反例已修正，增加实际总目标非增条件，参数/矩阵/试探预算不变。

状态详见docs/D92_MC_RESIDUAL8_RUN_PLAN_20260930.md。160support parent、旧6类/new0,2,5,10,20、K1,5,10,20，A缺失N/A；B0不替代A。全部state/gradient/trial NPZ坐标完整保存，报告完整事件及实际教师/双伴随开销，不新增成员hash链。新增地面统计0B；星载/部署项未测为N/A。未启动、没有新性能结果。

VERIFIED: Single math-driven joint MC-Residual8/BranchLocalRidge candidate locally tested and independently reviewed; projection-induced objective-increase P1 fixed, no parameter grid. Complete support matrix registered; input identity/output preflight VERIFIED. Root sole launch owner; not launched or claimed performant.

VERIFIED: MC-Residual8/LocalRidge single candidate launched once at runtime commit 191d7df111a58aa8435473582db42902f9b8d3a0. Independent post-state matched supervisor175794 and two child PID/argv/cwd, with12 completed support parents. No query/source sample/encoder access; two CPU lanes, no performance result yet.

MC完整训练机制collector聚焦测试通过；仅新增只读分析工具。实际160配置汇总待运行结束，不更改训练runtime。

ARTIFACTS_COMPLETE/VERIFIED：MC联合四row/160parent完整结束，主管及worker均退出，独立complete/state/物理配置计数读回一致。实际更新3544次、实际头拟合22257次。未读取query或源域样本；运行完成不代表性能改善。下一步独立support汇总与完整训练机制诊断；root唯一owner，目标ACTIVE。

VERIFIED：完整MC support独立汇总已单次启动，analysis release d92_mc_residual8_analysis_20260930_r01，PID203687/argv/cwd独立读回匹配，分析commit ccda587647a09bad5c766383c86dea886e8dabc1。本地等待handle72223仍有效；summary尚未产生。训练已全部结束，禁止重复publish或重复analysis。

VERIFIED：MC160 parents完整support分析及训练诊断完成。OOF96新增类任务相对R0：B -0.173611pp、C旧 +0.095486pp、C新 -0.023438pp、H +0.011084pp；绝对新旧差距增加0.184896pp。保留BranchLocalRidge，不晋级query。A=N/A。936信息阶段全部实际更新，保持硬约束未激活，不能解释为保护约束阻止训练。完整120行矩阵和4576阶段/17604曲线/1944教师折已保存。

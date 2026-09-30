# BranchLocalMargin固定局部核平方间隔与单样本代理support诊断

- run_id：`20260929-phase2-d92-branch-local-margin-support-m4-r01`
- group_id：`d92-branch-local-margin-support-development`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ARTIFACTS_COMPLETE（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保持LocalRidge训练折内核几何，比较最强错误类平方间隔目标；四臂逐折从零拟合，完整物理OOF及全部support内1-shot anchors。

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

## 本轮目标与比较口径

本轮直接基线是冻结的BranchLocalRidge。候选BranchLocalMargin保留其五块特征、训练折内局部径向核、带宽和trace匹配，改用最强竞争类平方间隔目标；margin、物理求和损失权重及RKHS正则系数均为1。四臂在完全相同的训练折和proxy anchor上重新拟合，原BranchRidge与BranchInteraction只作描述性对照。

沿用全部4个模型seed、4个目标RX、3个场景、5个support seed、K=1/5/10/20及新增类数0/2/5/10/20。旧类固定6个；K表示每个注册类的support物理样本数，总support=(6+新增类数)×K。共4800个parent，其中1200个真实K1仅有数值诊断，3600个parent有物理OOF，全部42000个anchor构成单样本代理。

标准OOF及单样本proxy分别按parent等权汇总，每个parent K=5/10/20相对LocalRidge要求新类及H严格提高、旧类退化不超过1个百分点；每K旧/新/H是否全部严格为正另行报告。所有新增类规模、RX×场景、模型seed及support seed分层完整保留，不增加全格正向门槛。只有旧对照改善而直接基线未改善，不满足本轮推进条件。

候选仅消费原有合法support缓存，所有折内状态从零拟合。新增source载荷、地面统计和特征前向均为0；未改变Phase1、received观测、capsule或物理划分，不触发数据重验证。独立验证按用户要求暂缓；本轮不是独立query泛化证据。

数值求解固定float64、最多1000个完整sweep，同时检查原始/归一化primal-dual gap与KKT残差。技术失败保留状态、已完成阶段和日志，不以held性能停止、改参或选择性重跑。实际逐sweep损失、正则、梯度、证书、求解状态、耗时写入结构化及紧凑日志；学习率和源域验证为N/A并注明原因。

当前状态：预登记完成，尚未启动。根Agent为唯一测试与launch owner。

LocalMargin support-only full matrix preregistered, not launched; direct baseline LocalRidge; source/query access disabled.

## 本地验证与发布准备

96项核心、入口、调度与汇总测试通过；其后仅新增首个sweep日志中的实际核参数，定点测试通过。覆盖独立凸优化对照、精确行块解、物理隔离、退化核、固定原控制、单query不变性、负测、失败状态保留和完整汇总口径。首轮两项失败分别来自合成文件读取守卫拦截库元数据及尚未生成的配置，修复后完整通过。

合成N=26/364/520、C=26分别在543/32/45个sweep取得收敛证书；开启tracemalloc和回调时耗时7.752/4.885/9.010秒，跟踪分配峰值3594864/12826368/20449290字节。这不是实际target耗时或RSS估计。对应头部数值状态158944/2224800/3178272字节。单样本代理可能占用较高求解成本，完整运行将如实报告。初始合成脚本使用错误标签类型，在拟合前被拒绝；证据保留，正确整数标签版本通过。

远端只读preflight已确认8组缓存绑定、身份、资源与新目录无冲突。正式运行仍以发布后startup记录的实际commit为准。

VERIFIED:96 focused tests and metadata logging test pass;3 synthetic sizes certified; query/source rows0; LocalMargin full support matrix ready for authorized launch.

## 启动核实

VERIFIED RUNNING: supervisor348892 and4CPUchildren argv/CWD match release5b7319acd; first trueK1 numerical parents complete, measured sweep logs growing; no source/query/GPU access.

首次读回已见4个lane的训练日志，包含实际tau、gamma、损失分量、primal/dual、gap、KKT、active数量与耗时。其余4个lane按原CPU队列等待，不启动重复run。

## 后续比较工具准备（不改变当前运行）

Benchmark tooling ready, not launched: 324 synthetic tests passed plus 2 focused reruns; independent entry review and root summary/audit review have no open P0/P1. Support runtime unchanged; complete support evidence required before paired query benchmark.

最新只读证据：`evidence/readback_1790696933018702700.json`。四个rx3 lane完成88至89/900个诊断单元；其余四个rx1 lane等待，未出现终止标记。此进度不是完整性能结论。

VERIFIED_all_8rows_4800parents_complete_analysis_pending_no_query_access

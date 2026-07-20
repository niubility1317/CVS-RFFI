# D81全面测试与125稳定性screen

## 实验登记

- 实验ID：`d81_comprehensive_125_20260720`
- 登记时间：2026-07-20 HKT
- 操作者：Codex
- 目标：停止新方法研发，对已锁定D81执行协议有效的全面测试；第一阶段完成`5 receivers×5 confirmation seeds×5 slices=125 jobs`稳定性screen，每个job评估3个`LEO_weak`场景。
- 方法：`D81 ground_nuisance_cauchy_center`。地面int8组件只读，用类中心化地面谱构造固定干扰子空间；target support对所有新旧类使用同一Cauchy稳健中心变换；query逐样本面对全部注册类独立评分。
- 比较对象：每个job同row的注册前D81旧类状态、注册后D81全注册状态，以及同一truth sidecar上的identity-only single-qKNN、ProtoNet CDA、最强合法target-support-only基线和direct ADV3B02锚；若某matched列尚未在本批落地，必须在结果中明确标记缺失，不能用跨row历史值替代。

## 125矩阵与数据

- receiver：`20-1,3-19,7-14,7-7,8-8`。
- confirmation seed：`713102,713103,713104,713105,713106`。
- slice：`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`。
- 每个job内部场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 数据根：N607`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`。
- 数据状态：D18母缓存30/30 cells已`VALIDATED_ONCE`，93600个物理样本对应93600份固定received IQ，跨场景物理ID重叠0；125只读取其中5×5 cells和已登记support前缀，不重建信道、不改变物理ID、scenario assignment、support/query split或`p2_min_v1`。
- K嵌套：K1取K10 support序列第1个物理样本，K5取前5个，K10取前10个；new5/new10取20类注册表前缀，new20取完整注册表。
- query：每TX每场景20个独立固定received-IQ样本；预测artifact先密封，独立scorer随后连接truth sidecar。query不得用于拟合、选参、早停、路由、回滚或状态更新。

## 协议与声明边界

- `protocol_schema=p2_min_v1`；无clean/raw/source样本访问，无query truth/role Oracle/真实batch类数/类配额/global reassignment/dense query graph。
- D81候选、公式、地面谱rank策略和所有超参数在seed713101开发cell后锁定；本次confirmation、K5/K1和new10/new20结果不得回流调参。
- 地面组件当前状态仍为`UNVERIFIED`和`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`。因此本批即使性能良好，也只能记为开发/稳定性证据；125不能替代完整`K∈{1,5,10,20}×new∈{2,5,10,20}`正式确认矩阵。
- 若完成但性能未达门槛，状态写为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不能写成项目完成。

## 本地版本与实现计划

- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`；共享主工作树存在大量无关改动，本批在隔离worktree`E:\type10-7\code\snapshots\d81wt`按路径实现、验证、提交，再精确同步。
- 根目录`E:\type10-7`不是Git仓库；本报告同时镜像到Git承载面，根目录副本只作现场实验记录。
- 复用既有sealed row pipeline生成每个job的before/after enrollment和apply package；不重复建设数据准入机制。
- 新增D81全K query evaluator：从after support按场景拟合一次锁定D81；同一fit产生old-only before state和all-registered after state；分别对before/after query逐样本评分。K1走既有D42 unit-covariance和D62 exact fallback，D81稳健中心在单样本下必须严格恒等。
- 新增125 thin launcher和汇总：固定125笛卡尔积、8个分片、每job独立输出、禁止覆盖、逐job事件日志、全量coverage和同row指标闭合。

## N607预检与资源计划

- 2026-07-20 10:23 HKT direct preflight PASS：host=`dell-DSS8440`，项目根可见，8张RTX3090均0%利用率、约10MiB显存。

## 本地验证记录

- `python -m py_compile`通过：`stage2_d81_query_evaluation.py`、D81 125 launcher、D81汇总器及两处复用脚本。
- 聚焦回归测试36/36通过：row pipeline、125 launcher、125汇总、D81地面干扰稳健中心及D81 probe。
- 合成D81真实拟合审计：K1/new20的before/after中心位移均为0，`k1_unit_covariance_fallback=true`；K5/new20最大中心位移0.076613，未触发K1 fallback。两者均无query参与拟合。
- 本地环境未安装`ruff`；本批以`py_compile`、聚焦`pytest`和`git diff --check`作为静态与回归证据。`pytest`结束后的Windows临时目录清理出现已知`WinError5`，测试进程退出码仍为0，且36项测试均已完成。
- 本地真实K1/new20烟测在打开support/query前即被authority拒绝：已签名authority锁定N607绝对cache路径，而本地镜像路径不同。该失败没有生成性能结果、没有访问query，也不表示D81或数据失败。保留现场目录`E:\type10-7\automation_reports\CV-SincNet\d81_comprehensive_125_20260720\local_smoke_k1_new20`，真实烟测转到authority所绑定的N607路径执行。
- 定向进程检查未发现相关训练或Phase2实验；`/dev/sdb`约11TiB，总占用2.8TiB，可用7.6TiB。
- SSH命令结束后本地`ssh.exe=0`，到N607 TCP22的`ESTABLISHED=0`。
- 计划GPU：8个shard分别固定GPU0–7，每shard串行运行15或16个job；任一GPU同时最多1个本批进程，低于每GPU最多2个训练实验的上限。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/source`。
- 计划输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_20260720`。
- 计划log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d81_comprehensive_125_20260720`。
- authority：为5×5 receiver/seed cell各生成一个unsigned lock，本机生产Ed25519私钥仅在本地签名；只把signed envelope同步回N607，私钥绝不上传。
- exact服务器命令、同步文件SHA、PID和逐GPU分配在本地验证与25个authority完成后补入本节，再启动125。

## 成功、失败与停止条件

- 技术完成：125/125 job完成、8/8 shard PASS、无覆盖/重复/缺失，所有before/after预测artifact和score绑定闭合。
- 每job必须保留同row：`old_acc_before_increment,old_acc_after_increment,seen_new_acc,H_old_new,average_forgetting,old_adaptation_gain`、全部逐类、三类混淆、逐场景、时延/MAC/显存/状态量和量化审计。
- K10门槛：after-old≥92%、min-old≥88%，new5≥92%、new10≥90%、new20≥86%。K5相对matched K10核心指标下降≤3pp。K1总体与每receiver旧类适应增益≥0，且相对同旧类query上的direct ADV3B02达到目标要求时才可通过。
- 任一协议、seal、hash、support/query不交、单观测、query隔离或K1恒等审计失败即fail-closed；不以部分job性能作整体声明。

## 结果

待125完成后补充：完整125-row结果表、逐receiver/seed/slice/scenario/类别分布、matched比较、置信区间、资源、异常、缺陷、最终判定和完整确认矩阵缺口。

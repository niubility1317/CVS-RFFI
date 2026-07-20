# D91跨折共识地面sigma头部实验报告

## 预注册设计

- 实验ID：`d91_crossfit_consensus_sigma_margin_20260720`
- 初始状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`
- 目标：提高压缩地面原型在头部域适应中的收益稳定性，保留D87降低旧类遗忘的信号，同时抑制其对新类的偶发伤害。
- 证据起点：D87相对D89仅4/15个outer row改变；3行提高旧类，其中rain/fold1同时损失10个百分点新类，clear/fold2只损失10个百分点新类。15/15行support sigma目标均下降，但support正确数与argmax均不变，故不能用support准确率或目标下降硬门选择D87/D89。
- 单一机制差异：保留D87的5,816B压缩地面v2组件、p90半径、rank13 sigma几何、20步类对称head优化；新增8个physical-rank OOF折的初始sigma梯度方向一致性。每折梯度先按Frobenius范数归一化，以所有非对角余弦均值裁剪到`[0,1]`作为残差倍率，不设可调阈值。
- 创新点：地面原型不再只提供“更新方向”，还与目标support的跨物理样本稳定性共同决定“信任多少”；这是方向共识收缩，不读取query表现，不按old/new角色选择，也不扫描门限。
- 效率假设：只在support侧增加8次初始sigma梯度和一个8×8 Gram矩阵；query仍为单个INT8 affine head、额外query MAC为0、持久状态不增加。
- 数据与声明边界：复用匹配`VALIDATED_ONCE`的`p2_min_v1`数据；单物理样本单LEO弱观测；反事实sigma view只读取固定received IQ数学表示；query test-only；无clean/source、query truth、role Oracle或class quota访问。v2组件仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，本轮强制非晋升。
- 运行矩阵：锁定development seed、K10/new5、3场景×5fold×7候选，共105行；INT8与FP32 matched同时保留。
- 成功门：相对D89注册后旧类`A>82.78%`且seen-new`N>=84.67%`，所有floor和混淆不恶化；若只降低遗忘但损失新类，或15/15预测不变，则淘汰，不进入seed2/125。
- 停止规则：不根据outer结果回调共识公式或增加阈值；完整报告同row总体、三场景、15行、逐类、混淆、量化、资源、机制强度和缺陷。

## 版本与执行计划

- 根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，实现工作树为`E:\type10-7\code\snapshots\d81wt`。
- 核心：`code/cvsrffi/stage2_d91_crossfit_consensus_sigma_margin.py`
- probe：`code/scripts/probe_d91_crossfit_consensus_sigma_margin.py`
- 测试：`tests/test_stage2_d91_crossfit_consensus_sigma_margin.py`、`tests/test_probe_d91_crossfit_consensus_sigma_margin.py`
- 环境：`ssr-gpu`
- 执行位置：本地锁定cell；不使用N607、不占用GPU、不建立SSH连接。
- 输出根：`E:\type10-7\automation_reports\CV-SincNet\d91_crossfit_consensus_sigma_margin_20260720\crossfit_consensus_ground_sigma_margin_head`
- 预期artifact：`training_log.jsonl`、`predictions.jsonl`、`predictions.receipt.json`、`D91_PROBE_METADATA.json`、完整性能汇总和本报告最终段。

## 启动与审计修复记录

- attempt0后台包装器已实际启动，但初次进程检查错误地只匹配了Miniconda路径，漏掉仍运行的Python进程；随后相同锁定命令启动attempt1。attempt1在最终写入时命中防覆盖保护并退出，未改写attempt0 artifact。attempt0随后形成105行、receipt和D91 metadata，证明模型运行完成；该并发过程不改变数据、配置或公式。
- 对attempt0作交付前审计时发现：预测已使用共识收缩残差，但D91 audit错误继承D87未收缩残差的哈希、最终support CE；资源账本也遗漏了为共识重复执行的8次LDA fit。该artifact因此只作实现诊断，不作为最终性能证据。
- 修复范围仅限证据闭包：重新计算收缩后残差哈希、support sigma/clean CE与argmax审计，并把重复8次LDA fit计入适配MAC；共识公式、模型输出路径、阈值（仍为零个）、20步D87拟合、数据和候选矩阵均不变。retry2使用新输出目录完整重跑。

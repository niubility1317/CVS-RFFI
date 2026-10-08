# CE＋urban中低仰角星地增强与信道泛化固定24行

- run_id：`20261008-phase1-urban-ce-manysig-m24-r01`
- group_id：`cvs-reference-ce-urban-channel-generalization`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

6组×4seed：CE、原星地增强、urban、受限通用线性信道扰动、urban＋扰动、urban＋扰动＋卫星分支抽样。全部scratch，固定预算与测试矩阵。

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

## 设计与判定

本轮按用户最新要求，以CE＋星地增强为主线，重点为urban中低仰角，同时检查跨接收机泛化。6组×4seed，全部使用更新的reference_response骨干、从零训练、E200×222更新、同一余弦学习率。L/U/V物理角色分别6300/56700/27000，U标签隐藏且CE不使用U梯度，V只读。固定E200，全24行冻结后再测试，不用测试成绩筛选、调参或选择性重跑。

|组|作用|
|---|---|
|ce|仅CE基线|
|legacy_leo|CE＋原登记high/mid/low_urban增强|
|urban|仅改为urban课程：早期high_urban，E41后mid_urban/low_urban|
|receiver_dg|CE＋受限线性信道扰动，隔离通用泛化收益|
|urban_dg|urban增强＋受限线性信道扰动|
|urban_dg_thin|上组＋卫星辅助分支50%抽样，保留主CE每步训练|

通用扰动不拟合任何target统计：对L中50%的样本施加随机复数FIR回波，延迟1/2/4采样点，总回波幅度不超过0.15，前40轮线性增加强度，并保持每条IQ的RMS。其余样本不变。不添加发射机非线性、PA或IQ不平衡扰动。这个限制只界定扰动范围，不保证身份信息不受影响，是否有益由固定对照证明。

Fast Trust＋RC4的历史结果支持保留干净样本、逐步增加难度和谨慎利用受扰样本的研发方向；本轮CE-only主线不启用其教师、伪标签、H/P路由、校准和额外loss。历史结果来自不同协议与架构，不能把旧权重或旧增益移植到新结果中。

所有星地组卫星CE系数0.68，E80起生效。thin组E80前跳过未进入loss的卫星分支，E80后独立按q=0.5保留整个卫星批次，保留时卫星CE乘2。主CE始终存在。已核实实际模型不含BatchNorm。卫星损失项获得期望权重补偿，但随机训练轨迹仍改变，不能声称每步完整梯度相同，性能与速度必须同时报告。

## 测试与泛化口径

显式采用practical residual/post_sync/noeq星地压力代理，符合用户urban场景要求，不混称默认LEO_WEAK或真实在轨验证。全部24个固定对照测试clean及六个既有VALIDATED_ONCE完整场景；每视图相同168000物理query。全部168个模型×视图预测固定后，独立scorer才打开truth，并用混淆矩阵独立复算。

主要指标为mid_urban和low_urban分别及等权平均Accuracy/Macro-F1、各场景最差RX；泛化指标为clean未见RX的Accuracy/Macro-F1/最差RX，以及非urban和high_urban场景的迁移稳定性。报告4seed成对差值、均值/标准差与正收益seed数，不能用urban提高掩盖clean泛化下降。只有两方面结果支持时才称为同时改善；计算下降单独报告，不替代性能判断。

成本报告包括训练总耗时（含源V）、推理时间、训练/推理CUDA峰值、参数量、实际增强次数与模型输入量；记录硬件和并行竞争口径。缺失值记N/A。完整step日志、紧凑epoch JSONL/CSV及信道计数保留。Phase2的K、support适应、新类注册和H指标本轮N/A。

## 发布与交接

PLANNED，正式性能尚未测量。新run独占输出，唯一owner为codex/root/urban-ce-20261008。控制器先等待原stack及repair的真实owner退出，再按每GPU总训练进程最多2、最多4新worker运行，保护健康任务。之后自动完成训练、全行冻结、预测、独立评分和同row报告。任何非有限梯度、权限/配置错误或非零退出保留产物并停止本run补位，不因低性能停止，也不自动重跑。

## 本地验证

6组各经过E1/80/131/132/200关键epoch真实原生循环的合成数据冒烟，保存/重载自身scratch checkpoint，无query访问。5项聚焦测试通过，覆盖配置越界拒绝、CE-only无隐藏loss/teacher、DG确定性/能量/梯度/扰动上界、thin保留及跳过、固定成对评分。独立代码审查P0=0、P1=0；审查范围为运行代码、资源隔离和truth-last边界。登记字段验证VALID，非性能证据。

## 实际发布状态

2026-10-08T08:21:14.171225+08:00独立核验：VERIFIED。控制器PID1173940，release commit `889d1c11fdc740b124dcdaa51d0ec741d9e1195d`，远端路径`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_urban_ce_20261008_r01`。本地30项相关测试通过；本地和N607各6组×5关键epoch自身scratch无query冒烟通过。24行配置逐项一致，无failure。

当前QUEUED/WAITING_PRIOR_OWNERS，正式训练0/24、正式测试0/24。等待实际旧owner 1170536, 4170278, 4170279, 4176081 退出，包含repair direct_controller接管路径。旧健康训练保持运行；本run不干预。正式worker启动后自动写resolved配置、PID/GPU/log和完整epoch/step证据。全行固定后自动完成168个模型×视图预测、独立truth-last评分、urban_generalization.json和成对对照。尚不能宣称星地/泛化/速度有正收益。

只读状态脚本为`evidence/inspect_remote.py`。恢复工作先核实已有queue/worker/产物，不重复启动；完成后回收scores/summary/paired_results/urban_generalization/resources/analysis，按原登记记录和Git交付。

# 训练期 RX 均值差软一致性：两范围三种子

状态：LOCAL_VERIFIED，未发布。新机制尚无训练或有效性结果，原目标 active。

只在训练期使用全 U 已知 RX 的五组低阶混合响应均值，构造同包的有限频率作用。作用是目标风格均值减原 RX 均值，经固定强度 0.1、幅度 ±0.5 dB、相位 ±0.025 rad 有界渲染；不减去单包响应、不把风格索引冒充真实接收机、不宣称分离纯硬件。原始 IQ 推理与默认头保持，无均值 bank 或新推理参数。

原普通 PL 的 raw 当前模型 eval 硬预测、0.95 阈值、weight1、全批归一化 CE 保持。新损失对全部 U 比较相同基础视图在均值差作用前后的软预测。raw 基础复用 PL teacher，H 基础使用 F(H) 的 detached 软预测，避免强迫受损 H 拟合 raw 硬伪标签。教师是当前学生，无 EMA；新增前向均 eval，学生保留梯度。30 个新增 U epoch 中 raw/H 各 15，H 使用已完成合法固定三副本三场景 raw cache。

U 与 LU 的标量均为 (nU*KL_U+64*KL_L)/(nU+64)，U 范围 detach KL_L，LU 才向 L 传播新梯度。每步复用原 T1 的 raw/As 输入选择 64 个真实 L，同样计算 teacher/student，保持范围消融前向成本一致。原 fullL 768/步、As 两次 CE/KL0、RDrop、浅中层跨 RX 及 SIG1/defaulthead 联合训练继续执行。

每个 seed 先用自己的实际 E20 父模型做 12 个源训练批的零更新梯度校准，raw/H 交替各 6 批；两范围使用同一系数 min(1,0.1*median(primary_norm)/max_scope_median(aux_norm))、444 步线性 ramp。这约束新增梯度剂量，不保证泛化。检查默认头与 SIG1 非零梯度、L 范围隔离、全模型 863730 参数与模型状态不变，PASS 后自动继续。均值银行仅拟合一次，model_seed=null，无 F/G 权重、V 或目标。

固定 RXMean-U、RXMean-LU × 2026092701/02/03 六模型；每行独立复用同 seed 合法 scratch E20/4440/13L7LU，再 +10+20 至 E50/11100/13L37LU。新训练共 180 模型 epoch/39960 次更新；共享父模型成本单列，不从协方差优胜模型或 F50 分支。没有新 PL/Sup 对照训练或预测。固定源合同、物理 ID、角色及已有 VALIDATED_ONCE 数据复用，不重验。

所有六行均固定 final E50，完成后逐行自动 source V raw+3H 诊断、raw clean+六 practical、cross_rx126000/cross_both42000、168000 包 ×7 视图预测，再独立 truth-last。源/目标分数均不改变剩余种子、配方、权重、epoch 或重跑决定。完整矩阵、精确命令及路径以 experiment.json 为准。已有目标基准暴露如实保留，不能称新的独立盲测。

原 raw clean≥85.981746%、六景>72.096197%及严格超过匹配 PL/Sup、原声明分层/范围的目标不变。即使此次 F0 三种子改进，也不能自动代表完整原目标达成。

队列最多四条本 run GPU worker，每卡本 run 一条、所有 owner 合计 CUDA+pending 最多两条，空闲显存至少10000MiB；独立 scorer 总并发1。健康其他任务不改动。技术错误停止新提交并保留活动产物；科学负结果不触发停种子。发布普通账号，一份代码 commit、一份 archive SHA 传输比对、remote compile，实际权重 smoke 是启动第一步并自动继续。

本地五项测试通过：U-only fit 权限、有限响应作用与同包独立性、两范围相同标量/不同 L 梯度、原 PL/SIG1/来源约束负例、22job 的三种子矩阵与预测/评分隔离。独立 P0/P1 审查 PASS；唯一直接 P1 已定点关闭：native labeled context 会自增步骤计数，校准进入前用4440+step，首批实际4441/call1。未增加额外审查门槛。

一次 N607 预检 VERIFIED：普通账户、新 run/release 根均不存在、三份 scratch E20/4440 的 source-only 元数据与合同匹配，已有其他 owner 作业保留。真实权重/梯度 smoke 仍待发布后的第一步，不用元数据预检代替。

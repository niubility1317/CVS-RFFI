# 局部分布与路径签名：默认分类头最小预登记

本轮仅 Phase1，目标为默认分类头 clean 三种子平均准确率达到 84%。共同主实验在任何新 target 访问前固定为 RFF 与 SIG_TIME，分别报告，不能用目标成绩重排、调参、选模或选择性重跑。基准此前已暴露，不称新的独立盲测。Phase2 不在本轮范围。

用户当前要求使用网络原默认分类头并随骨干联合训练；禁止另训、重训、替换分类头和身份/RX 诊断分类头。历史旧门控＋EMA＋cos 与 50 轮配方是这条已授权特征路线的固定对照，不把本轮 scratch50 成本冒充 E20 或 E20＋20。

八组 BASE_REF/MEAN/MEAN_VAR/NONLINEAR_MEAN/RFF/COV/SIG/SIG_TIME，固定 model seed 2026092701/02/03。先只派发首种子 8 行，其中 7 个新增独立 scratch50 骨干；BASE_REF 对应原 OLD_GATE_COS50 同种子、同物理源角色、source-only final50，来源与实际参数已有合法核对，仅作对照和复用原默认头预测，不初始化新模型。新模型只有自己的 EMA，无 checkpoint 祖先。逐 row 配置、命令及数据句柄唯一维护在 experiment.json，实验不消费报告或索引调参。

保留源 RX 1/3/4/6/8、三天；L_s=6,300，U_s=56,700（真实 TX 隐藏），V=27,000。保留历史已核实的原数据契约和固定 clean target physical split，不重建数据。每新增骨干从零固定 50 轮、33 有标签＋17 伪标签、222 步/轮、11,100 次更新。门控/教师/优化设置保持已授权历史配方：EMA .999、batch_neighbor/window2/minconfidence .8、strong agreement、rx_day quantile .86、lambda_u .16、lambda_ent .01、AdamW lr .0002、weight_decay .0001、clip5、cos 最小 1e-6、L128/U256、label_smoothing .01。关闭卫星训练；NoExplicitCFO 保留内部补偿。U_s 不因特征变化解封 TX；V 不反向传播、不更新 EMA、prototype、normalization 或持久状态。

新增能力只进入实际部署 z160。接入 t2 中层输出 96 通道、32 个位置，先用共享的 1×1 投影得到 6 维并经 tanh 有界化，再分为 4 个等长固定窗口。全部新增方案共享同一局部投影结构、112→160 投影、LayerNorm 和逐通道有界残差 gain（初值 .05，界限 ±.2）；均新增 18,976 个可训练参数，原默认 ResidualPhysicalHead/6×160 cosine/scale30 不变。参数数目相同不等于描述子的内在信息维度相同，后者如实披露。

MEAN 使用固定 14 个方向上的均值重复展开；MEAN_VAR 使用方向均值＋方差；NONLINEAR_MEAN 对均值计算 cos/sin；RFF 在同一固定方向逐位置计算 cos/sin 后取均值。固定方向 seed 2026101307，长度按 .5/1/2 循环；这是特征空间的数学映射，不是 IQ FFT。比较 NONLINEAR_MEAN 与 RFF 可区分均值后非线性和局部分布中的高阶差异。

COV 对相同 6 维路径计算均值＋协方差上三角，共 27 分量。SIG 计算 6 维一阶增量＋15 个二阶反对称量，共 21 分量；SIG_TIME 加归一化时间坐标，得到 7＋21=28 分量。COV/SIG 使用固定稠密正交映射扩展到统一 28 维；不新增可学习参数或类别索引。四窗口拼接到 112 维后进入原特征融合。有限维 RFF 和有限阶路径量不声称唯一表达、不声称任意信道/CFO/RX 不变。

训练冻结后执行不拟合参数的源诊断：完整 L_s 特征的 TX/RX 嵌套方差与谱；V 内只对新增特征分支的每窗口局部位置做固定置换，比较默认头和 z，原 IQ、原有其他路径与分类头权重不变。无优化器、backward 或另训头；不持久保存样本特征，不导出 Phase2。该顺序诊断不参与目标访问或候选排序。

首种子所有 8 行源 checkpoint 固定后，采用原默认头作扩展条件：每行终轮 L CE≤.5；各新增行相对固定同 seed BASE_REF 的 V 平均准确率下降≤1个百分点、最差源 RX 下降≤2个百分点。全部通过才派发另两个种子，不按 V 小涨分排列候选，不消费任何目标指标。源门槛不通过时，保留首种子自动预测/评分，标 SOURCE_SCREEN_NOT_PASSED，不重新选择或重跑。

每行 source 训练/复用→参数不更新的源诊断→固定默认分类头 clean 预测→fresh CPU 独立 truth-last 评分。每 row 168,000 个独立 all-registered-class 决策，预测固定前 scorer 不连接 truth。完整三 seed 为 24 行、4,032,000 条默认头预测、432 条 overall/RX/TX/day 指标，21 个新增骨干共 1,050 轮、233,100 次更新；3 个历史对照共 150 轮、33,300 次更新单列。未完成三 seed 不计算完整矩阵均值或声称达到 84%。

运行 ID：20261013-phase1-localdistribution-pathsignature-oldgate-e50-clean-manysig-m24-r01。代码/配置身份由 release_commit.txt 中 Git commit 记录，不使用设计、报告、配置或成员 SHA。环境使用 N607 既有 CVS-RFFI 普通账户 Python；CWD 为项目 releases/cvs_feature_sequence50_20261013_r01；命令 python -u -m experiments.cvs_feature_sequence50.dispatch。精确解释器、完整 CWD、输入/输出/日志路径与每 row 命令见 experiment.json.execution/rows。唯一 owner codex/root/sequence-features-20261013。每 GPU 最多 3 个总实验进程，计入其他 owner；至少 12,000 MB 空闲，不干预健康旧任务。

不可覆盖输出 root：runs/20261013-phase1-localdistribution-pathsignature-oldgate-e50-clean-manysig-m24-r01。预期逐行 source 配置、初始真实模型 smoke、固定最终 checkpoint、epoch/step 记录、源诊断 summary、默认 prediction/provenance/complete、18 条指标及 completion；整批固定汇总、配对差和全部负结果。运行产物不进入 Git。

技术停止规则：角色/来源/checkpoint 不匹配、U 真值暴露、target 拟合、输出碰撞、非有限数值或无法产生合法预测/评分时停止新派发，保留全部产物，已启动健康行完成其既定闭环。来源不明不使用旧权重兜底，无自动重跑或目标分数停止规则。Phase1 固定测试范围仅 clean。

本地数学、实际模型默认头/特征梯度/teacher/strict checkpoint roundtrip、真值隔离/错误 ID/额外读出/另训头拒绝和 source-only 扩展检查 PASS。正式启动第一步执行同契约真实 OLD_GATE_COS50 checkpoint 的 no-query smoke，通过后直接运行。唯一独立 P0/P1 审查与 N607 发布读回尚待，不宣称性能改善。

2026-10-14 唯一独立审查发现一项直接 P1：native train_ssdg 在结束时无条件拟合 RX/day/channel ridge probe。候选 source 入口已用 scoped native_default_head_only 禁用该调用并在退出时恢复；resolved/completion 记录实际禁用一次、head_fit_updates=0，source proof 检查同一证据。针对性负测将真实 native frozen_ridge_linear_probe 替换为异常哨兵，确认整个受控闭包不调用拟合函数，原生非必需 gate 不触发，函数恢复；检查 PASS。原默认分类头仍在原 optimizer 中随骨干联合训练。定点复审尚待。

唯一独立 P0/P1 审查与唯一 P1 定点复审 PASS（/root/sequence_p0_review）。已核实真实 native optimizer 收到全部新增参数、EMA 在安装后复制、默认头联合训练、source-only扩展、预测评分数量与实际发布入口。没有新增 blocker、重复全面审查或另训分类头。

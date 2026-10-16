# 非恒定相位形态：跨 TX/day 的 RX 相关源依据

PLANNED，尚未启动。固定候选／矩阵见[experiment.json](experiment.json)，此前来源与边界见[研究说明](../../../analysis/cvs_phase_curvature_source_basis_20261017.md)。本轮只有一个统计row，非IQ增强或训练；0模型/CP/NN/优化器/教师/头拟合/新IQ视图/target，不替代最终身份识别目标。F＋跨RX同TX／3SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine头正常联合训练／E20边界不变。

唯一问题：原接收IQ的非恒定相位形态，是否具有跨TX/day共享、可复现的RX相关分量。旧commonphase20、可训练feature phasecurvature、cycle20 covariance和孤立AWGN E20保持关闭。没有新loss/分支/补偿/分类器，不根据此分析直接宣称RX振荡器参数或域泛化。

只用原VALIDATED_ONCE载体精确fitL5040，RX3/4/6/8×day1/2/3×6TX、72格各70包。L6300及未用V/历史features反序列化，RX1 L1260在统计前排除；U只有角色ID、无IQ/TX标签。不使用U/V/heldRX1/target统计，不混同fullL6300或L_train5580/L_dev720。复用原source_contract/preparation句柄，仅核对角色，不新增数据验证/hash/allowlist。

固定输入窗口80:160，包含n80..157全部78个三点。Complex128：c[n]=z[n+1]^2 conj(z[n])conj(z[n+2])，q=c/(abs(c)+1e-12)，156维交错实虚坐标。常相位与线性CFO影响抵消；不做lag20周期假设、unwrap、wrap、过滤、clip、重采样或范围扫描。全部低能量点／零点／包保留，softmodulus、零点、逐包均值与离散程度保存。这个观测仍包含TX/信道/RX/均衡影响，不识别纯RX phase noise。

每cell全部70包均值、总体方差；18个heldTX/day交叉组同时排除heldTX和heldday，只用其他5TX×2day的4RX均值，先在各TX/day中心化，再平均RX位移预测held组。held组中心化4RX作为实际相对位移，与零RX位移比较156维均方误差，无拟合scale/projection/身份或RX分类器。固定条件：pooled MSE至少减少10%，每RX不恶化，18个TX/day组至少15个strict正收益。任何FAIL关闭固定统计/lag/window/TX/day/RX/threshold/seed，不救跑、不自动生成增强或E20。PASS仅源关联依据，仍需独立物理合理算子、身份保持与匹配训练检验。

本地29输入配置负例＋7科学反例、266合成IQ统计／5040合成metadata PASS，0realIQ/CP/NN/更新。核实常相位/线性CFO不变性、已知二次相位响应、窗口外变化无影响、零点保留、留出TX/day均排除，及无RX/跨day或TX反转/单RX恶化/仅14正组/不足10%/零风险反例。一次独立P0/P1审查PASS。零模型stage不加载checkpoint，无realCP smoke，不新增无关阶段门槛。

唯一owner codex/root/F-phase-curvature-source-basis-20261017，普通N607现有CVS-RFFI Python，CPU/OMP/MKL/OpenBLAS2，CUDA_VISIBLE_DEVICES空，无GPU，不干预旧健康任务。CWD PROJECT/releases/cvs_F_phase_curvature_fitL_20261017_r01；命令CVS-RFFI/bin/python -u -m experiments.cvs_phase_curvature_basis17.run。不可变输出PROJECT/runs/20261017-diagnostic-F-crossRX-phasecurvature-fitL-manysig-m1-r01，日志release/source_phase_curvature_basis.stdout.log。发布入口F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_phase_curvature_basis17.publish，CWD当前managed工作树。

Git commit为代码身份，实际写release_commit.txt/resolved；一次普通账户source句柄/空间预检、一份archive/一次传输SHA比较、remotecompile、一次launch、独立PID/CWD/argv/CUDA/loggrowth或自然终态回读。系统性角色/输入/非有限/输出碰撞/运行失败保留failure/partial，无自动重试或健康旧进程修改。

期望active/resolved/completion、72cell完整均值/总体方差、18fold所有预测/实际向量和风险、全部5040×156逐包观察及零点/soft统计/源metadata、私有物理ID和stdout；独立全观察/72格/18折/固定条件/成本复算、聚合图和同row报告登记/Git。新源统计393120三点计算，0新前向或更新；IQ/权重/物理ID/逐包向量/日志不入Git，只保留聚合。联合目标ACTIVE。

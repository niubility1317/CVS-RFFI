# 双教师与浅中层时序适配器：固定 E20＋10 预登记

状态：PLANNED。用户确认学生继承合法 clean F 的 E20 权重，追加 10 轮，累计 E30。此轮明确授权冻结原 F 与默认分类头并新增特征模块；没有新分类器。一个模型种子 2026092701，不宣称多种子稳定性。实际 code/config commit 写入 experiment.json 与 release_commit.txt；以下矩阵已固定，不按目标成绩改变。

研究问题：合法同 IQ 星地教师的额外判别能力，能否通过 t1/t2 浅中层适配器进入单个部署学生，同时保持 clean？已有同 IQ 源 V360 上，CSIL 相对纯 F B1 在 1440 条 urban 观测修复 257 条、新增错 66 条，净差 13.263889 个百分点。分层证据见 teacher_source_evidence.md。旧 epochReplay、nontrue KD 和输入 Jacobian 固定损失路线保持关闭。

## 固定矩阵

|行|首 L 视图|As 学生|可训练部分|clean KD|As KD|
|---|---|---|---|---:|---:|
|NOAS_FT|原 ordinary|无|原 identity 骨干，默认头冻结|0|0|
|NOAS_ADAPTER|raw clean|无|两个新适配器|0.25|0|
|S0|raw clean|完整 As(raw)|原 identity 骨干，默认头冻结|0|0|
|S1|raw clean|完整 As(raw)|两个新适配器|0|0|
|S2|raw clean|完整 As(raw)|两个新适配器|0.25|0|
|S3（固定主候选）|raw clean|完整 As(raw)|两个新适配器|0.25|0.25|

六行全部冻结原默认 6×160 cosine30 头，保留其全部参数。适配器行冻结原 F；梯度穿过固定后续层。两个并行 adapter 位于 t1/t2 的原 GN 前，r16、depthwise k5、GELU、无 bias，仅 B 输出权重零初始化，共 4768 参数；保持原池化、SIG_TIME_X3 tap、物理路径和 joint160。它包含非线性，不能称为可合并的标准 LoRA。原 F 为 900722 参数，部署新增参数仍须实际模块读回。

原 ordinary NoAs 参照保留，不用新的 raw/adapter NoAs 替代它；报告两种同预算 NoAs 及继承 B0 E20。NOAS_ADAPTER 匹配 S3 的 raw/adapter/cleanKD；S2→S3 分离 As 教师，S1→S2 分离 clean 教师，S0→S1 比较更新空间。As 行 raw/As CE 各 0.5，NoAs 首视图 CE 为 1；label smoothing=0.01、CE 温度 1。raw 与 ordinary、绝对 clean 权重及学生前向量不同，明确披露，不声称所有条件同时不变。

## 教师、来源与训练衔接

学生与 Tc 固定同 seed 纯 F B0 FINAL20：20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01/B0-s2026092701。来源为 scratch，原 model/ownEMA 均 206 keys，固定 FINAL20、4440 updates，完整 L6300/U56700/V27000，source RX1/3/4/6/8、day1/2/3、split392005。按固定 seed 和终轮选择，不按已见 target 最高分选取。Ts 是 20260930-phase1-native-baselines-practical-manysig-m5-r01/csil-s2026092701 固定 last200，scratch、9800 updates；相同物理源角色已核实，无 target 选模、无 oracle、更长或更干净输入。历史 Ts E200 训练成本单列，新教师训练为 0。初始化与教师载入执行已有来源和实际角色检查，不追加数据 hash/allowlist 重验。

Tc 与学生消费相同首视图，Ts 与学生消费同一完整 As tensor。两位教师 eval/停止梯度，不注册到学生 state，不进入部署。六行每次 L update 均各执行 Tc/Ts 128 包，未用的项作 shadow；NoAs 仍产生 As 与教师 shadow，学生 As 前向为 0。Ts 原 logits=5cos+5，乘 6 转到共同 cosine30 单位，固定偏置在 softmax 中抵消；Tc/学生 scale 不变。两项 KD 均 T=4，teacher||student KL，乘 T²，分母是全部 128；教师 argmax 错时仅关闭该条 KD，所有合法 L 的 CE 保留，不使用学生置信门控。没有依据 target 拟合温度。

每行继承 E20/4440，新增 10/2220，累计 E30/6660。重置空 AdamW，lr=2e-4、weight decay=1e-4，新 cosine10 到 1e-6，clip5，不冒充连续 cosine30。FT 行保持 CENTER_UNITS Sinc 的 LR1000/eps 反比、物理衰减和裁剪，头仍冻结；adapter 行 Sinc 更新为 0。原 own EMA 206 keys 严格继承，新增 adapter keys 与学生初始化相同；decay0.999、startup_average=False，新段计数 0→2220，累计 6660。原 U hidden-TX/batch_neighbor/强增强/ownEMA 规则不变，10 轮均启用 U，学生与 ownEMA 各 567000 包。原 ordinary 在全部行完整生成，固定父 epoch13 日程；raw 行只作 shadow。L/U 采样器使用各角色固定新 Generator，所有行规则相同，记录完整物理 ID 序列，不声称全训练 RNG 完全相同。

As 只从实际 raw L 生成，保留完整 practical 物理、场景、归一化和接收机 session，逐更新刷新。计数延续 epoch21 至30、step4441 至6660，保留原 FcrossRX namespace 字符串，不因此加入 CROSS_RX 机制。U 标签不解封，V 只健康诊断和以下预定 source 决策，不进入梯度。

## 源规则、正式测试与放弃条件

先固定唯一主候选 S3。完整固定 V1800×2 draws 的 mid/low 合计 7200 条，S3−S2 至少 +0.5 个百分点（净正确至少 +36）；full V27000 clean 的 S3−max(NOAS_FT,NOAS_ADAPTER) 至少 −0.2 个百分点（净正确不少于 −54）。源决策在任何正式评分之前固定。通过只表示源资格，不自动延长；失败关闭该固定 TEMPORAL16/T4/.25 配方，不用 target 调温度、系数、seed 或选择性重跑。科学失败不停止其他健康训练。

每行固定 FINAL30，训练后立即源资格和七视图预测；其他行尚未训练完成也不阻止有资源行的预测。固定 received IQ clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban，每视图 168000 包。预测过程不连接 truth；同 row 全部七视图完成且跨行源决定固定后，单独 CPU scorer 才连接 truth。已暴露基准不称新的独立盲测。

报告全部行 accuracy、macro F1、worst RX、配对修正/新增错，分别列每个 suburban 的绝对准确率及相对自身 clean 差距。模型种子数 1，样本 SD 不可估，不把两次 channel draw、RX/day 当种子。保留固定 B0 E20 clean-correct cohort 但注明不同累计成本。低于 60% 不单独证明物理上限，非零梯度或 KD 下降不代替性能。标准 LoRA、raw IQ 补充网络、feature keep 和恢复默认头联合训练均为条件性后续，不在本轮展开。

## 执行与产物

环境：N607 既有 ordinary account，既有 Python3.10/Torch2.1；CPU2，FP32，TF32 off。CWD 为 PROJECT/releases/cvs_F_dualTeacher_adapter_E30_20261019_r01。命令：CVS-RFFI/python -u experiments/cvs_dual_teacher_adapter30/runner.py --mode dispatch。唯一 launch owner：codex/root/F-dualTeacherAdapter30。GPU4/6/7 动态选空位，跨 owner 每 GPU 最多 2 进程、可用显存至少 12000 MB，使用既有全局调度锁；不改健康旧任务。

不可变输出根：PROJECT/runs/20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r01，各 row 独占 source/source_qualification/prediction/scoring。首步使用真实父 checkpoint 做 strict/no-query smoke，PASS 立即继续；仅一次 N607 preflight、一个 archive 的本地远端 SHA 比较及一次 remote compile。实际代码/config Git commit 是身份，不增加 seals、member hash 或额外审批。

系统性技术失败停止规则：真实角色/来源不匹配、非有限更新、冻结 F/头变化、视图/教师/U truth 隔离、预算或输出根不一致，只处理该 run owner 的精确进程，保留产物、不自动重试，不停止其他健康任务。源指标低或正式结果未达目标是科学结果，不是技术故障。

预期产物：逐 row strict FINAL30 checkpoint/lineage、resolved native args、完整 step 与 L/U ID 序列、actual channel/ordinary/teacher identity、EMA/optimizer 初始化与逐轮成本、真实 joint160、参数变化/冻结证据、source 12600 NN readout、七视图 prediction complete、独立 truth-last scores、同 row 报告和登记。分别核算学生 L（NoAs284160/As568320）、U567000、ownEMA567000、Tc284160、Ts284160、信道284160、source eval/qualification、formal1176002 以及墙钟；不按可训练参数数目宣称加速。

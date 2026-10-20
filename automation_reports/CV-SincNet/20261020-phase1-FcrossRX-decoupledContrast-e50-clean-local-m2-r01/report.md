# F＋跨 RX 同 TX：解除正样本竞争与正负耦合

状态：RUNNING／VERIFIED，两行已开始本机真实源训练，尚无本轮终轮或 clean 成绩。用户选择的原 F＋跨 RX 同 TX 对比学习为基底；历史 85.1139% 只作方法锚点，实际配对复用本机原 BASE 三种子，不重训基线、不继承基线权重。当前新机制同时改善源域弱 TX 14-7、20-19，并保护其他 TX 的假设尚待验证。

| 条件 | 浅、中层对比目标 | 新训练 | 测试 |
| --- | --- | --- | --- |
| BASE | 原 CROSS_RX SupCon | 复用同 seed 完整匹配结果 | 复用原 clean |
| Q_DCL | 负样本分母，乘 detached 原负样本概率 q | 首 seed 2701 从零 50 轮 | 自动 clean |
| DCL | 负样本分母，不乘 q | 首 seed 2701 从零 50 轮 | 自动 clean |

完整模型 seed 为 2026092701/02/03；首种子两行固定完成，再于读取任何新目标汇总前，逐条件决定是否原样扩 02/03。固定源门槛：V27000、固定 final50 的默认头，两个弱 TX recall 均须严格高于同 seed BASE；其他每个 TX 不低于 BASE−0.25 个百分点、整体不降、最差 RX 不低于 BASE−0.25 个百分点，全部类无零特征。两条初始行不论源门槛是否通过，都按固定矩阵完成 clean 测试；不利用新目标分数调参、重排或选择性重跑。

每模型原 F＝三个独立 SIG_TIME 描述子接实际 joint160，统一 CENTER25K＋物理单位 AdamW 修正；从零 50＝33L＋17U，222 step/epoch，共 11100 次原更新。RDrop alpha1、LS0.01、普通增强、旧门控＋ownEMA＋cos、默认头全部正常联合训练保持。不使用星地信道增强，只测试 clean；不存在额外分类头、头拟合、prototype 决策或推理映射。

只替换原 CROSS_RX 在 time_down／t2.pool 的对比项。原时间均值／总体标准差描述子、当前 L batch 的 detached 标准化、L2 normalize、temperature0.2、同 TX 不同 RX 正样本与不同 TX 负样本保持；同 TX 同 RX 排除。两次原 RDrop 各自计算浅／中层目标后取平均，复用同一份实际增强 IQ，无新增模型前向、IQ 读取、参数、优化器更新或教师。原 5 轮预热、10 轮线性升到 0.1，共 9990 启用 step／39960 层位前向目标。六类均按原锚点均值，不另加弱类权重。

对锚点设 P 为正样本 logsumexp，N 为负样本 logsumexp，M 为正样本 logit 均值，K 为正样本数。原 loss＝logaddexp(P,N)−M−logK；DCL＝N−M−logK；q＝exp(N−logaddexp(P,N))；Q_DCL＝stopgrad(q)×DCL。原梯度精确分为 q×DCL 梯度和 (1−q)×[P−M−logK] 梯度，两项系数在该点取常数。Q_DCL 检查去除正样本间竞争，DCL 进一步解除 q 缩放；二者差异包含梯度幅度变化。负的有限 loss 合法，不截断。方法参考 [DCL 原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf)，这里是多正样本有监督扩展，不能用论文图像结果替代 CVS 证据。

此前类内精度缓存诊断的三个模型均未通过固定门槛，结果完整保留；本轮不继承任何 V 协方差矩阵，不叠加失败的 gap、梯度投影、特征挑战、SAM 或旧类权重。L6300、隐藏 U56700、V27000 原角色／物理 ID／split392005、RX1/3/4/6/8、day1/2/3 不变。只复用现有 EXACT_MATCH／READY／VALIDATED_ONCE 数据句柄；Phase1 manifest 没有实际 p2 capsule_id／split_id 字段，不新造 ID 或重验。初始化只用本次随机 student 和自身 EMA，无外部 checkpoint 继承；一次合法匹配 BASE checkpoint 仅用于无 query 烟测并丢弃。

数学梯度分解、单正样本边界、变换不变性与 13 个协议负例通过。CPU／CUDA 真实 F 检查均通过：两次实际 IQ 相同，浅／中层与 Sinc 收到非零新梯度，默认头六组参数对新对比项均无直接梯度，主联合梯度与原流程逐项完全一致；实际物理 Sinc／默认头更新、EMA、导出一致。CPU 合法 CP 无 query 烟测一次通过，两个平台各两次合成优化步均丢弃，真实数据训练步为 0。开发时输入计数断言曾选错挂钩层，已改为原 RDrop 实际捕获／复用边界验证；未修改原执行路径，未启动真实训练或读取目标。唯一一次独立 P0/P1 审查 PASS。

代码配置身份为实际 Git commit，发布后填入。CWD＝IMMUTABLE_LOCAL_RELEASE/cvs_F_decoupledContrast50_local_20261020_r01；环境为已验证原生 Windows Python3.10／PyTorch2.10cu128、FP32／无 TF32、CPU2／workers0。本机物理 GPU0，全机至多 3 个 source/predict worker（计入其他 owner），启动需 RAM≥4.5 GiB，独立 CPU scorer 至多 1。唯一 owner＝codex/root/FcrossRX50-decoupledContrast-local-20261020；保留健康任务，不热改或重启。唯一不可覆盖输出＝LOCAL_RUN/20261020-phase1-FcrossRX-decoupledContrast-e50-clean-local-m2-r01。

命令：`LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_decoupled50.driver --profile LOCAL_PROFILE --metadata LOCAL_SOURCE_NATIVE_GATE_METADATA_R02 --base LOCAL_RUN/20261020-phase1-FcrossRX-decoupledContrast-e50-clean-local-m2-r01 --reference-base LOCAL_RUN/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --commit RELEASE_COMMIT --max-workers 3`。启动器第一步数学检查通过后立即继续；一个发布归档只比对一次，发布目录 compile 一次，启动后独立核实 PID／CWD／命令／GPU／日志增长。实际依赖与配置均来自已提交 Git；无 N607／管理员操作。

停止只限角色／物理来源／元数据／配置或输出冲突、非有限 loss／特征／梯度、原生预算／Sinc／EMA 错误、worker 技术失败；停止新派发并保留健康任务及全部产物。不因有限负 loss、科学低分停止，不自动或按目标重试。

每行固定 final50/defaulthead 预测 target clean168000，7RX×4day；完整预测固定后由独立 CPU scorer 首次连接 truth。记录逐 seed 配对差、均值、样本 SD、正差数、Macro-F1、逐 TX、最差 RX／day，完整 50 epoch／11100 step／11100 对比记录、q 的逐类实际值、U 实际曝光与接受率、训练／预测耗时、GPU 峰值和参数成本；无真实 U TX。n1 无样本 SD，不冒充三种子或独立盲测；不作 Phase2／星地声明。预期产物见 experiment.json；实现、配置、检查与后续汇总提交推送 Git，并独立核对远端 OID。

实际执行提交：`8da6a824434d239f30e3d3796c345b4b31d3ef7a`。两行 PID／CWD／命令／GPU0／逐步日志增长已独立核实，详见 [启动核验](launch_verification.json)。当前仍处原 5 轮预热；不把启动正确性当作新机制或目标性能证明。初次推送遭远端解包失败，独立核对分支后同提交重试成功，远端 OID 已核实一致。

两条原进程均已完成首个启用轮次 E6，各执行 222 个新目标 step，epoch 回调显示无临时 hooks 残留，随后继续进入 E7。实际 RDrop 同 IQ 复用、原调用计数、新目标权重／有限非零 loss、无额外模型前向及默认头辅助梯度边界已核实，见 [首个启用轮次证据](first_active_epoch.json)。这只证明机制在实际训练中运行；尚无终轮或新目标 clean 性能结论。

首个完整 U epoch（E34）已逐条复核每行 native 日志前缀 1..7548：各 222 个 U 批次、56,700 次实际曝光，Q_DCL 接受 878（1.5485%）、DCL 接受 873（1.5397%）；U 真值可用标记、星地增强及非有限跳步均为 0。原进程继续进入 E35，详见 [首个 U epoch 证据](first_U_epoch.json)。接受率不是伪标签正确率，也不能推出弱 TX 的真实曝光或目标收益；未修改任何门控或扩种子规则。

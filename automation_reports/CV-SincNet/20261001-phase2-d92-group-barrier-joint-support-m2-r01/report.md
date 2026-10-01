最新终态：TRAINING_COMPLETE；[evidence/runtime_readback_1790870516458140900.json](evidence/runtime_readback_1790870516458140900.json)（2026-10-01T16:00:55Z）。Four support rows complete; immutable9518 runtime unchanged; source-fixed query method; accuracy/OOF/resource analysis not yet read.

# D92 GroupBarrierJoint：解析注册与小适配器联合学习

当前状态：RUNNING／首次发布与运行状态 VERIFIED。尚无完整诊断或 query 准确率结论。

- run_id：`20261001-phase2-d92-group-barrier-joint-support-m2-r01`；唯一 launch owner：`root`。
- [逐行配置、数据和继承记录](experiment.json)；[状态事件](events.jsonl)。
- [数学设计](../../../docs/D92_GROUP_FACTORIZED_JOINT_DERIVATION_20261001.md)；[方法解释](../../../docs/D92_GROUP_BARRIER_JOINT_IMPLEMENTATION_DECISIONS_20261001.md)；[相关验证](../../../docs/D92_GROUP_BARRIER_JOINT_PIPELINE_VALIDATION_20261001.json)；[必要源代码审查](../../../docs/D92_GROUP_BARRIER_JOINT_ENTRY_REVIEW_20261001.md)。

本轮机制是保留当前路径实际 Margin B 的旧类条件函数，单独解析拟合新类 Ridge，用全部合法训练 support 学习 Bernoulli gate，并将完整内层梯度传回继承 B 的小 adapter。C 不重新训练旧类列，数学上保留 B 的旧类内部排序；逐旧点／逐新类约束保护训练 support 的原间隔，不保证 query 零遗忘。

有限障碍预算固定为 `ζ=N*1e-4/(m_old*q_new)`。原目标近似界只属于精确中心路径驻点，不能解释为准确率或遗忘界。原 Margin query run 的紧约束隐式梯度失败仍保留在原记录中；这个候选使用独立名称和独立 run，不修改其 runtime，不停止、重启或热修改健康行。

当前先执行固定算法的 support 信息诊断：同一 parent 的 A/B/C 使用配对物理旧类 outer-held support。A 来自匹配的冻结地面模型包；B 仅用 old outer-train；C 继承这一实际 B 后注册 new outer-train。inner-held 参与优化，不能冒称独立验证。K=1 没有独立 held，三阶段 held 准确率为 N/A；oneshot proxy 单独报告。新增 0 类时 C 为同一实际 B，新类准确率、H 和新旧差距为 N/A。

诊断后按已经固定的同一候选另建完整 query 重复基准：4 行、2400 parents、原 5 个 support seeds、全部原 receiver/scenario；所有预测固定后独立 truth-last 评分。support 或 query 成绩均不回流调参或选择性重跑。当前重复使用的基准不能称为全新独立确认；新增独立数据验证按用户要求暂缓。

合规 Phase1 来源仍是 source-only scratch final200，地面权重固定。卫星方法只读取合法目标 support 的冻结 raw 特征和匹配地面汇总包；不读取源域样本或逐样本源特征，不 reload checkpoint，不训练 encoder，不读取 query IQ、标签、truth 或评分。practical residual、post_sync、no equalization、25 MHz 固定，既有 VALIDATED_ONCE 数据不重建、不重验。

四行来自已固定的最小两个源模型 seeds 2026092701/02 与 rx3/rx1 两 cohort 的完整交叉。每行 40 parents：对应 receiver 19-1 或 20-19、practical_high/low_urban 两场景、4 个 K、5 个新增类数、support seed 2026092711。共 160 parents。六种 seed 角色、每行命令、完整 split ID 和缓存身份见 experiment.json，model seed 不代替其他随机性。

## 完整 K×新增类数报告位置

以下是本轮固定矩阵，准确率单位为百分比，变化量为百分点。旧类数固定为 6；support 数为当前完整 split 的标称数量，OOF 子路径实际训练／held 数另存 fit_stages。每格还将按 row、receiver、场景和 model seed 分层。当前均无真实结果，禁止猜填。

| K | 旧类数 | 新增类数 | 注册总类数 | 旧／新 support 数 | A 旧类 | B 旧类 | C 旧类 | C 新类 | B−A | B−C | 新旧绝对差 | H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0 | 6 | 6／0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 2 | 8 | 6／2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 5 | 11 | 6／5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 10 | 16 | 6／10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | 6 | 20 | 26 | 6／20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 0 | 6 | 30／0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 2 | 8 | 30／10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 5 | 11 | 30／25 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 10 | 16 | 30／50 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 6 | 20 | 26 | 30／100 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 0 | 6 | 60／0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 2 | 8 | 60／20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 5 | 11 | 60／50 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 10 | 16 | 60／100 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | 6 | 20 | 26 | 60／200 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 0 | 6 | 120／0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 2 | 8 | 120／40 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 5 | 11 | 120／100 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 10 | 16 | 120／200 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 20 | 6 | 20 | 26 | 120／400 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

理想目标保留为：旧类适应提升≥10个百分点、注册后旧类下降≤1个百分点、新旧准确率绝对差≤3个百分点。这些是逐步改进的理想目标，不是每轮启动硬门槛。最终旧类和新类统一面对全部注册类竞争，不能用旧／新 query 的真实角色进行推理分流。

## 执行与成本

N607 普通用户，CPU 两 lane，每 lane 两 BLAS threads，CUDA_VISIBLE_DEVICES 为空。B 与 C 的固定优化预算、实际参数、kernel、gate Newton/line-search 及 buffer guard 见配置。`167772160` 字节 guard 仅覆盖 gate 的 H＋Cholesky 因子缓存；它不是总训练状态、过程 RSS 或星载内存上限。

保留 CVS 风格详细训练文本、完整数组档案、逐步完整事件、去大数组的 compact JSONL/CSV、每阶段训练/预测成本、峰值过程 RSS、实际可训练坐标和驻留／部署状态字节。实际新传输字节与地面包文件字节分别报告；未知的真实通信、星载训练/推理耗时、内存和能耗写 N/A。参数较少和解析内层不自动等于总计算量较小。

首次独立读回（2026-10-01T15:04:28Z）：supervisor PID1004125，start_ticks11465111；rx3两个模型PID1004188/1004189实时训练，rx1两行等待原排程。实际 argv/CWD/CPU2环境、每行resolved config和启动输入来源见[evidence/runtime_readback_1790867129563692100.json](evidence/runtime_readback_1790867129563692100.json)。immutable runtime为 `9518d46d2f763e5b080b5337e23155c9de883e09`，source preparation parent仍81985ed；之后的记录/分析器提交不改变这个运行。完整成本尚N/A，当前进程RSS只作途中证据，不代表星载或最终峰值。一个 run 只有 root 启动。每行独占目录；技术失败只失败该 lane，保留完整产物并让健康行按原计划结束，不自动重试，不因性能弱停止。

## 下一步

固定源码push/独立OID、唯一publication及live process读回均已完成。继续原排程；诊断产物闭合后详细分析，完整query入口在盲态源码并行准备。query测试、独立truth-last评分、报告及Git交付分别记录，成绩不回流调参。Goal仍ACTIVE。

发布证据：[evidence/publication_20261001.json](evidence/publication_20261001.json)；源码交付证据：[evidence/source_delivery_20261001.json](evidence/source_delivery_20261001.json)。初次运行时未访问query、truth或源样本，checkpoint/encoder未reload，既有数据未重建/重验。

独立support分析已在experiment.json预登记：源码73aa1b31f、被分析fit runtime9518d46d2、仅完整四行原产物，无query/truth/fit，root唯一执行，独占新输出。分析尚未启动，accuracy/资源结果仍N/A；已运行query方法不受后续诊断分数影响。

独立分析实际启动 VERIFIED_RUNNING（2026-10-01T16:29:14Z），child PID1055923/start11979916，wrapper1055916；argv/CWD/CPU2环境见[evidence/analysis_runtime_1790872215329109400.json](evidence/analysis_runtime_1790872215329109400.json)，源码仍73aa、fit runtime仍9518。完整summary尚无，accuracy和最终成本仍N/A；过程中RSS不是最终峰值或星载证据。软件source tar为1208320 bytes，只是本次SSH源码发布包，不能算source样本或卫星链路实测字节。发布与独立读回见[evidence/analysis_publication_20261002.json](evidence/analysis_publication_20261002.json)。

独立分析r01终态FAILED：[evidence/analysis_runtime_1790872597649626100.json](evidence/analysis_runtime_1790872597649626100.json)，仅compact引用格式接口错误，未生成summary/成绩。20项合成检查的限定修复见[ABI说明](../../../docs/D92_GROUP_SUPPORT_ANALYSIS_ABI_FIX_20261002.md)；原训练和query版本/参数不变。新r02分析已预登记但尚未启动；保留失败release/log/全部原state，不覆盖、不重训练，真实指标仍N/A。

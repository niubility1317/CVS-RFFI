# ProtoFrame joint 与 support 入口直接正确性检查

日期：2026-10-02。结论：在下述冻结源码范围内，未发现会改变数学目标、泄漏 held 教师标签、破坏实际 B→C 继承、错误连接 A/B/C 物理样本，或使失败路径冒充完成的 P0/P1。

这是一次有界、QUERY-BLIND 的只读检查。只新增本文，不修改作者代码，不运行数值、Conda、Git、SSH 或训练。未读取真实数据、原型包、权重、cache、预测、评分、日志、experiment_registry、automation_reports 或 handoff。主 Agent 最终告知共 111 个 distinct 合成 cases 通过，其中本轮 entry/control 为 34 PASS；并告知隔离源码闭包验证为 34 modules/45 files、四行只读元数据 preflight 为 VERIFIED。本文没有执行这些检查或打开其证据文件，亦不把它们写成真实性能证据。主 Agent 提供的合成证据 prefix 为 `1790881631294080000`（34）与 `1790881160942242100`（先前 86）。

## 检查范围

主要范围是 [新 joint core](../code/cvsrffi/d92_proto_frame_joint_local_ridge.py) 和 [support evaluator](../tools/evaluate_d92_proto_frame_joint_probe.py)，以及相应 literal synthetic 测试。evaluator 作者已明确 FREEZE，最后的 ABI 修订仅为完整工作键集合的显式零初始化与严格 SUM/MAX 聚合。

仅为确认调用能连接到上述接口，读取了新 [supervisor](../tools/run_d92_proto_frame_joint_probe.py)、[preflight](../tools/preflight_d92_proto_frame_joint_probe.py)、[publisher](../tools/publish_d92_proto_frame_joint_probe.py) 的源码。既有 cache reader、Ground A packet、gate 和原 BranchLocalRidge 的内部实现不重新审查；primitives 是本检查者负责的文件，本次没有自审。未查看当前运行配置、真实预登记或产物。

## 数学与继承

| 项目 | 已核对的实际调用与边界 |
|---|---|
| 当次 B→C | core `prepare_proto_frame_joint_training`，303–316 行，核对 B 类型、mode、旧类序、run/row/split/外层 scope/fold/trial/K、资源、旧物理 ID、标签、五块原始特征和 Q；C anchor 为该 B 的实际 theta。evaluator 540–548 行直接传该路径刚拟合的 B state，不读取历史 B。 |
| 折内 prior | core 326–342 行：只从当前 inner-train 取旧类 raw/labels，固定实际 B theta，重新拟合 old inner-train 解析头；held 仅作为无标签 score 端点。最终 full problem 的 prior 则直接评分实际 full B，并绑定其 final ref。 |
| 顺序状态与 new0 | core 321–324、557–560 行先识别 exact reuse；fit 返回原 B 对象。evaluator 511–512、543–544 行跳过无新增类的 C 重拟合，保留同一 B 对象和分数。 |
| 固定 old 尺度 | core `_fixed_geometry` 与 prepare 的 `oldraw/ol` 调用固定各折原始 old inner-train 的 tau/gamma。theta 试探没有重估尺度，也没有增加 nuisance head。 |
| 旧条件函数 | core 434–445、531–540 行使用冻结实际 B 的旧类条件分布。C adapter 只改变新条件头与 gate 的核几何，不重拟合或改写该 B 分类函数。 |
| 全类竞争 | core 541–554 行逐单物理样本评分，计算全部旧、新注册列。公开 C predict 使用两组稳定最大值及旧 raw B winner 比较；它没有按真实 old/new role 路由，不读真实类数量或 query 配额。 |
| 原 margin 与完整约束 | core 433–440 行保留 prior 的原负 margin、零 margin 和 ties；阈值按完整 old×new pair 构建。没有 clamp 原 margin、丢弃物理行或只保留一个新类最大值。 |

这里的“保持旧条件函数”只表示 C 的 old-only 相对类别顺序继承实际 B。新类仍参与统一竞争，因此不保证最终旧类零遗忘；旧 support 的 barrier 条件也不构成任意 query 的保护证明。

完整五方向链位于 core 410–457 行：new Ridge 的 train/train block 和所有 train/held cross block 都携带 Jacobian；新头的 log-softmax 导数同时进入阈值和 held 分数。`_gate_jvp`（365–407 行）重新构造实际 sqrt(D) SPD 系统，将五个真实 RHS 与 intercept Schur RHS 合并求解，包含阈值变化、自由 b、dK alpha 和 dL alpha。它没有把阈值固定、删掉自由 b、调用 centering VJP，或只保留一个核端点。额外 gate JVP factor 与六列 RHS 均单列收费，没有宣称复用 gate 未公开的最终因子。

## 一次更新、失败和状态

core 464–496 行把每个物理 inner-held 的全类 score/Jacobian 回填到统一 ID 位置，先跨 fold 求每类 CE，再计算 RMS。真实 outer-held 诊断没有传入 prepare/fit。

core 570–605 行只在初点计算一次 GGN 方向，之后最多 12 次完整真实目标回读。固定步长 `.125 * .5**trial`，接受式为初点 CE 加 `1e-4 * eta * g·d`；逐次保存实数形式比较是否成立、128 倍机器精度比较容差及 observed increase。接受最多一次，使用该最后接受点；全部拒绝时保留初点，技术失败则抛出，不以 B 或另一参数点替代失败 C。

K1 没有 OOF，core 605–607 行仍拟合完整 final head。Q 无信息、tau0 或零核分支不删除物理 head；连续核 Jacobian 无信息时 GGN 方向为零，完整 C new head/gate/free b 仍保留。单新类的 new conditional 是常数，不把它误作整个 gate 的零函数。

core 472–496、628–638 行保留已完成 fold、失败的当前数组、最后接受 theta 和累计实际工作。evaluator 的 failure archive 允许保存非有限失败状态，成功 archive 仍要求有限。失败档案设备本身再失败时，core 保留主要异常和内存 partial，另记 archive/log 错误，不把次要报告错误替换为已成功训练。

INITIAL、GRADIENT、TRIAL、STEP、FINAL 与 FAILURE 回调保存完整数值档案引用；full C 档案包含实际 B 的 raw/state 数组。resident walker 按 ndarray 对象身份去重，识别 Mapping 和 dataclass；报告范围是保留状态的数值数组，不是 Python/LAPACK/RSS 总内存，也不是已实现的最小部署 serializer。

## A/B/C 物理配对与入口工作账

evaluator `path` 的 train/held mask（475–486 行）先固定外层 support 划分。B 只拟合该路径 old train；C 用该路径全部注册 train 并继承该 B。outer held 特征只归档和推断，不进入 prepare/fit。

evaluator 588–601 行的 A 仅调用显式匹配 Ground packet 的原六类 float32 头，读取同路径 old held。A 不是 B0/R0；没有 packet 或 K1 时明确 N/A。B/C/A 的 old held 物理 ID 在 pooled assessment 和诊断中精确配对。

evaluator 620–631 行先输出完整 A、R0 B/C、candidate B/C 固定预测，persistence callback flush 后才做 support-held 标签连接。support 标签仍是本次合法训练/诊断输入；入口没有 query 文件或 query truth 读取。结构 C 的 public predict 用于固定分类决定，诊断不以发生共同 log offset 舍入的 score argmax 偷换该决定。

最终 evaluator 的 `WORK_KEYS`、`_empty_work`、`_work_merge` 与 `_account_work`（339–364 行）要求声明键集合完整，按 core ABI 对准备、stage、score 计 SUM，对 peak 计 MAX。准备工作只计一次；嵌入 stage 的 preparation 描述不重复收费。Ridge 额外 spectral check、拒绝 trial、gate forward 与额外 gate JVP 的 factor/RHS 都来自实际 operation audit，没有套用五次 factor 或旧方法固定 factor 公式。

public structural predict 会额外重算几何。入口为它记录真实调用数和耗时，内部工作量明确 N/A；`score_actual_work` 只覆盖 score 调用，没有冒充总过程的完整 FLOP 或能耗。无传输测量时，wire/deployment/增量传输均有缺失口径。`minimum_deployment_numeric_state_bytes` 沿用字段名，但其值的实际 scope 是 current full retained state；不能据它宣称已经实现最小部署压缩。

失败 evaluator（640–662、809–815 行）分别记录已完成计数与失败 fit 的部分工作，保留档案，不将未知失败工作填成零。完整 marker 对新独立分析仍声明 REQUIRED；训练完成不等于独立数学分析、query 预测、truth-last 测试评分或真实性能完成。

## 控制面连接与未测范围

supervisor 159–169 行传显式 run/row/实际 runtime commit、ground summary/deployed 状态和同 config 的 gate 资源；242–294 行先做全行输入元数据绑定，独占新输出、CPU 2 lane/BLAS 2，只启动声明行，不自动重试或干预其他任务。完成时独立重读各 row marker、selected support 物理集合、SUM/MAX phase 账和 archive manifest。

publisher 16–44、48–75 行包含新 core/ground/evaluator 的静态闭包，隔离 import 禁止 np.load/torch.load 及 native encoder 模块；实际发布仍由 root 执行。此检查没有运行 import 验证、读远端 post-state 或证明真实输入可用。

尚未核实真实 source/cache/packet/ground-summary 绑定、设备、实际预算消耗、运行状态、数值求解成功率或准确率。它们由 root 的既有输入权限、合成验证、唯一发布和独立分析路径负责。本文不新增审批、receipt、数据重验、参数组合、选模依据或运行干预要求。

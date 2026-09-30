# AffineJoint：解析截距与 LocalRidge 联合微调实验

状态：`ARTIFACTS_COMPLETE`。run 为 `20261001-phase2-d92-affine-joint-support-m2-r01`，release 为 `d92_affine_joint_support_20261001_r01`。root 是唯一 launch owner。配置已生成、登记并单次发布；实际进程与训练日志已独立核实，没有完整真实性能结论。

## 数学改动与依据

本轮只有一个结构变化：在 AJLR 的残差核头中引入不受惩罚的类别截距 b，并让微调使用该完整分类函数的梯度。保留实际 B 函数先验 m、旧 support 的核尺度、当前函数坐标 adapter、类 RMS 交叉熵、近端惩罚及原有限回溯预算。只比较原 BranchLocalRidge（R0）和顺序 AffineJoint（R_AFFINE_seq）；没有参数网格、温度扫描或手动新旧类偏置。

解析头求解：

```text
min_{g,b} 0.5||M + g(X) + 1b − Y||² + 0.5||g||²_RKHS
A = K + I, E = Y − M
F = A⁻¹E, z = A⁻¹1, s = 1ᵀz
b = (1ᵀF)/s, alpha = F − zb
f(x) = m(x) + k(x,X)alpha + b
```

因此 `(K+I)alpha+1b=E`、`1ᵀalpha=0`。这解除原无截距残差函数在旧参考测度上的强制均值限制，不保证旧类、新类或 H 上升。对于固定 U、均衡旧类 support、零 prior 的 B 阶段，数学截距为零，与原 B 函数等价；浮点运算顺序不承诺逐位一致。C 阶段残差 `Y−M_B` 的列均值一般不为零，才是本轮要检验的情况。

完整伴随保留 `g_b=1ᵀG`，满足 `A T+1 eta=LᵀG`、`1ᵀT=g_b`，再沿中心核、原始几何和 adapter 反传；不能沿用无截距梯度。详细推导、退化分支和 q gauge 的适用条件见[数学审计](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md)，实现见[核心说明](D92_AFFINE_JOINT_CORE_20261001.md)。q gauge 等价要求同一原始核、U、尺度、support 和 prior，不能由此授权重新估计尺度或省略几何梯度。

依据包括 [GPML 第 2 章 §2.7](https://gaussianprocess.org/gpml/chapters/RW2.pdf) 的均值函数处理、[representer theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf) 的有限核展开，以及 [R2D2 §3.2](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf) 的可微闭式 ridge。本文公式是在这些基础上对当前残差目标的推导；文献不是本项目性能证据。C 使用旧 support 且不继承后验协方差，仍是正则化函数延续，不能称为精确顺序 Bayes。

## 固定输入、矩阵与权限

[实际配置](../configs/d92_affine_joint_support_20261001.json)沿用已有 source-only 未适应缓存与 `p2_min_v1/VALIDATED_ONCE` capsule 身份。practical residual、post_sync、noeq、25 MHz 固定。未改变 received IQ、物理 ID、receiver/TX、scenario、K、support/query 或 schema，因此不重建或重验数据；当前只读 preflight 核查实际输入绑定、可用性和输出冲突。

固定两个合规 source-only final200 model seeds：2026092701、2026092702。split/data seed 均为 2026092705，augmentation 为 2026092707，support 为 2026092711，evaluation 为 null（确定性计算）。来源核查继承固定 Phase1 元数据证据；此 run 不加载 checkpoint 或更新 encoder，不接收源域样本、源域逐样本特征或历史目标适应状态。新增地面摘要传输为 0 B。

四个 cohort/model rows，各 40 个 parent，共 160。旧类 6 个，新增类 0、2、5、10、20 个，K 为 1、5、10、20。沿用 pilot 已声明的两个 receiver/cohort 和 high、low_urban 场景；不冒充全接收机或完整 query 矩阵。

当前 run/row/物理 fold 内，C 继承刚训练出的实际 B 函数和 U_B；不读取旧 AJLR、跨 run、跨 fold 或跨 model 的目标适应状态。C inner prior 在冻结实际 U_B 上只重拟合旧 inner-train 头，无嵌套 B 训练。inner-held support 标签属于合法监督训练，不是独立验证；outer-held support 在分数固定后评分。query、truth、role、真实 batch 类数、配额与全局重排均不进入方法。预测逐样本面对全部注册类。

## 报告与资源

报告完整 K×新增类数的 A、B、C 旧/新准确率、H、注册旧类下降、新旧类绝对差及 receiver/scene/model 分层。此次 support pilot 没有 ground A，A 与 B−A 为 N/A；B0/R0 不替代 A。true K1 仍拟合完整解析头，但没有独立持出准确率；proxy train-K1 不更新 adapter。新增 0 精确复用 B 状态与分数。

10/1/3 个百分点是用户的理想方向，不是每轮硬门槛。此试验不自动晋级，也不形成独立数据或 query 泛化结论。旧 AJLR 独立分析 r02 尚在运行；本方案在该分析完整评分前已由数学审计确定，没有根据部分分数调整。

CPU-only、两 lane、每 lane 两 BLAS 线程；不干预 GPU 或健康 AJLR 分析。每阶段至多 4 次归一化梯度更新、每次至多 12 次减半试探，初始步长 0.125。实际活动 adapter 参数 `736r≤5888`，解析系数和 C 个截距另外报告。前向每次三角求解 C+1 RHS，伴随 C RHS，记录实际 RHS、元素和 n²r 工作代理；后者不是测得 FLOP。所有拒绝试探均计费。

保留完整 CVS 文本、各损失/梯度/步长/方法状态/耗时、紧凑 JSONL/CSV 与数值 NPZ。报告训练和推理耗时、实际 CPU RSS、常驻/部署状态字节与新增传输量；真实星载、GPU、完整模型传输及未测费用为 N/A。少量参数和 SFT 名称不证明省算力。

## 已完成验证与执行边界

79 个不同相关检查通过：core/entry/ops 58 项（7.09 s，证据 `pytest_utf8_1790801239413599000`）、独立汇总 10 项（109.86 s，`pytest_utf8_1790802427189642100`）、报告器 11 项（0.57 s，`pytest_utf8_1790802979858337900`）。测试包含独立 primal/saddle oracle、完整有限差分、非零截距伴随、真实回调顺序、合法 B 状态绑定及篡改拒绝。唯一[独立 P0/P1 审查](D92_AFFINE_JOINT_P0_REVIEW_20261001.md)无未解决问题。读取预先生成的代码和合成测试，不以性能决定通过。

本次实际 preflight 为 VERIFIED：四个缓存绑定匹配，新 run/release/archive 不存在，CPU 96 核、loadavg 约 1.7、磁盘剩余约 6.95 TB；未访问样本值、query 或 checkpoint。发布前代码、配置、预登记和验证证据进入 Git 并核对远端 OID。发布单次执行，随后独立读回 PID/argv/cwd、实际 startup 和训练日志。技术失败只终止所属 lane，无自动重跑；健康任务不因性能弱而停止。完整 160 parents 结束后才独立汇总，不依部分结果修改方法。

RUNNING/VERIFIED：AffineJoint已单次发布启动，实际runtime81a226a8d1cef34ea817ada87070bd89912d027d；supervisor451253及workers451265/451266的PID/argv/cwd独立读回一致。实际生效算法、缓存身份、run/row和CPU/BLAS参数匹配。首次读回16/160parents，两个rx3运行、两个rx1待排队，详细训练文本已增长；尚无完整性能结论。query/source样本0，encoder/checkpoint不加载，新增地面摘要0B。禁止重复publish/停止/重启/热修改；旧AJLR分析r02保持原PID394399/handle99201，目标ACTIVE。

COLLECTOR_VERIFIED/VERIFIED：独立只读AffineJoint训练诊断工具已完成，12个不同合成检查通过（9.02s），累计91个不同相关检查。两阶段目录/训练引用快照与完整训练档案提取、13项RHS/intercept费用、gCE=gZ−Z及实际B→C状态诊断已覆盖；不读取外层评分、不做新拟合/求解/SVD/前向。真实采集尚未执行，当前训练release/runtime不变，run继续RUNNING；不能据此宣称性能改善。

ANALYZER_READY/VERIFIED：未来Affine独立分析器的完整中心化VJP以代数等价的求和/外积展开，O(n³+hn²)改为O(n²+hn)，保留移动reference项、非零g_b及原有核验与容差；64MiB数组预算/64entry缓存按实际nbytes管理，每lane释放，缓存命中仍核查文件状态且不跳过数学核验。17个新增不同检查及10项既有summary回归通过，累计108个不同检查。首轮1项fixture混淆非均匀q的伴随零方向，原独立矩阵oracle已一致；仅修正fixture后受影响8项通过。健康训练runtime81a226a8d1cef34ea817ada87070bd89912d027d及AJLR分析r02不变；Affine分析尚未启动，无真实加速或性能结论，目标ACTIVE。

ARTIFACTS_COMPLETE/VERIFIED：AffineJoint四row/160parent训练已结束，supervisor及全部workers退出；完整marker/state/manifest元数据和实际计数独立读回一致。实际更新2592次、头拟合35988次、latent SVD3240次。完整数学档案仍由独立分析核验；完成不代表性能改善，query/源样本0。下一步唯一analysis release d92_affine_joint_analysis_20261001_r01，root sole owner，目标ACTIVE。

ANALYSIS_REPAIR_READY/VERIFIED：Affine分析器严格阶段比较已补齐真实入口schema/method；B→C顺序、全部字段、数学核验和容差保持原样。真实evaluate→core写盘及篡改回归在内的11项相关测试通过（106.31s），其中1项新增、10项既有回归，累计109项不同检查。r01失败和完整训练产物保留，新release d92_affine_joint_analysis_20261001_r02尚未启动；提交推送核对版本后单次独立分析，不重跑训练，目标ACTIVE。

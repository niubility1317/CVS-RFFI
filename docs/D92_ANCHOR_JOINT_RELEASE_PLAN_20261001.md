# AJLR 联合微调与 LocalRidge 的固定 support 试验

状态：`ARTIFACTS_COMPLETE`。新 run 为 `20261001-phase2-d92-anchor-joint-support-m2-r01`，group 为 `d92-anchor-joint-support`，release 为 `d92_anchor_joint_support_20261001_r01`。root 是唯一 launch owner。[逐行配置](../configs/d92_anchor_joint_support_20261001.json)维护实际矩阵；已独立核实启动，完整 AJLR 性能结果尚未产生。

数学方案见[完整推导](D92_JOINT_AFTER_FCR8_DESIGN_20261001.md)。仅比较原 BranchLocalRidge（R0）与预先声明的顺序 AJLR（R_AJLR_seq），不新增 reset 路径，不扫描 rank、学习率、损失权重或温度。

| 改动 | 数学依据 | 要验证的实际问题 |
|---|---|---|
| C 继承实际 B 分类函数 m，并求解残差 g | 最小化 `0.5||m+g−Y||²+0.5||g||²_RKHS`，得到 `α=(K+I)⁻¹(Y−M)`、`f=m+kα` | B 参数继承是否能进一步成为分类函数继承；不保证旧类准确率保持 |
| 固定旧 support 的物理中心测度与 B 核尺度 | `P=I−1qᵀ`、`K=γPRPᵀ` 为半正定核，参考点随 U 重新映射 | 避免新增样本单独改变旧核中心与尺度；不能冻结参考点梯度 |
| 用最终残差 LocalRidge 头反传全类监督 | 平滑 CE 按类均值后取 RMS，解析求解器的完整伴随梯度 | 使优化目标与最终全类竞争相连，避免只优化最大错误类间隔 |
| 固定字典，学习当前 support 的函数坐标 | `U=U_anchor+ZWᵀ`；精确算术下平均预切向函数位移平方为 `||Z||²` | 用可解释的函数近端代替参数组合搜索；不是完整 Fisher 自然梯度 |
| 有限回溯且完整计费 | 最多 4 次更新，每次最多 12 个减半试探；接受首个实际 Armijo 且非增的试探 | 检验有限预算内能否实现目标下降；不声称收敛或准确率必升 |

非零均值残差公式参考 [GPML 第 2 章 §2.7](https://gaussianprocess.org/gpml/chapters/RW2.pdf)，有限核展开参考 [Schölkopf、Herbrich、Smola 的 representer theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)，闭式 ridge 可微求解参考 [R2D2 §3.2](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)。这些资料支持公式构造与可微求解，不是本项目性能证据。AJLR 正则化残差 g；B 先验不必属于当前 U 的同一 RKHS。C 重用旧数据且不继承后验协方差，不能称为精确顺序 Bayes。

地面 source-only Phase1 与未适应缓存固定。卫星只使用合法 target support；不加载源域样本、源域逐样本特征、query、历史目标适应状态或地面统计包。仅允许本 row 当前旧物理 support 的 B→C 继承。固定 `practical residual/post_sync/noeq/25MHz`，复用 `p2_min_v1/VALIDATED_ONCE` 数据；方法变化不触发重建或重复数据验证，必要的缓存与 checkpoint 来源身份按原协议核对。

每个内层头只用旧 inner-train 原始特征计算固定 τ/γ；当前训练字典 SVD 只定义 adapter 坐标。inner-held 标签是合法训练监督，outer-held 只在分数固定后评分。C 内层 prior 头在冻结实际 U_B 上用旧 inner-train 重拟合，不重新训练 B；U_B 本身可能使用了合法 inner-held 监督，因此内层目标不能称为独立验证。

四个 model/cohort rows、两个 model seeds，每 row 40 个 parent，共 160；旧类 6 个，新增 0/2/5/10/20，K=1/5/10/20。完整报告逐 K×新增类数的 A、B、C 旧/新/H、注册旧类下降、任务绝对新旧差和 receiver/scene/model 分层。A 在本 support 试验中缺失，A 与 B−A 均为 N/A；B0/R0 不能替代 A。true K1 没有独立持出准确率，但仍拟合完整 B/C 闭式头；proxy trainK1 不训练 adapter。新增 0 精确复用 B。

最坏预算是 1800 条路径、3240 次 baseline/preparation/stage，最多 648 个信息阶段、2592 次接受更新、31752 次内目标、95256 次内头、3240 次最终头、864 次 prior 头，加上 3240 次 R0，共 102600 次头拟合。该数字是上限，实际 Cholesky 次数按退化分支测量；没有隐藏 nuisance 头。student、prior、baseline primal、baseline EDF 和 CE adjoint 三角求解分别计费。距离、核与 adapter 前向按实际执行计量，参考距离是 raw 距离的解释子集，不重复相加。

CPU-only、两 lane、每 lane 两 BLAS 线程，不干预 GPU 任务。实际活动参数为 `736r≤5888`；保存 U、V0、B prior、α 与 support 几何的唯一缓冲区字节数。保留完整 CVS 文本、各损失、梯度、核统计、接受/拒绝、耗时、紧凑 JSONL/CSV 和完整 NPZ。源域验证为 N/A 并说明访问禁止。额外地面统计传输为 0 B；真实星载计算、GPU 内存和模型完整传输未测则为 N/A。两路径 AJLR 与三路径 FCR 的总工作不同，不能用总 wall time 直接证明微调本身更省算力。

数值验证：22 项核心合成检查通过，证据 `.codex_tmp/pytest_utf8_1790789398064171500`；首轮标签 dtype 问题已修复。入口 4 项、汇总 5 项、启动器 23 项均通过，共 54 个不同相关检查；[唯一 P0/P1 审查](D92_ANCHOR_JOINT_P0_REVIEW_20261001.md)结论为 NO_UNRESOLVED_P0_P1。资源修改后受影响核心 3 项通过（1790789671575009100）；实际路径元数据 6 项通过（1790789874716344800），完整复算 1 项通过（1790790062262768900），10 种篡改拒绝通过（1790790155613702700）。正式发布以已推送版本为准，启动后独立读回 PID/argv/cwd 与产物。技术失败只保留该 row 的诊断，无自动重试；不因低性能停止健康任务。完整 160 parent 后才运行单次独立分析，不根据部分结果更改方法。

LOCAL_VERIFIED/VERIFIED：AJLR单一数学结构已实现，54个不同相关检查通过；core22、ops23、entry4、summary5。整数dtype、summary唯一参考pair、实际1800路径和原生bool验收边界均已修复；完整正常复算与10种篡改拒绝通过，独立审查NO_UNRESOLVED_P0_P1。实际B函数prior、固定旧物理参考测度与tau/gamma、残差闭式头、全类CE伴随联合微调；只有R0/R_AJLR_seq。完整四row/160parent预登记与既有source-only缓存身份preflight已核实，新run/release/archive无冲突。未发布、未启动，尚无真实性能；A与B−A=N/A，目标ACTIVE。

RUNNING/VERIFIED：AJLR已单次发布启动，实际runtime a1a003f59e8ed14640a252ada8290a03af3420c5。supervisor341635、workers341647/341648的实际PID/argv/cwd独立匹配；两个rx3 row训练中，两个rx1待排队。CPU两lane/BLAS2、query/source样本不读、encoder/checkpoint不加载。首次readback1790790609；完整160parent尚未结束，尚无真实性能分析；禁止重复publish/重启/热修改。54不同相关检查与唯一P0/P1已完成。A与B−A=N/A，目标ACTIVE。

ARTIFACTS_COMPLETE/VERIFIED：AJLR完整四row/160parent已结束，supervisor及全部workers退出；完整marker/state/实际计数独立读回一致。实际更新2592次、头拟合36564次、latent SVD3240次。未读query/源样本；完成不代表性能改善。下一步单次独立support分析与完整训练诊断，root sole owner，目标ACTIVE。

ANALYSIS_R01_FAILED/VERIFIED：完整训练产物保留；首次独立汇总PID386856已退出，无summary/output。实际阶段流为prep B→fit B→prep C→fit C，汇总器重建顺序错误，6项回归检查通过。仅修复分析器，保持严格流核对；预登记新analysis release d92_anchor_joint_analysis_20261001_r02，root sole owner，尚未启动，不重跑训练。

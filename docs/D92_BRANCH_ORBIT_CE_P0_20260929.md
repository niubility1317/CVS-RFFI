# D92-BranchOrbitCE-v1 独立 P0/P1 审查

日期：2026-09-29。结论：**两项 P1 已修复并定点核对，当前范围无未解决 P0/P1。** 审查者 `/root/source_aux_feasibility` 仅编辑本文，未修改作者代码、配置或实验记录。

范围为 `code/cvsrffi/d92_branch_orbit_ce.py`、冻结配置和三个 support specs、`docs/D92_NEXT_AFTER_METRIC_20260929.md`，以及新 export/evaluate/summarize/run/prepare/preflight/publish/analyze 入口。仅读取代码、设计、合成测试和指定只读预检记录；未读取任何 query 成绩、正式 benchmark 结果或结果型 handoff，未加载真实 IQ/特征/权重、源样本或地面摘要，未执行远端、镜像、提交或实验启动。

## 发现与关闭记录

| 级别 | 发现 | 状态 |
|---|---|---|
| P1，已关闭 | 轨道归一化的自核由 16 个带正负项的核值求和，而 cross-kernel 使用另一种累计路径。近抵消轨道的真实范数很小时，舍入误差可被归一化放大，导致巨大负自核、非对称 Gram，破坏 PSD 和 CE 的 L/梯度依据 | 核心改为同一轨道表示的 thin-QR 等价计算；平方和自核与稳定 cross-kernel 一致；新增均值及张量抵消反例通过，审查者原反例复核通过 |
| P1，已关闭 | CE 异常在 `_fit`/`_fold` 补充作用域之前抛出，失败 JSON 最初仅有 split ID 与无上下文的迭代轨迹，缺 arm、fold/anchor、实际 train K/physical IDs；当前 episode 已成功 stage 也仅留内存 | 异常逐层补齐 arm、actual train K/IDs、parent/fold/trial/held IDs，保留本 episode 之前成功的完整 stages；首折、后续折和 proxy 失败合成测试覆盖 |

第一项由审查者运行全合成定点复现确认，不涉及任何真实数据或调参。精确输入如下：

```python
r = np.random.default_rng(8)
v = r.normal(size=(3, 2, 160))
v /= np.linalg.norm(v, axis=2, keepdims=True)
z = np.stack((v[:, 0], v[:, 1], -v[:, 0],
              -v[:, 1] + 1e-8 * v[:, 0]), axis=1)
zero = np.zeros_like(z)
b, a = core._blocks(z, np.zeros((3, 96)), zero, zero, zero)
s, d = core._scale(b, a)
g = core._kernel(b, a, b, a, s, s, True)
```

原实现的显式轨道均值范数平方约为 `6.2e-18`，但归一化 Gram 对角约为 `[2.08e8, -2.08e8, 2.29e8]`，最大不对称误差约 `5.20e6`。输入符合公共 API 的有限 float64 形状；问题不是非法数据、优化参数或 query 表现。只对最终 Gram 对称化或截断负自核不足以纠正错误尺度，应实现同一冻结轨道公式的稳定计算。

修复使用 `B^T=QR`、`T=RA/4`，完整轨道由 `[mean(B),mean(A),QT]` 表达；Q 列正交，所以 tensor 自核为 `||T||_F²`，cross-kernel 为对应 Q/T 内积组合。它改变数值求值路径，不改变轨道公式、norm floor、损失或冻结超参数。审查者重放原公开输入：自核为 `[6.214344706652299e-18,6.220001593087898e-18,6.249997943967213e-18]`，归一化对角在 float64 精度内等于 3，最大不对称为 0，最小特征值 `2.476235003932883`。训练与逐条推理复用同一稳定表示，未引入谱截断或 jitter。

## 已核对的科学与实现边界

**物理输入与轨道。** exporter 从 `SupportIQ` 的只读 ZIP_STORED mmap 按已注册 support ID 白名单取值；元数据阶段核对全 support/query 索引互斥，未获取 query IQ 或 truth。每物理原始 received 仅取一次，确定性生成 `(I,Q),(-Q,I),(-I,-Q),(Q,-I)`，没有第二次 LEO realization。四个完整分支出口来自同一冻结模型，FFT96 只对原观测计算一次。`transform_support` 每次 native forward 恰为一物理记录的一视图，实际 received forward 为 4N，另有 4 次合成 smoke，分别记账。

**checkpoint 与缓存。** 新 cache schema 与旧单视图 cache 分开，成员严格为四个 `[N,4,160]` 分支、`[N,96]` FFT 及必要绑定元数据。exact native loader 沿用 scratch final200、source-role EXACT_MATCH、SHA、类顺序和无继承/无目标接触核验；参数/buffer 始终冻结，Tensor↔NumPy 采用已验证的 list 桥，不走曾失败的 `.numpy()` ABI 路径。CPU evaluator 仅消费该 support cache 与 capsule manifest，不再次加载模型或 IQ。当前候选不使用 source 或 ground 统计。

**核与 physical 等权。** 冻结目标是完整交互特征 `phi=[B,A,vec(BA^T)]` 的四视图均值，而不是 `mean(B)⊗mean(A)`；归一化后通常自核为 3，零/下限轨道按原公式处理。每 physical 对应一个轨道嵌入和一个 CE 项，损失质量是 N，不是 4N。`single_ridge` 直接调用原交互 `_fit/_score_rows` 的 view0 路径，避免对照公式漂移。完整 2×2 四臂均保留，不根据 row/K 的成绩选臂。

**CE 目标和梯度。** 在 train-only 中心化特征 X 上优化 `sum_i CE(X_i^T W,y_i)+0.5||W||²`，无额外可学习截距、温度或旧新类权重。令 `W=X^T alpha`、`G=XX^T`，真正 RKHS 梯度为 `X^T(alpha+P-Y)`，范数由对应 Gram 二次型计算；系数更新 `(1-eta)*look-eta*(P-Y)` 等价于该 RKHS 参数梯度步，不能把系数欧氏梯度错称为真实梯度。softmax Hessian 算子范数≤1/2，所以 PSD G 下 `L=1+0.5*trace(G)` 是合法上界，强凸常数为 1。固定 momentum、初始化、梯度容差与最多 2000 步只使用训练状态；held/query 不选择步数。上面的数值 P1 正是维护这个 PSD 前提所必需的修复。

**K1 与物理隔离。** 公共 full fit 在所有 K 使用同一公式。真实 K1 probe 仅数值诊断，OOF/proxy 均 null；没有声称四个坐标视图提供独立类内样本。标准 fold 按每类 physical ID 排序位置模 `min(K,3)`；每 fold 重新计算 train-only kernel、中心化和拟合状态。proxy 仅使用当前 parent support，穷尽每类的全部排序位置 anchors；每 trial train K=1，held K=parent K−1。全 cache 跨 row 复用的是冻结特征，未复用拟合状态或借外部 support 训练。

**预测与不变性。** 公共 state 复制数值数组并冻结；query 每条独立计算自身四视图表示与固定 support 的核，面对全部注册类，不使用 query 群体统计或真值。old membership 只影响指标报告；输出类表重排不改变物理预测，精确平局按物理 class ID 字典序处理。有限 C4 不变性只能说明坐标变换规则，不保证所有连续相位、信道或真实新旧类误差改善。

## 完整诊断、日志与汇总

预登记为 8 rows、4800 parents：1200 个真实 K1 数值诊断，3600 个标准 OOF parent，42000 个穷尽 anchor proxy。proxy 每臂 7879200 次 held occurrence 包含大量相关复用，不能当独立样本数。两 ridge 臂全矩阵实际分解次数应为 105600；两 CE 臂共 105600 个 fit，包括第 0 步即收敛的退化 fit。CE optimizer steps 从每个实际 stage 累计，不以 2000 上限乘拟合数冒充实测值。

核心/入口保存 iteration0 和每次更新的真实 loss_data/loss_ridge/total、RKHS gradient、LR、L、momentum 和 elapsed；阶段记录 train IDs、actual K、次数、converged 和终值。无 epoch 的方法记录 null；source validation 记录 null 及禁用原因。stdout 抽取第 0 步、每 25 步和末步；完整结构化 JSONL/CSV 与 full trace 保留每步。非收敛为技术失败，不切换 ridge、不改容差、不产生 completion marker。

汇总从标准 OOF 逐 physical 记录、proxy confusion/classwise NLL sums/counts 与 paired 四格计数复算各指标，并核对 train/held 映射、全部 anchors、parent/train K、core 实际次数及收敛轨迹。proxy 先在 parent 内平均全部 anchors，再按任务汇总；K1 数值有独立 `parent_numerical` 层，不混入 held 指标。未校准 softmax NLL 不用于温度或方法选择。

预声明筛选分别对标准 OOF 和 proxy 的每个 parent K5/10/20，要求 orbit_ce 对 `single_ridge/single_ce/orbit_ridge` 三个 controls 的 new/H 均为正差且 old≥−0.01；额外报告 old/new/H 是否同时严格正差。旧类下降但在护栏内不能称为用户“全面提升”目标完成。all-old 任务及完整分层另报，不自动晋级、不按 K 拼接 control，也不把 proxy 解释成正式 K1/query 结果。

## 编排与预检

新输出每 row 独占，缓存不覆盖旧产物。原模型来源、SHA、capsule 和六类 seed 角色从既有 source-only 契约继承；四个 model seeds 均保留。runner 串行导出，GPU0 每次一个冻结模型进程；CPU probe 最多四 lanes、每 lane 两 BLAS 线程，CUDA 隐藏。技术失败仅失败所属 lane，健康任务结束，未自动重试。publisher 核对已 push commit、发布路径未提交变更和独占落地路径；analysis 依赖闭包包含新 summary/evaluator/exporter 与复用 helper，并要求完整八行终态。

指定只读证据 `E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-orbit-ce-support-m4-r01/evidence/orbit_preflight_1790673620028626500.json` 为 VERIFIED：GPU0 当时仅占 1 MiB，两个既有 capsule 分别 900/300 splits，run/release/archive 路径均不存在，未读 IQ/源样本/评分。该快照不是实验启动或完成证据；本审查没有重复远端预检。

## 验证证据与限制

owner 最终报告核心 32、exporter 18、entry 10、summary 70、编排 16 项测试通过，共 146 项；核心新增修复测试包括均值/张量近抵消、第一折后续臂、第二折及 proxy 失败的完整上下文。审查者检查对应测试源码，并仅为新发现的近抵消问题执行必要复现与修复后读回；未重复运行未变测试套件。已定点核对最终设计文件第 61 行的 thin-QR 平方和说明及第 143 行的修复后合成成本：C26K20 orbit CE 463 步、fit 约 0.476 秒、solve 约 0.340 秒，梯度低于冻结容差；文档明确不将修复前后墙钟差解释为机制提速。本文 UTF-8/空白检查通过，审查结束并停止编辑。

本审查验证权限、数学实现与证据一致性，不保证真实 support 或 query 泛化提升。发布与运行后的 artifact 读回、成本和科学解释仍由 launch owner 负责。

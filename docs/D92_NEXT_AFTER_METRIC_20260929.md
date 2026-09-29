# D92-BranchOrbitCE-v1：固定观测轨道与分类损失

日期：2026-09-29。状态：核心、冻结配置和合成测试已实现；本文件不包含真实 OrbitCE 实验结果。设计者仅使用合法 support 诊断，未读取 query 评分、正式重复实验报告或排名。实验登记、发布、入口和评分由其他职责方完成。

## 依据与选择

BranchMetric 的完整 support 诊断没有技术失败。常规物理留出 A 通过，但单样本 proxy B 未通过。相对 interaction ridge，proxy 的 H 在 parent K=5、10、20 分别变化 −0.131、−0.031、−0.094 个百分点，旧类分别下降 0.665、0.736、0.834 个百分点。new_count=2 时 H 改善，new_count=20 时 H 分别下降 0.689、0.623、0.600 个百分点。证据来源为 `20260929-phase2-d92-branch-metric-support-m4-r01/results/support_summary` 下的 `summary.json`、`report.md`、`by_parent_k_newcount.csv`。

这些现象支持研究“单个原型面对更多竞争类时的决策方式”，但不能证明相位变化是失败原因。K1 的类内度量严格退化为 kernel NCM；继续调整同一散度系数不能解决这一机制缺口。本设计不使用这些结果扫描 λ，也不按 K 拼接已知方法。

| 路线 | 机制与可检验性 | 本次决定 |
|---|---|---|
| 局部邻域或类内度量 | 多样本时学习局部几何；单样本缺少独立类内变化，类邻域也可能受竞争类数量影响 | 不继续沿已失败 proxy 的同一机制调系数 |
| 固定单视图核上的多类交叉熵 | 直接优化所有注册类的竞争概率，K1 也能拟合判别函数；不能消除单观测坐标敏感性 | 保留为必要 `single_ce` 对照 |
| 完整分支交互表示的 C4 轨道池化，再用多类交叉熵 | 将同一 received 的确定性相位坐标变化映射为一个物理表示，并优化分类竞争；两项作用可由 2×2 对照分离 | 唯一候选 `orbit_ce` |

已有 MVRidge 使用过四相位 identity 与同一 FFT，并以 view scatter 进入 ridge。因此本方法不把有限相位轨道称为新物理信息；增量是对完整五块交互表示作轨道核池化、归一化与分类损失的联合检验。`D92_EXISTING_GROUND_FEASIBILITY_20260929.md` 记载的合法 ground summary 仅覆盖 identity 分支，不能推出其他分支的类内协方差，本方法不读取或使用它。

## 输入权限和物理单位

Phase1 全部冻结。训练输入仅为当前 capsule 的合法 support。每个物理 received IQ 生成以下四个确定性坐标视图：

\[
T_0(I,Q)=(I,Q),\quad T_1(I,Q)=(-Q,I),\quad
T_2(I,Q)=(-I,-Q),\quad T_3(I,Q)=(Q,-I).
\]

每个视图经同一冻结网络取得 `z_id,t_emb,f_emb,pa_local`，每块维度 160；缓存形状均为 `[N,4,160]`。FFT96 仅从原始 received 计算一次，形状 `[N,96]`，四视图共享它。所有分支与 FFT 先按现有实现归一化，特征量与损失均无物理量纲。禁止读取 source 样本、source 样本级特征、clean IQ、query、第二次 LEO realization 或源原型训练头。

四视图仍计作一个物理样本，K 不变，损失质量为 N 而非 4N。视图不作为独立类内样本或真实噪声协方差，也不能作为 K1 的独立持出。新增 exporter 必须通过 support ID 白名单读取；旧的全 IQ exporter 不适合作为此入口。单个支持样本需要四次冻结分支前向；导出器另列合成正确性检查成本。

## 核与训练目标

令 `unit` 为现有零安全归一化，FFT 使用其历史归一化下限，其余块沿用 `1e-12`。第 v 个视图：

\[
B_v=\operatorname{unit}([\operatorname{unit}(z_v),4\operatorname{unit}(f_{FFT})]),\qquad
A_v=[\operatorname{unit}(t_v),\operatorname{unit}(f_v),\operatorname{unit}(pa_v)]/\sqrt3.
\]

定义完整显式表示与现有交互核：

\[
\phi_v=[B_v,A_v,\operatorname{vec}(B_vA_v^\top)],\qquad
k_0(x,y)=K_B+K_A+K_BK_A.
\]

轨道表示先对完整交互表示求均值，再归一化：

\[
\bar\phi(x)=\tfrac14\sum_v\phi_v(x),\quad
\bar k(x,y)=\tfrac1{16}\sum_{v,w} k_0(T_vx,T_wy),
\]
\[
\psi(x)=\frac{\sqrt3\bar\phi(x)}{\max(\|\bar\phi(x)\|,10^{-12})},\qquad
\kappa(x,y)=\frac{3\bar k(x,y)}{\max(\sqrt{\bar k(x,x)},10^{-12})\max(\sqrt{\bar k(y,y)},10^{-12})}.
\]

这里不是分别平均 B、A 后再相乘。正常非零样本的对角为 3，与完整非零 single 核一致；零块或小于下限的轨道不满足严格能量匹配，必须披露数值统计。不对 Gram 矩阵作谱截断或加 jitter。

数值实现不能用 16 个大自核项相加求轨道范数：近抵消会使实际很小的正范数被舍入为负值，错误放大归一化。实现对每个物理样本取 thin QR 分解 \(B^\top=QR\)，令 \(T=RA/4\)，则完整平均交互矩阵为 \(QT\)。直接计算 meanB、meanA，自核为 \(\|\operatorname{mean}B\|^2+\|\operatorname{mean}A\|^2+\|T\|_F^2\)。两个样本的交互内积用 \(\sum_{r,s}\langle Q_{x,r},Q_{y,s}\rangle\langle T_{x,r},T_{y,s}\rangle\) 求得。Q 正交使自范数无需大项抵消，R 与 A 的消减在小量表示 T 中完成，再求平方和；这是原公式的等价分解，不含经验性能阈值、秩筛选或强制 Gram 对称。该表示也覆盖秩亏和全零 B。训练端分解在每次评分调用中复用。

对当前训练折的 N 个物理样本，`single` 使用视图 0 的现有 k0，`orbit` 使用 κ。设 \(\mu=N^{-1}\sum_i\psi_i\)、\(X_i=\psi_i-\mu\)、\(G=H K H\)。中心化只使用当前训练 support。类数 C 为全部注册类，所有类使用相同公式，无旧新类权重或真值配额。

\[
\mathcal L_{CE}(W)=\sum_{i=1}^N[-s_{i,y_i}+\log\sum_{c=1}^C e^{s_{ic}}]+\tfrac12\|W\|_{\mathcal H,F}^2,
\qquad s_i=X_i^\top W.
\]

CE 使用物理求和，不取均值；λ 固定为 1，温度固定为 1。没有额外学习截距，截距仅由训练特征均值诱导为 \(-\mu^\top W\)。ridge 对照使用 \(\tfrac12\sum_i\|s_i-(e_{y_i}-\mathbf1/C)\|^2+\tfrac12\|W\|^2\)，精确复用旧 single interaction ridge 实现。更改 K 自然改变数据项相对正则项的质量，这是固定求和目标的含义，不再按 K 调 λ。

表示定理给出 \(W=X^\top\alpha\)、训练 logits \(G\alpha\)，惩罚为 \(\tfrac12\operatorname{tr}(\alpha^\top G\alpha)\)。查询仅使用其自身视图，分数为：

\[
k_c(q)_j=k(q,x_j)-N^{-1}\sum_l k(q,x_l)
-N^{-1}\sum_l k(x_l,x_j)+N^{-2}\sum_{l,m}k(x_l,x_m),\quad s(q)=k_c(q)^\top\alpha.
\]

实现采用参考样本差分后中心化以降低常量核的消减误差，与以上 dual 式等价。query 不参与均值、训练、停止条件或全局重排。各样本独立面对全部类，按物理 class ID 字典序解决精确平局。

## CE 数值求解与失败处理

零初始化、固定步长的强凸加速梯度仅优化训练目标。令 P 为逐行 softmax，R=P−Y，则 RKHS 梯度为 \(X^\top(\alpha+R)\)，真实梯度范数为：

\[
g_{\mathcal H}=\sqrt{\operatorname{tr}[(\alpha+R)^\top G(\alpha+R)]}.
\]

softmax Hessian 的算子范数不超过 1/2，因此
\(L=1+\tfrac12\operatorname{tr}(G)\) 是训练目标梯度的 Lipschitz 上界，强凸常数为 1。固定 \(\eta=1/L\)、\(\beta=(\sqrt L-1)/(\sqrt L+1)\)。在 lookahead 系数 a 上更新
\(\alpha_{t+1}=(1-\eta)a_t-\eta(P(Ga_t)-Y)\)，再令
\(a_{t+1}=\alpha_{t+1}+\beta(\alpha_{t+1}-\alpha_t)\)。这在 RKHS 参数中是相同的加速迭代。λ1 保证函数/权重唯一，不保证奇异 G 下 α 唯一。

每一步重新测量当前迭代的训练真实梯度；仅当
\(g_{\mathcal H}\le10^{-7}(1+\sqrt N)\) 停止，最多 2000 次更新。不以 held/query 精度选步，不保留“验证最优”状态。失败抛出 `OrbitCEConvergenceError`，保留已有有限 trace，不回退 ridge、不修改阈值或另扫参数。非有限失败注明尝试到的 iteration 与缺失测量，不能伪造该步梯度。异常逐层附上 arm、训练 ID、train K、scope、parent K、fold/trial、held ID，并保留同一 episode 此前全部 `completed_stages`，使后续折或 proxy 失败仍可追溯已完成计算。C1、全零或全常量训练特征得到零函数，CE 可在第 0 步收敛。

## 四臂与可反驳假设

| arm | 表示 | 目标 |
|---|---|---|
| `single_ridge` | 现有单视图完整交互核 | 现有 ridge1，作为精确回归锚点 |
| `single_ce` | 同一单视图核 | 固定物理 CE+λ1 |
| `orbit_ridge` | 归一化完整轨道核 | 固定物理 ridge1 |
| `orbit_ce` | 同一轨道核 | 固定物理 CE+λ1；唯一候选 |

必须分别回答：CE 是否改善单样本类别竞争；轨道是否降低无关坐标变化；联合候选是否超过各单机制。若 `orbit_ce` 仅超过 `single_ridge`、却不能超过 `single_ce` 或 `orbit_ridge`，联合机制证据不足。禁止在失败后换成胜出的 control、按 K 拼接或调整 λ。

潜在失败包括：有限四相位不覆盖实际载波相位；坐标变化可能携带有用身份信息，池化会将其删除；共享 FFT 和确定性轨道不能模拟独立接收噪声；轨道能量归一化可能移除可靠性信息；CE 的训练竞争改进不保证持出校准或旧新类同时改善；K1 的其他未知干扰仍未被观测。上述问题都不能由合成不变性测试证明不存在。

## 完整 support 诊断和筛选

沿用相同完整模型/cohort/K/new_count 矩阵和原 support capsule。对 K≥2，每类按物理 ID 排序并按位置模 `min(K,3)` 分折；每臂、每折均只用训练 ID 拟合。常规 OOF 保留每个物理 held 的类、预测、NLL、正确性以及五项配对差值。

1-shot proxy 独立记录。parent K≥2 时，对每类同一排序位置选一个训练 ID，遍历全部 K 个位置；其余 K−1 个合法 support 作持出。精确保存 parent_k、proxy_train_k=1、trial、train/held ID、coverage。每个 trial 每臂仅保留 confusion、逐类 NLL sum/count、metrics；配对压为四格正确性计数。来自更大 support 池的 proxy 不冒称正式 K1 结果，不改原 capsule。真实 K1 仅检查输入数值，OOF 和 proxy 都为 null，不拟合诊断头。

科学筛选不降低原全面目标。A（标准 OOF）和 B（单样本 proxy）分别在 parent K=5、10、20 比较唯一候选对三个 controls；每个 parent K 的新类准确率与 H 均须严格改善、旧类退化不得超过 1 个百分点，且技术与覆盖检查完整。这个旧类容忍度是研究筛选口径，不能称为“新旧类全面提升”；若旧类点估计下降，报告必须明确标出，不能据此宣告用户最终目标完成。各 new_count 分层及 all-old 单列完整报告，不能只展示有利 strata；A 通过但 B 失败仍不足以推进全面目标。预定主效应配对另有 `single_ce_minus_single_ridge` 和 `orbit_ridge_minus_single_ridge`，仅解释机制，不用于改选候选。所有置信区间与汇总单位遵循父实验预登记，不从 proxy 重复 held 次数制造独立样本量。

不新增 source validation：其指标为 null，原因为无 source 样本或特征库。后续真实 query 必须另按既有 prediction 后 scorer 协议执行；本次不启动、不读取 query 结果。

## 接口、日志与成本

核心 `code/cvsrffi/d92_branch_orbit_ce.py`；冻结配置 `configs/d92_branch_orbit_ce_frozen_20260929.json` 与 `FROZEN_CONFIG` 精确一致。API：

```python
fit_branch_orbit_ce(*, z_id, fft, t_emb, f_emb, pa_local,
                   support_labels, support_ids, classes,
                   old_classes=(), arm='orbit_ce') -> BranchOrbitCEState
probe_branch_orbit_ce(*, z_id, fft, t_emb, f_emb, pa_local,
                     support_labels, support_ids, classes,
                     old_classes=()) -> dict
state.score(z_id=..., fft=..., t_emb=..., f_emb=..., pa_local=...)
state.predict(...)
state.audit_dict()
```

纯数学核心无 I/O。state 为冻结 dataclass，数组只读；数值状态含七个 float64 数组和两个标量，bytes 为
\(8[N(736V+C+3)+C+2]\)，V=1 或 4，不含 Python、注册信息及 audit 开销。C26、K20 的 orbit 状态 12,367,904 bytes，single 状态 3,182,624 bytes。

probe 顶层沿用 `k/support_count/classes/old_classes/folds/oof/paired/oneshot_proxy/standard_factorization_count/factorization_count/optimizer_steps/fit_seconds/numerical`。四臂每折 2 次 Cholesky、2 次 CE 优化。完整矩阵标准折 ridge 21,600 次，proxy ridge 84,000 次，合计 105,600 次；CE head 同为 105,600 个，实际 CE steps 累加，退化头仍计 head 数。真实 K1 数值诊断不另 fit。最大步数的总理论上界 211,200,000 不作为预期耗时估计。

stage 保存作用域、parent/train K、精确训练 ID、arm、物理 loss mass、factorization_calls、iterations、optimizer_steps、converged、终值 loss_data/loss_ridge/loss_total、gradient_norm、stationarity_residual、gradient_tolerance、learning_rate、L、momentum 与耗时。`steps` 从 iteration0 连续记录每步上述实际损失与梯度；ridge steps=[]。结构化完整轨迹不丢步，紧凑 JSONL/CSV 去数组，文本可每 25 步展示而不改变结构化记录。此方法无 epoch，写 null，不虚构训练 epoch。

2026-09-29 本机在 thin QR 稳定分解修复后重测合成成本：seed930 的独立随机特征，不是物理性能证据，BLAS 限 2 线程。C26K1 的 orbit CE 收敛 91 步、fit 约 0.0113 秒；C26K20 收敛 463 步、fit 约 0.476 秒、solve 约 0.340 秒，终值真实梯度 2.344e-6，小于冻结阈值 2.380e-6。C26K20 single CE 同为 463 步、fit 约 0.405 秒。耗时随机器负载变化，不将本次与修复前的墙钟差异解释为机制提速。真实相关特征的条件与成本可能不同；触发固定 cap 是技术失败，不能按性能选择重试配置。

32 项核心测试覆盖显式完整张量核、PSD、近抵消轨道的归一化与交互项范数、独立 primal CE 解、真实 RKHS 梯度、精确旧 ridge 回归、C4 视图排列不变性、逐 query 批次无关性、物理等权、零/常量/C1、不可变状态与 bytes、输入拒绝、完整物理 OOF/proxy、JSON 序列化、迭代 cap、非有限失败轨迹及后续折/proxy 异常的上下文保留。合成 C4 不变性仅验证公式和实现，不代表真实 held 性能。

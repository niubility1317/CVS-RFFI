结合 `9a6eb5720` 的代码、四个 seed 的原始 JSON/CSV、训练步骤记录和组合诊断，我建议把多解耦路线进一步调整为：

> **多个独立网络分别学习“明确的扰动增量作用于当前信号状态后，会怎样改变特征”，再用经过验证的作用约束身份学习。**

这里需要明确分开三件事：

1. **施加了什么变化**：扰动增量或作用参数。
2. **变化施加在什么信号上**：接受作用的样本自身状态。
3. **这种变化应如何约束身份识别**：保持类别判别能力，以及在合理范围内降低接收条件敏感性。

**当前最值得优先推进的是 L；T 需要先修正参数对照、作用模型和损失尺度；R 需要从质心迁移改成真实分布统计监督；LT 暂时不应成为主要扩容方向。**

下面先解释新结果真正支持什么，再给出数学、通信和射频指纹层面的原因，以及下一版可以直接落实的结构与验证顺序。

---

## 一、新结果已经提供了什么证据

### 1. 这次完成的是作用网络诊断，尚未验证新的身份识别收益

本轮使用四个既有 `multi E200` 身份骨干，冻结参数，重新拟合辅助网络。每个 seed 使用：

- 1,920 个辅助拟合包；
- 960 个独立辅助审查包；
- clean 和源 `practical_mid` 两种条件；
- 合计 14,400 次辅助更新。

提交的 256 条 L/T 指标、24 条 R 指标及汇总统计能够相互复核。但这里的审查包只对**新辅助拟合**留出，原身份骨干训练见过这些源包；四个 seed 也使用相同的物理包划分。

此外，`legacy_self` 是按旧损失形式重新进行的 200 步拟合，没有加载原 E200 辅助网络权重。因此，本轮不能直接解释为“原在线辅助网络已经被完整重新评估”。原 `multi_disentangle` 的身份训练实现也没有随这次诊断一起更新。依据：[本轮报告](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/automation_reports/CV-SincNet/20261009-phase1-multi-action-audit-manysig-m4-r02/report.md)、[实际运行逻辑](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_action_audit/runner.py)。

### 2. L 在源 LEO 条件下的改善值得继续追踪，但需要正确归因

下表使用“另一 TX 在相同干预参数下提供的作用码”。其中：

\[
\operatorname{skill}
=
1-\frac{\text{作用预测 MSE}}{\text{零作用预测 MSE}}.
\]

**这是作用预测的误差改善比例，不是身份识别准确率。**

| 条件与分支 | 原目标 `legacy_self` | 分块＋G 后目标 `block_z_self` | 跨 TX 拟合 `block_z_cross` |
|---|---:|---:|---:|
| clean，L：G 后 skill | 40.25% | 39.25% | 37.88% |
| 源 `practical_mid`，L：G 后 skill | 11.54% | 14.67% | **16.66%** |
| clean，T：G 后 skill | 11.05% | 12.15% | 11.67% |
| 源 `practical_mid`，T：G 后 skill | 10.76% | 11.29% | 10.61% |

源 `practical_mid` 的 L 从 11.54% 到 16.66%，应拆成：

- **改变拟合目标：+3.127 个百分点，4/4 seed 改善。**
- **进一步改用跨 TX 拟合：+1.988 个百分点，4/4 seed 改善。**

因此，不能把全部约 5.115 个百分点都归因于跨 TX 迁移。

对应的 margin 变化预测 MAE 为：

\[
0.45288\rightarrow0.45113\rightarrow0.44468,
\]

同样支持源 LEO 条件下的局部改善。不过，L 的全局 \(h\)-skill 同时从 12.12% 降到 7.22%，说明这里存在**不同拟合目标之间的取舍**。依据：[原始作用指标](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/automation_reports/CV-SincNet/20261009-phase1-multi-action-audit-manysig-m4-r02/results/action_metrics.csv)。

T 分支本身也包含有效信息：正确作用码通常优于零预测和打乱作用码。问题在于，**新增分块目标和跨 TX 拟合，没有产生与 L 相同的额外收益。**

### 3. R 的结果明确说明：特征拟合改善与身份方向改善是两回事

| 条件 | R 的 \(h\)-skill | 真实组 G 变化 MSE：零作用 → 学习作用 | 真实组 margin 变化 MAE：零作用 → 学习作用 |
|---|---:|---:|---:|
| clean | 22.57% | 0.14763 → 0.13315 | **0.76575 → 0.97666** |
| 源 `practical_mid` | 22.09% | 0.60478 → 0.46588 | **3.06153 → 2.86389** |

clean 下，G 变化 MSE 四个 seed 都改善，但 margin MAE 四个 seed 都恶化。源 `practical_mid` 下，两类指标则都改善。

**所以，“能预测一部分接收条件变化”已经得到支持；“这些预测可以直接加强身份解耦”仍需独立检验。**依据：[R 原始指标](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/automation_reports/CV-SincNet/20261009-phase1-multi-action-audit-manysig-m4-r02/results/receiver_metrics.csv)。

还有一个范围需要注意：本轮 clean 和 LEO 条件分别拟合辅助模型。它尚未证明**同一套 L/T 网络能够同时覆盖两种条件**。

---

## 二、通信链路决定了：应建模接收条件的作用，不能任意删除“域信息”

### 1. `equalized=1` 并不意味着信道、频偏和接收机影响都已消失

WiSig 原论文明确说明，其均衡处理包括：

1. 从 25 Msps 重采样到 20 Msps；
2. 利用 L-STF 检测前导、估计和补偿频偏；
3. 利用 L-LTF 估计信道并进行 MMSE 均衡；
4. **重新施加频偏，用于指纹识别**；
5. 重采样回 25 Msps，保留前 256 个样本。[arxiv.org](https://arxiv.org/html/2112.15363v2?utm_source=chatgpt.com)

这对当前设计有直接影响。

首先，T 处理的频率和相位变化仍可能与 TX 身份证据重叠。观测频偏大致包含：

\[
\Delta f_{\mathrm{obs}}
=
\Delta f_{\mathrm{TX}}
-\Delta f_{\mathrm{RX}}
+\Delta f_{\mathrm{Doppler}}
+\Delta f_{\mathrm{residual}}.
\]

它既包含发射机信息，也包含接收机和动态条件。已有 WiFi 射频指纹研究同样观察到，CFO 会漂移，但在特定时间和环境范围内仍可能提供身份区分信息。[arxiv.org](https://arxiv.org/html/2412.07269v2?utm_source=chatgpt.com)

其次，在均衡后的 IQ 上施加 FIR，得到的是**后处理表示上的受控线性变化**。它不能自动代表真实信道变化经过同步、信道估计、MMSE 均衡和重采样后的全部效果。

因此，当前 L/T 的语义最好保持为：

- L：指定数字线性增量造成的特征变化；
- T：指定数字相位演化造成的特征变化；
- R：条件匹配后，真实源 RX 数据之间的统计差异。

这些语义足以开展有效研究，不必先把它们解释成纯硬件潜变量。

### 2. 更强的不变性可能删除身份证据

在线性近似下：

\[
X(\omega)
=
R_r(\omega)H_c(\omega)T_y(\omega)S(\omega).
\]

对于任意可逆的 \(K_y(\omega)\)，令：

\[
T'_y=K_yT_y,\qquad H'_c=H_c/K_y,
\]

仍可产生相同观测。这说明，仅凭有效频率响应，一般无法唯一决定哪部分属于 TX、哪部分属于传播链路。

从局部几何看，若：

\[
\delta x=J_y\delta\theta_y+J_d\delta\theta_d,
\]

而身份表示被要求满足：

\[
J_FJ_d=0,
\]

那么，只要某部分 TX 差异方向也落在 \(\operatorname{col}(J_d)\) 中，这部分身份证据就会同时被压掉。

这不是说域不变性不可用，而是说明其目标应该是**在可验证的接收条件范围内降低不必要的敏感性，同时保留类别区分**。域不变表示的理论研究也明确讨论了非可逆压缩造成的信息损失，以及过强不变性带来的限制。[proceedings.mlr.press](https://proceedings.mlr.press/v89/johansson19a.html?utm_source=chatgpt.com)

因此，下一版不能用“域预测越差”或“所有作用方向都被消除”作为主要成功标准。

---

## 三、最根本的数学问题：\(h\) 是否足以决定扰动后的特征？

这是我认为比“分支容量是否够”更值得优先检验的问题。

当前模型希望：

\[
D_k(h,p)\approx E(A_k(p)x)-E(x),
\qquad h=E(x).
\]

但存在精确作用模型，至少需要满足：

\[
\boxed{
E(x)=E(x')
\Longrightarrow
E(A_k(p)x)=E(A_k(p)x')
}
\]

也就是说，**被 \(E\) 压缩成同一表示的信号，经过相同扰动后，也必须仍能由同一表示预测。**

身份特征不天然满足这个条件。

### 一个简单反例

令：

\[
x=(a,b),\qquad E(x)=a,
\]

扰动为：

\[
A_p(a,b)=(a+pb,b).
\]

那么：

\[
E(A_px)-E(x)=pb.
\]

若只知道 \(h=a\) 和 \(p\)，不知道被编码器丢掉的 \(b\)，就无法预测这个变化。增加网络深度或宽度也不能补回缺失信息。

在平方误差下，即使允许任意函数，最优作用预测仍是：

\[
D^\star(h,p)=\mathbb E[\Delta h\mid h,p],
\]

其不可消除的风险为：

\[
\inf_D\mathbb E\|\Delta h-D(h,p)\|^2
=
\mathbb E\operatorname{tr}
\operatorname{Cov}(\Delta h\mid h,p).
\]

这是一个**状态信息是否充分**的问题，不只是拟合能力问题。

### 对多解耦架构的直接修改

我建议改成：

\[
\boxed{
\widehat{\Delta h}_{k,b}
=
D_k\!\left(h_b,s_k(x_b),\xi_{k,a}\right)
}
\]

其中：

- \(b\)：接受作用的样本；
- \(s_k(x_b)\)：接受作用样本自身的局部状态；
- \(\xi_{k,a}\)：由另一个样本的受控观测提供的扰动增量；
- \(D_k\)：每种因素独立的作用网络。

这里要明确区分：

> **扰动码描述“发生了什么变化”；样本状态描述“这个变化作用在什么对象上”。**

允许 \(s_k(x_b)\) 保留与 TX 有关的波形状态是合理的。相同滤波器对不同频谱的信号产生不同变化，相同相位演化经过不同非线性特征提取器，也会产生不同特征响应。

真正需要跨 TX 共享的是**作用规则和增量语义**，并非强迫所有条件输入都完全没有身份信息。

现有结果尚未证明 \(h\) 的状态信息不足。下一轮应通过 \(D(h,p)\) 与 \(D(h,s,p)\) 的匹配对照来检验这一假设。

---

## 四、当前人工 L/T 的参数可以解析估计，不能把所有困难归因于“观测不足”

这一点对 T 尤其重要。

### 1. T 的人工配对具有精确代数关系

当前人工变化为：

\[
x'[n]=e^{j\phi[n]}x[n].
\]

因此：

\[
x'[n]\overline{x[n]}
=
|x[n]|^2e^{j\phi[n]}.
\]

只要幅度非零，且相位变化没有跨越主值范围，就有：

\[
\arg\left(x'[n]\overline{x[n]}\right)=\phi[n].
\]

本实现的相位幅度满足：

\[
|\phi[n]|\le0.08+0.10+0.06=0.24<\pi.
\]

于是可以先计算相位差，再拟合：

\[
\phi=Bp_T,\qquad
B=
\begin{bmatrix}
0.08\mathbf 1&0.10u&0.06u^2
\end{bmatrix},
\]

得到：

\[
\hat p_T
=
(B^\top WB)^{-1}B^\top W\phi.
\]

**原始接收噪声已经包含在 \(x\) 内，并与信号一起被数字旋转。它并不是这两个配对观测之间的独立噪声。**

因此，80 点窗口条件数约 45.76、全 256 点约 4.73，说明短窗口对额外误差和有限精度更敏感；它不能直接证明当前确定性人工配对存在不可消除的参数估计噪声下限。

这也意味着：让两个观测分别经过网络，再做 \(N(v_1)-N(v_0)\)，可能把一个具有明确代数结构的问题变得不必要地困难。

### 2. L 同样具有线性回归形式

当前：

\[
x'-x
=
0.06\left(c_1Sx+c_3S^3x\right),
\]

其中：

\[
c_1=p_0+jp_1,\qquad c_3=p_2+jp_3.
\]

令：

\[
X_L=[Sx,\ S^3x],
\]

当其满列秩时：

\[
\hat c
=
\frac1{0.06}(X_L^HX_L)^{-1}X_L^H(x'-x).
\]

实际实现应检查矩阵条件数，并使用稳定的最小二乘求解。STF 有周期性，并不自动意味着延迟 1 和延迟 3 的两列不可区分。

以上关系来自当前明确的人工变换，依据：[L/T 变换实现](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_disentangle/physics.py)。

### 下一版应有三个分开的对照

| 增量来源 | 检验的问题 |
|---|---|
| 真实生成参数 \(p\) | 给定正确增量，作用网络能否预测特征变化？ |
| 解析配对估计 \(\hat p_{\mathrm{analytic}}\) | 实际观测视图是否保留了足够的增量信息？ |
| 神经估计 \(\hat p_{\mathrm{neural}}\) | 学习式估计器相对解析关系增加了什么能力，或引入了什么损失？ |

人工增强时，参数本来就是已知的。可以先独立训练好“给定参数的作用网络”，再研究神经估计器的价值。

这些解析关系只适用于当前同包受控变化。真实跨 RX 的两包信号一般没有这样的精确对应，不能照搬。

---

## 五、当前“已知参数 oracle”有结构偏差，不能作为性能上界

新报告最容易误读的结果之一是：clean T 的参数 oracle 平均为：

\[
h\text{-skill}=-2.55\%,\qquad
G\text{-skill}=-3.21\%.
\]

这不意味着真实参数没有用。

### 1. 这个 oracle 强制了不一定正确的奇对称性

当前受限算子对固定 \(h\) 的 \(q\) 满足：

\[
D(h,-q)=-D(h,q).
\]

而 oracle 将 \(p\) 直接补零到 8 维，于是进一步强制：

\[
D(h,-p)=-D(h,p).
\]

但真实作用通常包含偶数项。

即使 \(E\) 是恒等映射，常相位旋转也有：

\[
\Delta(p)=(e^{j\phi(p)}-1)x,
\]

其偶部为：

\[
\Delta_{\mathrm{even}}(p)
=
\frac{\Delta(p)+\Delta(-p)}2
=
(\cos\phi(p)-1)x,
\]

一般不为零。

对对称的 \(p\) 分布，任何严格奇函数预测器的平方误差，至少包含：

\[
\mathbb E\|\Delta_{\mathrm{even}}(p)\|^2.
\]

与此同时，神经观测码：

\[
q(p)=N(v(A_px))-N(v(x))
\]

对物理参数 \(p\) 不必奇对称。因此，当前 oracle 并非仅仅删除了参数估计误差，还可能改变了可表示的函数范围。依据：[参数对照实现](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_action_audit/actions.py)、[受限算子实现](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_disentangle/model.py)。

**建议保留零增量恒等性，但去掉不必要的同状态反号约束。**例如：

\[
a(p)=\operatorname{MLP}(p)-\operatorname{MLP}(0),
\]

可以满足 \(a(0)=0\)，同时允许偶数项。

对相位族，逆作用应在变换后的状态上检验；当前两 tap FIR 族则不对组合封闭，不能把 \(-p\) 当成精确逆滤波器。

### 2. 分块归一化使近乎不变的块获得了很大权重

clean T 在拟合角色中的三块作用能量约为：

\[
s_{\mathrm{base}}\approx3.99\times10^{-3},
\]

\[
s_{\mathrm{PA}}\approx8.44\times10^{-5},
\]

\[
s_{\mathrm{ref}}\approx3.48\times10^{-7}.
\]

当前逐块除以这些能量，再平均。于是 reference 块中很小的绝对误差，会被大幅放大。

分块模型确实改善了近乎不变 reference 块的相对误差，但全局 \(h\)-skill 明显下降，G 后改善又很有限。这说明“每块相对误差一样重要”本身需要论证。

参数 oracle 的优化也明显更困难：clean T 前 20 步平均梯度范数约 1510，末 20 步仍约 41.6；对应 learned 模式末段约为 0.6。多数作用模型在第 200 步前仍持续下降。依据：[步骤记录](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/automation_reports/CV-SincNet/20261009-phase1-multi-action-audit-manysig-m4-r02/results/audit-s2026092701/clean_actions_steps.csv)，上述数字为四 seed 同名记录汇总。

因此，应先排查参数尺度、初始输出尺度、偶部表达和损失权衡，再讨论最终容量。

### 3. 当前谱也不支持优先扩大 rank

该算子的所有输出仍落在最多 16 维的固定全局子空间中。对目标差分矩阵，其最佳 16 维投影能够保留的平均能量为：

| 分支 | clean | 源 `practical_mid` |
|---|---:|---:|
| L | 81.65% | 80.87% |
| T | 92.83% | 91.51% |

而当前若干模式的 \(h\)-skill 只有约 5%–20%。

最佳投影能量不是当前模型一定能达到的成绩，但当前成绩距离这个几何上界很远。**现在没有充分依据把主要瓶颈归结为 rank 8 太小。**

还要区分两个概念：一个因素只有三个参数，并不意味着所有样本的作用向量都落在同一个三维空间。不同样本上的局部作用方向可以不同。

---

## 六、下一版 L/T：明确增量、保留样本状态、精确处理已知前端

我建议保留独立的 L、T 网络，并尝试以下结构：

\[
\widehat{\Delta h}_k
=
J_k(h,s_k)\xi_k
+
Q_k(h,s_k)\operatorname{vec}_{\mathrm{sym}}
(\xi_k\xi_k^\top).
\]

其中：

- \(J_k\) 描述状态相关的一阶作用；
- \(Q_k\) 描述必要的二阶变化；
- 零增量自动产生零作用；
- 作用方向可以随接受作用的样本变化。

这是一个待验证的候选，不需要一开始就把二阶部分做大。先用一阶与一阶＋二阶对照，检查增加表达能力是否改善独立源审查结果。

局部方向监督本身已有 Tangent Prop 等工作作为先例；这里更有价值的问题是，这些方向能否按明确的 RF 增量分别学习、跨样本迁移，并对身份判别提供额外收益。[proceedings.neurips.cc](https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf?utm_source=chatgpt.com)

### 一个更直接、优先级更高的实现改动：29 维固定响应精确计算

当前：

\[
h=[h_{\mathrm{base}},h_{\mathrm{PA}},r(x)],
\]

末 29 维 \(r(x)\) 是固定可计算的 reference response。

人工增强时，接受作用的 \(x\) 和参数 \(p\) 都已知，因此可以构造：

\[
\tilde h_k=
\left[
h_{\mathrm{base}}+\widehat{\Delta h}_{k,\mathrm{base}},
\quad
h_{\mathrm{PA}}+\widehat{\Delta h}_{k,\mathrm{PA}},
\quad
r(A_k(p)x)
\right].
\]

这样，网络优先学习其余 320 维响应。

这一改动与新数据中的两个问题直接对应：

- L 使 reference 模板路由改变的比例约为 30.83% / 31.98%；
- T 的 reference 差分能量非常小，逐块逆能量加权却会放大它。

**能够精确计算的固定前端响应，可以先从自由回归任务中拿出来。**这样不会改变身份骨干的原始前向定义，也保留了多个独立作用网络。

需要将额外计算 \(r(A_px)\) 的成本计入比较；这一做法适用于可生成的数字动作，不意味着真实跨 RX 可以得到精确逐包反事实。

另外，当前 delay jump 诊断需要一个明确修正：模板 bank 以 20 个样本为周期，却使用普通绝对差。应改用：

\[
d_{\mathrm{circ}}
=
\min(|d_1-d_0|,\ 20-|d_1-d_0|).
\]

例如 \(19.9375\rightarrow0\) 是模 20 的邻近变化，不能记成约 20 个样本的物理时延跳变。即使修正后，这个量也只是模板路由变化。依据：[固定参考模块](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_reference_identity/model.py)、[当前物理诊断](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_action_audit/action_checks.py)。

---

## 七、R 应转向条件分布作用，先解决目标不一致和弱基线

新 R 已经修正了原始 IQ 均值抵消、旧贡献刷新和未经验证的 L/T 残差扣除问题。这些旧问题不应继续作为当前诊断的主要批评。

但还剩下更深的统计问题。

### 1. 当前拟合的对象仍然是 \(G(\bar h)\)

R 训练使用：

\[
\widehat{\Delta h}_R\approx\bar h_{r'}-\bar h_r,
\]

以及：

\[
G(\bar h_r+\widehat{\Delta h}_R)
\approx G(\bar h_{r'}).
\]

真实组级身份统计却是：

\[
\bar z_r=\frac1{n_r}\sum_iG(h_{r,i}).
\]

一般而言：

\[
G\left(\frac1n\sum_i h_i\right)
\ne
\frac1n\sum_iG(h_i).
\]

报告已经量化了这一差异。clean 下，质心 margin 变化与平均逐包 margin 变化的差距约为 **0.728**，而真实组 margin 变化绝对均值约为 **0.766**。这个近似误差与待解释的信号处于相近量级。

因此，即使质心迁移相当准确，真实群体 margin 仍可能预测不好。当前 R 的误差不能全部归因于作用网络，也不能只靠提高拟合权重解决。依据：[R 诊断实现](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_action_audit/receiver.py)、[clean R 记录](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/automation_reports/CV-SincNet/20261009-phase1-multi-action-audit-manysig-m4-r02/results/audit-s2026092701/clean_receiver.json)。

### 2. 改成逐包施加共享作用，再聚合

下一版应计算：

\[
\boxed{
\hat\mu^z_{y,r\rightarrow r',c}
=
\frac1{n_{y,r,c}}
\sum_i
G\!\left(
h_i+D_R(h_i,q_{r\rightarrow r',c})
\right)
}
\]

目标为真实目的源 RX 的统计：

\[
\mu^z_{y,r',c}
=
\frac1{n_{y,r',c}}
\sum_jG(h_j).
\]

margin 同样先逐包计算，再取平均。

这里还有一个关键要求：

\[
q_{r\rightarrow r',c}
\]

应尽可能在多个 TX 间共享，或由不包含当前被评估 TX 的组估计。**共享网络参数，并不自动意味着共享了同一个 RX 作用。**如果每个 TX 的组描述产生完全不同的 \(q_R\)，模型仍可能学到 TX×RX 特定关系。

训练目标可以包含实际均值特征、各竞争类别 margin，以及固定目的源 RX 类别锚点：

\[
\mathcal L_R
=
\|\hat\mu^z-\operatorname{sg}(\mu^z_{\mathrm{dest}})\|^2
+
\lambda_m
\|\hat{\boldsymbol m}-\operatorname{sg}(\boldsymbol m_{\mathrm{dest}})\|^2
+
\lambda_a\mathcal L_{\mathrm{anchor}}.
\]

这仍然是统计分布约束，不会恢复唯一的逐包配对。

### 3. 当前 `fit_mean` 基线几乎重复了零预测

代码同时枚举 \(r\rightarrow r'\) 和 \(r'\rightarrow r\)，所以：

\[
(\bar h_{r'}-\bar h_r)+(\bar h_r-\bar h_{r'})=0.
\]

对全部有向关系求均值，理论上自然接近零。当前 `fit_mean` 的预测范数约为 \(10^{-8}\)，因此它没有提供比零作用更强的比较。

应增加：

\[
\bar\Delta_{r,r',c}^{(-y)}
=
\operatorname{mean}_{y'\ne y}
\left(
\bar h_{y',r',c}-\bar h_{y',r,c}
\right),
\]

即按**有向 RX 对 × 条件**、利用其他 TX 估计的平均作用。

如果神经 R 不能超过这个基线，复杂网络的必要性就尚未得到证明。

### 4. 输入统计和目标统计的估计误差应分开

设：

\[
\hat\mu_r=\mu_r+\epsilon_r.
\]

则观测差分：

\[
\hat\Delta
=
\Delta+\epsilon_{r'}-\epsilon_r.
\]

如果作用码、输入均值和目标差分来自同一批组样本，样本波动会同时出现在输入和目标中。

在组内独立同分布、二阶矩存在的条件下：

\[
\operatorname{Cov}(\hat\Delta)
=
\frac{\Sigma_{r'}}{n_{r'}}
+
\frac{\Sigma_r}{n_r}.
\]

因此，下一版应通过独立分袋分别估计作用描述、接受作用的源样本和目的统计，并用重复分袋检查稳定性。条件匹配还应检查源 day 的组成，避免把组间采样比例差异当成稳定 RX 作用。

本轮锚点损失仅做了损失值和输入梯度诊断，尚未参与身份优化，不能据此宣布锚点方案有效或无效。

---

## 八、跨 TX 迁移需要补齐对照，不能只看 own/cross 差距缩小

当前 donor 只保证 TX 不同，没有同时匹配 RX、day 和接收质量。

所以：

\[
\text{cross error}-\text{own error}
\]

混合了换包、换 TX、可能换 RX，以及作用码估计状态变化等效应。

建议加入以下递进对照：

| 作用码来源 | 保持什么 | 主要检验什么 |
|---|---|---|
| 当前包，相同 \(p\) | 所有样本条件 | 自身作用拟合 |
| 同 TX、不同包，相同 \(p\) | 身份，尽量匹配 RX/条件 | 换包造成的损失 |
| 不同 TX、匹配 RX/条件，相同 \(p\) | 接收条件 | 跨身份的增量迁移 |
| 不同 TX、不匹配条件，相同 \(p\) | 仅增量 | 联合状态变化下的迁移 |
| 保持条件、打乱 \(p\) | 波形或身份背景 | 是否真正使用增量信息 |

一个具体例子说明为什么不能只看差距：

clean L 从 `block_z_self` 到 `block_z_cross` 后，own/cross 的 G-skill 差距从约 **6.34 pp** 缩到 **1.74 pp**。但 own 和 cross 的绝对成绩都下降了。

源 `practical_mid` 的 L 才呈现较有意义的变化：cross 提高，own 基本保持。

此外，当前 `block_z_cross` 实际是用 cross 输入**替换** own 输入，并非额外加入 own＋cross 联合损失；每包的 \(p\) 和 donor 也在缓存时固定，200 步重复使用。依据：[作用拟合与 donor 构造](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_action_audit/actions.py)。

下一轮可以重新采样干预和 donor，或为每包固定多个预登记增量，以检验更广的作用范围。各对照必须使用相同样本曝光与生成日程。

如果要检验作用规则是否能泛化到未参与拟合的 TX，可以只在**辅助作用模型**内做 TX 交叉留出。身份骨干仍然使用全部合法源 RX 和源训练数据，不必牺牲原身份训练的数据覆盖。

---

## 九、身份训练要以判别相关误差为中心，并单独验证虚拟动作的梯度

### 1. \(h\)-MSE 和 G 后 MSE 都不足以单独决定准入

当前分类头还会对特征归一化。因此：

\[
\|z_{\mathrm{pred}}-z_{\mathrm{true}}\|^2
\]

仍不完全对应分类决策误差。

我建议增加归一化身份特征与各竞争类别 margin 的作用预测：

\[
\bar z=\frac{z}{\|z\|+\epsilon},
\]

\[
\boldsymbol m_y(z)
=
\big(C_y(z)-C_j(z)\big)_{j\ne y}.
\]

作用拟合目标可采用：

\[
\mathcal L_{\mathrm{act}}
=
\mathcal L_{\mathrm{feature}}
+
\lambda_z\|\bar z_{\mathrm{pred}}-\bar z_{\mathrm{true}}\|^2
+
\lambda_m
\|\boldsymbol m_y(z_{\mathrm{pred}})
-\boldsymbol m_y(z_{\mathrm{true}})\|^2.
\]

其中 \(E/G/C\) 在作用拟合阶段固定，但 G 和 C 对预测特征的梯度保留。

这能帮助区分：

- 特征坐标误差较大，但分类几乎不受影响；
- 特征误差不大，却恰好改变了关键类别边界。

本轮 R 的 clean 结果已经说明这种区分是必要的。

### 2. 精确的虚拟动作，也不等价于真实增强的骨干梯度

在当前参数点：

\[
E_\theta(x)+
\operatorname{sg}\!\left[
E_\theta(Ax)-E_\theta(x)
\right]
=
E_\theta(Ax).
\]

前向值可以相同，但对前段参数的导数不同：

\[
\nabla_{\theta_E}\mathcal L_{\mathrm{real}}
=
J_{\theta_E}E(Ax)^\top g,
\]

\[
\nabla_{\theta_E}\mathcal L_{\mathrm{virtual}}
=
J_{\theta_E}E(x)^\top g.
\]

这是虚拟特征训练自身的机制，不是实现错误。

本轮梯度诊断使用的是**精确差分**，并未测量 learned action 的梯度；参考也只是单项有标签 CE，不是完整的 clean＋LEO＋pseudo 原生目标。`practical_mid` 文件中的 `clean_CE_reference` 实际对应该 LEO 输入上未再施加 L/T 的 CE。依据：[梯度诊断实现](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/experiments/cvs_multi_action_audit/runner.py)。

首次整合时，值得明确比较：

- 真实 IQ 路径正常更新 E/G，虚拟动作只更新 G/分类器；
- 真实 IQ 路径正常更新 E/G，虚拟动作也更新 E/G。

这样才能判断收益来自后段判别鲁棒性，还是虚拟梯度对前段学习确实有帮助。

### 3. 真实增强 CE 与学习作用可靠性应分开

建议总体形式为：

\[
\mathcal L_{\mathrm{id}}
=
\mathcal L_{\mathrm{native}}
+
\lambda_{\mathrm{real}}\mathcal L_{\mathrm{real\ IQ}}
+
\lambda_{\mathrm{virt}}
\sum_k w_k(c)\mathcal L_{\mathrm{virtual},k}
+
\lambda_R\mathcal L_{\mathrm{R,group}}.
\]

这里：

- 真实增强 CE 使用独立权重；
- \(w_k(c)\) 根据源端校准中的作用误差、判别相关误差和稳定性确定；
- 新增网络不自动增加总辅助梯度预算；
- R 保持组级监督；
- 原 U 数据继续遵循既定伪标签门控。

当前正 skill 是在 960 包聚合后得到的，它不等于逐包可靠性。在线整合后，还必须检查参考编码器变化是否导致作用模型失配。新贡献时间戳和版本隔离机制在本轮固定参考诊断中没有经历实际的在线坐标漂移验证。

---

## 十、LT 的当前证据支持先完善主效应

本轮四 seed 平均：

| 指标 | clean | 源 `practical_mid` |
|---|---:|---:|
| 四角残差能量 | \(1.13\times10^{-4}\) | \(1.10\times10^{-4}\) |
| 真实 L→T 与 T→L 的 \(h\)-MSE | \(2.16\times10^{-9}\) | \(2.38\times10^{-7}\) |
| `block_z_cross` 加性组合预测 \(h\)-MSE | \(3.01\times10^{-3}\) | \(2.96\times10^{-3}\) |

这里有一个直接可解释的比较：

如果真实 L/T 主效应已经完全知道，那么加性组合留下的误差就是四角残差。当前学习组合的误差却约为这一水平的 **27 倍**。

而把当前加性算子改成顺序执行，改善幅度极小。依据：[组合诊断](https://github.com/niubility1317/CVS-RFFI/blob/9a6eb5720/automation_reports/CV-SincNet/20261009-phase1-multi-action-audit-manysig-m4-r02/results/audit-s2026092701/clean_composition.json) 及四 seed 同名记录。

因此，当前大部分预测困难还没有被证明来自缺少 LT 网络。

同时：

\[
h_{11}-h_{10}-h_{01}+h_{00}
\]

包括有限增量组合、编码器曲率等影响，不等同于非交换性，也不自动成为唯一的物理交互分量。功能 ANOVA 中交互的归属还依赖分布和中心化条件。[proceedings.mlr.press](https://proceedings.mlr.press/v108/lengerich20a.html?utm_source=chatgpt.com)

后续若增加 LT，必须保持：

- 相同的 \(x_{11}\) 曝光；
- 相同的 L/T 主效应权重；
- 相同的总辅助预算；
- 无 LT、零 LT、学习 LT 的明确对照。

当前阶段，LT 可以继续保留为诊断任务。

---

## 十一、我建议下一轮这样推进

### 第一阶段：修正作用问题的定义和对照

先完成一组范围明确的源端改进：

| 项目 | 具体改动 | 要回答的问题 |
|---|---|---|
| 参数估计 | 真实 \(p\)、解析配对估计、神经估计分开 | 损失来自观测估计还是作用模型？ |
| oracle 函数范围 | 保留零增量，允许偶数项；加入 \(0,+p,-p\) | 现有奇对称性是否造成可测偏差？ |
| 固定前端 | 349 维全学习，对照精确 29 维＋学习 320 维 | 已知前端是否占用了不必要的拟合能力？ |
| 状态信息 | \(D(h,p)\) 对照 \(D(h,s,p)\) | 中间身份特征是否足以决定作用？ |
| donor 对照 | 增加同 TX 不同包、匹配 RX/条件 | 跨 TX 迁移损失能否与换包效应分开？ |
| R 统计 | 逐包作用后聚合、独立分袋、方向条件均值基线 | 是否学到稳定且跨 TX 共享的 RX 条件作用？ |

固定 200 步可以继续作为等预算对照，但不能把它当成能力上限。应同时报告学习曲线和优化稳定性；若预算耗尽时仍明显下降，应保留“预算内尚未充分拟合”的解释。

### 第二阶段：用同一套网络联合覆盖 clean 和源 LEO

当前两个条件是分别拟合的。下一版需要验证：

\[
D_L,\ D_T,\ D_R
\]

各自能否在同一组参数下覆盖已登记的源条件，并检查分层表现。特别要防止源 LEO 改善是以 clean 作用质量明显下降为代价。

其中，L 是最适合优先进入这一阶段的分支。

### 第三阶段：开展最小但有区分力的身份训练矩阵

在作用模型达到可接受的源端证据后，建议依次比较：

| 对照 | 主要目的 |
|---|---|
| 原生身份基线 | 确认最终识别是否改善 |
| 相同额外真实视图＋CE/一致性 | 排除仅由增强和额外监督带来的收益 |
| 单个统一作用网络 | 检验多个独立网络的必要性 |
| L 独立作用网络 | 确认当前最有证据的分支是否带来身份收益 |
| L＋T 独立作用网络 | 检验分工后的互补性 |
| L＋T＋R 分布作用 | 检验真实跨 RX 统计约束的额外贡献 |

LT 根据前面的组合证据再加入，不必与所有尚未解决的问题同时展开。

继续保留原来的 clean＋LEO 拼接、旧伪标签门控、cosine 调度和源数据契约。额外视图、辅助更新、解析计算、模型参数和梯度预算需要单独核算。设计与选择规则在源端确定后冻结，再进行独立目标评分。

---

## 最值得作为下一版核心假设的内容

已有 RF 多因素表示工作采用因素分类、重构和表示交换，因此，“增加多个编码器”本身很难成为充分的研究贡献。[arxiv.org](https://arxiv.org/html/2508.12660v1?utm_source=chatgpt.com)

你这条路线更有价值的研究假设可以写成：

> **对于可观测的接收条件增量，将增量语义与接受作用样本的状态分开建模，由多个独立网络学习可迁移的条件作用，并以判别相关误差和真实组统计验证这些作用，能够为身份骨干提供超出同等增强与统一作用网络的训练信息。**

落实到下一版，我最优先推荐的组合是：

\[
\boxed{
\text{明确增量的独立 L/T}
+
\text{接受作用样本的状态条件}
+
\text{固定 29 维响应精确处理}
+
\text{R 的逐包作用后统计监督}
}
\]

这几项分别对应本轮已经暴露的参数对照、函数限制、损失尺度和统计目标问题。**先把这些环节做对，再检验它们能否共同改善身份骨干，是目前推进多解耦域泛化最有依据的方向。**
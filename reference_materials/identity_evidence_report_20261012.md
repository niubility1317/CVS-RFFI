**我核对后的核心判断是：目前最有证据支持的瓶颈，是辅助目标与跨接收机身份判别所需要的信息不够一致。模型确实在更新，部分方案也有小幅收益；但大量纠错被新增错误抵消，后续更强、更复杂的约束没有持续改善主要混淆。**

因此，后续研究应优先回答两个问题：

1. **当前特征空间里，究竟有没有被现有分类器充分利用、又能跨接收条件保持区分力的身份证据？**
2. **新增约束是在帮助模型使用这些证据，还是在同时削弱接收条件变化和身份差异？**

下面结合最新代码、实验结果、数学结构和射频观测条件展开。

## 一、我核对了哪些进展，结果究竟说明什么

我查看的是实际研究分支：

[`codex/report-physical-ssl-clean-delivery-20261011`](https://github.com/niubility1317/CVS-RFFI/tree/codex/report-physical-ssl-clean-delivery-20261011)

本次固定到提交 [`51820aee`](https://github.com/niubility1317/CVS-RFFI/commit/51820aee7690e09ca9afd239231095b9ecf1e7f2)，时间为 **2026-10-12 01:47 UTC，即北京时间 09:47**。核对覆盖了 **11 组已经完成的 E20 训练矩阵，共 261 个新模型**，包括原始评分统计、源侧诊断和相关训练实现。

最新九行条件交互实验已经启动，但这个仓库快照中**尚无最终测试成绩**。它们需要单独保留为待验证项，不能算作已经失败或已经有效的实验。[九行条件实验状态](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-conditional-fishr-manysig-m9-r01/report.md)

### 1.1 严格地说，目前有小幅收益，但没有形成持续进展

下面列出最有解释力的几组结果。准确率均为**完整三个种子的 clean 均值**。

| 实验与方案 | 准确率 | 应如何理解 |
|---|---:|---|
| 原始纯 CE E20：P0 | **80.0504%** | 当前指定参考基线 |
| cross-style：同路径 CE 续训 | 79.8883% | 判断该批新增机制的必要对照 |
| cross-style：SSL | 80.1657% | 使用独立同伴教师的 U 监督 |
| cross-style：浅、中层 F_BOTH | **80.4302%** | 相对同批 CE **+0.5419 个百分点**；相对 SSL **+0.2645** |
| cross-style：JOINT | 80.3484% | 加入组合后低于 F_BOTH |
| scratch-boundary：JOINT | 78.7946% | 从零训练并未自动解决问题 |
| scratch-calibrated：DELAYCAP_JOINT | 79.9339% | 校准、限幅后恢复到基线附近 |
| bidirectional：DELAYCAP_POS / BIDIR | 79.9470% / 79.1216% | 更强的双向变化反而退化 |
| 最新 energy：MARGIN / CE | 80.1010% / 79.9806% | 保留敏感度幅度后，仍在约 80% 附近 |

来源：[cross-style 完整结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-cross-style-manysig-m33-r01/test_results.md)、[从零训练结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-boundary-style-manysig-m27-r01/test_results.md)、[校准结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-calibrated-style-manysig-m36-r01/test_results.md)、[双向增强结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-bidirectional-style-manysig-m18-r01/final_analysis.md)、[最新 energy 结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01/final_analysis.md)。

这里需要特别区分：

- **+0.5419 个百分点是 F_BOTH 相对同批 CE 的收益，其中包含 SSL 的贡献。**
- Fishr 在同批 SSL 上的增量是 **+0.2645 个百分点**。
- F_BOTH 相对原 P0 的收益是 **+0.3798 个百分点**。

F_BOTH 相对同批 CE、相对同批 SSL 的三个种子配对均为正，这部分小收益值得保留。但目前没有证据支持“继续叠加更多机制，就能将这些收益持续累加”。

### 1.2 “模型没有变化”与原始预测记录不符

我重新核对了原始混淆和修复／破坏计数。

以三个种子合计为例：

- cross-style JOINT 修复了 P0 的 **13,851** 次错误，同时破坏了 **12,349** 次原本正确的预测，净增加 **1,502** 次正确。
- 最新 CE_ENERGY 修复了 **11,451** 次错误，同时破坏了 **11,803** 次正确预测，净减少 **352** 次正确。

所以，目前很大一部分现象是：

> **模型在改变判断，但这些改变缺乏足够稳定的净收益。**

这些计数是三个模型对同一组测试样本的决策汇总，不是新增了独立物理样本。[cross-style 原始统计](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-cross-style-manysig-m33-r01/evidence/test_results_analysis.json)、[energy 原始统计](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01/evidence/test_results_analysis.json)。

---

## 二、最具体的瓶颈：主导身份混淆基本没有被解决

将三个种子的总体混淆矩阵相加，得到：

| 模型 | 错误总数 | TX `14-7 ↔ 20-19` 双向混淆 | 占全部错误 |
|---|---:|---:|---:|
| P0 | 100,546 | 61,340 | **61.01%** |
| cross-style F_BOTH | 98,632 | 59,903 | **60.73%** |
| cross-style JOINT | 99,044 | 59,769 | **60.35%** |
| CE_ENERGY | 100,898 | 61,381 | **60.83%** |
| MARGIN_ENERGY | 100,291 | 60,910 | **60.73%** |

这比“辅助损失有没有下降”更接近当前识别瓶颈：

**大约六成错误集中在同一对发射机之间，而新增机制基本保留了这个主导错误结构。** 相应原始矩阵见上面的 [cross-style 统计](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-cross-style-manysig-m33-r01/evidence/test_results_analysis.json)与 [energy 统计](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01/evidence/test_results_analysis.json)。

这支持一个重要判断：当前各种变化，尚未充分增加模型在困难接收条件下区分这些身份的能力。

但它还不能告诉我们，究竟属于哪一种情况：

1. 原始观测里有稳定证据，当前编码器没有提取出来；
2. 编码器已经提取出部分证据，分类器主要依赖了更容易但不稳定的方向；
3. 新增不变性约束同时削弱了那部分微弱的区分证据；
4. 某些接收条件下，现有短窗口本身就缺少足够可靠的信息。

**这四种原因需要不同的实验，不能统一用“再加一种正则化”处理。**

这里的特定 TX 混淆只能用于解释已经冻结的结果。后续训练应对所有类别采用同一规则，不能根据目标集这张表给这两个 TX 硬编码特殊权重。

---

## 三、数学上的第一处关键问题：很多实验仍在同一种有限变化空间内调整

最新路线虽然经过了多次改造，但大量变化仍围绕浅、中层的**通道缩放与平移**展开。

当前虚拟作用大致是：

\[
H'_{ict}
=
a_{ic}H_{ict}
+
\operatorname{sg}(\sigma_{ic})b_{ic},
\]

其中 \(a=1,b=0\)，每个样本、每个通道有两个虚拟坐标，沿时间维保持不变。

因此，相应敏感度为：

\[
g_{a,c}
=
\sum_t H_{ct}
\frac{\partial \ell}{\partial H_{ct}},
\]

\[
g_{b,c}
=
\operatorname{sg}(\sigma_c)
\sum_t
\frac{\partial \ell}{\partial H_{ct}}.
\]

这是每层一个 **\(2C\) 维通道仿射方向的观测**。[实际敏感度实现](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_e20_boundary_style/multilevel.py)

它可以反映一部分接收变化，但并没有直接覆盖所有可能重要的变化，例如：

- 局部时间结构的改变；
- 跨通道相关关系的改变；
- 具有记忆的滤波作用；
- 复数响应中的耦合；
- 随输入幅度变化的非线性作用。

这些变化可能部分投影到通道均值和尺度上，但**目前没有证据表明，这个投影已经覆盖未知 RX 的主要错误方向**。

这就解释了为什么以下改进可能都合理，却不产生大的性能突破：

- 从最后一层移到浅层、中层；
- 从辅助头梯度改成主身份敏感度；
- 从有偏平方估计改成独立 A/B 估计；
- 从逐样本归一化改成固定层尺度；
- 增加限幅、延后标定、改变候选搜索。

它们显著改善了**如何在这个变化空间中训练**，但没有自动扩展**这个空间能够解释什么**。

另外，当前方法需要准确命名。原 Fishr 匹配的是损失梯度的域间方差，并据此讨论局部损失几何；最新版已经变成了“主身份损失对虚拟特征作用坐标的敏感度约束”。这种改造可以研究，但不能直接把原论文的解释移植过来，进而宣称已经分离了 TX 与 RX 的物理因素。[proceedings.mlr.press](https://proceedings.mlr.press/v162/rame22a.html?utm_source=chatgpt.com)

---

## 四、数学上的第二处关键问题：域统计一致，不足以保证身份对应正确

### 4.1 混合方差丢失了一部分身份对应关系

当前域内梯度方差可以写成：

\[
\operatorname{Var}(g\mid r)
=
\mathbb E_y
\left[
\operatorname{Var}(g\mid y,r)
\right]
+
\operatorname{Var}_y
\left(
\mathbb E[g\mid y,r]
\right).
\]

它既包含类内变化，也包含类均值之间的散布，所以不能说 Fishr 完全不涉及身份。

但这个统计量仍然不充分。

假设不同 RX 下，六个类别的条件分布只是重新排列。各 RX 混合后的均值、方差，甚至完整协方差都可以相同，但“同一个 TX 对应哪个区域”已经改变。

这意味着：

> **把对角方差升级成完整协方差，也不能替代对同一身份跨 RX 对应关系的约束。**

这是统计量的信息边界。它不表示真实模型已经发生了精确的类别置换，但足以说明“域矩匹配得很好”不能单独证明身份泛化变好。最新仓库已经用条件几何诊断正面研究了这个问题。[条件几何报告](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-conditional-geometry-manysig-m3-r01/report.md)

### 4.2 最新检测到的 TX×RX 交互确实存在，但不能全部视为污染

源侧完整诊断中，固定层尺度后的原始 margin 敏感度能量分解为：

| 分量 | 三种子、两层的范围 |
|---|---:|
| TX 主效应 | **35.38%–54.35%** |
| RX 主效应 | **2.81%–3.93%** |
| TX×RX 交互 | **10.69%–17.82%** |
| 单元内散布 | **32.14%–48.95%** |

这里测的是**分类损失对虚拟作用坐标的敏感度**，不是物理硬件参数，也不是最终身份表示的方差。因此不能将这些比例解释成“多少特征被 RX 污染”。[完整源侧解释](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_source_conditional_geometry/SOURCE_INTERPRETATION.md)

这些结果支持：不同 TX 对接收条件有不同响应，仅用一个与身份无关的共同 RX 偏移，无法解释全部结构。

但下一步“把交互压低”仍然存在风险。

设一个简化作用模型为：

\[
M_{y,r}=A_rv_y+b_r,
\]

其中 \(v_y\) 表示身份响应，\(A_r\) 表示接收作用。

双重中心化后的交互为：

\[
B_{y,r}
=
M_{y,r}
-M_{y,\cdot}
-M_{\cdot,r}
+M_{\cdot,\cdot},
\]

因此：

\[
\boxed{
B_{y,r}
=
(A_r-\bar A)(v_y-\bar v)
}
\]

这说明，**同一个接收作用施加在不同身份上，本来就会产生交互。**

降低 \(B\) 可以通过两种途径完成：

- 学习抵消接收作用；
- 缩小身份差异 \(v_y-\bar v\)。

第二种途径也会降低损失，却可能损害识别。

因此，最新 conditional9 的价值，在于检验条件约束能否改善身份关系。**仅看到交互损失下降，仍然不足以判断成功。** 需要同时检查类间区分、跨 RX 身份对应，以及最终身份头的修复／破坏。[条件交互实现](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_scratch_conditional_fishr/multilevel.py)

---

## 五、增强路线的核心反例：样本更难，不等于更新更有用

### 5.1 双向增强已经明确证明了这一点

最新双向风格实验中，增强实际执行得很充分：

- 反向候选被选择约 **91.5%–94.4%**；
- clean 仅被选择很少次数；
- 所选候选 CE 增大，margin 降低，源批准确率也下降。

但三种设置、三个种子，共 **9/9 配对**的 BIDIR 相对 POS 都下降：

| 设置 | BIDIR 相对 POS 的平均变化 |
|---|---:|
| DELAY | −0.6524 个百分点 |
| CAP | −0.8335 个百分点 |
| DELAYCAP | −0.8254 个百分点 |

这足以否定“只要让源样本更难，泛化就会更好”的设计原则。[源轨迹](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-bidirectional-style-manysig-m18-r01/source_trajectory.md)、[完整结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-bidirectional-style-manysig-m18-r01/final_analysis.md)。

不过，它还不能证明增强一定违反物理规律。更准确的解释是：**这些更难的方向，没有体现出相应的跨接收机训练收益。**

### 5.2 风格统计在射频任务中可能承载身份

MixStyle 的原始动机来自视觉特征统计。原论文也指出，靠近输出的统计可能包含类别信息，混合这些统计会破坏类别空间。[arxiv.org](https://arxiv.org/pdf/2104.02008?utm_source=chatgpt.com)

射频指纹任务更需要谨慎：我们希望识别的，恰恰可能是幅度分布、频谱形状和非线性失真中的微小差异。

所以：

\[
\text{特征均值、标准差}
\]

不能自动等同于：

\[
\text{与 TX 无关的接收风格}.
\]

同 TX、同 day、跨 RX 的供体规则减少了明显错误，但仍不能保证两包只差一个可任意交换的 RX 因素。

### 5.3 风险网络已经学会排序，也没有解决更新效用

另一个已经完成的源侧诊断中，冻结身份模型上的 RANK_STATE 风险排序准确率均值已达到约 **82.23%**。后续真实小步更新效用实验，仍没有显示风险加权能够稳定优于均匀策略。这是旧父模型上的独立源诊断，不应冒充当前 E20 的直接配对结果，但它同样说明：**难度预测准确，不足以证明采样权重有用。**[风险学习结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261011-phase1-rx-risk-learning-manysig-m3-r01/learning_results.md)、[更新效用结果](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261011-phase1-rx-update-utility-manysig-m3-r01/utility_results.md)。

数学上，动作 \(a\) 的局部更新效用更接近：

\[
U(a)
=
L_{\mathrm{ref}}(\theta)
-
L_{\mathrm{ref}}(\theta+\Delta\theta_a),
\]

\[
U(a)
\approx
-\nabla L_{\mathrm{ref}}^\top\Delta\theta_a.
\]

而当前样本难度只衡量 \(\ell_a(\theta)\) 有多大。

**损失值本身没有告诉我们，这次实际参数更新会帮助还是伤害其他接收条件下的身份判别。** 对 AdamW，还必须考虑预条件和已有动量，不能简单用原始梯度大小替代更新效果。

---

## 六、大量 U 已经用了；剩余问题是它提供了多少独立纠错信息

最新版已经覆盖全部 **56,700 个 U**，使用另外两个原始 E20 教师的固定概率均值，采用 ALL soft KL。当前不能再用“U 没覆盖”“高阈值把 U 全过滤了”“EMA 锁死学生”等说法解释结果。[最新实际训练设计](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01/report.md)

问题在于，U 的监督主要来自：

- 相同身份网络；
- 相同合法 L；
- 相同源 RX 和日期；
- 不同随机种子的已有判决。

对应目标为：

\[
\mathcal L_U
=
\mathbb E_{x\sim U}
D_{\mathrm{KL}}
\left(q_T(x)\,\|\,p_\theta(x)\right).
\]

它能够利用更多样本、平滑决策、降低部分随机误差，但当教师共享某种接收条件偏差时，KL 也会鼓励学生保留该偏差。

源 V 的同伴诊断给出了一些直接证据：

- 三模型 ALL3 教师准确率为 **97.2296%**，有一定互补；
- 三个学生共有 **203 个高置信错误**，LOO2 教师只修复 **6 个**；
- 三个原模型都判断错误的 V 物理样本有 **491 个**。[同伴教师完整分析](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-peer-audit-manysig-m3-r01/analysis.md)

这不能用来推算 U 的真实错误率，因为 U 的 TX 标签仍然不可见。但它支持一个研究判断：

> **当前教师提供的独立纠错信息有限；增加同类教师或重复蒸馏，未必足以修复共同错误。**

教师准确率并不是学生的数学上限。只是当前路线还没有证明，学生从 U 中获得了哪些教师原本缺少的、能够跨接收条件使用的身份证据。

同时，更多 U 仍来自同一组源接收环境。**样本数增加与接收环境支持范围增加，是两个不同的变化。**

---

## 七、一些工程问题确实存在过，但修正它们主要恢复了已有水平

这一点很重要，否则研究容易反复回到已经检查过的原因。

### 7.1 辅助梯度曾经过弱，也曾经过强

早期源诊断中，原始 Fishr 对编码器的梯度范数比很小；提高预算后，确实获得了更大的梯度。

从零训练时，又出现了另一种情况：初始化阶段的标定系数在后期失配，中层辅助梯度远大于 CE。延后标定与限幅对此进行了实质修正，性能也从明显退化恢复到了基线附近。[早期梯度诊断](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-error-audit-manysig-m3-r01/analysis.md)、[校准漂移诊断](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-boundary-style-manysig-m27-r01/source_calibration_drift.md)。

但最新版 CE_ENERGY 的联合限幅平均系数约 **0.963–0.993**，多数时候并没有大幅削弱辅助项。因此，其结果不能继续主要归因于“限幅压得太死”。[energy 源轨迹](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01/source_trajectory.md)

### 7.2 CE 与 margin 的逐样本归一化等价问题，最新版已经修正

当前使用的平滑 margin 是：

\[
m=s_y-\log\sum_{j\ne y}e^{s_j}.
\]

于是存在精确关系：

\[
\ell_{\mathrm{CE}}=\operatorname{softplus}(-m),
\]

\[
\nabla_\xi\ell_{\mathrm{CE}}
=
-(1-p_y)\nabla_\xi m.
\]

若分别进行逐样本 RMS 归一化，且不处于数值 epsilon 主导区域：

\[
\frac{\nabla_\xi\ell_{\mathrm{CE}}}
{\operatorname{RMS}(\nabla_\xi\ell_{\mathrm{CE}})}
\approx
-
\frac{\nabla_\xi m}
{\operatorname{RMS}(\nabla_\xi m)}.
\]

整体变号不改变方差。因此，仅将归一化后的 margin 改为 CE，基本没有引入新的幅度信息。

**energy6 已经改用固定层尺度，避免了这一问题。** 但其最终结果仍在基线附近。这说明这个修正是必要的数学澄清，却不足以解决主要识别瓶颈。[最新 energy 敏感度代码](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_scratch_energy_sensitivity/multilevel.py)

### 7.3 统计无偏，也不保证优化有用

源矩诊断显示，旧的小批平方估计相对全 L 参考存在明显膨胀；独立 A/B 交叉估计已经显著修正这个偏差。

但这里有两个层次：

\[
\text{是否准确估计这个统计量}
\]

和

\[
\text{降低这个统计量是否改善身份泛化}.
\]

前者解决后，后者仍然需要独立证据。有符号 A/B 损失偶尔为负也不自动是 bug；无偏同样不代表低方差或每一步方向可靠。[矩估计分析](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-e20-moment-audit-manysig-m3-r01/analysis.md)

### 7.4 热启动已经不能承担主要解释

最新版已经有：

- 随机初始化学生；
- 全身份网络更新；
- 正常的 `2e-4 → 1e-6` cosine 日程；
- 完整 U 使用；
- 浅、中层约束；
- 混合二阶反向传播。

没有发现“整体断图、没有进入优化器”这样的确定阻断问题。[实际训练循环](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_scratch_energy_sensitivity/source.py)

所以，**从训练好的基线继续训练仍然适合快速验证增量机制，但它已经不是当前平台的充分解释。** 从零训练也试过了，仍然没有解决主要混淆。

20 轮是当前固定探索预算，不能自动称为收敛。另一方面，也没有充分证据支持把失败统一归因于“训练轮数不够”。

---

## 八、从通信与物理观测出发，为什么更复杂的解耦网络未必得到更多信息

### 8.1 目前可利用的是有限、经过处理的前导观测

WiSig 的 Id signal 为包开始的 256 个 IQ 样本。原始采样率为 25 Msps，约对应 10.24 μs；equalized 版本经历了检测、重采样、信道估计和 MMSE 均衡等处理。不同 RX 独立运行，跨 RX 捕获没有严格时间对齐保证。[CORES Lab - UCLA](https://cores.ee.ucla.edu/downloads/datasets/wisig/?utm_source=chatgpt.com)

当前参考支路又明确使用 `[80,160]` 窗口，即四个 20 点周期和 12 个前导音调。[参考响应代码](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_reference_identity/model.py)

这些观测可以支持有限响应建模，但不能预先保证恢复完整 PA 记忆、任意接收机非线性或纯 TX 硬件参数。

### 8.2 重复观测能降低估计噪声，但不会自动增加独立激励

例如，若拟合一个候选记忆多项式：

\[
u[n]
=
\sum_{p,m}
c_{p,m}s[n-m]|s[n-m]|^{p-1},
\]

其可辨识性取决于设计矩阵的列是否足够独立。

在有限、周期性的前导激励下，不同阶数和时延对应的列可能高度相关。四次重复有助于估计一致性，但不会自动把独立激励自由度扩大四倍。

因此，增加网络、增加阶数、增加低秩分量之前，更应该先测：

- 设计矩阵的奇异值和条件数；
- 跨重复周期的可复现性；
- 跨 TX、RX、day 的稳定部分；
- 拟合残差是否主要来自噪声、截取或均衡误差。

否则，所谓“更强物理分支”可能只是在拟合一个缺乏辨识依据的高容量残差。

### 8.3 TX 与 RX 的作用天然可能重叠

在局部线性近似下：

\[
H_{\mathrm{eff}}
=
H_{\mathrm{RX}}
H_{\mathrm{CH}}
H_{\mathrm{TX}}.
\]

只观察 \(H_{\mathrm{eff}}\)，通常无法唯一决定各部分如何分配。

更重要的是，**部分 TX 身份本来就体现在线性响应中**。如果把所有频谱形状变化都作为接收条件删除，就可能连同身份证据一起删除。

同样，训练后加一个数字滤波 \(A(x)\)，一般只是对已有接收观测做后处理。它并不自动等价于真实更换 RX，因为真实更换还会影响模拟链路、噪声、非线性及后续估计误差。

所以，多解耦路线最需要建立的是：

> **某个有限作用模型能够解释哪些真实变化，以及在这些变化下哪些身份关系能够被保留。**

物理命名本身不能替代这个证据。

---

## 九、下一阶段我建议怎样调整研究重点

我不建议现在再把更多 Fishr、采样、候选搜索和辅助网络一起叠上去。更有价值的是按下面的顺序缩小不确定性。

### 9.1 先固定可比较的对照，把当前九行实验完整收尾

P0 与新 scratch CE 已有明确训练路径差异：

- P0 使用历史 native 训练器；
- 新 scratch 使用 RX×day 的 `DomainStream` 轮转采样；
- 随机状态重置与样本顺序也不完全相同。

因此，P0 的 80.0504% 与新 scratch CE 约 79.44%–79.55% 的差异，混合了训练器、采样和优化轨迹变化。不能全部解释成随机噪声，也不能全部解释成从零训练不好。[新训练循环说明](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-early-geometry-manysig-m30-r01/report.md)

后续每个核心假设都应有同代码路径、同学生更新预算的对照。教师成本和额外前向另计。

还有一个需要落实的输入约束：**最新主路径仍会使用原 `ExcitationResponse` 显式拼接的 CFO 分量。** 这与当前要求不一致，后续实验应统一到不使用该身份输入的版本，并建立对应对照；现有证据不能把全部平台归因于它。[当前模型构建](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_stage2_rx_risk/model.py)、[输入拼接实现](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/experiments/cvs_reference_identity/model.py)

### 9.2 优先做“稳定身份可读出性”诊断

这是我认为最值得先做的一步，因为它直接区分“没有好特征”与“已有好特征没有被使用”。

在固定的浅层、中层、最终表示上，用合法 L 重新估计：

\[
S_{\mathrm{TX}},\quad
S_{\mathrm{RX}},\quad
S_{\mathrm{interaction}},\quad
S_{\mathrm{within}}.
\]

注意：这里研究的是**实际表示**，不能直接套用之前敏感度统计的比例。

可以用收缩后的广义特征值问题检查稳定线性方向：

\[
S_{\mathrm{TX}}w
=
\lambda
\left(
S_{\mathrm{within}}
+\alpha S_{\mathrm{RX}}
+\beta S_{\mathrm{interaction}}
+\varepsilon I
\right)w.
\]

它回答的是：

> 当前坐标中，是否存在身份均值差异较大、其他变化相对较小的线性方向？

这只是诊断工具，不能直接升级成最终物理解耦方案。尤其要注意：

- 交互不全是噪声，不能依据这个比值把它全部删除；
- 六个全局 TX 均值构成的 \(S_{\mathrm{TX}}\) 秩最多为 5，只覆盖有限的均值判别空间；
- 分母应使用半正定、经过收缩的估计，不能直接把可能不定的 A/B 交叉矩阵拿来求逆；
- 投影与读出要在 L 内交叉拟合，不能在同一批上找方向再证明方向有效；
- 冻结骨干已经见过源 RX，这只能检验新统计读出的迁移，不能称未知 RX 验证。

**如果稳定方向已经存在，优先改进读出与训练目标；如果它们在当前表示中缺失，再投入新特征分支。**

### 9.3 下一种身份约束，应直接检验跨 RX 类别对应

相比继续只匹配混合矩，可以先验证一个简单、明确的关系任务。

例如，查询来自 RX \(r\)，用独立合法 L 中其他 RX 的样本构建各 TX 原型，定义：

\[
\mathcal L_{\mathrm{rel}}
=
-\log
\frac{
\exp(s(z_i,p_{y_i,-r})/\tau)
}{
\sum_c \exp(s(z_i,p_{c,-r})/\tau)
}.
\]

它直接要求查询在跨 RX 参照中仍属于正确 TX，同时保留其他类别作为竞争项。

但也要先检查“一个原型、共同度量”的假设是否成立。如果 RX 对身份空间产生较大旋转，合并原型可能本身就不合理。可以先用冻结特征比较合并原型与分别保留 RX 原型的读出，判断问题来自统计噪声还是坐标关系。

联合训练时保留原身份 CE，并分别观察：

- 关系任务是否改善；
- 实际身份头是否改善；
- 类间 margin 是否缩小；
- 修复是否超过破坏。

这样可以防止只训练好了另一个辅助头。

### 9.4 多解耦网络应回到有限作用预测，并先验证迁移

我仍然认为“多个网络解释不同作用”有研究价值，但应先让每个网络证明自己的作用模型有效。

例如：

\[
\widehat{\Delta h}_k
=
D_k\big(\operatorname{sg}(h),a_k,q_k\big).
\]

第一阶段固定参考编码器，只训练作用预测。初版优先使用已知受控参数 \(a_k\)，减少自由状态估计的歧义。

评价重点应包括：

- 未参与作用模型拟合的 TX 上，是否仍能预测变化；
- 不同 day、接收质量下是否仍然成立；
- 是否优于简单线性／低秩作用基线；
- 拟合改进是否覆盖真实跨 RX 残差的一部分。

通过之后，再把其解释的有限变化用于身份学习。

这样，多网络的价值来自**可迁移的作用知识**。若它只在已拟合 TX 上重构得好，就不应继续提高其身份约束权重。

### 9.5 U 应优先支持新的结构证据

当前同伴 soft KL 可以保留为已验证的小幅增益基础，但新增 U 利用方式应明确回答“新增了什么监督”。

优先考虑：

- 同包已知操作的响应预测；
- 重复前导之间的稳定成分与估计不确定性；
- 条件响应模型在更多合法源样本上的覆盖；
- 在确有不同特征依据的教师之间，验证独立纠错能力。

同包一致性本身也保留 RX 信息，不能自动完成 TX/RX 分离；不同包、不同 RX 更不能因为伪标签相同就被当作严格反事实配对。

### 9.6 已登记的工程矩阵可以验证优化假设，但应单独解释

仓库已写出更大学习率、短预热、SAM、label smoothing、隐藏层 mixup 等后续设计；这些在当前快照中尚未实现发布。[工程后续设计](https://github.com/niubility1317/CVS-RFFI/blob/51820aee7690e09ca9afd239231095b9ecf1e7f2/automation_reports/CV-SincNet/20261012-phase1-scratch-conditional-fishr-manysig-m9-r01/evidence/engineering_next.md)

这些实验可以回答：**初始化阶段的优化轨迹和置信度正则是否有问题。**

即使有效，也应与“发现了更好的物理身份空间”分开归因。若无效，也不必因此放弃物理作用建模。

---

## 十、我对当前路线的最终判断

目前的证据支持下面这个优先级：

| 判断 | 证据强度 | 对后续工作的含义 |
|---|---|---|
| 新机制确实改变了预测和梯度 | 高 | 不再把整体无更新作为主要解释 |
| 小幅收益存在，但后续复杂化没有持续增益 | 高 | 保留有效对照，避免把组合默认视为进步 |
| 更强增强不一定更有用 | 高 | 评价实际更新效用与身份保留 |
| 混合矩不能充分约束身份对应 | 数学上明确 | 增加直接的跨 RX 身份关系检验 |
| TX×RX 交互可能承载正常身份响应 | 数学和物理上明确 | 不应只凭交互存在就全部抑制 |
| 当前特征缺少稳定身份信息 | 尚未证实 | 先做冻结表示的可读出性诊断 |
| 当前短窗口已达到不可突破上限 | 没有证据 | 不能据约 80% 平台宣布信息论上限 |

**我建议下一轮把资源优先投入“身份证据在哪里、为何没有被保留和使用”这一问题。** 当前已经投入了大量工作去精确估计统计量、校准梯度和搜索困难变化；下一步最需要补的是这些操作与真实身份区分之间的证据链。

多解耦路线可以继续，但每个新增网络都应带来一个明确、可检验的增量：解释此前解释不了的接收作用，恢复此前无法稳定读取的身份关系，或者提供教师共同错误之外的新监督。只有这样，多网络、特征增强和采样机制才有机会产生持续的性能收益。
# D92-BranchRidge-v1：冻结分支增强

日期：2026-09-29。本方法仅依据固定 Phase1 代码、科学协议和获准的 support-only 探查选择；研发者仍未读取任何历史或当前 query 成绩。用户希望所有 K 的旧类、新类和 H 都改善，这是后续验收目标，不是现有支持证据已经证明的结论。

## 1. 支持证据与唯一选择

唯一证据目录为 `E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-support-probe-m4-r01/results/support_summary/`。其 `summary.json` 核实 4800 个 episode、1200 个 K1 数值诊断、3600 个完整物理 OOF 和 64800 次固定解析分解。所有分支和 ridge 参数在该探查之前固定，无 query 数据或成绩输入。

以下来自 `by_k.csv`，是含新类任务上 `zfft_aux - zfft_duplicate` 的逐 episode 等权支持 OOF 均值，单位为百分点：

| K | 旧类差 | 新类差 | H 差 | B_zfft→A 的 held 线性重构 R² |
|---|---:|---:|---:|---:|
| 5 | +3.684 | +7.351 | +6.881 | 0.280 |
| 10 | +4.608 | +9.162 | +8.420 | 0.359 |
| 20 | +4.734 | +10.629 | +9.391 | 0.423 |

旧类单独任务上，相同 duplicate 对照的旧类差分别为 +2.042、+2.632、+2.149 个百分点。相对于未重复的 zfft 背景，含新类任务的旧/新/H 支持均值也均为正。接收机×场景分层的均值支持该方向，但单个 episode 存在负差，不能声称全面逐 row 胜出。支持 draws 复用物理记录，并非独立重复，未据此构造独立样本置信区间。

重复背景已控制一个明确的等效正则变化：`[B,B]` 配 ridge=1 等价于 B 配 ridge=0.5。辅助组相对这一固定控制仍有支持增量，足以优先推进已检验的 `zfft_aux`，不必再增加未经探查的分支选择器、度量训练或分数混合。重构 R² 只表示固定线性探针的可预测性，不证明互信息或 TX 因果信息。

选择**单一固定方法**：把探查中的 zfft_aux 特征和解析目标原样用于当前 row 全部真实 support，生成可部署头。旧 D92 保留为独立比较基线，不将其分数或任何旧 adapted state 接入本方法。重复背景只是研发控制，不作为运行时候选。方法选择发生在授权 support 探查之后，后续重复基准必须如实记录这一开发历史，不能称为未见过目标域 support 的盲测试。

## 2. 冻结特征与训练公式

四个 Phase1 checkpoint 实际均为 `lite_d/no_dac/feat_joint`，time/freq/PA/stats 启用，DAC 关闭；exact provenance、SHA、final200 与架构核验继续生效。用原始单条 received IQ 做一次冻结原生 `return_aux` forward，取 `z_id/t_emb/f_emb/pa_local`，各 160 维；FFT96 使用已有同一 received descriptor。禁用 DAC 不得开启。没有新 LEO view、clean IQ、源样本或逐样本源特征；无需地面原型或摘要。

令 `u(v)=v/max(||v||₂,10^-12)`，对每个物理样本定义

\[
B_i=u([u(z_i),4u(FFT96_i)]),\qquad
A_i=[u(t_i),u(f_i),u(p_i)]/\sqrt3,\qquad X_i=[B_i,A_i]\in\mathbb R^{736}.
\]

最后不整体 unit，也不做训练集标准差缩放。辅助组仍采用三分支等权，没有 receiver、TX、K、新类数专属参数。全零块保持零；近零块沿固定 norm floor 处理。通常 B 和 A 各单位能量；零块会改变该值，运行日志记录实际范数，不据此切换公式。

设 C 为实际注册类数，每类 K 个当前 row 物理 support，N=CK。目标 `Y_i=onehot(y_i)-1/C`。只求解

\[
\min_{W,b}\frac12\sum_{i=1}^{N}\|X_iW+b-Y_i\|^2+\frac12\|W\|_F^2.
\]

ridge=1 对应物理损失之和，不除以 N 后仍保留同一 ridge。截距不惩罚。所有类等物理权重，old membership 只记录角色，不参与优化或预测。用当前 row support 估计均值后，`Xc=X-X̄`、`Yc=Y-Ȳ`，`W=(XcᵀXc+I)^-1XcᵀYc`、`b=Ȳ-X̄W`。float64 Cholesky，D≤N 时 primal，否则用等价 dual `W=Xcᵀ(XcXcᵀ+I)^-1Yc`；不换目标、不扫描 ridge。

固定特征能量最多为 2，故解的 SPD 矩阵最小特征值至少为 1，条件数上界 `1+||Xc||F²≤1+2N`。计算中出现非有限值即技术失败，不静默改算法。全零输入得到零 W 和常数截距，预测按物理 class ID 稳定打破平局。C=1 也合法。

## 3. K1 与诊断边界

K1 采用同一公式：全部 C 个 1-shot 标签可约束共享判别边界，不必声称估出了类内协方差。它不能提供同类跨物理变化信息；也不能用同一记录的视图伪造验证。本次 K1 仅有数值证据：三个实际辅助出口均非零，辅助臂和 duplicate 的平均能量差在浮点舍入量级。**K1 的预测增益尚未验证**，K≥5 的 OOF 增量不能代替 K1 证据。

正式拟合所有 K 都只执行一次全 support 解析求解。既有完整支持探查已验证固定公式，不再重复六臂或 OOF；没有运行时模型选择。K1 `oof=null`、原因 `K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT`；其他 K 同样 `oof=null`、原因 `NO_CV_FIXED_CONFIG`，不能误称已执行本次 OOF。未来如单独诊断 OOF，必须完整移除 held physical 并重新计算全部中心/Gram/head，不复用全 support 监督状态。

## 4. Query 与缓存边界

support 允许缓存旧探查导出的原始 float32 五块，键为 checkpoint SHA、capsule、physical ID 和固定单 view contract；必须逐 row 选择合法 support，不能继承 probe 头或跨 row 状态。新完整特征缓存不包含真实 query 标签或角色。query 只做同一冻结原始 forward 和同一特征公式，每条独立计算 `s=XW+b`，面对全部注册类；不读批次类集合、配额、角色或 truth，不更新任何模型/均值/头。

W/b 以不可变 float64 保存，class registry 只读；score 内逐行点积以避免批次矩阵调用改变数值路径。score 接口不接受 labels。输出列保持调用方 registry 顺序，预测平局以物理 class ID 排序解决。prediction artifact 冻结后才能由独立 scorer 连接 truth。两 cohort 都结束后统一解释；无 query 反馈驱动参数、停止或重跑。

## 5. 实现 API、日志与成本

```python
fit_branch_ridge(*, z_id, fft, t_emb, f_emb, pa_local,
                 support_labels, support_ids, classes, old_classes=()) -> BranchRidgeState
state.score(*, z_id, fft, t_emb, f_emb, pa_local)  # N×C
state.predict(*, z_id, fft, t_emb, f_emb, pa_local)  # physical class IDs
state.audit_dict()  # JSON-safe
```

只复用 support probe 的固定特征构造和解析 solver，不调用其六臂遍历或统计选择。audit 包含实际损失 data/ridge/total、梯度与正常方程残差、真实 train K/N/IDs、solver/分解维度、中心/Gram/solve/objective/总耗时、feature/Gram/cross/W/b 字节。optimizer steps=0、LR/epoch=null，source validation=null 并说明无 source 输入。顶层 fold_count=0/folds=[]/oof=null，单次分解、selection=`fixed_no_selection`。

持久数值头仅 W[736,C] 与 b[C]，实际 `nbytes=8×737×C=5896C`：C=6 为 35,376 B，C=26 为 153,296 B。此口径不包含 class ID/audit 的 JSON 或 Python 对象开销，字符串与完整 artifact 实长另记。query 点积每条约 736C 乘加。每 row 一次 min(N,736) 阶 Cholesky；不保存 support feature bank。

当前新增 query 分支出口需冻结模型推理；已经合法缓存的 support raw blocks 可以直接复用。未来同次 forward 导出多个出口不增加 backbone 次数，但不能把本次尚未提取的 query 宣称零成本。原始五块每条 2944 float32 字节，记录真实 cache 和模型包大小、加载/提取/拟合/预测/写日志时间、forward 数与 RSS。新增 source payload=0、ground statistics=0；已有完整模型包与新增传输分列，不猜部署状态。

验证覆盖：与 probe zfft_aux 同一目标/解一致；展开平方目标及有限差分梯度；K1/C1/全零/近零/SPD；类与物理样本排列；query 逐行/分块/重排完全相同；只读 W/b；实际字节；shape/type/finite/label 负测；JSON 无 NaN。正式 query 全 K 旧/新/H 验收另由 root 执行，本文不承诺结果。

实现状态：`code/cvsrffi/d92_branch_ridge.py` 和冻结配置 `configs/d92_branch_ridge_frozen_20260929.json` 已完成；配置顶层为 `{"algorithm": FROZEN_CONFIG}`。新核心 19 项测试及复用 probe 的 21 项测试合计 40 项全部通过。对所有 support 特征完全相同的退化，显式输出数学上精确的 W=0、b=0，消除浮点 onehot 均值残留造成的假平局破坏；该判断没有阈值，不影响一般解。

本机 `ssr-gpu` Python、2 BLAS threads、seed=20260929 标准正态纯合成 C=26/K=20 的一次全拟合：外层 0.025249 s，核心 0.024347 s，dual 520 阶单次分解；分类梯度残差 1.88×10^-13，正常方程残差 4.17×10^-14，W/b 实际 153,296 B。这是合成实现成本，不是实际数据性能；未启动新真实实验。

# D92 后备设计：共享协方差的循环相位边缘化

日期：2026-09-29。状态：独立后备设计，尚未实现、测试或启动；不改变正在运行的 BNNA。本设计未读取历史或当前 target scores、结果解释或评分索引。名称暂记为 Orbit Shared Covariance（OSC），不创建正式 run 或宣称性能提升。

## 1. 决策与协议边界

唯一候选是：保留同一 received IQ 的四相位有序特征，先在每类真实 support 内做有限循环对齐，再由所有注册类共同估计一个共享收缩协方差；query 对四种相对相位进行固定均匀边缘化，所有类别使用同一条打分公式。无需梯度优化，无地面摘要依赖，不训练 Phase1。

已实际核对本机 `E:/type10-7/项目.md` §5.3、§5.3.1、§5.3.2、§5.4：普通地面原型仅可作不可训练类别锚点或冻结判决，不能据此拟合协方差、LDA 或持久分类头；绑定冻结的量化聚合摘要才有独立的固定分布锚点/确定性虚拟特征点/normalization 权限。本候选不读取两者，不构造 source proxy、伪源样本、源特征库或新 source 统计，因而不需要扩大任何原型权限。

唯一监督输入为当前 row 的真实 support 标签。四个视图来自同一 received IQ 的确定性相位旋转，不新增物理样本、K 或 LEO 观测。query 只运行冻结特征提取与下述逐样本公式。无跨 query 状态、排序、配额或训练反馈。

## 2. 代码依据与机制差异

只读核对的实现位置：

- `code/scripts/probe_d92_registration_balanced_covariance.py::build_d92_fit`：D92 在 support 中估计共享协方差，并接入 full/block 组件；其固定 old/new 组权重各为 0.5。
- `code/cvsrffi/stage2_d42_unified_shrinkage_lda.py::_fit_equal_prior_lda`：统一等先验仿射判别，以及少样本时球形协方差退化。
- `code/cvsrffi/stage2_ablation_executors.py::fit_stage2_ablation`、`::_metric`：原路径先由 old support 适配 metric，再进入内部 head 留出；本设计不沿用这一提前拟合顺序。
- `code/cvsrffi/stage2_d92_summary_joint.py`：已实现的 SGJoint 是固定摘要归一化、单个联合向量的收缩 LDA 和 support CV。
- `code/cvsrffi/stage2_d92_bnna.py`：BNNA 最终对适配后的四视图求均值，以单个类原型打分。

OSC 保留四视图的**相对对应关系**。BNNA 的均值会消掉这部分信息；MVKME 的逐视图平均映射不保留循环顺序。OSC 的 log-sum-exp 作用在四种整组匹配上，而不是先把四视图压成一个均值；一般情况下它不是一条单一仿射分类头。新增的是冻结编码器对同一个物理观测的响应结构，不是新观测或额外统计独立样本。若冻结编码器已经完全相位不变，四视图相同，额外机制自然失效，不应期待收益。

共享协方差只是对保留下来的结构提供统一距离。本设计不宣称任意可逆线性坐标变换本身带来新信息；其区别来自循环匹配边缘化，以及有限物理 support 下明确的收缩估计。

## 3. 唯一公式

设当前拟合集合每类有 n 个物理 support，共 C 类；d=256，V=4。所有物理 ID 和类 ID 按 UTF-8 字符串排序以固定求和及 tie 顺序。

### 3.1 固定特征

对每个物理样本 i，在完整 received IQ 上构造相位 0、π/2、π、3π/2 的四视图。冻结原生 Phase1 编码器给出 identity160：

`z[i,v] = unit(identity160(phase_rotate(received_i, v*pi/2)))`。

`f[i] = unit(received_fft96(received_i))`，只计算一次。FFT96 沿用当前 received descriptor 的固定实现；归一化 `unit(x)=x/max(||x||_2,1e-12)`。

`x[i,v] = concat(z[i,v], f[i]) / sqrt(2)`。

不再归一化 x，不重估 FFT 权重。非零块具有相等总能量；零向量保持零。每个 x 是 256 维，四视图 FFT 部分完全相同。

### 3.2 每类的物理 medoid 与有限循环对齐

定义两个完整四视图结构的距离：

`D(i,j;s) = (1/4) * sum_v ||x[i,v] - x[j,(v+s) mod 4]||^2`。

每类 medoid 是最小化 `sum_j min_s D(i,j;s)` 的真实 support；精确同值时取物理 ID 最小者。medoid 自身固定 shift=0。其余样本取最小化 `D(medoid,i;s)` 的 shift，精确同值取最小整数 s。

对齐后 `a[i,v]=x[i,(v+s_i) mod 4]`，模板为 `mu[c,v]=(1/n)*sum_{i:y_i=c} a[i,v]`。不归一化模板，不迭代重分配，不把 medoid 换成生成的样本。类别之间允许不同相位坐标原点，最终边缘化消除这种相位坐标的选择。

上述对齐是当前训练集合内的估计。它不保证连续任意相位不变，也不保证 medoid 离散选择稳定；不能把四相位 invariance 夸大为对任意信道扰动鲁棒。

### 3.3 一个共同协方差形状

对 n>=2：

`S = sum_c sum_{i:y_i=c} sum_v (a[i,v]-mu[c,v])(a[i,v]-mu[c,v])^T / (4*C*(n-1))`。

该式每类等权，没有 old/new 分组权重。四视图只平均贡献协方差的方向；用于收缩的有效物理残差数严格为 `nu=C*(n-1)`，不是 `4*C*(n-1)`。S 是对齐后的经验方向估计，不声称它在离散 medoid 选择之后仍是无偏 Wishart 估计。

令 `t=trace(S)`，`energy=mean_{i,v} ||a[i,v]||^2`。n=1 或 `t <= 64*eps_float64*energy` 时，固定 `Q=I_d`，记录 ZERO_PHYSICAL_RESIDUAL。energy=0 时也固定 Q=I。否则：

`R = d*S/t`，`lambda=d/(nu+d)`，`Q=(1-lambda)*R+lambda*I_d`。

这里使用 d 个单位协方差方向作为固定收缩强度，与真实物理残差数 nu 平衡，不扫描 lambda。该规则是结构性正则化选择，不宣称统计最优。Q 正定，平均特征值为 1；在精确算术下 `condition_number(Q) <= nu+1`。该边界来自 `trace(R)=d` 和 R 半正定，不依赖 target 成绩。

最终使用参考协方差 `Sigma=Q/d`。其 trace 为 1，对应固定单位能量联合特征的尺度；只适配 covariance shape，不另拟合噪声总量或温度。它是一个预先固定的 Gaussian surrogate，输出不能直接宣称为已校准真实后验概率。S 仅浮点误差导致的不对称用 `(S+S.T)/2` 修复；Cholesky 失败或非有限值直接报技术错误，不静默换算法。

### 3.4 循环边缘化判别

每个 query 独立提取四个 x[q,v]，对所有注册类计算：

`ell[c,s] = -(1/8) * sum_v (x[q,v]-mu[c,(v+s) mod 4])^T Sigma^-1 (x[q,v]-mu[c,(v+s) mod 4])`。

`score[c] = logsumexp_s(ell[c,s]) - log(4)`。

注册类 prior 均为 1/C，其公共项省略。共享协方差 determinant 项也为公共项。精确最高分并列按物理类 ID 排序取首项。

这是四种整组对应关系的几何平均 Gaussian expert，再对相对相位作均匀混合；`1/4` 的视图平均阻止把相关视图当成四个独立物理观测。使用 log-sum-exp 的数值稳定实现。固定 state 下，query 四视图的循环重排仅置换 s，因此 score 精确算术不变。任意交换非循环位置不承诺不变。

可编译 `W[c,v]=Sigma^-1 mu[c,v]` 和 `b[c]=-(1/8)*sum_v mu[c,v]^T W[c,v]`。删除对所有类别和 shift 相同的 query 二次项后：

`score[c] = logsumexp_s( (1/4)*sum_v x[q,v]^T W[c,(v+s) mod 4] + b[c]) - log(4)`。

最终只保存 W、b、类表与审计信息；不持久保存训练 support、medoid、协方差或任何源状态。删除公共 query 项不改变 softmax、NLL、argmax；审计应明确使用了这一分数约定。

## 4. 最小预算、K1 和完整 physical CV

参数全部如上冻结：四视图、等块能量、d=256、解析收缩、固定尺度、无优化器、无候选网格、无温度搜索、无早停。不存在需从成绩决定的 rank、步数或学习率。

- K1：只拟合一次。每类唯一物理样本为 medoid，模板是其完整四视图，Q=I。无 holdout、无 CV、无验证指标；四视图不充当独立验证集。
- K>=2：F=min(K,3)。每类按物理 ID 排序，序号 modulo F 指定 fold。每折整组排除 held 物理样本及其四视图。该折重新选 medoid、重新循环对齐、重新算模板、S、nu、lambda、Q、W、b。禁止提前全 support 对齐或估协方差。
- 每个 fold 类内训练样本数一致；K/F 不整除时不同 fold 的 n 可以不同，必须按实际 n 计算 nu、lambda。
- 本候选仅一个公式；CV 只产生 OOF 诊断，不选候选或调参。完成 F 折后，用全部 support 从头拟合唯一最终 state。因此总预算 F+1 次解析拟合，K1 为 1 次，零梯度步。

OOF 按物理样本各一次记录 NLL/预测；报告 macro-class NLL、old/new 分组 NLL及训练 n。old/new 标识只能用于这一诊断，不影响拟合或 query 规则。不能把后续根据这些诊断开发新方法称作本方法的独立验证；若未来修改公式，需另行冻结和登记。

## 5. 输入、传输与计算核算

拟议核心接口：`fit_orbit_shared(support_identity_views[N,4,160], support_fft[N,96], support_labels, support_ids, classes)`。fit 不接收 query、teacher、ground 或 source 路径。评分接口 `state.score(identity_views[B,4,160], fft[B,96])`，逐样本不可变。old membership 由入口仅供 OOF 分组审计使用。

输入验证：类 ID/物理 ID 唯一且为显式字符串；每类相同正整数 K；标签与类表一一对应；float 数组维度严格、全部有限。按稳定物理 ID 排序归约。零特征允许，NaN/Inf 拒绝；极端幅值导致平方溢出时拒绝，不改变尺度规则。

资源以实际 nbytes 和文件字节报告，不把以下公式当已测量时间：

- 地面新增 source payload：0 B；无需原型或摘要。已有冻结模型包单列实际文件大小，若已部署，新增模型传输为 0 B。
- float32 特征 cache：每个物理样本 `4*(4*160+96)=2944 B`，元数据 UTF-8 字节和文件容器开销另计。与 BNNA 同一四相位/FFT 定义，可在 checkpoint、capsule、schema 和 view 规则完全一致时复用已有合法 cache；不能仅凭相同形状假定兼容。
- 最终 float64 数值 state：`8*C*(4*256+1)=8200*C B`；C=26 时 213200 B。类表/审计 JSON 另计；只读字节缓冲，score 不追加缓存。
- 临时 covariance：每个 256×256 float64 矩阵 524288 B。实现应记录实际同时存活的 S/Q/Cholesky/solver 数组、训练特征及 RSS，不能仅报一个矩阵即宣称峰值内存。
- 每次拟合的 medoid 直接计算量为 `O(C*n^2*4^2*d)`；残差 covariance 为 `O(4*C*n*d^2)`；分解为 `O(d^3)`，多右端求解为 `O(4*C*d^2)`。使用 NumPy/BLAS/Cholesky，无自动微分框架，无 N×C×d query 张量。
- 编译后每 query 打分约 `16*C*d` 次乘加及 C 次四项 log-sum-exp；C=26 时主要内积约 106496 次乘加。编码器仍为每物理样本 4 次冻结前向，FFT 一次；不额外跑 LEO。
- 全流程分别计 encoder/FFT/cache 读写、每 fold/final medoid/covariance/solve、query score、prediction write；optimizer_steps=0，epoch/lr/gradient 不适用，不伪造训练日志。

## 6. 退化、风险与可证伪验证

四视图完全相同时，循环分量相同，机制严格退化为同一特征均值及共享收缩度量的普通判别。所有输入为零时 W=b=0，按物理类 ID 稳定并列裁决。K1 没有物理残差证据，因此不估计非球形协方差。缺少类别、非有限值或输入 provenance 不满足时应失败，不调用源数据修复。

该机制可能失败的具体原因包括：Phase1 四视图响应已近乎相同；相位结构主要为无关信道变化；medoid 在小 K 下不稳定；只有四个相位不足以覆盖连续偏移；模板平均仍可能模糊多峰类别；高维收缩偏强。以上是预先列出的机制风险，不来自 target 结果，也不构成运行中换公式或停机条件。

实现前后的本地验证只使用合成输入：

1. 直接 quadratic 公式与编译 W/b 公式的分数差仅为每 query 的公共常数，softmax/预测相同。
2. 固定 state 的 query 循环视图重排、query batch 分块和 query 顺序不改变相应逐样本输出；评分前后 state 字节不变。
3. class/support 输入排列等变；各物理样本的视图必须整体进出 fold。
4. 对 held physical 样本作任意扰动，对应 trainfold 的 medoid、shift、模板、Q、W、b 完全不变。
5. K1 没有 CV，四视图相同退化、全零并列、低秩/精确相消及近零残差都符合固定规则。
6. 正定性和 `cond(Q)<=nu+1` 在浮点容差内成立；物理自由度没有乘 4；不同 fold 使用实际 n。
7. 构造均值相同但循环对应关系不同的有限合成例子，验证实现确实读取相对相位结构。它只证明该例子的功能差异，不证明对真实数据有提升或不存在所有线性表示。
8. 测量实际 nbytes、分阶段耗时和 RSS；不读取任何 target truth 或历史分数来设测试阈值。

科学验收只能在日后获得真实运行授权并按 truth-last 固定预测后进行。本次交付仅为后备机制设计，不登记、不启动新实验，也不改变 BNNA。

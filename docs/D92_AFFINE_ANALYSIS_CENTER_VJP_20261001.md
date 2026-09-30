# Affine 分析器中心化 VJP 的 rank-one 展开

本文件交付未来独立 Affine 分析器的纯数学 helper。它已集成到未来独立摘要器，只替换代数表达式；候选核心、方法配置和已运行 release 保持原样，真实性能尚未知。未读取真实数据、query、运行产物或结果；数值测试由 root 统一执行。

## 接口和完整公式

新增 `tools/d92_affine_analysis_math.py`，公开接口为：

```python
center_kernel_vjp(bar_K, bar_L, q, gamma) -> (bar_R, bar_Q)
```

`bar_K` 为 `n×n`，`bar_L` 为 `h×n`，`q` 为 `n`，`n≥1`、`h≥0`，输出分别为 `n×n` 对称矩阵与 `h×n` 矩阵，dtype 均为 `float64`。函数不修改输入，不归一化 `q`；调用者继续核验物理 reference、权重与各 head 的绑定。`gamma` 必须是有限标量。形状或非有限输入抛出 `ValueError`，不可表示的运算抛出 `FloatingPointError`。数学结构允许一般输入 `bar_K`，最后对原始对称核的伴随对称化。

记 `e=ones(n)`、`f=ones(h)`、`P=I−e qᵀ`，原始 train/cross 核分别为 `R`、`Q`。完整前向为：

```text
K = gamma P R Pᵀ
L = gamma (Q − f qᵀ R) Pᵀ
```

设 `A=bar_K`、`B=bar_L`。完整反向为：

```text
bar_R = gamma sym(Pᵀ A P − q fᵀ B P)
bar_Q = gamma B P
```

helper 独立展开为以下求和与外积，不读取 core 的反向结果：

```text
C = B − (B e) qᵀ
J = A − q (eᵀ A) − (A e) qᵀ + (eᵀ A e) q qᵀ − q (fᵀ C)
bar_R = gamma sym(J)
bar_Q = gamma C
```

`−q(fᵀC)` 是 cross 核对移动 reference 的 train 核梯度。必须保留；`q` 固定并不表示 reference 在当前 adapter 下不移动。此接口不计算 `q` 的梯度。

## Affine cancellation 仍独立检查

若 `A=−sym(T alphaᵀ)`、`B=G alphaᵀ`，且 saddle 约束为 `eᵀalpha=0`、`eᵀT=fᵀG=g_b`，完整展开随后满足：

```text
bar_R = gamma A
bar_Q = gamma B
```

helper 不用这两式省略 reference 项，不假设 `g_b=0`。root 集成时只替换 `verify_companion` 中构造 `P` 和稠密乘法的表达式，继续保留现有 saddle 方程、`g_b`、完整中心化 cancellation、距离反向、adapter 反向和最终 `g_Z` 检查。现有覆盖、容差和实际训练工作量字段不变；分析器运算不计入方法训练成本。

## 复杂度和浮点边界

旧表达式显式保存 `P[n,n]`，`Pᵀ A P` 的稠密乘法为 `O(n³)`，`B P` 为 `O(h n²)`。展开只对 `A` 与 `B` 做行列求和，再广播外积；时间和额外空间均为 `O(n²+h n)`，没有 `P`、矩阵乘法或 `n×123616` 特征。输出本身需要 `n²+h n` 个 `float64` 元素，即 `8(n²+h n)` bytes；这是输出大小，不是进程峰值内存声明。

实数算术下两种表达式相等。浮点求和和乘法的顺序不同，不保证位级一致。近抵消时，误差相对于参与抵消的量而非精确结果缩放；不能要求接近零的伴随仍有固定相对误差。现有摘要器的数值核验容差没有放宽。有限输入也可能在求和或乘法时溢出，helper 报错而不裁剪、重标定或伪造有限值。

`n=1,q=[1]` 时 `P=0`，两项伴随为精确零；空 held 产生 `bar_Q[0,n]`，其 reference cross 项为零。零 `gamma` 在形状和有限性检查后返回同形状零伴随。函数对一般固定权重仍使用原值；生产物理 `q` 的合法性由已有摘要器检查。

本 helper 不处理距离、带宽或头拟合。`tau=0` 的原始 complete-feature equivalence、`gamma=None` 的零核、K1/rank0/new0、actual B 继承均保持既有分支语义。root 只应在原 `verify_companion` 的 active 分支调用。

## 合成验证和当前状态

新增 `tests/test_d92_affine_analysis_math.py`，数值检查包括：

1. 独立显式 `P` oracle，覆盖 `n=1`、空 held、非均匀 `q`、含零权重、one-hot 和不同输出形状，确认输入不变。
2. 正负、大、小伴随值；维持已有分析器的 norm/shape 容差，并核验尺度线性。
3. 独立完整前向的 train、cross 和联合方向有限差分，确认漏掉移动 reference 项会改变结果。
4. 非零 `g_b` 的 affine saddle cancellation；同时比较完整 `P` oracle，确保没有通过省略 `g_b` 得到伪等价。
5. 近抵消的 gauge 分量、精确单样本边界，以及一般输入矩阵和未归一化固定权重不会被静默改写。
6. helper 源码无矩阵乘法节点、运行路径不构造 `np.eye`，以及形状和非有限值拒绝。

直接运行该测试文件只执行标准库 AST/UTF-8 检查，不导入数值库或运行测试。主 Agent 已完成集成及项目环境串行数值验证：8 项通过（0.33 s）。首轮近抵消 fixture 错用全 1 向量，独立稠密 oracle 比较已通过；仅把伴随零方向改为 q，保留原断言和全部容差。未测量真实加速比例；完整证据见[分析验证](D92_AFFINE_ANALYSIS_VALIDATION_20261001.md)。

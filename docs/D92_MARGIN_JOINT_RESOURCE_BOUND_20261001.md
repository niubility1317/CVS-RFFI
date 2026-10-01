# MarginJoint 的资源边界与预算候选

本文从冻结源码和字面矩阵推导资源上界，供调用方区分单头资源、阶段工作量和进程内存。**`max_factor_buffer_bytes` 约束单次 C QP 头的指定 factor 数组，`max_transitions` 约束单次头的工作集循环次数；两者都不是整个阶段的内存或成功收敛保证。**本文不选择实际 run、预算或硬件，不改变核、精度、误差尺度、目标或优化路径，也不使用真实数据、产物或成绩。

依据是[纯头实现](../code/cvsrffi/d92_margin_qp_head.py)、[联合 core](../code/cvsrffi/d92_margin_joint_local_ridge.py)、[evaluate](../tools/evaluate_d92_margin_joint_probe.py)、[run 的结构预算](../tools/run_d92_margin_joint_probe.py)和[prepare](../tools/prepare_d92_margin_joint_probe.py)。数学背景见[结构推导](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)、[working-set 可行性](D92_MARGIN_JOINT_SOLVER_FEASIBILITY_20261001.md)、[纯头说明](D92_MARGIN_QP_HEAD_IMPLEMENTATION_20261001.md)及[联合实现说明](D92_MARGIN_JOINT_IMPLEMENTATION_20261001.md)。这里只引用这些文档的数学与实现章节，不据其状态段落推断任何实际运行结果。

## 1. 授权矩阵与单头维度

令新增类数为 `s`，注册类数为 `C=6+s`。一个每类有 `k` 个合法训练点的 C 头保留全部物理行，因此

\[
n=k(6+s),\qquad m=6k,\qquad q=m(C-1)=6k(5+s).
\]

`q` 是全部旧 train 点面对全部竞争列的约束数。当前工作集大小记作 `a`。独立性要求给出

\[
a\le\operatorname{rank}(Q)\le\min\{q,(C-1)\operatorname{rank}(P_O)\}\le q.
\]

低秩核可以缩小独立工作集，但不能预先据此删除约束或默认 `a` 很小。正定 Gram 下，全部 pair 可以独立，保守内存推导必须允许 `a=q`。

授权父矩阵为 `K={1,5,10,20}`、旧类 6、`s={0,2,5,10,20}`。下表给出以 `k=K` 作为包络时的 `n / q`；实际 inner train 更少。`s=0` 直接复用实际 B，没有新增 C QP，故 `q` 记为“—”。

| 父 K | 旧点 m 上界 | 新增 0 | 新增 2 | 新增 5 | 新增 10 | 新增 20 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 6 / — | 8 / 42 | 11 / 60 | 16 / 90 | 26 / 150 |
| 5 | 30 | 30 / — | 40 / 210 | 55 / 300 | 80 / 450 | 130 / 750 |
| 10 | 60 | 60 / — | 80 / 420 | 110 / 600 | 160 / 900 | 260 / 1500 |
| 20 | 120 | 120 / — | 160 / 840 | 220 / 1200 | 320 / 1800 | 520 / 3000 |

因此授权矩阵的通用包络是 `n≤520`、`m≤120`、`C≤26`、`q≤3000`。这是维度界，不是实测 active 数。K1、零 gamma、tau0 和 rank0 仍可能执行完整 C 头；只有 new0 在进入 QP 前复用 B。

## 2. 当前 factor guard 的准确口径

纯头 `_Ledger.reserve/factor` 与 forward 循环执行以下检查：

\[
F(n,a)=16n^2+16a^2\le B,\qquad B=\texttt{max\_factor\_buffer\_bytes}.
\]

float64 每元素 8 字节。第一项是 `A=I+K` 的输入与 `chol_A`，第二项是当前 `Q_W` 的输入与 `chol_working`。先检查 `16n²`，每次构造工作矩阵前检查完整 `F(n,a)`；`factor()` 再检查 retained factor 数组加本次输入/输出。上一工作集的矩阵和 factor 在重建前释放。返回只读状态前释放 `A/Q_W` 输入，为原 factor 与冻结 factor 两份留出空间；不过 `_frozen` 还可能产生布局转换副本，不能据此声称返回隔离全过程已由双份额度覆盖。

分解时存活的指定 factor 输入/输出、成功头的不可变返回状态、整进程 RSS 是三个范围。成功头中 factor 子集为 `8(n²+a²)` 字节；完整不可变返回状态还包含 K/M、响应、系数和约束数组。RSS 又包含全部同时存活状态、scratch、复制、库 workspace 和 Python 内存，不能用前两种范围替代。

超出额度返回 `FACTOR_BUFFER_LIMIT`，保留技术失败，不截断工作集或丢掉其他约束。这个检查并不保证其余数组尚未分配：输入 K/M 已先复制并校验，core 的几何和 kernel 也已存在。QP 伴随复用已冻结因素，没有新的分解；B 解析头、C preparation 的 old-only B prior 以及 R0 baseline 的因素不受这个 C QP guard 约束。

在 `a=q`、新增 20 的通用包络下，指定 factor 数组所需字节如下。MiB 固定为 `2^20` 字节。

| 父 K | n | q | F(n,q)，字节 | MiB，约 |
|---:|---:|---:|---:|---:|
| 1 | 26 | 150 | 370,816 | 0.354 |
| 5 | 130 | 750 | 9,270,400 | 8.841 |
| 10 | 260 | 1500 | 37,081,600 | 35.364 |
| 20 | 520 | 3000 | 148,326,400 | 141.455 |

最后一行精确为 `141.455078125 MiB`。该数仅覆盖上式中的四个数组，不能作为所需 RAM。

当前 evaluate 对 `K>1` 做三条 outer OOF 路径和 K 条 one-shot proxy 路径；不是另外再训练一个父 K 的 full-support 路径。outer 每类 train 数为 `K−ceil(K/3)` 或 `K−floor(K/3)`。因此本入口在 `K=20,s=20` 的最大 final C 头是 `k=14,n=364,m=84,q=2100`，其指定 factor 上界为 `72,679,936` 字节，即 `69.31298828125 MiB`。inner 头比该 final 更小，proxy 为 `k=1`。这个更紧的界只适用于当前 evaluate 调用路径；不能取代 core 接受 full `k=20` 时的通用包络。

### 可讨论的 factor 容量候选

对通用 `n=520`，可容纳的工作集维度为

\[
a_{\max}(B)=\left\lfloor\sqrt{B/16-520^2}\right\rfloor,
\]

前提是 `B≥16·520²`，且实际仍受 `a≤q` 和独立性限制。

| 候选 B | 在 n=520 时可容纳的 a | 可由维度界确认的覆盖范围 |
|---:|---:|---|
| 16 MiB | 882 | 授权矩阵 K≤5 的全部指定 factor；不能覆盖通用 K20 全工作集 |
| 64 MiB | 1980 | 授权矩阵 K≤10 的全部指定 factor；不能覆盖通用 K20 全工作集 |
| 80 MiB | 2229 | 覆盖当前 probe 的 `n≤364,q≤2100` 指定 factor；不覆盖 core 的 full K20 包络 |
| 160 MiB | 3196，实际最多 q=3000 | 覆盖通用 `n≤520,q≤3000` 的全部指定 factor |

这些是容量候选，不是默认值、建议采用值或实际资源配置。较小容量可能成功处理 active 较少的头，也可能在数学可行的头上正常产生技术资源失败。160 MiB 也不能排除条件数、Jacobian 资格或 transition 失败。

## 3. guard 没有覆盖的内存

| 范围 | 当前实现与可计算尺寸 | 与 B 的关系 |
|---|---|---|
| `Q_W` 构建表达式 | `_working_matrix` 先生成 class 内积，再进行 `P_old` advanced indexing 与乘法；未承诺原位复用时，保守按内积、gather、结果三个 `a×a` float64 数组计 `24a²`，另有布尔和索引临时数组 | guard 只为最终输入和 factor 预留 `16a²`，没有逐项预留表达式临时数组；该 `24a²` 不是实测峰值 |
| factor 返回隔离 | `_frozen` 先执行 `np.ascontiguousarray(value)`，再 `contiguous.tobytes()`。若 factor 非 C-contiguous，可同时存在原 factor、布局转换数组和 bytes 副本 | 删除 factor 输入只释放一份位置；布局转换临时数组没有单独 reserve/observe。它不能算作已由 ledger 双份额度确认的峰值 |
| 原核、prior、响应与解 | 复制 K/M、R、`P_old`、alpha、scores、slack、multipliers、pair 索引，以及返回只读副本 | 不在指定 factor scope |
| 旧列 response RHS | `selection` 和 `selected_solve` 各 `n×m`，`P_old_raw` 为 `m×m`；audit 单列 `16nm+8m²` 的相应数组字节 | 不受 B 限制，也不能与 solve 临时峰值机械相加 |
| triangular solve 输出 | ledger 按一次 solve 的两个显式输出记录 `2·rhs.nbytes`；最大 old-response RHS 给出 `16nm`，通用包络为 998,400 字节 | 仅观测 `peak_explicit_solve_temporary_bytes`，没有由 B 限制；不含 caller RHS 或其他同时存活数组 |
| 循环检测与 ledger | `seen` 保存每个签名的 `V.tobytes()`，最多 T 份；还保留 working tuple、rho 字符串、事件、solve/factor 元数据 | signature 数值 payload 至多 `8TmC`；Python 容器、working 索引和元数据另计，没有统一内存 guard |
| core 几何和 cache | original/adapted b/a、train/held 距离、radial、K/L、adapter cache、Z/U/H/W、多个 fold 和 actual B 状态 | 不受 B 限制；正则 VJP 上游也是额外数组 |
| 完整档案 | `_Recorder` 隔离复制每个数值记录；无 callback 时一直保留。evaluate callback 再复制数组并写压缩 NPZ | 不受 B 限制；内存档案、压缩工作区、文件缓存、磁盘量与传输量须分别说明 |
| 数值库与进程 | Cholesky/eigen/condition/BLAS workspace、布局转换、NumPy 临时数组、Python 对象和分配器缓存 | 没有由 factor 字节代理给出进程峰值保证 |

例如通用最大 `mC=3120` 时，每份 V 签名 payload 为 24,960 字节；T 达到约束数量量级时，仅这个 payload 就达到数十 MiB，不能因紧凑 primal 当前 V 很小而忽略累计签名。

多个头的 factor 也会同时存活。完成单头的 factor 子集只保留两个因素，其字节至多 `B/2`，但三折 accepted cache 与下一试探 cache 可以共存。Python 赋值返回前，上一 rejected trial 的 `tc` 还可能存活。只计一个 C fit 中这三类 cache 的 factor，在当前头的已声明输入/输出阶段、暂不计布局转换、档案和 actual B，当 f 个 fold 的下一头正在形成时，保守计数为 `B+(3f−1)B/2`；f=3 时是 `5B`。这是限定对象和时段的粗上界，不是整个阶段的内存上界。factor 返回隔离、QP 构建 scratch、档案复制和其他数组都在其外。

`margin_qp_peak_factor_buffer_bytes` 取每头及各层的 MAX；各种 work counter 累加 SUM。MAX 正确表达指定 scope 的最大单头占用，却不能代表所有共存头之和。`max_simultaneous_forward_cache_bytes` 是已返回 accepted/trial cache 的显式数组计数，也不包含求解过程中所有暂存对象。RSS/GPU 峰值不能由这些字段相加或相乘猜得。

## 4. transition 上限和单头工作量

记 `T=max_transitions`。循环每次都扫描全部 q 条约束；移除负乘子、加入 blocker、无 blocker 的 MOVE 和最终 KKT candidate 都消耗循环机会。工作矩阵每次重建并 Cholesky，不使用未来可能实现的 rank-one 更新成本。最后还从原方程重建全 KKT/gap；超限或数值条件未解析时退出技术失败。

**凸性和已知零 residual 可行点没有为当前 active-set 实现提供一般多项式成功收敛保证。**反复移除/加入、退化和数值循环都可能使任意所给 T 不足。给定 T 可以封顶所做工作，不等于在该工作内必然取得最优头或可微头。

| 单次 C forward 头项目 | 由源码得到的保守次数或维度量 |
|---|---|
| kernel 谱检查 | 1 次，维度 n，谱 cubic proxy 为 `n³` |
| factor attempts / condition estimation | 分别最多 `1+T`；1 个 A 加最多 T 个非空 working 系统，失败计已尝试部分 |
| Cholesky 维度尺度 | `n³+Σ a_t³≤n³+Tq³`；不是测量 FLOPs 或时间 |
| 全约束扫描 | 最多 `T+1`，包括最终 readback 或超限后的最终紧凑快照 |
| triangular 调用 | 最多 `8+4T`：常数、base、old-response、final 各两次；每次循环至多 multiplier 与 blocker-independence 两组 |
| triangular RHS 列计数 | 最多 `2(1+m+2C)+4T`，按每次 triangular 调用计列 |
| triangular dense-work proxy | 最多 `2n²(1+m+2C)+4Tq²`；不能替代构造、乘法、谱、条件估计和扫描的工作 |
| 紧凑快照乘法 proxy | 最多 `(T+1)m²C`；循环另有 `P_old@dV` 等乘法，不能把此字段当总工作 |

成功的正则 QP VJP 复用 A 和 working 因素，最多有 6 次 triangular 调用、没有新的 factor 或 spectrum；非空 working 的 RHS 列计数为 `4C+2`。无 working 时只需两次、`2C` 列。普通 VJP 资格失败仍是技术失败，不因增加 T 或 B 自动消失。

可以用 `T=q_max+2`、`2(q_max+2)`、`4(q_max+2)` 作为三个透明的技术工作封顶候选。在通用 `q_max=3000` 包络下，它们分别是 3002、6004、12008。`q+2` 仅能容纳“至多 q 次加入、一次无阻塞全步、一次终止检查”的无移除路径；有移除或再次加入时没有该成功保证。每个候选都可直接代入上表和下一节公式，预先知道最坏已允许工作量；这里不评价这些值的实际速度、内存适合度或成功率，也不把任一值写入配置。

## 5. 4×12 外层对整个阶段的放大

一次 informative B 或 C 阶段最多 4 次梯度评估、48 条 line-search 候选和 4 次接受。初始 forward 为 1 次，每个候选最多增加 1 次 forward；梯度使用已接受状态的 forward cache，final objective 也复用 cache。因此，f 折下的头数上界是

\[
H_{\rm stage}=49f+1,\qquad V_{\rm stage}\le4f,\qquad
\text{C QP factor attempts}\le(49f+1)(T+1).
\]

末尾 `+1` 是最终 full-train 头，不是额外优化候选。f=3 时，C 最多 147 个 inner QP 加 1 个 final QP，即 148 个 forward 头；最多 12 次 head VJP、72 次伴随 triangular 调用。各 forward 的实际 n/m/a 不同，使用共同 q 上界只是保守包络；实际成本应按各 archive 的真实尺寸、RHS 和 ledger 累加。拒绝的 trial 也完整收费；4 次接受不意味着只有 4 次 QP。

B 使用解析 affine 头，同样最多 148 个 forward 头，但每头是它自己的单 factor；C preparation 最多另拟合 f=3 个 old-only B prior。Final prior 复用实际已完成 B，不另拟合一次 full B。若连同本路径的 B0/C0 两个基线头，一条 informative B→C 路径的 head 总数上界是 `2+148+3+148=301`，factor attempts 上界是 `153+148(T+1)`。该式不含基线诊断、预测、adapter/几何/SVD、归档或库工作；更不表示这些头同时运行或 factor 峰值相加。

K1 或 core 的其他 no-information 分支不执行外层更新，C stage 仍做 1 个 final QP；已有 preparation 工作不能反向清零。new0 没有 C QP，返回实际 B。

当前 evaluate 的单个父 split 还有三条 OOF B→C 路径和 K 条 proxy B→C 路径。仅对 `s>0`，其 C QP forward 结构上界如下；每个 split 是否真的达到上界不由本文猜测。

| 父 K | 路径数 | C QP forward 上界 | C QP VJP 上界 |
|---:|---:|---:|---:|
| 1 | 1 条 full-support 路径 | 1 | 0 |
| 5 | 3 OOF + 5 proxy | `3·148+5=449` | 36 |
| 10 | 3 OOF + 10 proxy | `3·148+10=454` | 36 |
| 20 | 3 OOF + 20 proxy | `3·148+20=464` | 36 |

任一行可再乘 `(T+1)` 得到 C QP factor attempts 的保守上界，不能由此乘出进程内存。多个 row、receiver/scenario、support seed 或实际选中 split 的整体工作，应像 run 的 `selection_budget/budget_for_spec` 一样按明确 selection 逐项 SUM；本文没有读取或假设真实 spec 的行数、并发或 selection。

## 6. 使用这些界的限制

factor 容量候选和 transition 候选属于不同 scope，应分别表达，不能把“可容纳所有 Q factor”写成“能在 T 内成功”。预算失败保留已发生的工作、数值状态和最后接受状态，不证明数学不可行。完整 KKT/gap 证书只确认返回的固定核头，正则 VJP 另需明确资格；结构证明不能代替生产求解器测试。

这些上界不证明星载可行性或省算力。训练/推理时间、RSS/GPU 峰值、常驻状态、归档大小、序列化部署和新增传输，应在各自真实 scope 下测量；未测项保持 N/A。本文完成的是只读源码推导与文档静态检查，没有运行数值环境、测试、真实评分、Git 或远端操作，也没有改动现有方法或运行产物。

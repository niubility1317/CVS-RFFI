# Margin约束解析头：组件交付与验证

本次完成独立的解析头组件和规则活动集上的反向传播，尚未完成整个MarginJoint–LocalRidge方法，也未启动该方法的真实support实验。组件验证不能说明准确率、H、query遗忘或星载资源得到改善。

## 与联合方法的关系

BranchLocalRidge仍负责解析适应；后续adapter将通过解析头的导数获得监督信号。本组件接收固定的实际B阶段分数M、全部训练support的核K和标签，不读取数据、checkpoint、query或历史评分。C阶段的新类列由上层按已注册类别补齐，不能用真实query类别数量构造。

训练目标是全部旧、新support上的平方误差，加λ=1的RKHS函数范数惩罚；常数截距不受惩罚。旧类训练记录对全部其他已注册类保留约束：

\[
f_{i,y_i}-f_{i,j}\ge d_i,\qquad
d_i=\min_{j\ne y_i}(M_{i,y_i}-M_{i,j}).
\]

d_i使用原值，不能截断为零。零残差是可行起点。约束保证训练旧类的最小margin不低于该B锚点；d_i>0时保持这些训练记录的正确分类，d_i=0时仍有平票问题，d_i<0时不保证分类不变。该结论不延伸到held support或query。

具体原始问题、对偶、紧凑原始坐标及条件核子集关系见[结构推导](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)和[求解可行性](D92_MARGIN_JOINT_SOLVER_FEASIBILITY_20261001.md)。代表定理与解析岭层的依据分别是[Schölkopf等的代表定理](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)和[R2D2](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)。本项目的margin、B→C继承和资源约定是本项目设计，不能归为这些论文已验证的效果。

## 求解与梯度

实现保留全部物理训练记录和全部旧类竞争约束。活动集从零残差可行点开始，在紧凑坐标中移动；每轮扫描全部约束，检查阻挡约束的独立性。不会预先构造整个q×q对偶Hessian；活动集因子最坏仍可能达到q×q，不能宣称消除了最坏二次内存或三次分解开销。

成功返回前，使用原K、原标签和完整约束独立重建α、b与分数，并检查原方程、截距方程、原始/对偶可行性、互补性、原对偶间隙与紧凑状态。没有jitter、伪逆、无约束回退或删除困难约束。

反向传播只在全部紧约束与工作集一致、严格互补且活动矩阵可认证时返回。它包含自由截距的g_b和活动约束乘子的响应，复用已完成的因子；不能把固定乘子或漏掉截距的导数作为完整梯度。该设计依据[OptNet第3节的KKT隐式求导](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)，不继承论文的GPU吞吐或免费反向计算结论。

无法认证的秩、活动集切换点或数值条件返回显式技术失败。调用者必须传入有限正整数`max_transitions`与`max_factor_buffer_bytes`；这些是计算上限，不是预测超参数。预算耗尽不意味着已求得最优解。

## API与状态

代码：[d92_margin_qp_head.py](../code/cvsrffi/d92_margin_qp_head.py)。三个API均使用keyword-only参数：

- `fit_margin_qp_head(K=..., M=..., labels=..., old_indices=..., max_transitions=..., max_factor_buffer_bytes=...)`。
- `predict_margin_qp_head(state, L=..., M=...)`返回完整`M + L @ alpha + b`分数，上层不得再次加入M。
- `margin_qp_head_vjp(state, L=..., G=...)`返回K、L的伴随及资格/残差证据。

返回的数值数组使用不可写、无输入别名的byte-backed存储；公开audit读取返回独立副本。失败保留已发生的计算记录和当时的原始状态，不把技术失败改成成功预测。new0直接继承实际B属于上层职责；调用本组件重新拟合不等于继承B。

## 验证与修复

主Agent在已验证`ssr-gpu`环境串行执行[test_d92_margin_qp_head.py](../tests/test_d92_margin_qp_head.py)，结果为**30 passed in 0.63s**。

证据：`E:/type10-7/.codex_tmp/pytest_utf8_1790834175912648700.stdout`与同前缀`.stderr`。先前28个用例的通过记录保留；最终数值验证是上述30项，不把先前检查重复计入。

检查覆盖独立有限特征原始解、单/多活动约束、完整列竞争和置换、奇异/零核、输入拒绝、不可写状态、预算/分解失败、原方程读回、规则点梯度差分，以及遗漏截距或乘子响应的错误梯度反例。它们仅使用手工合成矩阵，不使用目标数据或实际成绩。

独立只读审查发现1项P1：MOVE后失败快照的rho/V/工作集已更新，slack仍来自移动前。现于每轮可能失败操作前及预算退出时重建同一原始状态，移除上一工作集的方向/乘子快照；新增两项回归分别覆盖移动后预算耗尽和下一轮Cholesky失败。独立有限特征重建确认rho=1/12、V=[1/180,−1/180,0]与slack=[0,0]相互一致。数学、容差、API和预算含义未改变。其余限定项未发现新增P0/P1。

此前19项结构数学证书仍是独立的小规模证书；本次没有重复运行或将其当成30项求解器检查的一部分。完整联合方法的几何、B→C继承、优化循环及部署入口仍须各自验证。

## 实际成本口径

audit记录实际谱检查、分解尝试/完成、条件估计、三角求解次数与RHS数量、全约束扫描、状态重建、活动集变化及事件。新快照重建的扫描和P_old@V工作代理已计入，包括最后预算失败的额外重建。

维度代理不是实测FLOPs或训练时间。`peak_factor_buffer_bytes`仅统计该组件明确持有的因子输入/输出，不包括核、返回状态、BLAS工作区或进程总内存；求解临时数组有独立scope。返回数组与部署头α+b的数值字节单列，不能代替完整序列化模型大小或链路传输量。

实际联合可训练参数、完整训练/推理耗时、星载内存/显存、硬件和新增星地传输仍为N/A。本次0.63s是测试耗时，不是训练耗时。

## 恢复与下一步

组件状态为`COMPONENT_SYNTHETIC_VERIFIED`，整个方法状态仍为`IN_IMPLEMENTATION`，无真实实验run。上层联合核心由query-blind worker开发；根Agent负责串行验证、登记、唯一launch与独立评分。已有健康Conditional分析继续使用原PID617023/start7407753/source0145234及local handle55338，不因本次组件交付修改或重启。

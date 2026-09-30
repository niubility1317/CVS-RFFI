# MC-Residual8 独立 P0/P1 正确性审查

日期：2026-09-30。结论：**NO_UNRESOLVED_P0_P1**。本次发现的一个 P1 数学正确性问题已完成定点修复并独立复现核验。结论针对本次单一冻结候选的实现与合成正确性，不代表目标域性能已经提高，也不构成新的审批、数据重验或性能门禁。

## 范围与权限

工作树：`E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt`。审查依据为[冻结设计](D92_JOINT_AFTER_PROTOTYPE_DESIGN_20260930.md)、当前 `E:/type10-7/项目.md` 和本轮合法 target support SFT 联合优化授权。

审查覆盖五个 prepare/run/preflight/publish/analyze 编排脚本、`d92_margin_constrained_residual8_local_ridge.py`、evaluate/summarize 入口及其必要依赖和合成测试源码。审查者只修改本文；没有实施候选、启动训练、SSH、运行 Conda、读取 query 或 query 派生评分，没有读取 root handoff、总 registry、历史目标评分或 ABC。只对已发现的投影数学问题执行直接 Python 数值复现，其余测试由主 Agent 统一运行。

## 已解决的 P1：球投影后的实际位移可能成为上升方向

原设计先将 `d0=-g/||g||` 投影到保持风险半空间，再将 `theta+t*d` 投影到两个 Frobenius 球。第一步的 `g·d<=0` 不能保证第二步后的实际位移 `Delta` 满足 `g·Delta<=0`。只检查 `L_trial<=L_current+1e-4*g·Delta+tol_L` 时，正的 `g·Delta` 可以允许总目标微升。

最小合成反例使用两个标量球 `[-1,1]×[-1,1]`，可嵌入 U 与 V−V0 的各一个坐标：

- `theta=(1,0)`、`g=(-1,1)`、`a=(1,-2)`、`t0=t=1/8`。
- 取冻结物理松弛口径 `k=14,C_prev=6`，`epsilon=1/(28*sqrt(6))=0.0145802960879951`，`b=epsilon/t0=0.1166423687039609`。
- 半空间投影给出 `d=(0.3061711862154112,0.0947644087557251)`，`g·d=-0.2114067774596861`。
- 两球投影后 `Delta=(0,0.0118455510944656)`，因此 `g·Delta=0.0118455510944656>0`。
- 令 `s=Delta_y`、`M=(1-5e-5)/s`，局部光滑目标 `L(x,y)=3-x+y-M*y^2` 从 `2` 增至 `2.000000592277555`，仍小于原 Armijo 上界 `2.000001184555110`。
- 非负保持风险 `R_keep=(1+0.5*(x-1)-y)^2` 的起点梯度为上述 a；试探风险 `0.9764492148918004` 低于固定上限 `1.014580296087995`。这不是任意线性保持风险或任意松弛造成的反例。

反例仅否定原接受条件的一般单调性保证，不证明实际 support 矩阵必然发生这种状态。

修复位置：`code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py:487` 的 `_trial_acceptance` 同时检查 Armijo、`L_trial<=L_current+tol_L` 和实际保持风险，等价接受上界为 `min(L_current,Armijo_rhs)+tol_L`；`:548` 的实际优化循环调用该条件。冻结配置新增 `objective_nonincrease_required=true`。`tools/summarize_d92_mc_residual8_probe.py:326` 起独立重算三项条件和拒绝原因。

独立重放上述数值给出 `armijo_pass=true`、`keep_pass=true`、`objective_nonincrease_pass=false`、`accepted=false`，拒绝原因 `OBJECTIVE_INCREASE`，`tol_L=5.684345252788914e-14`。新增合成负测另覆盖真实球投影导致实际方向翻转的场景。该修改属于接受条件的数学正确性修正，没有调整 rank、松弛、步长、迭代或搜索参数。

## 关键正确性结论

| 审查面 | 结论与证据位置 |
|---|---|
| 表征与两套反向 | U=0 精确 R0 前向保留活 U 导数，V 首步数据梯度为零；exact GELU、切向投影、饱和与范数保持 VJP 符合冻结公式。总目标与保持风险分别执行 score 伴随；适配距离伴随乘 0.5。核心 `:98`、`:126`、`:205`、`:355`。 |
| 训练目标 | 每类损失先跨全部内折累计，再计算任务类 RMS 与旧类保持 RMS；保持 margin 的错类最大值来自全部当前注册类。新类不承担旧教师保持项；教师错误时 q=0。核心 `:345`、`:376`。 |
| 教师和物理隔离 | B 复用首次 R0 折内缓存；C 用冻结 B adapter，只在当前对应旧 inner-train 重拟合教师旧头。完整旧 B 头没有替代折内教师。B→C 核对同 row/context、旧物理 ID、标签与原始 feature；outer-held 只在 fit 完成后评分。核心 `:252`、`:266`；入口 `:376`、`:408`。 |
| 保存、失败与有限预算 | 初始、anchor、完整 U/V、总梯度、保持梯度、方向、所有已执行 trial 和 final 坐标保存在独占 NPZ；教师 score/q 同样保存。JSON 保留引用、元数据及摘要。拒绝试探不替换接受缓存，技术失败保留已接受状态及部分计费；最多 4 次迭代、每次 3 次试探。 |
| 成本 | B 预付初始头仅在 preparation 计费一次；共享 C preparation 的教师头单列计费；task/keep 两套三角求解分别计数。入口 `:387`、`:403`；汇总 `:201`、`:373`、`:423`。11776 个参数、94208 B 仅为 adapter 参数，常驻头/raw/标签/继承状态另计。未测部署传输与显存等项目保留 N/A。 |
| 完整合法汇总 | 完成标记与来源先核对，再读取本轮 support held score。汇总独立核对 NPZ、物理折、教师 q、类 RMS、方向/球投影、三项接受条件、最终接受状态和实际计数。完整 4 row、160 parent、三路径及 K×新增类数、receiver/scenario/model 分层全部覆盖。A 为 N/A，B0 不冒充地面 A。 |
| 编排与保护 | 固定矩阵和单候选，无参数组合搜索；运行与发布路径拒绝已有输出，不覆盖 support 缓存。CPU 发布不干预健康 GPU 任务；preflight 只读。主候选保持 R_MC_seq，reset_init 只作预声明对照。 |

K1、单类、单旧教师无法定义错类 margin、Nnew=0、零块、零带宽、近重复以及标签置换边界均有相应合成检查。单旧教师情况下关闭保持项；真实 K1 不伪造独立持出或学习；Nnew=0 复用对应 B。

## 验证证据与解释边界

主 Agent 统一验证共 48 个不同合成用例：core/编排首轮 36 个、2 个新增核心用例、entry/summary 10 个。审查者读回首轮与入口/汇总的输出，不另开全测试：

- `E:/type10-7/.codex_tmp/pytest_utf8_1790770662322837500.stdout`：`36 passed in 2.69s`。
- `E:/type10-7/.codex_tmp/pytest_utf8_1790771643704337400.stdout`：`10 passed in 17.62s`；同前缀 stderr 为空。
- 源码覆盖两目标全链有限差分、映射逐样本 bitwise 一致、近重复显式高精度 interaction 距离、教师折隔离、B/C 继承、状态坐标无损保存、日志/CSV 原生有限 JSON、实际计费、非法条件及汇总篡改负测。
- 审查者针对上述 P1 的直接数值复现和修复 helper 读回已记录于本报告，未重新运行完整 pytest。

当前没有未解决的 P0/P1。有限差分和合成隔离检查证明所测场景的实现一致性；内层三项接受条件只保证浮点容差内的总目标非增及固定保持风险上限，不保证收敛、support OOF 提升、query 性能、旧类下降≤1 个百分点或新旧差≤3 个百分点。teacher B 参数已经接受合法旧 support 监督，内 held 是训练监督；外层 support held 也不能被称为 query 或独立目标数据确认。正式固定 support 矩阵的运行、实测资源及完整结果由主 Agent 后续记录，本报告不宣称已启动或完成该矩阵。

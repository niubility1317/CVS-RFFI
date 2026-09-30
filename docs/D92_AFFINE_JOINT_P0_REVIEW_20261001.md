# AFFINE_JOINT 独立 P0/P1 审查

日期：2026-10-01。结论：**`NO_UNRESOLVED_P0_P1`**。

本次对新 AFFINE_JOINT 代码草案完成一次独立只读审查，未发现可证实、会导致下一次实验越权、方法执行错误、覆盖既有产物、无法启动或产生非法预测的 P0/P1 问题。该结论是所列代码与合成测试的正确性审查，不是性能结论、真实数据验证或远端运行完成证明。

## 范围与依据

审查依据为本机当前 [项目协议](E:/type10-7/项目.md)、[八项最小流程](E:/type10-7/tools/optimizer_workflow_contract.md)及本次用户对合法目标 support 监督适应的授权。当前本机协议优先于快照内历史 optimizer 条款；本次没有沿用历史性能门槛或额外来源授权链。

审查范围包括新 [核心](../code/cvsrffi/d92_affine_joint_local_ridge.py)、[入口](../tools/evaluate_d92_affine_joint_probe.py)、[runner](../tools/run_d92_affine_joint_probe.py)、[准备工具](../tools/prepare_d92_affine_joint_probe.py)、[发布工具](../tools/publish_d92_affine_joint_probe.py)、[分析工具](../tools/analyze_d92_affine_joint_probe.py)、[独立汇总](../tools/summarize_d92_affine_joint_probe.py)、[固定结构配置](../configs/d92_affine_joint_frozen_20261001.json)及对应合成测试。同时对照了 [截距数学审计](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md)、[实现映射](D92_AJLR_AFFINE_IMPLEMENTATION_MAP_20261001.md)和新 [CORE](D92_AFFINE_JOINT_CORE_20261001.md)、[ENTRY](D92_AFFINE_JOINT_ENTRY_20261001.md)、[SUMMARY](D92_AFFINE_JOINT_SUMMARY_20261001.md)说明。

仅新增本文档，未修改核心、配置、入口、其他方法或健康运行。没有执行 Git、Conda、测试、SSH、发布或实验；没有读取真实 run、query、outer 成绩、实验总索引或主任务交接。新候选处于代码草案阶段，本次没有新 run/spec/launch 证据。

## 核查结果

| 核查项 | 可从代码及合成测试确认的行为 |
|---|---|
| 无惩罚截距与完整伴随 | `_solve_affine_head` 对 `A=I+K` 使用一次 Cholesky 和合并 `[E,e]` 右端，形成 `z`、`s`、截距及 canonical `alpha`。`_affine_adjoint` 明确包含 `g_b=sum_rows(G)`，满足 `A T+e eta=L.T G`、`e.T T=g_b`；完整中心化 VJP 再传递到两端几何及 adapter。独立 primal/saddle oracle、非零 `g_b` 反例和 B/C 的 Z 方向有限差分覆盖了新增数学行为。未发现遗漏截距梯度或惩罚截距的实现。 |
| q-gauge 的适用边界 | q 等价测试固定原始核、尺度、U、support 及 prior；比较 alpha、完整 score、ridge 项和完整梯度，并检查截距的必要变换。配置和说明没有将该等价性扩展为重新估计 tau/gamma、更换 support 或删除旧参考 raw 几何导数的权限。均衡 B 的固定 U 等价与不平衡低层反例分开处理，没有声称 C 准确率必然保持。 |
| 当次实际 B→C | 入口真实调用顺序为 `prepare B → fit B → prepare C → fit C`，传入同一物理路径刚产生的 B 对象。核心检查 run、row、scope、split/fold 绑定及旧 support 的物理 ID、标签、原始特征；C anchor 精确取实际 U_B，final prior 包含实际 B 的核、截距和类列映射。C inner prior 在冻结实际 U_B 下只拟合该 inner-train 的旧 support 头，无嵌套 B 优化。该 inner-held 监督被明确记为训练，未冒充独立验证。入口没有加载历史目标适应状态的接口。 |
| 退化与复用 | 真正 K1、train-K1 proxy 和 rank0 不更新 adapter，但仍执行完整解析头；真正 K1 不生成独立持出指标。`tau=0` 且 gamma 有效时使用原始完整特征等价核并执行 Schur 解，不误作零核；缺尺度零核使用 `intercept=mean(E)`、`alpha=E-intercept`，无虚构分解。`new0` 在 C preparation/fit 前精确复用 B 对象和分数。测试区分这些分支及其实际费用。 |
| 输入与推断边界 | Phase1/cache 固定，入口沿用 support-only cache 与 source-only checkpoint 来源绑定，未增加 checkpoint 重载、source 逐样本输入或 query 拟合接口。外层 held support 在拟合完成后评分；inner-held 标签只用于合法 support 训练目标。score 逐记录计算，面对全部注册类，按物理 class ID 确定并列顺序；没有 query role、真实 batch 类数、配额或全局重排。 |
| 预算、失败与真实资源 | adapter、函数坐标、原始旧 support 尺度及 4×12 first-acceptable Armijo/nonincrease 预算不变。拒绝 trial 的 forward、head 和求解全部累计，最后接受 cache 不被拒绝 trial 替换。前向为每次 triangular 调用 C+1 个 RHS，伴随为 C 个，记录真实 `r`、`n*r`、`n*n*r`，并声明其是工作量代理而非实测 FLOPs。解析截距 C、独立 contrast C−1、解析系数与梯度可训练/实际更新参数分开计量；resident/deployment 数值 buffer 包括实际截距和 B prior，Schur/RHS 训练缓存不伪装成部署需要。异常保存已完成上下文及不完整归档，无自动重试或数值 fallback。 |
| 真实事件与完整重算 | 入口保存实际 preparation/INITIAL/GRADIENT/TRIAL/STEP/FINAL 回调、完整 NPZ、文本、紧凑 JSONL/CSV。独立汇总核验当前 run/row/path 的全部数值引用，重算 Schur 方程、完整 prior+residual+截距分数、非零 g_b 伴随、中心化 VJP、CE/proximal Z 梯度、trial 接受条件、RHS 费用及实际数值状态字节。它检查全 160 parents、1800 paths、全 proxy anchors、parent-first 聚合、事件及文件清单和 row/run 总数，不从部分结果生成最终结论。 |
| 集成与产物保护 | 新 method/schema/config 与旧方法分开；runner 传递明确 run/row，发布清单包含新核心及复用的纯计算依赖，不发布历史适应状态。新输出使用不可覆盖创建，失败不写成功 marker；独立分析先检查完整四行和 160 parents，再读固定的外层 support 结果。没有在本候选中添加参数搜索、source 特征、重验 IQ、125 早期门槛或自动晋级。 |

上述检查对应的合成测试为 [核心测试](../tests/test_d92_affine_joint_local_ridge.py)、[入口测试](../tests/test_evaluate_d92_affine_joint_probe.py)、[编排测试](../tests/test_d92_affine_joint_orchestration.py)和[汇总测试](../tests/test_summarize_d92_affine_joint_probe.py)。汇总负测包括截距、Schur、合并 RHS、`g_b/eta/T`、Z 梯度、actual B 绑定、费用、事件、分数和归档元数据篡改；入口测试观察真实 callback 顺序，未用汇总器自身生成期望事件。

## 验证证据与结论边界

主任务提供的既有串行验证为 core/entry/ops **58 PASS，7.09 s**（合成证据前缀 `.codex_tmp/pytest_utf8_1790801239413599000`）以及 summary **10 PASS，109.86 s**（前缀 `.codex_tmp/pytest_utf8_1790802427189642100`）。本审查读取测试实现并独立检查公式与调用链，没有重复执行这些测试，也没有读取上述测试运行产物；通过状态为主任务报告的证据。此前 bare JSON、scalar 0D 保存及 summary scalar shape 兼容修正在当前代码中可见。

本次未核实真实 cache、IQ、checkpoint 文件、数据来源事实或远端运行状态，不能以本审查替代已有数据结论或当前输入绑定。支持集诊断不是 query 泛化结果；A 和 B−A 仍为 N/A。未归档完整 final C 在 Z=0 的头时，注册本身与后续 adapter 对外层分数的分解继续为 N/A，汇总不会额外拟合补造。部署包、新增传输量、星载/GPU 性能等未测项保持 N/A；数值 buffer、参数数和求解代理不能证明实际省算力。

未发现未解决 P0/P1；没有新增性能硬门槛、固定审批或重复审查/测试要求。10/1/3 个百分点方向仍是用户给定的理想目标，不是本轮自动晋级条件。

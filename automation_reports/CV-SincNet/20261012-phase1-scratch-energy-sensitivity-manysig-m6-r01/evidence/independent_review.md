# 独立 P0/P1 发布正确性审查

结论：**PASS，未关闭 P0/P1 为 0。**

范围为本包设计及源优化器诊断的 `next_experiment_design.md`、固定尺度/真实导数/校准实现、训练与持久状态、评估/日志分析、发布控制和六行登记。审查者未参与实现，仅编写本文件。未修改的原基础机制沿用已有核验，不重复全量检查。

## 科学契约与真实路径

- 六行登记逐项等于 `design.config`：MARGIN_ENERGY、CE_ENERGY 各三种子。每行 scratch20/4440，累计 26640 次身份更新；保持原 identity9 DELAYCAP_JOINT 的优化、style、U KL、联合 cap 与数据角色。固定历史参考只在独立报告使用，无目标反馈。
- CE 分支实际调用 `cross_entropy(..., reduction='none').sum()` 对虚拟 a/b 坐标求导，`create_graph=True`；没有分离置信度因子或额外 batch 平均。margin 分支保留原平滑 margin。两层由原生 time_down/t2.pool hook 提供，混合二阶梯度仍连接身份网络。
- `_pair` 的父类 margin 标签仅用于共享采样/方差函数的旧 guard；动态调用的 `self.sensitivities` 确为本包实现。输出 metadata 替换为 raw_sensitivity/实际 objective/固定层尺度；未暗中调用旧逐样本归一化。
- 每层仅有一个固定源 L RMS。以 float64 累计全部 1440 次样本读出及全部坐标平方和，分离标量后用于所有样本/RX；后续不动态更新，不逐样本自归一化。非法/零尺度拒绝，无默认 1 或重新采样回退。有限总体 A/B 估计及其负值、两层固定分母 2 保留。

## 时序、状态与数据

- 第 444 次 optimizer 更新完成后、下一次 L/U 抽样之前，两遍读取同 12 对 A/B60：第一遍固定尺度，第二遍固定损失系数；合计 48 次前向、2880 次 L 读出、0 身份/头更新。首次辅助步骤为 448，训练辅助仍为 999 次。
- 启动仅写 PENDING 校准状态；完成校准后发唯一 ready 事件。主线程自检发现的“初始 resolved 提前 export 未校准状态”已移除，并由两臂真实 `source.train` 入口回归核对到首批 objective 边界（合成 fixture，0 更新）。预检/集成已把身份参数分组放到校准之前，和生产路径一致。
- final checkpoint、auxiliary_state 与 resources 显式保存 `fixed_sensitivity_state`；模型严格恢复目标、归一化方式、444 时点、两层尺度和系数。缺失/错误状态拒绝，不用空 state_dict 代替。评估额外核对 checkpoint 状态与自身 calibration.json 一致；普通推理不安装虚拟敏感度 hook。
- 学生无初始化 checkpoint，原 E20 两教师仍排除当前 seed 并核验完整合法来源。L6300 用于监督/校准，U56700 的真实 TX 隐藏，V27000 仅预定无梯度健康验证；无 EMA、星地增强、U 风格或可训练辅助头。

## 记录与发布

- 源分析核对精确两层尺度、float64 RMS 重算、每步 objective/normalization/scales、RX×TX 幅度单元、二阶图、完整预算与 raw/effective cap。新增校准成本按 48/2880 单列，历史参考为 24/1440，不宣称同计算成本。
- 逐行固定 final20/4440 后自动预测 168000 个 clean query，prediction 固定后独立 truth-last 评分；六行 1008000 条决策。主比较 CE_ENERGY−MARGIN_ENERGY，历史对照固定全部三种子，缺种子不补值或重跑。
- 独占 run/release/owner/output，旧健康任务保持；每 GPU 至多 3 个总进程/预留，UNKNOWN 不自动重试，不覆盖旧产物。60 个运输依赖全部存在，包含新包及 identity_style，预检与调度入口正确。

## 已核实证据

登记 `automation_reports/CV-SincNet/20261012-phase1-scratch-energy-sensitivity-manysig-m6-r01/evidence/` 的五份结果均为 PASS：

- `energy_sensitivity_local.json`：六行预算、两臂原生 checkpoint 往返、非法来源及状态拒绝；评估与 calibration.json 错尺度拒绝。
- `energy_sensitivity_integration_cuda.json`：分组先于校准的两臂原生目标/反向、早期 CE 路径与实际前向预算。
- `energy_sensitivity_math_cpu.json`、`energy_sensitivity_math_cuda.json`：两目标浅中层一阶及混合二阶数值核对、CE 恒等关系、样本幅度保留、48/2880 同批校准、状态/RNG/mode/grad 保持、非法状态拒绝、活二阶反向与两臂训练入口。所列一阶最大绝对误差约 8.53e-12，混合二阶最大绝对误差约 6.43e-9；这些是合成局部检查，不代表性能结果。
- `reference_checks.json`：两个固定历史比较与缺种子行为，仅合成数据。

本次仅本地只读审查和配置/证据核对，并创建本文件；未修改实现、启动训练、使用 GPU、操作远端或读取目标成绩。PASS 不表示正式训练已完成、识别性能已提高或 84% 目标已达成。

审查时间（UTC）：2026-10-12T00:50:35.999364+00:00

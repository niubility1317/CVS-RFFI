# 固定 CENTER_25K 与跨样本贡献：源检查结果

**执行完成并独立核实；预登记源可用性条件未通过，不扩为完整训练。** 固定 BASE 首种子 2026092701/final50，actual code commit `cc43ab0012fdee8bd4c42f347ae9ce699debb7df`。4 个固定 L90 梯度批次与分离 L180 评价，RAW_HZ／CENTER_25K × SAME_L／HELD_L 共 80 次贡献比较。所有 4 个矩阵单元都是 0/20 accepted；CENTER_25K 的 HELD_L active 与 accepted auxiliary kernel-change cohorts 均 0/4。

| 坐标 | 评价样本 | 接受组 | 平均 CE 贡献范围 | base 改变滤波核数／24 | auxiliary 再改变核数／24 |
|---|---|---:|---:|---:|---:|
| RAW_HZ | SAME_L | 0/20 | -2.03e-08 至 1.15e-08 | 0 至 0 | 0 至 0 |
| RAW_HZ | HELD_L | 0/20 | -5.74e-09 至 5.16e-09 | 0 至 0 | 0 至 0 |
| CENTER_25K | SAME_L | 0/20 | -1.26e-08 至 1.6e-09 | 12 至 13 | 0 至 3 |
| CENTER_25K | HELD_L | 0/20 | -5.65e-09 至 3.45e-09 | 12 至 13 | 0 至 3 |

CENTER_25K 恢复了部分实际滤波变化：base 更新改变 12 至 13 个核，auxiliary 相对 base 最多再改变 3 个核，最大核差 `1.1920928955078125e−7`。RAW 两类更新均未改变核。中心坐标的 HELD_L 最大平均 CE 贡献只有 `3.453654473517531e−9`，低于固定 `1e−6` 容差；这些微小正负量不能解释为可用收益或确定伤害。辅助梯度非零（约 .00124 至 .00555），但“有梯度／有实际核变化”仍不等于其他样本获得可分辨学习收益。

两个坐标的当前批次 CE 完全相同；4 个数值为 .0113936、.00189436、.00753934、.00366156。此为接近源拟合饱和的 final50、eval-mode、L-only 假设步骤，不能据此否定所有早中期训练信号，也不允许由此改容差／尺度扫参或直接扩训练。模型读出仍为原默认头；没有新训练、头拟合、U/V/目标 IQ 或 truth 访问。

完整成本：540 unique L IQ、6300 source metadata，18,004 packet network forwards、160 临时 AdamW、240 kernel 合成；实际 optimizer／EMA／head-fit 更新均 0。真实 checkpoint no-query smoke PASS；唯一 P0/P1 审查 PASS；archive SHA 与 remote compile VERIFIED；独立 PID/CWD/argv/log-growth/GPU 检查确认 CPU-only，并在 summary 完成后核实自然退出。所有模型／buffer／moment／gradient 保持。矩阵与 320 个逐 RX 贡献见 CSV，完整 8 cohorts/80 groups 保存在 result_evidence.json。

本轮保留固定 CENTER_25K 和 SIG_TIME 各自已有正路线，但未证明二者组合收益，更不能相加旧增益。下一轮优先在实际早中期训练检验持续的特征学习信号；末期微小 CE 增量筛选不再单独承担机制晋级依据。84% 默认头三种子目标仍未完成。

以下保留原预登记，不追溯改写阈值、矩阵或预算。

---

# 固定 CENTER_25K 与跨样本特征贡献：源检查预登记

运行 `20261015-phase1-SIGTIME-center-heldcredit-sourceL-manysig-s2026092701-r01`。只检查固定匹配 BASE 首种子 2026092701 的 scratch50/final-only checkpoint。父运行 `20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01`；完整源数据角色和祖先来源核对后才加载。默认 6×160 分类头结构和权重保持，0 实际训练／头拟合／EMA 更新，不访问 U/V/目标 IQ。84% 三种子目标仍未达到。

固定矩阵 RAW_HZ／CENTER_25K × SAME_L／HELD_L。CENTER_25K 原样复用历史已核实的 25,000 Hz 尺度和原构造频率 origin，不扫描尺度、温度、间隔或分类头。只替换 primary identity Sinc 的 48 个坐标；其余模型参数保持，native AdamW 的 step 和状态逐参数保留，center 的 m×25000、v×25000²。它是冻结 BASE 的假设坐标切换，不能冒充历史 CENTER 的完整优化轨迹或数值问题单一因果证明。

4 个 L90 批次使用原 90×70 源 cells 内 ordinal 0、17、35、52。HELD_L180 使用每个 cell 的 ordinal 68、69，完整物理 ID 和全局 base_index 均与当前梯度批次分离；这些样本仍是已训练 L，不是未见样本、未见 RX 或源 V。只读取 540 条唯一 IQ，另读取 L6300 元信息。固定 ORDER 使用实际部署 SIG_TIME pre-gain160，anchor 接图，positive/negative 停梯度，margin/coefficient=.1，全部合法 triplet 平均含满足项。auxiliary W/gain 梯度必须 None；main CE 默认头梯度须非零。

每个 anchor RX 分别比较 same saved native AdamW 下 B 和 B+aux，独立 clip5，并用相同更新前默认 W 评价另外 4 个源 RX；mean credit>1e−6 且每 RX>=−1e−6。每次临时前向先清空参数派生滤波／归一化 W 缓存，结束恢复原缓存及 live 参数／buffer／optimizer／gradient／RNG。原因是 native cache 用 tensor._version，functional_call 两套不同权重可能版本相同；实际 CosFace 聚焦检查已经复现。旧 CREDIT 结果与产物保留，其机制归因须补记该实现限制，不能据此删负结果或重跑目标。

源可用性阈值预先固定：CENTER_25K 的 HELD_L 至少 2/4 个批次各有 accepted 且梯度>1e−8 的 RX group；至少 2/4 个批次的 accepted auxiliary 还须实际改变 Sinc kernel（total 与 base 对比）。任一失败则本候选不直接扩为训练矩阵。160 次临时 AdamW，18,004 次 packet network forwards（含 4 个零 IQ smoke），0 实际更新；额外 kernel 合成单独统计，不计作 IQ 前向。全部结果保留，不用目标分数调参或选父权重。

代码/config 身份为 release_commit.txt 对应 Git commit。普通 N607 项目 Python，CWD `releases/cvs_SIGTIME_center_heldcredit_sourceL_20261015_r01`；发布命令 `python -X utf8 E:/type10-7/.codex_tmp/native_polyphase_runner.py experiments.cvs_sigtime_heldcredit.publish`；运行命令 `python -u experiments/cvs_sigtime_heldcredit/runner.py resolved_request.json`。CPU 2 线程、GPU 0，唯一 immutable output `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-center-heldcredit-sourceL-manysig-s2026092701-r01/source_diagnostic`。一次预检、一个 release archive 的本地／远程 SHA 比较和 remote compile；launcher 第一步严格 checkpoint no-query smoke，PASS 直接执行。独立 PID/CWD/argv/GPU/log-growth 检查，唯一 P0/P1 审查记录在本报告，禁止重复候选审查或报告审批。

仅发生本 run 的系统技术错误才停止：契约／来源／元信息／状态不一致、非有限值、依赖缺失、越权数据或头拟合、live 状态改动。保留输出，不自动重启，不影响其他健康进程。预期 runtime、8 个完整 ordinal 结果和 summary；本轮没有实际训练，source-only 明确范围不新增测试集访问。

唯一独立 P0/P1 审查（2026-10-15）PASS：固定 BASE 来源、坐标与实际 AdamW 状态、same/held L 分离、参数缓存恢复、输出保护和发布入口均无下一次源检查的直接阻断缺陷；核对 80 assessments／160 临时步骤／18,004 IQ 前向／240 kernel 合成。聚焦 native CosFace 缓存碰撞、修正恢复与非法 role／重叠 ID 拒绝检查通过；代码编译通过。

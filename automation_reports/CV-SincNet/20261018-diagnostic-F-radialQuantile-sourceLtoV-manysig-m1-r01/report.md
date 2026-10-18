# 全源 256 点：共同轻度径向幅度关系的 L→V 反证

状态：ANALYZED_SOURCE_ONLY_SCIENTIFIC_NEGATIVE。原 JSON 故障已从保存统计恢复；源检验失败，固定假设关闭。无 IQ 增强或身份识别结论。

## 固定问题与权限

唯一假设：源 RX 之间存在跨 TX/day 可迁移的轻度共同径向幅度关系，足以支持后续独立 IQ 身份保持检验。数学探针为 g_a(z)=|z|^(a−1)z，再作 RMS 归一化；幅度下限未激活时，去中位数的 log 幅度分位数恰按 a 缩放。它不是已识别的接收机硬件，也不能证明 PA/TX 身份保持。本 row 不生成变换 IQ、不训练。

完整原 L6300 和 V27000；RX 1/3/4/6/8、day 1/2/3、6 TX、split392005、全部原 256 点。每角色 90 个 TX/RX/day 单元，每单元 L70/V300。无训练 RX 留出，无 L5580/dev720；U56700 TX 真值隐藏，未使用。既有 VALIDATED_ONCE 源载体复用；只核对现有 role_ids 及原生 EXACT_MATCH。原 source_cache 含未使用 U/features 容器的反序列化不计作消费。零 RMS 包保留并记录。

源契约：PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json。预处理：PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701。没有模型、父权重、教师或 checkpoint；后续训练约束仍为 F＋跨 RX 同 TX 对比学习、默认 cosine 头联合训练、完整源数据和 E30。

## 统计和固定控制

每个原包统计 log(max(|z|/RMS,1e−6)) 的 .1/.2/.3/.7/.8/.9 分位数，减 .5 中位数；所有点、所有包参与，不切片、不筛样本、不改 IQ。L/V 分别按 90 单元取均值。

18 个 (TX,day) 统计折：共享 scalar a 仅由其他 5 TX、其他 2 day 的 L 拟合；每折保留全部 5 RX。RX1 作为参照观测，不是留出的训练域。用同 TX/day 的 RX1 V 形状预测其他 4 RX 的 V 形状；V 仅用于读出，不能拟合系数或更新持久状态。RX1 的零误差不计作正域。

固定控制为恒等 a=1、L-only 共享六维加性偏移、同 TX/其他 2 day 的 scalar 关系。它们分别检验微小数值改动、纯偏移和 TX 条件关系能否解释结果。

以下全部成立才通过：

- 恒等误差 RMS ≥0.01；共享 SSE ≤0.8×恒等、≤0.9×加性偏移。
- 18 折至少 15 折优于恒等；每个非参照 RX 的共享 SSE 同时不高于恒等和加性控制。
- 共享 SSE ≤1.05×TX 条件 SSE。
- 全部 72 个共享系数均在 [0.9,1.1]；不截断或重拟合。
- 每源单元幅度下限比例 ≤0.01，总零 RMS 包为 0；SSE 数值容差 1e−20。

失败关闭固定的全 256 点、这些分位数、RX1 参照、幂律族、轻度范围与源问题；不以更换指数、窗口、分位数、参照、RX 或阈值挽救，不自动训练。通过仅说明源分布关系；随后仍需真实有界 IQ、默认头身份保持和匹配 E30 识别证据。不得称为 A_g、A_s 或联合目标完成。

## 执行、成本和交付

Run：20261018-diagnostic-F-radialQuantile-sourceLtoV-manysig-m1-r01。Row：RADIAL_POWERLAW_L_TO_V。Owner：codex/root/F-radial-quantile-source-20261018。代码身份为实际 Git 提交；publisher 从 HEAD 写入 release_commit.txt 和归档内登记，不要求额外配置哈希。

环境：N607 既有普通用户 CVS-RFFI Python/PyTorch/NumPy，CPU 2，GPU none；CWD：PROJECT/releases/cvs_F_radial_quantile_source_20261018_r01。命令：CVS-RFFI/bin/python -u -m experiments.cvs_radial_quantile18.run。唯一不可覆盖输出：PROJECT/runs/20261018-diagnostic-F-radialQuantile-sourceLtoV-manysig-m1-r01。发布入口：experiments.cvs_radial_quantile18.publish。

预算：33,300 原源包，199,800 包分位数实数，1,080 单元分位数实数；144 scalar 拟合、72 加性向量，每种方法 432 非参照预测坐标，4 种方法共 1,728。0 新 IQ 视图、0 checkpoint、0 网络前向、0 更新、0 教师、0 U 训练、0 target。原缓存只加载一次；记录实耗时间，不占 E30 身份训练预算。

源角色缺失/不匹配、非有限数、输出冲突或异常时保留产物并停止本 run；不自动重试。只操作本 run，不停止或修改其他任务。唯一 ordinary preflight、一次归档传输 SHA 对比、一次远端编译；运行首步直接执行数学/权限 smoke，PASS 后继续。该阶段不消费 checkpoint，不增加无用模型 smoke。发布后核对本 run 的 PID/CWD/argv、CPU 环境、日志增长或自然完成。

产物：resolved_config/checks/active/source_ids（私有）、observations.npz（私有）、cell_quantiles.npz（私有）、fitted_readout.npz（私有）、result.json、completion.json 或 failure.json。独立 recount 只从保存的包分位数重建全部单元/拟合/预测/判定，不重新读取原 IQ；它不是独立原 IQ 分位数重放。公开仅汇总统计、验证和报告，不提交 IQ/ID/日志/权重/私有 runtime。

## 发布前验证

本地数学及权限检查 PASS：37 个合成 IQ，幂律分位数关系最大误差 4.440892098500626e−16；共同轻度关系通过，纯加性、TX 专属、过强系数、幅度下限污染均被拒绝；V 污染不改变 L 拟合系数，9 项输入/权限反例通过。0 真实数据、0 checkpoint、0 网络。新增整个包编译通过。

唯一独立 P0/P1 review PASS：未发现导致下一次实跑错误、越权、破坏、无法启动或无法生成合法结果的缺陷；覆盖数学关系、18 折 L-only、V 读出、全部 RX、零包保留、U 不访问、发布依赖。未修改文件或访问真实输入/远端。

REJECTED_EXTRA_GATE：泛化设计审批、seal 或重复报告审查不适用；按用户已授权研究及项目最小工作流直接执行。

## 技术故障与固定统计恢复

首次执行的数学/权限 smoke PASS，33,300 源包统计及全部拟合数组已保存；在写 result.json 时，nonzero_identity_error 为 NumPy bool，标准 JSON 拒绝序列化，原进程已退出。独立原始产物证实失败层为汇总保存，不是源输入、统计或权限错误。原 failure.json、日志及全部数组保留，不覆盖原发布、不重新读 IQ、不重跑训练。

最小修复：core/recount 中该比较显式转 Python bool；新增完整正负结果 JSON 序列化回归。原失败复现后，修复验证 PASS。恢复仅消费原保存的统计，原全部 fitted_readout 值必须在 1e−12 内复现；另存不可覆盖 json_recovery_v1 子目录，再由独立 recount 全量复算。原 scientific 阈值、范围、数据及方法不变。原 source_cache 加载 1 次；恢复增加原 IQ/缓存/CP/NN/训练/target 均为 0。

最终结果见 [源检验完整分析](source_analysis.md)。

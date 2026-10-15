# E20 真实证据责任：R02 技术恢复

**PLANNED。** 主机制与 R01 完全相同：当同包实际分支证据加入后提高身份交叉熵，辅助项只把修复梯度送回负责组件；默认分类头和部署特征继续按主交叉熵正常联合训练。源 INITIAL 3/4、FINAL 0/4 支持从零 E20 验证，尚未证明完整泛化收益。84% 默认头三 seed clean 均值目标仍未完成。

## 修复依据与范围

[R01](../20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01/report.md) 在已启动三行的 step16 诊断比较处退出，没有完整 epoch、冻结预测或评分。独立退出证据和不可变原产物保留。CPU 真实构造器未复现；本地 CUDA 使用相同 FULL_FP32 策略、实际模型、合成 L128×16 复现了两个 Sinc 坐标的 autograd 影子与真实更新差异。实际 backward 梯度推演及实际 postclip 梯度单步 oracle 均与 188 个真实参数逐位一致。数值与环境边界见[CUDA 复现标量](cuda_shadow_regression_evidence.json)，不把本地复现冒充远端逐包复现。

R02 只修复诊断参照：保留原两分支 autograd 临时推演，明确其为近似反事实，记录其与真实更新的差异；真实更新一致性改由相同 optimizer/state 和实际 postclip 梯度复制生成的 oracle 验证。显式复制 grad/None，不再次裁剪，不修改 live 参数、梯度、optimizer 状态或 RNG。真实比较仍用逐元素 atol=2e-7、rtol=2e-6。每个固定探针由 2 次增为 3 次临时 AdamW，即每模型 8 次增为 12 次；真实更新、损失和日程均不增加。近似增量不宣称是精确的实际辅助增量。修复仅接受一次针对该直接问题的 P0/P1 复审，不重复既有源机制或主矩阵审查。

## 固定科学矩阵与预算

三 seed 2026092701/02/03，各四臂 BASE_CE、BASE_SSL、JOINT_GAP、ROUTED_GAP，共 12 个全新 scratch20 模型，每行 20 epoch、222 step/epoch、4440 次实际更新；全矩阵 53280 次。R01 部分状态不继承。BASE_CE 为 20L/0U，真实关闭 U/EMA；其他三臂 13L+7U，保留原 oldgate/ownEMA。每行均为 SIG_TIME、CENTER25K、residual_fusion、NoExplicitCFO、原 joint160 和默认 cosine 头；AdamW lr2e-4、wd1e-4、clip5、cosine min1e-6、实际 label_smoothing=0。历史 M15 实际 .01 的配方差异继续披露，新基线独立训练，不复用旧分数。

辅助仅在第 1 至 13 轮每 16 step 启用，共 180 次，四个 L180 bank 循环、720 个当前训练包，每 active 模型额外 32400 次网络 packet 前向；基线不建 bank、不读额外 IQ。900 包载入中的 held180 不用于主训练梯度；当前 bank 与普通 L loader 可以重叠，不能称未见 holdout。JOINT/ROUTED 使用完全相同损失标量、mask、720 分母、数据和日程，区别只在组件边的梯度去向。辅助 W/三个 gain 停止梯度，主 CE 正常联合训练；不新增参数、分类头或拟合诊断头。四个真实更新探针仍为 16/896/1776/2880，仅位于 label 阶段。后 7 轮关闭辅助并保留原 U 强视图 Gaussian。L6300/U56700 隐藏 TX/V27000 及原 RX/day/TX/physical split 不变，V 无梯度或状态更新。

先完成首 seed 四行的源健康和机制信号：固定 final20、L CE≤.5；BASE_SSL/JOINT/ROUTED 的源 V 整体/最差 RX 较 BASE_SSL 下降不超过 .01/.02；两 active 臂 scheduled180、eligible≥95%、至少两个固定探针非零辅助梯度、positivegap call fraction≥1%。只据这些预登记源规则扩另两个 seed，不等待或读取 target 分数。未通过则保留首 seed 四行并完成其已授权测试，不把部分矩阵当完整三 seed。

逐行 final20 冻结后自动 clean168000 原默认头预测，全部 prediction 固定后由新的独立 CPU scorer 连接 truth；18 CM/row、全矩阵 216 指标和 2016000 决定。汇总实际 seed 数、均值/样本 SD、MacroF1、最差 RX/day、预定配对及纠正/新增错误。目标基准已经暴露，不称新盲测，不将分数回流参数、候选排序、扩种子或重跑；本轮 clean-only，不追加卫星部署声明。

## 执行与停止

run `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02`，owner `codex/root/evidence-responsibility-train20-20261015`，实际 commit 由 `release_commit.txt` 固定。普通 N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_evidence_responsibility_train20.dispatch`，不可变输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r02`。GPU 需至少 12000 MB 空闲、每卡最多 3 个总实验进程（含其他 owner）；不停止或热改健康任务。每行完整配置、路径、种子含义和预期产物见[experiment.json](experiment.json)。

沿用输入合法性、聚焦负例、真实合法 checkpoint no-query smoke、一次定向直接缺陷复审、一次归档传输比较/remote compile 和发布后 PID/CWD/argv/GPU/日志增长核对。相同诊断指纹若再次出现，保留状态并停止盲目重试。既有科学停止规则不变：Input/role/provenance mismatch, forbidden Utruth/targetfit/headfit, outputcollision, nonfinite, wrongcallback/actualstep or inability to close legalprediction/scoring: stop newdispatch, preserve partial and healthyactive rows, reconcile before retry. Sourcehealth/signal failure freezes firstseed4 scope and still completes their clean predictions/scores; no seed02/03 expansion. No targetscore feedback or healthyprocess termination.。历史正收益结构和其他机制负结果继续保留。


发布后 **RUNNING / VERIFIED**：实际执行 commit `3a5cff49091deef46cad5d787e5daea7b4c0dfb9`，归档单次比较、remote compile、real-checkpoint no-query preflight 均 PASS。首 seed 四行独立核实 CWD/argv/GPU 与日志增长，均完成至少 3 个源训练 epoch/666 实际更新；四个 step16 实际梯度 oracle 误差均为 0，autograd 近似影子差异如实保留，JOINT/ROUTED 辅助范数分别 .383151/.218327，W/三 gain 辅助梯度均 None。四个真实 worker 的 200 个初始 state 张量逐项相等，已检查来源后读取各自 scratch 初始快照，未读 IQ。BASE_CE 的 U/EMA 实际关闭、两基线额外 IQ 为 0，两 active 臂各 41 次辅助/7380 额外 packet 前向，与第 3 轮预算一致。另 8 行等待固定源规则；未完成训练、预测、评分或 84% 目标。见[启动和进展证据](launch_evidence.json)与[实际初值证据](actual_initial_state_evidence.json)。

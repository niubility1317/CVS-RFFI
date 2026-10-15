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


## 完整三种子结果

**ANALYZED / VERIFIED**。12 行均完成从零 E20、默认头 clean168000 预测及独立 CPU truth-last 评分，216 份指标及 2016000 条决定齐全；调度器和所有 worker 自然退出。实际执行 commit `3a5cff49091deef46cad5d787e5daea7b4c0dfb9`，三个种子内四臂初始 state 均逐项相同。

|条件|seed01|seed02|seed03|均值 ± 样本 SD|Macro-F1 均值|
|---|---:|---:|---:|---:|---:|
|BASE_CE|82.1113%|80.2179%|81.3000%|81.2097% ± 0.9499 个百分点|0.807659|
|BASE_SSL|81.9381%|79.9423%|81.6440%|81.1748% ± 1.0775 个百分点|0.807283|
|JOINT_GAP|82.0869%|79.6286%|81.5679%|81.0944% ± 1.2957 个百分点|0.806419|
|ROUTED_GAP|81.9238%|79.7768%|81.6613%|81.1206% ± 1.1712 个百分点|0.806674|

ROUTED−BASE_SSL 的均值差为 **-0.0542 个百分点**，1/3 个 seed 为正；ROUTED−JOINT 为 **+0.0262 个百分点**，2/3 为正。默认头 clean 三种子均值 84% 目标**未达到，整体目标继续 active**。

|配对（三 seed 合计）|纠正错误|新增错误|
|---|---:|---:|
|ROUTED_GAP−BASE_SSL|1680|1953|
|ROUTED_GAP−JOINT_GAP|1908|1776|
|JOINT_GAP−BASE_SSL|1714|2119|
|BASE_SSL−BASE_CE|2330|2506|

完整源日志分析覆盖 240 个 epoch、53280 个训练步及全部 callback、48 个真实更新探针；全文 stdout 扫描无致命异常/OOM/Killed，数值记录有限。两 active 臂每 seed 的 180 次辅助均执行，后 7 轮不再增加辅助曝光；每个 SSL 模型完整处理 396900 个 U 包，纯 CE 为 0。原默认头全程联合训练，辅助 W/三个 gain 无梯度；所有实际梯度 oracle 误差为 0，近似 autograd 推演的差异另列，不能当作精确辅助增量。源统计和 Sinc 改变说明机制执行及部署特征学习，不直接替代未知 RX 泛化证据。

本轮实际训练 53280 次更新，辅助额外 L 网络前向 194400 包、144 次临时 AdamW。R01 的三行在诊断抛错前各执行至第 16 次原更新，另计 48 次失败技术开销；原部分状态保留且不继承。初始化/no-query smoke/合成回归属于另行披露的技术验证，不充作主训练或性能结果。

本结果仅对应当前 ManySig 物理划分、Phase1、默认头、clean 视图，K 不适用；不作卫星部署或新的独立盲测声明。历史 M15 的实际 LS=.01 与本次实际 LS=0 分开，只有本轮匹配对照用于配对。本轮数据/日程/参数不因测试结果调整，不追加选择性重跑；CENTER25K 等历史正收益路线保留为独立证据。相近研究与后续归因问题见[创新边界](related_work_boundary.md)。

证据：[12 行 seed 指标](seed_metrics.csv)、[逐 seed 错误变化](paired_error_changes.csv)、[完整指标与混淆矩阵](result_evidence.json)、[全日志源分析](source_training_evidence.json)、[独立终态](completion_evidence.json)。模型、IQ、原始预测及日志保留在原不可变远端输出，不纳入 Git。


## 当前判断与分组边界

当前具体配方不支持晋级：相对匹配 SSL，ROUTED 三 seed 合计纠正 1680 个错误、引入 1953 个错误，净增加 273 个错误；总体均值下降 .0542 个百分点、只有 1/3 为正。相同损失 JOINT 也未形成总体正收益。两者在源训练中均有充足非零信号，但这项同包风险修复未在本轮形成总体泛化增益。不能由此推断所有责任分配方法均无效，也不据测试分数扫描本损失系数、启用轮次或挑种子重跑。

有一项次要正结果须保留：ROUTED 的逐 seed 最差 RX 均优于 SSL，均值差 **+0.2458 个百分点（3/3 正）**；逐 seed 最差 day 的均值差为 **-0.1802 个百分点（0/3 正）**。二者方向不同，不用局部分组增益替代总体目标或重排候选。明细见[最差组证据](worst_group_evidence.json)。

指标记录中的 `stage=repair` 是沿用的展示字段；本轮实际行为由新 scratch 初始化、20 epoch、4440 次更新和无祖先 checkpoint 的证据确定，不表示继承旧模型续训。原机器产物字段保持。

后续研究先依据合法源证据解决归因和监督内容：现有 INITIAL 资格来自混合 RX 的源包，而非真正源 RX 留一；同标量路由也没有隔离梯度大小。应先核对既有跨 RX/集合监督/观测预测负结果，再在原默认头和合法 L/U 约束下形成下一项可证伪机制。既有 CENTER25K 等正收益基础保留，整体 84% 目标继续 active。本报告没有发布新的训练候选或修改当前已冻结模型。

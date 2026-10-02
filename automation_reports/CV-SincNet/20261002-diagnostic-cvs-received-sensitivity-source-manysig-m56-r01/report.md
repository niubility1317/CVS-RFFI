# CVS 完整源 V 接收变换与身份路径诊断

状态 PLANNED，尚未发布。这是零优化的冻结源诊断，没有新模型或新候选，不重跑既有 clean。总体识别性能与真正 RFF physics aware 目标仍未完成。

依据是新 query 之前已经完成的公共源物理诊断：整体相位约束成立，但固定 80 kHz CFO、两径、镜像及三阶接收失真仍改变表征。本轮检查这些性质在完整真实源 V 上是否成立，使用源标签，不使用 target 成绩或分层。固定 residual_fusion 与全 FP32 equivariant_memory 各 4 个 E200 权重，不更新、继承训练、重选或改写原产物。加载前逐个核对完整 source 契约与 scratch/resolved/completion/payload。

56 行为 2 网络×4 seed×7 变换，每行 27000 包、6 TX×5 RX×3 day，共 90 单元、每单元 300 包。相位 0.37 rad、相对 CFO ±80 kHz、零历史边界一抽头 `(0.15+0.08j)` 接收 FIR、镜像 `(0.02+0.02j)`、三阶系数 −0.03，及原始 identity。除 identity 外逐包重做 RMS。它们作用于已均衡/crop/RMS 的 received IQ，不能当作 TX 参数干预或完整 WiSig 均衡器重建。CFO 仍为相对频偏；LTI 第一包样点使用零历史，不循环。范围沿用原公共物理诊断，负 CFO 是固定对称对照，尚未看本轮结果。

记录整网 logits/嵌入与 5 个真实执行节点：时间、频率和行为投影，以及完整分类器的 base_norm、pa_norm。前 3 个节点位于 stats 注入之前，后 2 个覆盖完整现有分类路径；不能把局部分支性质归给整网。只使用返回 None 的观察 hook，不禁用/替换分支。各模型采用其已登记的原始数值策略；残差历史默认精度与新整网全 FP32 有差异，不能称为同精度结构因果对比。重复 identity、实际 backend flags、源 E200 准确率复现及模型 state 完全未变均读回。

指标为同包预测一致率、原始/干预源准确率、配对差值、logit 最大误差、单位嵌入和各节点距离，保留全部 TX/RX/day 单元。模型及数据角色固定，无优化器、训练增强、域骨干、额外损失、源重新选模或 target 访问。每 GPU 最多 2 个任务，单冻结推理 worker，不干预其他进程。异常保留原产物，不自动重发。

物理不变、接收干扰敏感和实际身份识别分别解释。此诊断既不恢复唯一 TX PA/IQ 参数，也不能证明任意 RX 不变；还未产生结果，指标当前 N/A。

## 启动核实

VERIFIED：新 release 固定提交 `636fa61d928442919b4bb67e3697bf0afd6f0d35`；worker PID 1322824，GPU 0，CWD/argv/run-root/实际配置及日志增长独立读回一致。源 RX/day/数据角色和零优化、无 target 已读回。既有训练/预测进程未修改。启动快照有 21 行完整源结果，其余由同一 worker 继续，尚不能把全矩阵写为完成。

## 全矩阵完成与独立复算

VERIFIED：worker 自然退出；56 行各 27000 包、90 单元，全部 5040 单元、按包加权均值和最大值已独立复算。8 个原始源 V 准确率逐个完全复现，零优化、模型 state 完全不变、数值策略与原源训练一致，backend flags 恢复。没有 target 访问或候选重排。

数值为四 seed 均值 ± 样本标准差；差值始终对同权重、同包原始输入配对。

| 网络 | Received 变换 | 源 V 准确率（%） | 配对变化（百分点） | 预测一致率（%） | 单位嵌入距离 |
|---|---|---:|---:|---:|---:|
| residual_fusion | identity | 98.0694 ± 0.0967 | +0.0000 | 100.0000 | 0 |
| residual_fusion | constant_phase | 62.3852 ± 4.4717 | -35.6843 | 63.0185 | 0.82867781 |
| residual_fusion | cfo_plus | 31.1546 ± 3.9941 | -66.9148 | 31.3519 | 1.1964667 |
| residual_fusion | cfo_minus | 34.6917 ± 2.9547 | -63.3778 | 34.8546 | 1.263786 |
| residual_fusion | received_lti | 97.4352 ± 0.0584 | -0.6343 | 98.6120 | 0.18377849 |
| residual_fusion | received_image | 98.0843 ± 0.1154 | +0.0148 | 99.7398 | 0.05473373 |
| residual_fusion | received_cubic | 98.0463 ± 0.1261 | -0.0231 | 99.3574 | 0.15831344 |
| equivariant_memory | identity | 98.0944 ± 0.1877 | +0.0000 | 100.0000 | 0 |
| equivariant_memory | constant_phase | 98.0944 ± 0.1877 | +0.0000 | 100.0000 | 9.6739535e-07 |
| equivariant_memory | cfo_plus | 30.7889 ± 1.0072 | -67.3056 | 31.1861 | 1.0846366 |
| equivariant_memory | cfo_minus | 24.5620 ± 0.9863 | -73.5324 | 24.9574 | 1.086018 |
| equivariant_memory | received_lti | 96.5444 ± 1.2158 | -1.5500 | 97.7065 | 0.2412177 |
| equivariant_memory | received_image | 98.0843 ± 0.1837 | -0.0102 | 99.6500 | 0.083610403 |
| equivariant_memory | received_cubic | 98.2000 ± 0.1756 | +0.1056 | 99.3028 | 0.17022067 |

## 逐路径响应

距离为节点输出归一化后对原始输入的同包平均欧氏距离；近零向量使用固定 eps=1e-4，不能把局部距离当作因果贡献或恢复的硬件参数。

| 网络 | 变换 | 时间投影 | 频率投影 | 行为投影 | 完整 base | 完整 physical |
|---|---|---:|---:|---:|---:|---:|
| residual_fusion | identity | 0 | 0 | 0 | 0 | 0 |
| residual_fusion | constant_phase | 0.567597 | 9.192113e-06 | 0.2954083 | 0.860899 | 0.2925872 |
| residual_fusion | cfo_plus | 0.9032509 | 0.2289806 | 0.5997537 | 1.214823 | 0.5914084 |
| residual_fusion | cfo_minus | 0.8775543 | 0.2352981 | 0.5784294 | 1.285654 | 0.5772077 |
| residual_fusion | received_lti | 0.1216849 | 0.06375781 | 0.1733675 | 0.1718347 | 0.2194843 |
| residual_fusion | received_image | 0.03267553 | 0.02746179 | 0.07114005 | 0.0465059 | 0.07639934 |
| residual_fusion | received_cubic | 0.05414372 | 0.06817605 | 0.1347943 | 0.1560112 | 0.1382242 |
| equivariant_memory | identity | 0 | 0 | 0 | 0 | 0 |
| equivariant_memory | constant_phase | 4.414958e-07 | 4.277979e-07 | 7.419008e-07 | 8.832614e-07 | 8.651993e-07 |
| equivariant_memory | cfo_plus | 0.7404208 | 0.1628956 | 0.7509939 | 1.046109 | 0.8927096 |
| equivariant_memory | cfo_minus | 0.7349173 | 0.1655714 | 0.801374 | 1.016103 | 0.9520587 |
| equivariant_memory | received_lti | 0.07579205 | 0.04994244 | 0.2741477 | 0.147057 | 0.3279787 |
| equivariant_memory | received_image | 0.03001404 | 0.02174683 | 0.08591304 | 0.06197145 | 0.09912078 |
| equivariant_memory | received_cubic | 0.02894574 | 0.05294955 | 0.1532386 | 0.1512581 | 0.1682302 |

整体相位约束在真实源 V 上也成立：equivariant_memory 的公共相位预测完全一致，全部真实源包最大 logit 误差 7.5340271e-05。但固定正负 CFO 同时改变全部身份路径与预测，说明常相位不变不能替代频偏处理。相对频偏可能包含身份信息，因此不能仅按该扰动把 CFO 全删除；下一步需在原源数据上验证逐包同步的可用性、相位斜率估计歧义以及保留身份信息的条件。

LTI/RX 镜像/RX 三阶也只是本轮已处理 IQ 上的规定变换，不能由这些数值分解真实 TX/RX 失真。原始 source 源选择、既有 clean 预测与评分全部固定，未重新拟合、修改或重跑。下一轮结构/参数/候选只能由源证据及明确物理假设确定，不能读目标分层决定。没有新候选，因此 default clean 为 N/A，旧完成的测试不重复。整体目标仍未完成。

[全 56 行](evidence/source_all56_rows.csv) · [全 5040 单元](evidence/source_all5040_cells.csv) · [独立复算](evidence/analysis_validation.json) · [远端最终读回](evidence/final_readback.json)。

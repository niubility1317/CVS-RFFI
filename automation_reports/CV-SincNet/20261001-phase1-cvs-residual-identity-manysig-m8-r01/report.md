# CVS 基础网络研发：残差物理融合

run_id：`20261001-phase1-cvs-residual-identity-manysig-m8-r01`。状态：LOCAL_VERIFIED，独立 P0/P1 审查 PASS，完整 11 项测试经独立实跑全部通过。当前没有新候选 E200 或 clean 测试结果。

## 已实现与研发依据

基准和初始 CVS 候选继续在原不可变 release 运行。新研发读取它们全部可用源日志，固定共同 E60 对两个 seed 配对分析；原 CVS 源 V=96.909%、最差 RX=91.731%，共享复数候选=92.806%、80.713%。低训练 CE 与较弱验证并存，但不能从中断定具体根因或目标性能。

原 CVS 总参数 382146，实际身份 CE 参数 317665。64481 个参数只用于其他输出；删除它们后，相同 RNG 的训练 logits 和有效梯度精确一致。这只是等价压缩控制，不声称提高准确率。

新候选保留 native 时域、频域、PA 的所有特征提取，把多级 head 改为 `LN(base)+tanh(gain)*LN(PA_local+PA_delta)`，gain 初始化为 0.25，仍用 scale30 无 margin 余弦分类。增益为零时精确退回通用分支，CE 同时更新两个分支。没有硬相位不变性，也没有额外任务或损失。

| 候选 | 参数及有效 CE 参数 | 卷积/矩阵 MAC |
|---|---:|---:|
| residual_fusion | 164225 | 9708836 |
| residual_fusion_moment | 164417 | 9708836 |

第二候选只增加三个均值/波动池化的 192 个零初始化参数。参数较总 native 减少约 57%，较其有效 CE 参数减少约 48.3%；MAC 只减少约 1.56%，FFT 仍为两次。CPU 合成训练分别约 210.24 ms 和 192.86 ms，不当作 GPU 速度结论。原 head dropout 随 head 移除，性能变化不能仅归因于残差相加。

## 训练、选择与边界

2 候选×4 固定 seed，共 8 行，全部 scratch。保持原物理划分：L6300 训练、U56700 未用、V27000 只评估；200×50 步、batch128、AdamW2e-4/wd1e-4/cosine1e-6、FP32、不裁剪梯度。无输入/信道增强、域骨干、PL/teacher/EMA 或额外 loss。父 run 是源日志研发参考，不是权重来源。六类 seed、逐行 config/output/log 见 [experiment.json](experiment.json)。

旧 owner 的 QUEUED 行全部启动或终态后，新 owner 才参与空槽调度，避免双 launcher 抢同一槽；无需等待旧训练全部结束。每 GPU 仍最多两个总训练任务并要求 12 GB 空闲显存。没有停止、重启或热修改旧任务。

目标暂不预测或评分。在首次当前 clean 测试前，联合原四个与新两个 CVS 候选的四 seed E200 源指标，冻结一个最终候选。规则为 0.5×V accuracy＋0.5×worst_RX accuracy，最高分 0.2 个百分点内优先较小 MAC，再较少参数。四个基线均保留。联合结果写独立 `research_selection.json`，旧 `source_selection.json` 保留。缺失记录 N/A，不使用 target 成绩挑替代模型。

## 验证与证据

11 个聚焦检查已覆盖真实 CE 更新、三分支/新增益梯度、无效参数等价性、单样本推理与 features、退化输入、scratch/target/augmentation 禁止输入、新进程导入、profile 副本隔离、队列等待和源选择。expanded run 中一项被已有 pytest 临时目录 ACL 阻断；改为模拟 artifact 读回并单独 PASS，其余 10 项 PASS。无需修改系统权限或重跑已通过模型检查。

详见 [设计分析](../../../docs/CVS_RESIDUAL_IDENTITY_RESEARCH_20261001.md)、[本地验证](evidence/local_validation.json)、[参数/MAC/源诊断](evidence/development_evidence.json)和[完整可用源日志](evidence/source_curves_readback.json)。CPU 只证明机制与测量路径可执行；正式 GPU 时间和显存尚未测量。

## 交接

下一步完成独立 P0/P1 审查、提交/push/远端 OID 读回，发布新 CVS-only release，执行 N607 Torch2.1 CPU CE 冷检查，再独立核实调度器与等待/训练状态。已有 release/run 时先核实，不重复发布。新源训练完成后核实完整日志及资源，再冻结最终 CVS；当前识别提升尚未证实。

独立审查证据：[independent_review.json](evidence/independent_review.json)。当前只核实本地代码与源研究权限，远端发布证据待补。

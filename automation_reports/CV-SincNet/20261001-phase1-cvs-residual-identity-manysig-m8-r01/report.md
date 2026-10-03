# CVS 基础网络研发：残差物理融合

run_id：`20261001-phase1-cvs-residual-identity-manysig-m8-r01`。状态：SOURCE_TRAINING（独立读回 VERIFIED），独立 P0/P1 审查 PASS，完整 11 项测试经独立实跑全部通过。当前没有新候选 E200 或 clean 测试结果。

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

## N607 发布状态

代码 `ff469598cdccb6ee40ec21bb1e304a050b3a5a00` 已提交并 push，发布前独立核实远端 OID 相同。新 release `cvs_residual_identity_20261001_r01` 的传输 SHA、compile、fresh entry 和两个候选的有限 CPU CE 反向传播均在 N607 Torch2.1 环境实际通过。随后提交新 dispatcher，独立 `/proc` 读回 PID `725183`、CWD 与 argv 一致。

当前状态：`SOURCE_TRAINING`；行状态 {'RUNNING': 8}。它只等待原 launch queue 排空及实际空槽，不停止、重启或改变原任务。运行证据见 [独立读回](evidence/source_launch_readback.json)。尚无这两个新 CVS 的正式 E200 或 clean 结果。

实际启动行的 resolved config 已核实：纯身份 CE、augmentation=false、domain_backbone=false、extra_losses=[]、target_access=false，源角色 6300/56700/27000、每轮 50 步；逐轮日志包含真实梯度和三个物理分支的有效 CE 参数。

下一步只读核实新行启动后的有效参数、checkpoint smoke、源角色、逐轮梯度及资源测量。全部 source 完成后联合六候选冻结；若旧 selection 尚未存在，保留本批 source_selection 并补做联合冻结，不重复启动 source。

## E200 源研发完成与联合冻结

8 行均独立核实完成 E200/10000 更新，scratch 无继承，augmentation/domain/extra losses/target access 均关闭，逐行完整 200 轮记录已保留。实际 completion 与 source-only 选择均已读回。按原规则从8个 E200结果重算双候选选择，再与原四候选联合重算，均与自动 research_selection 一致：选中 residual_fusion。源训练结束发生在四基准 clean launch 之前，目标成绩未进入选择。

|候选|四seed源V均值|四seed最差源RX均值|总参数|Conv/Linear MAC/包|
|---|---:|---:|---:|---:|
|orthogonal_pa|97.898148%|93.986111%|382146|9862436|
|moment_pool|97.975926%|94.268519%|382338|9862436|
|orthogonal_moment|97.891667%|94.032407%|382338|9862436|
|shared_complex|94.883333%|86.990741%|168682|3119232|
|residual_fusion|98.069444%|94.495370%|164225|9708836|
|residual_fusion_moment|98.046296%|94.439815%|164417|9708836|

新选中候选源V=98.069444%，最差源RX=94.495370%；原native源V=97.930556%、最差源RX=94.134259%。相对增益分别为0.138889及0.361111个百分点，只能作源域结果。源筛选的0.2个百分点平分区间偏好较小MAC、再较少参数，不能改为按目标测试挑选。

选中模型总参数164225，全部有CE梯度；比原总参数382146少约57.03%，比原有效CE参数317665少约48.30%。Conv/Linear MAC从9862436降到9708836，减少约1.56%，FFT仍为两次；参数减少不代表同比例计算或时间减少。此时 clean 尚未测，目标“结构改进同时提升性能且轻量化”仍待一次冻结确认。

[完整源研发读回](evidence/source_research_complete.json)、[联合冻结](evidence/research_selection.json)、[源选择重算及边界验证](evidence/source_completion_validation.json)。最新用户要求四基准先测试已另行完成；新候选的独立确认将只新增4预测并只读复用原16基准预测。

## 冻结候选最终确认已完成

source-only run自身从未读取目标；其已冻结residual_fusion由独立子run `20261001-phase1-cvs-selected-clean-manysig-m20-r01`完成四seed clean确认，并只读复用16基准预测。最终accuracy78.4543%±0.8436%，较原CVS配对+2.2271个百分点，全部四seed正提升；总参数减少57.03%，有效CE参数少48.30%。完整结果与资源/限制见[确认报告](../20261001-phase1-cvs-selected-clean-manysig-m20-r01/report.md)。未选中的候选不做目标测试，目标成绩不回流source选择。该source矩阵及关联确认现已完成分析。


## 2026-10-03完整测试补齐

本批全部8个固定E200模型已纳入[全量clean测试报告](../20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01/report.md)，每模型168,000个测试样本；此前未晋级且未测试的候选也已补测，已有预测复用。304行预测固定后统一独立truth-last评分，完整分类决定复算通过。测试结果见汇总报告和逐seed/RX/TX表；历史源域选择记录保持原样。

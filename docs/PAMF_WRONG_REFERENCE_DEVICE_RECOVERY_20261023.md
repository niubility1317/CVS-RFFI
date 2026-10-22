# WRONG_REFERENCE 设备构建技术恢复

本次只恢复严格监督物理结构面板中 `WRONG_REFERENCE` 的执行错误，不提出新结构、种子或训练预算，不使用目标分数决定修复或重跑。

## 已复现的根因和修复

原 `20261022-phase1-PAMF-supervised-physics-m24-r01/WRONG_REFERENCE-s2026092701` 在首段模型构建时失败：原生构建器已把参考响应 bank 放到 CUDA，固定相位仍在 CPU，二者相乘触发设备错误。原 CPU 架构测试没有覆盖“先在 CUDA 构造原生模型、再安装结构变体”的真实训练次序。

本机按真实调用次序复现了相同错误。修复仅将原 CPU Generator、固定种子 `2026102208` 生成的同一 FP64 相位向量传到 bank 所在设备。相位分布、随机值、参考谱幅度约束、维度、可训练参数、数据集合、标签、增强、损失、种子和选模规则保持原登记。

修正后的实际 CUDA 模型构建、默认联合分类头反向以及原结构测试通过。恢复发布仍先运行真实固定 checkpoint 的合法 L_s 无 query smoke，再执行错误参考结构的合法 L_s 前向与反向；该 checkpoint 不用于初始化恢复训练。

## 有界恢复矩阵

- 控制器：`20261023-phase1-PAMF-physics-device-control-r01`。
- 技术替代：`20261023-phase1-PAMF-physics-device-recovery-r01`。发布前已确认的失败行是 seed `2026092701`，替代其从零 F、raw/G-L 七景预测和独立评分，共 3 个阶段；保持原 50 轮、11100 更新、纯 CE、默认星地拼接 0.68、LS0、无 U/EMA/CRIC/RDrop、原默认头。
- 专用时延替代：`20261023-phase1-PAMF-supervised-physics-timing-recovery-r01`。原专用时延尚未启动或测量，原 9 结构、F/GF、5 批量、5 轮、450 case-round 全部保持，只把失败行的 F 来源映射到技术替代权重。
- 其余尚未启动的同结构 F 和预测消费者只切换执行 release，原配置和输出目录保持。已经运行或完成的任务不重启，不改科学配置。不新增 G 训练、源数量扫描或已完成统一 500 case-round 的复测。

原失败 F、阻塞的下游及未测原专用时延保留历史，按明确映射从有效闭合中排除。所有新输出使用不可覆盖根目录。逻辑结构矩阵仍为 8 变体 × 3 seed；技术替代不是额外独立模型或新方法结果。完整评测和报告按替代映射取值，不能把旧失败或缺项填成零分。

## 执行、资源和交付

发布命令见三个 run 的 `experiment.json`；代码以实际 Git commit 为身份。唯一新 owner 源 smoke PASS 后正常 handoff 原 `20261023-phase1-PAMF-PL-Gfactor-control-r02`，保留健康 PID、其他 owner 和所有逐任务科学 release。基础每卡 3 条、全机最多 26 compute、GPU0/6 的受限第四 F 槽、显存/CPU/RAM/IO guard 和 G-heavy 每卡最多 1 条保持。

仅正式专用时延需要独占卡且排除 GPU2/3；不停止训练获取计时卡。两次相同未解决技术指纹停止新提交。旧已替代统一时延错误不重复恢复。

每行全部预登记任务、raw/G-L 七景决策冻结后，独立 CPU scorer 才连接 truth。保持每景/RX/TX/day、Accuracy/Macro-F1、净纠错、三 seed 和成本口径；模拟错误参考控制不证明完整硬件解耦。完成后归入 IEEE 实验章及新的不可覆盖导出包，原 r06 及历史包保留。

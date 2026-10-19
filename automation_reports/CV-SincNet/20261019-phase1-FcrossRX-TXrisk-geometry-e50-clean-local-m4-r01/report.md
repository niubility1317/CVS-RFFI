# 原 F＋跨 RX 同 TX 对比学习：弱发射机特征优化

当前状态：RUNNING，BASE／WEIGHT／GEOMETRY 三行已完成并冻结 50 轮，BOTH 正在训练。历史 85.1139% 为原三种子方法结果；本次实际增益以同 run 的本机 BASE 配对比较。

固定原配方从零训练 50 轮（33L＋17U），保留三路 SIG_TIME、统一 Sinc 物理更新、旧门控、ownEMA、cos、RDrop alpha=1、LS0.01、普通增强和默认分类头联合训练。训练关闭星地信道增强，冻结后只测 clean。两次 RDrop 前向复用同一实际增强 IQ。无历史权重继承。

| 首种子 2026092701 | 特征处理 |
| --- | --- |
| BASE | 原浅层 time_down／中层 t2.pool 跨 RX 同 TX 对比学习 |
| WEIGHT | 基于此前源 V 的固定类风险权重，仅作用于特征对比项 |
| GEOMETRY | 原机制＋实际部署 joint160 的跨 RX 同 TX 聚合、不同 TX 分离 |
| BOTH | 风险权重＋joint160 几何约束 |

新机制不增加参数、投影或分类头。原特征对比权重最大 0.1；joint160 最大 0.05；沿用 warm5／ramp10／温度 0.2。权重和弱类均由原三种子源 V 固定，未依据新目标结果调节。

先完成四行源训练。相对 BASE，弱两类平均召回至少增加 0.2 个百分点、整体准确率至少 −0.2 个百分点、最差 RX 至少 −0.5 个百分点、任一类至少 −1 个百分点的变体，原样扩展种子 2026092702／03，并配套扩展 BASE；若无变体通过则不扩展。此决定在任何新目标预测前固定。全部首种子与获准扩展行自动 clean 预测、独立 truth-last 评分，负结果全部保留。

数据复用既有 READY／VALIDATED_ONCE：L=6300、U=56700（TX 真值隐藏）、V=27000；clean 目标 168000。缓存补齐原时序邻近门控必需的 eq_i／sig_i／base_index，未导出 U TX。源 V 只作固定晋级判断；目标 clean 才是性能验证。每模型 11100 次更新；原无标签阶段曝光 963900。

本机 GPU0 最多 3 个训练／预测 worker，计入其他 owner；独立 CPU 评分最多 1 个。新派发前至少 4.5 GiB 空闲系统内存。异常停止新派发并保留健康任务及输出，不按性能自动重跑。

实现与运行配置身份为 Git commit，实际执行 commit 写入不可变 release_commit.txt。命令、CWD、占位路径、矩阵与预期产物见 [experiment.json](experiment.json)。正式输出在本机新不可覆盖 run root；训练、预测、评分和实际成本逐行保留。

验证已通过：[实际网络与协议负例检查](checks.json)、[原生 L/U/EMA/Sinc 合成后端检查](backend_checks.json)、[独立 P0/P1 检查](review.json)。合成检查没有训练数据／目标读取，其权重不被正式训练继承。尚无本轮性能结果，不能据此宣称提升。

执行提交：`31467086c3ced30237f3ac86e91eae68645425d7`，已推送并独立核对远端一致。不可变本机 release 编译通过。启动证据见 [launch_verified.json](launch_verified.json)；首次观察三行各约 965.5 MiB 专用显存，初始 checkpoint 已生成，正式性能尚待训练与 clean truth-last。

首三组阶段更新：各 50 轮、11100 次实际更新；U 曝光均为 963900，接受数分别为 BASE 15127、WEIGHT 15126、GEOMETRY 14900，约 1.55% 至 1.57%。实际 Sinc 物理裁剪各 11100 次，默认头联合训练、单独拟合头为 0，未使用星地增强，未访问目标数据。完整 50 轮／11100 步分析及 BOTH 启动证据见 [source_first3_complete.json](source_first3_complete.json)。此为源训练执行证据，尚无本轮 clean 性能结论；扩种子决定仍等待 BOTH 完成。不同并行密度的 worker 耗时不能解释为算法加速。

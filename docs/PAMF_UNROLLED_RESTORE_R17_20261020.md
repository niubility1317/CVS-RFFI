# R17：三步物理展开与学习先验

完整持续目标保持：单一恢复 IQ 供原 PAMF 全部分支，合法 L/U 全量训练、0 TX/F 训练信息，冻结原 F50/defaulthead 验证输入对应 TX。正式 low urban≥70%，其余五个受损场景分别距同 F/seed 原始 raw clean≤5pp；恢复 clean 损伤单列。N607 多 GPU，保留其他 owner。

来源为用户六类恢复报告 R3（3～5 步、共享/不共享、观测估计作用、软一致性）。显式数值一致性与学习先验的组合参考 [MoDL 原论文](https://arxiv.org/abs/1712.02862)，本轮是接收链适配，不是 MRI 论文或报告梯度步公式的逐式复现。R06/R14 已完成源域已知作用诊断：更准作用改善波形，但固定 prior 下的身份收益小，不能视作部署上限。R16 平均改善同时伤害 TX14-7，说明仍须检验表示/求解过程，不能只调平均损失。尚未对新展开模型做已知作用反事实；该项如后续需要单独披露，不能称本轮 oracle 上限。

新增 `unroll_shared` / `unroll_unshared`，均从 scratch 学习 Core(incoming=9)。三步，各做2次 warm-start CG，总6次；R07 coarse/action、native variance、SCALE²冻结。设初值x0=coarse；每步输入当前received2、coarse2、当前估计2、单位RMS的Aᵀ(r−Ax)2、截断log(native variance)/8一通道；只用当前观测估计的A，不用模拟器真实参数或场景/TX路由。先验为coarse+SCALE×Core，随后求软加权MAP，以上一步估计作CG初值。方差与先验强度固定，不使用硬Ax=r约束。跨步反向传播不detach，不额外辅助损失。共享一Core；不共享三Core，初始全克隆同一Core，故两种完整初始输出相同。新初始解与R16单次6CG不同，因为2步重启的数值路径不同；不宣称与旧模型相同初始化。

只继承R07完整冻结上下文，R16/R15训练后prior不初始化新模型。model seed2701在加载父后重置；data seed2026102001、R07原20轮缓存（15%clean/85%residual）、6300L/56700U、20轮/4940更新、batch256、AdamW2e-4→2e-5、warm247、WD1e-4、clip1、FP32与R16一致。每行L126000/U1134000曝光；新prior E20，继承coarse20+action20，组件E60；每样本三次prior前向，计算量不等于单次prior，另报参数/训练时长/显存/延迟。IQ NMSE唯一训练目标，0TX/F/teacher/校准，单份IQ派生全部原分支。

固定最后20轮；每候选source1800V七预测先固定，再独立CPU评分。每候选须同时满足相对R16 U-Net IQ单次参考、R15 U-Net FFT参考和原R10 native的原双护栏：mean6NMSE≤1.05倍、cleanNMSE≤.02；low/midurban各≥+.5pp；clean相对raw下降≤.1pp；最差低urban TX≥−1pp；其他五场景最差变化≥−.5pp。实现用两次现有资格函数（R16/native与R15/native）取交集。源V标签用于选择，0V状态更新，选择过程不称完全无标签。

两种都可晋级，不事后改主候选。仅在全部合格候选中，按source低/中urban平均准确率最高、低urban最差TX准确率最高、低urban最差RX准确率最高、最后名称字典序固定选一。选定后冻结。无合格候选则0target；若选定，自动对“选定展开+固定R16 U-Net IQ单次参考”两行做原168000物理记录×七视图确认，每行七预测固定后独立truth-last。旧单次参考仅target-only加载，不续训；其来源source-only/0target和完整祖先已核实。正式rawclean84.6761904762%，其他五场景下限79.6761904762%。既有基准暴露，不称新盲测，不使用目标反馈调参。

N607池0/1/5/7择空卡，最多两条本run训练/预测worker，每卡一条、全owner cap2/free≥12GB；capacity4共享锁到CUDA PID可见，子进程不重复锁。唯一owner、不可覆盖root/release、实际2Ls新/旧checkpoint+原F+所有步梯度smoke是入口第一步，PASS继续。技术失败保留输出/停失败行/pending暂停/健康任务继续，无自动重试。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A17-01|报告R3|三步共享/不共享、软物理约束、单IQ|models.py/design.py|verified|三次Core/各2CG、初始输出一致/各步梯度/逐包独立；参数{'unroll_shared': 835842, 'unroll_unshared': 2507526}|CG适配，非报告逐式复现|
|A17-02|逆问题正确性|warm-start实线性MAP与真伴随|shared weighted_map|verified|稠密实系统一致、两步目标下降、有限梯度；旧默认兼容检查通过|不硬投影|
|A17-03|无监督/来源|全L/U、0TX/F/target训练、仅R07冻结父|runtime.py/training.py|verified|两行实际20轮/4940；每行L126000/U1134000曝光；TX标签/F权重/F输出/target读取及冻结父更新均0|R16/R15仅源参照；两行仅R07父，新prior scratch|
|A17-04|身份保护|三参考护栏、固定源排序、无合格0target|evaluation.py/controller.py|verified|第三参考最差TX/角色/性能与最差RX排序/未选target负测通过|V标签选择单独披露|
|A17-05|原F/正式对照|新展开与旧单次不同schema；七预测后truth|evaluation.py/runner.py|implemented|实际新/旧checkpoint和原F smoke通过；两新行source七预测先固定再CPU评分；正式胜出+固定单步两worker正在预测|完整正式七视图及truth-last结果仍待证；默认头不拟合|
|A17-06|运行保护|单owner/共享容量/不可覆盖/N607发布|controller.py|verified|43tests/compile/help/一次P0P1/预检/单归档比对/远端编译及PID/CWD/参数/GPU/日志增长均有实际证据；原训练worker自然退出，原controller自动启动正式两行|无额外gate；正式原worker继续监测|
|A17-07|完整报告|三郊区三urban/clean、TX/RX/day/损伤/成本|run汇总|pending|完整日志和预登记条件工作|源成绩不替代正式70/5|
|A17-08|有效作用边界|新展开已知作用反事实/多seed/独立物理族/逐细节真实|后续确认|deferred|已有R06/R14源诊断；新模型及外部稳定性须另证|保持完整目标|

当前为 **5 verified、1 implemented、1 pending、1 deferred、0 rejected、0 blocked**。只有不共享行通过全部原定源条件，已在目标访问前固定选择；source low/mid urban为86.8333%/91.4444%，逐TX low urban均优于R16单次及R15 FFT。完整[源域固定汇总](../automation_reports/CV-SincNet/20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17/source_interim.json)与[运行报告](../automation_reports/CV-SincNet/20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17/report.md)已更新。正式结果尚无，最高风险是这些源收益能否迁移至固定目标RX并满足70%/五场景5pp，不能提前宣布目标达成。

本阶段实际执行 `python -X utf8 .codex_tmp/observe_unrolled_r17_20261020.py`、`python -X utf8 .codex_tmp/collect_unrolled_r17_source_interim_20261020.py` 与 `python -X utf8 .codex_tmp/record_unrolled_r17_source_20261020.py`。它们核实同一原进程/源域固定产物并更新公开登记，不重新启动训练、改变方法或读取中途目标评分。已有43项聚焦检查与真实checkpoint检查保持原证据，本阶段不增加重复审查或额外发布门槛。实现仍为warm-CG的接收链适配，非原报告梯度步/MoDL MRI逐式复现。

# 四种对比方法接入 DURA：冻结读出方案

用户于 2026-10-23 要求四种对比方法接入 DURA，比较恢复前后的性能。本方案复用已经训练、冻结并通过星地拼接审计的 RIEI、DRIFT、IB-RFF 和 SDG；各方法的 Full 和自身 CE 全部保留。不得依据目标成绩挑选接入方法、种子、场景或恢复器。

## 矩阵与权限

覆盖 516 个既有模型，三种子 2026092701/02/03：原四方法主矩阵 156 个，加源数量矩阵 360 个。主矩阵包括 F0/F1/F2 接收机组合、chrono 跨日期、ManyRx、IB-RFF 的 label01/label05，以及 F0 普通 PL。源数量覆盖 m=1/2/3/4/6，自然预算监督、自然预算普通 PL、固定 L1080/V5400 监督。标签比例面板只有原已训练的 IB-RFF，不虚构其他方法。无新 F 训练、无新 G 训练、无五源扫点、无另训头。

每个 F 都报告 raw、DURA-L、DURA-LU，形成 1548 个模型×读出单元。raw 决策逐物理 ID 直接复用。DURA 使用现有 17 个完全匹配源数据契约的 L/LU 完整谱系，各角色固定 seed01；原 F0 LU 使用已固定 DURA v1.0。所有消费者共享该契约的同一 G 和相同 received IQ。G 的全部祖先维持 TX-free/F-free/target-free。

- 监督 F＋DURA-L 是严格 L-only 系统。
- 监督 F＋DURA-LU 使用了源域 U IQ，只能作为增加恢复器无标签数据的单独对照。
- 普通 PL F＋任一 G 均属于半监督系统，PL 训练机制与阈值 .95 原样保留。
- 每个 F 保留原方法网络、原默认联合头、原生训练预算与选模规则。只增加冻结公共预处理，不开展恢复后再训练、微调、分类头拟合或目标适应。

源数量曲线仍固定六个目标 RX/144000 包，F0 等五源主配置使用原七个目标 RX/168000 包；两条曲线不得拼接。ManyRx 使用其原十类/123021 包。chrono 同时完成 main 与原 source_unseen_date 诊断任务。每个任务均为 clean＋六种 practical，不生成新目标 IQ。

## 共享恢复执行与统计

34 个契约×G角色任务各自恢复全部合法测试视图，FP32/TF32off、reuse=True/graph=True/native=False，结果不额外归一化。每个缓存通过同源 L 的 eager/reuse 对照后才读取目标 IQ。缓存仅保存恢复 IQ 和物理 ID，不接触 TX 真值或任何 F；仅是同一冻结函数计算的复用。与在线 G→F 的差别是把相同 FP32 中间 IQ 写入后读取，分类前不改变数值、顺序或权重。缓存阶段计入每卡一个 G-heavy 限制。

每模型的 raw/L/LU 七视图全部固定后，独立 CPU scorer 才连接真值。统计 Accuracy、Macro-F1、每 RX/TX/day、RX×day CM、净纠错、clean 损伤、同 seed Full−CE、PL−Sup，以及 L/LU 数据权限差异。三 F seed 的误差条条件于一份固定 G，不冒称三份独立完整系统。所有负收益、零 PL 接受率和非单调结果保留。

源训练成本按原方法预算复用，新增成本只有 34 份共享恢复、516 份分类读出与独立评分；G 的完整祖先成本与缓存成本分别报告，不重复记为每个消费者的新增训练成本。推理吞吐来自共享队列运行，只作执行成本，不替代独占正式延迟。

已登记的 `20261022-phase1-DURA-external-Lonly-readouts-m24-r01` 继续承载 F0 监督 24 行，增加 LU 读出。另登记 readouts-m492、IQ-cache-m34 和 analysis-m516，合计 1067 个冻结阶段任务；无新增模型训练。

## 伴随技术恢复与发布

发布前发现原 core 两条读出在 source_physics 的 NumPy2→Torch2.1 索引处失败，错误为 `Could not infer dtype of numpy.int64`。源侧 IQ、权重和训练已完成，失败发生在目标预测前。修复仅将 NumPy 索引/整数显式转换为项目已验证的 tensor buffer bridge，选取的物理 ID、11 项干预、计算与模型权重不变。

两条失败读出和两条依赖评分转入新的不可覆盖 recovery-m2；旧失败、输出和错误证据保留，旧汇总由新的 core-factorial-recovery 替代并指向恢复后的 scores。只迁移尚未启动的同缺陷源诊断消费者的执行 release，科学配置和原输出根保持。健康训练/预测原 PID 接管，绝不为切换执行版本重启。新增恢复阶段 5 个，总追加 1072 个任务。

当前 code/config 以发布时 Git commit 为准。远端普通账号、现有合法句柄、当前 owner 和健康 PID 由发布前后核实。资源继续原负载策略：基础每卡 3，0/6 的第四 F 受原 guard 约束，全机 ≤26，G-heavy ≤1/card，不为腾卡停止健康任务。两次同指纹技术失败暂停新派发。已替代旧统一计时的 r02 不重测。

源码入口为 external_dura/external_dura_extension、publish_external_dura，worker 仍通过统一 pipeline，原其它 owner 与科学 release 保持。完成后取回冻结评分与成本，补齐同契约方法×G 表、源数量图、IEEE 实验章节与新不可覆盖导出包。

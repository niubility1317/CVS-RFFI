# 特征增强版多解耦三种子探索

纯 CE、无半监督、无星地训练增强；保持当前身份结构、cosine分类头与学习率日程。48行均从零训练E200/44400步，不加载旧权重。

## 实验矩阵

| 配置 | 关系流 | 弱Style | Fishr | 机制 |
|---|---|---|---|---|
| native | none | False | none / 0.0 | 无作用网络 |
| random_S | random | False | none / 0.0 | 无作用网络 |
| S | balanced | False | none / 0.0 | 无作用网络 |
| S_raw5 | balanced | False | raw / 0.05 | 无作用网络 |
| S_dir5 | balanced | False | direction / 0.05 | 无作用网络 |
| LTR_S | balanced | False | none / 0.0 | linear,temporal,receiver |
| LTR_S_A | balanced | True | none / 0.0 | linear,temporal,receiver |
| LTR_S_F3 | balanced | False | direction / 0.03 | linear,temporal,receiver |
| LTR_S_F5 | balanced | False | direction / 0.05 | linear,temporal,receiver |
| LTR_S_F10 | balanced | False | direction / 0.1 | linear,temporal,receiver |
| LTR_S_AF3 | balanced | True | direction / 0.03 | linear,temporal,receiver |
| LTR_S_AF5 | balanced | True | direction / 0.05 | linear,temporal,receiver |
| LTR_S_AF10 | balanced | True | direction / 0.1 | linear,temporal,receiver |
| joint_noR | balanced | True | direction / 源选强度 | linear,temporal,receiver |
| joint_shared | balanced | True | direction / 源选强度 | linear,temporal,receiver |
| joint_direct | balanced | True | direction / 源选强度 | receiver |

每配置seed为2026092701、2026092702、2026092703。先完成13种固定配置39行；以SAF3/5/10完整源V三seed选择强度，并选择同强度SF，未选4配置12行只保留源结果。再从零训练noR/shared/direct的9行。最终测试36行，每行clean及六practical。

固定控制逐行冻结即测试；需要源选的候选等待必要源选择，未选候选不读target。所有预测固定后独立truth-last评分，目标结果不回流任何健康训练、排序或参数。

B48为6TX×2RX×4不同物理PID，同一天；30block公平轮换。每8个原生step一次，5550关系调用、266400clean包次，不等于独立样本数。所有关系对照CE权重0.1。Style只复用nativeL原有CE。

Fishr使用当前cosine头6×160参数方向梯度；raw梯度作同流程对照。每桶RX/day/clean独立计数去偏EMA，E1–20积累，E20成熟源L定标，E21–40升权，之后固定。EMA只保存梯度统计，不是教师。

L/T预测用于筛选并在真实IQ训练；合法困难IQ不因代理预测失准被丢弃。R保留独立角色分袋、判别margin分布和尾部风险，noR对照仅关闭身份约束。共享作用核心hidden50与独立hidden48参数差约0.1612%，保持候选和真实视图预算。

最高尚未验证的科学问题是弱Style与方向Fishr是否在不损害身份间隔的情况下改善困难RX；不能用辅助loss下降替代识别收益。分支关闭仅依赖性探针，非严格因果结论。

报告中旧U/EMA/LEO配方按用户明确要求覆盖为pureCE和clean-only Fishr；DSU、困难重采样和第四网络按首轮条件延后。完整要求见[设计追溯](traceability.md)和[原报告](design_report.md)。

本地CPU/CUDA聚焦与原生训练集成验证通过，独立P0/P1审查通过。E20全部成熟关系批次中位数定标；保存每桶统计、计数、年龄及各分支加权梯度。日志重复annotate按epoch幂等，避免覆盖稀疏观测分母。详见evidence。

## 发布核验

**VERIFIED / RUNNING**。新训练已实际启动，其余行由同一控制器按资源容量排队；保持既有健康任务。

执行代码提交：`d4723b1aa227fce6368fd25680bb74b1190557d7`。控制器PID：`3235663`。远端预检PASS，独立读回确认进程、目录、命令、纯CE/noU/noEMA/noSat实际参数与日志增长。

| 已启动row | GPU | PID | 日志字节 | 两次读回增长 |
|---|---:|---:|---:|---:|
| native-s2026092701 | 2 | 3235684 | 14938 | 5369 |

发布证据：[归档参数](evidence/package.json)、[回执](evidence/submit.json)、[首次读回](evidence/readback_initial.json)、[再次读回](evidence/readback.json)。

设计追溯25项verified、3项按报告首轮条件延后（困难采样、DSU替代、第四网络）、0项rejected/blocked。严格落实设计，旧半监督/LEO配方按用户明确要求替换为pureCE和clean-only Fishr。

正式训练和测试尚未完成；最高剩余风险是源V及目标识别收益是否成立，尤其是弱Style是否削弱时间分支身份信息、Fishr是否与多解耦重复约束。完整源选择、分支探针和冻结后的独立测试将给出证据。

## 每卡 4 进程调度调整

用户明确要求每张卡 4 进程，覆盖本轮原每卡 2 进程上限。新控制器统计同卡其他任务及 pre-CUDA 预留，保留至少 12 GB 空闲条件，使用所有可用卡槽。仅接管调度器，既有训练 PID、科学 release、配置和源选/自动测试流程保持原样。

本地接管与容量检查 PASS；独立 P0/P1 调度审查无阻断项。发布状态暂为 PLANNED，实际生效以独立读回为准。

调度接管 **VERIFIED**。控制器 PID `3250081`，原训练 PID `3235684` 保持运行；本轮当前 17 个训练进程。每卡总占用为 `{'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4}`，上限 4 已实际生效。科学代码提交仍为 `d4723b1aa227fce6368fd25680bb74b1190557d7`，调度代码提交 `7b71ecc7b94518bd09d82245a5093abbf62172ec`。证据见 [接管与容量读回](evidence/capacity4/handoff_readback.json)、[行状态读回](evidence/capacity4/readback.json)。

## 诊断故障与恢复

已核实 15 行在 E200 后的时间分支诊断发生相同错误，最终权重保留；11 行继续健康训练。新 run `20261010-phase1-feature-disentangle-manysig-m48-r02` 补做诊断和测试，不重训已完成模型。

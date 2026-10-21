# 1 至 6 个源接收机：四种域泛化方法的正式对比

状态：PRE_REGISTERED，尚未宣称训练或测试完成。run `20261021-phase1-dg-sourcecount1to6-m432-r01`。逐行配方唯一维护于 `experiment.json`；可执行矩阵为 `experiments/phase1_dg_sourcecount6/matrix.json`。

## 研究问题与预定解释

研究源接收机数量由 1 增至 6 时，RIEI、DRIFT、IB-RFF、SDG 的实际泛化性能、相对自身 CE 的收益，以及普通 PL 的增益如何变化。保留所有负收益；不要求 Full 必须胜过 CE，不根据目标结果调整配方、组合、预算或重跑。

独立 ResNet18 CE 和 PAMF 不属于本矩阵。各方法自身骨干的 CE 是匹配消融，仍保留默认头和联合训练，以回答收益是否来自 DG 机制。PAMF-T1-E50 及恢复器的其他 owner 任务保持原计划。

## 数据与接收机

ManySig：6 个 TX、12 个 RX、4 个采集日期，equalized=1；中心 256 点、逐包 float32 unitRMS。源日期固定 D1/D2/D3（2021-03-08/15/23），目标包含 D0/D1/D2/D3，D0 是更早的未见日期，不能称为未来日期。

预定嵌套顺序为 `(1,3,4,6,8,11)`，取前 m 个，m=1/2/3/4/5/6。沿用历史源集合顺序并追加 RX11，未根据目标分数排名。所有 m 的目标固定为 RX `(0,2,5,7,9,10)`，共 6 RX×6 TX×4 日期×1000 包＝144000 个物理样本。未选中的候选源 RX 保持不用，不能随 m 改为目标 RX。物理接收机名称和 TX 映射由一次 builder 读回 `data_status.json`。

| m | 源 RX 索引 | 自然 L_s | 自然 U_s | 自然 V | 固定 L_s | 固定 V |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 1 | 1260 | 11340 | 5400 | 1080 | 5400 |
| 2 | 1,3 | 2520 | 22680 | 10800 | 1080 | 5400 |
| 3 | 1,3,4 | 3780 | 34020 | 16200 | 1080 | 5400 |
| 4 | 1,3,4,6 | 5040 | 45360 | 21600 | 1080 | 5400 |
| 5 | 1,3,4,6,8 | 6300 | 56700 | 27000 | 1080 | 5400 |
| 6 | 1,3,4,6,8,11 | 7560 | 68040 | 32400 | 1080 | 5400 |

自然预算使用既有逐 TX/RX/day/eq 的连续 sig_i 7%/63%/30% 角色，训练池标注率 10%。固定预算在原 L_s 前缀内平衡选取每 TX/RX/day `60/m` 个标签，在原 V 前缀内每格取 `300/m` 个验证标签；每个 m 都恰为 L=1080、V=5400，同等训练标签和验证标签成本。1080 可被全部 6×3×m 整除，避免 m=4 时 1260 标签无法逐格平衡的问题。未选 L 标签隐藏进入 U，未选 V 排除；不转作训练。固定预算 U=12600m−1080，全部不用于训练。

固定预算仅监督，保持各方法 epoch 上限，因此同一方法不同 m 的 minibatch 更新数和有标签曝光一致（IB 保留原生源 V 早停，实际预算另报）。自然预算的 epoch 相同但更新数随数据量变化，不能称为等计算量。V 只验证/选模，不反向或更新持久状态。

## 矩阵与复用

主矩阵：6 个 m×4 方法×2 版本（Full/自身 CE）×2 训练方式（监督/普通 PL）×3 模型 seed＝288 个条件实例。

固定 L/V 对照：6 个 m×4 方法×2 版本×3 模型 seed＝144 个实例。总计 432。仅一条预定源 RX 链；三模型 seed 不能冒充三组接收机选择，结论限定于该链和固定目标集合。

自然 m=5 的监督 24 行与 PL 24 行和旧 run `20261019-phase1-external-dg-concat-pl-m219-r03` 的 F0 实际数据及科学配方一致，共 48 行复用源域冻结模型及七视图预测。先逐行核实完整配方、源物理角色、默认头、源选模与 checkpoint 来源，再按新目标 144000 ID 截取全部七视图，全部固定后重新独立评分。不重训，不复用旧目标评分作为新结果。其他 m 与固定标签预算全部从零，共 384 次新训练。IB 第二阶段只继承同 run、同 seed、同契约第一阶段。

模型 seed 为 2026092701/02/03，数据角色 split_seed=392005，卫星增强 seed=2027，测试观测 seed=392005/receiver_seed=2027。无 support 阶段。模型 seed 不改变源 RX、日期或物理角色；SDG 原生随机增强跟随模型 seed。

## 训练预算与联合增强

| 方法 | 预算 | batch | 优化器 / LR | 冻结规则 |
|---|---|---:|---|---|
| RIEI | E100 | 64 | SGD，1e-4，momentum=0 | 固定末轮；Full 保留原交替更新 |
| DRIFT | E100 | 256 | Adam，1e-4 | 固定末轮 |
| IB-RFF | 两阶段，各最多 E200 | 128 | Adam，1e-3 | 每阶段源 V NLL；LR plateau patience5，早停 patience10 |
| SDG | E200 | 32 | Adam，1e-3 | 源 V 最小 NLL；Full 保留原双视图对比机制 |

RIEI、DRIFT 的 E100 是用户明确指定的实现预算，不能写成论文披露总轮数；其余沿用已核实原生配方。Full/自身 CE 使用同一最大预算、优化器、增强与源数据规则。跨方法不声称相同 FLOPs 或 GPU 时间。

所有行从首个 batch 使用默认 clean＋satellite 拼接，目标项 `CE_clean + 0.68 CE_sat`，无额外星地一致性。保持原 LEO_WEAK 场景；概率 0.30/0.60/0.80 按最大预算相对进度 20%/45% 分段，IB 各阶段独立缩放。Full 在此之上使用自身原生机制。

PL 行同时使用上述星地拼接和最普通置信度伪标签：当前联合模型 eval/no_grad 生成硬标签，固定阈值 0.95、权重 1、U/L batch 比 1；mask 后 CE 按整个 U batch 归一化。无 EMA 教师、复杂门控、近邻、强一致性或另训头，U TX 真值不导出。报告实际接受数/曝光数、逐 RX 接受率及被选 checkpoint 的遥测；接受率为 0 时不声称获得伪标签监督收益。U 与 L 拼接会影响训练态 BN，这一效应不等同于被接受的伪标签梯度。

单源时真实接收机分类和对抗项只能是单类，RIEI/DRIFT 相应项退化，不人工用 TX/日期虚构多个接收机域；其余原生项保留。SDG 的两个原生增强视图属于其自身定义，并不冒充两个物理接收机。全部模型默认身份头与骨干联合训练，冻结后直接预测。

## 测试与报告

每行源冻结后自动预测 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。所有行共享已有 received IQ；仅建立新位置 wrapper，不重新模拟。每行七视图预测完成后，独立 CPU scorer 才连接 truth。

主指标为目标 RX 等权 Accuracy；同时报告 all-6-class Macro-F1、最差 RX、每 RX/TX/day/RX×day。三 seed 报均值±样本 SD，Full−CE 和 PL−监督按同 seed/源集合配对，逐个 m/预算/场景保留正负差。七视图来自同一物理样本，不计作七份独立样本。不把已有暴露基准称为新的独立盲测或真实在轨验证。

新完整训练成本与历史复用成本分列；报告每阶段实际 epoch、minibatch/optimizer steps、L/U/卫星曝光、时间和显存。固定预算与自然预算单列曲线，不混成一个排行榜。

## 执行、环境与停止

唯一 owner：`codex-root-sourcecount6-20261021`。N607 普通账户、既有 SDG-SEI Py3.8/torch1.11＋隔离 NumPy1.23.5，不安装共享包。Git commit 是代码与配置标识，实际值保存在 `release_commit.txt` 和本地 `package.json`。一个不可覆盖 release/run/log root，路径及精确解释器见本地 `local_artifacts/phase1_dg_sourcecount6_20261021/runtime.json`；CWD 为该 release_root。

命令：`python -u -m experiments.phase1_dg_sourcecount6.driver dispatch --config runtime.json`。launcher 先一次准备数据并做真实源样本 checkpoint/no-query smoke，PASS 后立即继续训练和自动测试。用户最新要求其他显卡也启动：物理 GPU0 至 7 均可选择，只在实际无其他计算进程的卡上放一个本任务 worker；不停止或修改其他健康研发。队列内 30 s 轮询用于释放卡和接续，人工/Agent 长任务状态读取间隔 30 min，无变化不输出；技术失败、完成或重要状态改变才提醒。

两个相同的技术失败指纹停止尚未启动行，已有健康行结束；保留全部 partial。错误角色/输出碰撞/非有限/无法生成合法预测是技术失败，负收益不是停止理由。技术恢复另立不可覆盖 run，先核实所属进程和产物。

预期产物：data_status/data_handles、smoke/result、pipeline_state、逐行 source freeze 或 reused_source、七视图 prediction/complete、独立 scores、complete、summary.csv/json、逐行独立重算 audited_rows、final_report、completion。最终结果回写 IEEE 实验章节，不填造未完成结果。首次运行后按已完成同方法日志和实际可用 GPU 更新 ETA，估计不作为截止保证。

## 发布与运行核验（2026-10-21 15:04，UTC+8）

VERIFIED：代码 commit `9500192b6a91365016006ea78fbebfda2afb2bed` 已提交、push，并独立核对远端 OID；一个发布归档本地/远端 SHA 匹配，远端编译通过。聚焦协议负例测试 3 项通过；一次独立 P0/P1 审查发现派生矩阵过期，修正后与生成器完全一致，未增加重复全量审查。

源数据构建 VALIDATED_ONCE，18 份源缓存/角色句柄，目标始终 144000 个物理样本；已复用全部接收 IQ。49 个真实源样本训练目标/checkpoint/no-query smoke 全部 PASS，随后自动开始正式训练。owner PID 69072 的 PID/CWD/argv 与六条首批 worker 已读回；训练日志增长已独立确认。GPU0、1、4、5、6、7 已启动，GPU2、3 继续其他恢复器研发，释放后可接入。首批 IB-RFF CE/Full 已完成源训练、七视图预测与独立评分，其余仍运行；当前不是全矩阵完成。

ETA：按历史完整同方法三种子实测时长推算，6 张卡持续空闲约 8–12 小时（中心约 9 小时），即约 2026-10-21 23:00 至 2026-10-22 03:00（UTC+8）；若 8 卡持续可用约 6–9 小时，若降至 4 卡约 11–17 小时。包含七视图预测/评分和收尾，IB 早停、共享 CPU/存储与故障会影响时长；这是估计，不是保证。后续只按实际吞吐校准，不改变科学矩阵。

已创建本线程每小时监控（automation `1-6`），无变化保持安静，仅完成、技术失败、需要输入或 ETA 显著变化时通知。完成后汇总并更新既有 IEEE 实验部分。执行证据与 ETA 明细分别见本地 `postlaunch_verified.json`、`eta_estimate.json`。

# LoRa Phase1 四种对比方法训练修复与三种子复验

- run_id：`20261122-phase1-lora-full-external-m12-r02`
- group_id：`phase1-pamf-lora-transfer`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

排查源域欠拟合和目标接近随机水平的三种外部对照，以 SDG 为参照。修复空掩码 PL 对训练批统计的影响，稳定 DRIFT 分离损失，调整 RIEI 优化器并延后 IB-RFF 第一阶段早停判断。保持原 LoRa 物理样本、标签预算和测试视图。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 原跑核查与变更

读取旧补跑 12 行的 1,304 条源域 epoch 记录和 25 份 stdout（1,514 行）。训练和冻结预测均完成。三种接近随机水平的方法在源域时已出现异常。

| 方法 | 旧跑证据 | 本次变更与源域验证 |
| --- | --- | --- |
| RIEI | 最终源 V 9.79%–10.46%，TX CE 约 2.30。 | 保留两套优化器、交替更新和原损失，将 SGD 改为相同学习率的 Adam。种子 2026092701 的 20 轮诊断在分层源 V 上为 35.42%，只提高 SGD 学习率的对照仍接近随机。 |
| DRIFT-v2 LoRa | 最终源 V 9.63%–10.02%，三行原生辅助损失末轮为 −133.09、−123.25、−129.65。 | 只在平方差分离项中对两个特征半部做 L2 归一化，保留 0.02 权重及其他损失。未乘权重的负平方差下界为 −4。25 轮诊断源 V 为 34.17%。 |
| IB-RFF | Stage 1 三行均在第 12 轮早停，最低源 NLL 均在第 2 轮。零 HSIC 的第一阶段已近随机。 | Stage 1 前 25 轮照常留存最低源 NLL 检查点，延后降 LR 和早停计数。保留原 FAN、原 dropout 和两阶段损失。原配置绕过早停的诊断在第 20 轮为 27.42%。 |
| SDG | 三行源 V 峰值 50.58%–52.00%，原生监督和对抗对比项与作者入口一致。 | 保留原生双视图、优化器和权重，在同批输入上复验共享 PL 修复。 |

短跑诊断只使用原源 L、源 practical 缓存和按 TX×RX 固定抽取的 1,200 个源 V 包。用于区分具体训练机制，完整复验继续使用原 12,000 个源 V 包。四种方法在置信度筛选时以评估态读取 U，只将通过 0.95 的 U 拼入训练态前向，PL 损失仍除以原 U 批大小。复验额外记录各原生损失分量、DRIFT 特征范数及伪标签接受率。

本地真实源 IQ 的训练更新、checkpoint 回读和有限 logits 验证四种方法全部通过。训练配置和原数据角色隔离。原补跑及主实验按原身份保留。DRIFT 有界做法和 RIEI 优化器属于 LoRa 移植调整。

## 复验矩阵与读出

原三种子共 12 行，每行冻结目标预测 15 台接收机、30,000 包、7 个场景和 2 种读出。独立 scorer 收齐矩阵后连接原归档真值，报告准确率、Macro-F1、S6、逐接收机与净纠错。原批目标已在前次评分中暴露，本轮为同条件复验。

## 发布与启动读回

发布提交 `360df1809b15794fea27774c9f4e10c0d5f5ec60`，本地 HEAD 与远端分支 OID 一致。N607 归档传输 VERIFIED、编译 PASS。控制器 PID `176226` 的工作目录、命令行和日志已核实，四种真实源 IQ 检查均 PASS。

首次读回时 12 行排队，GPU 0 有 23,436 MiB 余量并已有两个 CUDA 进程，其余允许使用的 GPU 也已达到并发上限。独立评分已在本机启动，会等待训练和 12 行预测全部完成再读取原测试归档。结果写入 `local_artifacts/20261122-phase1-lora-full-external-m12-r02/readout/`。

[发布凭据](../../../local_artifacts/20261122-phase1-lora-full-external-m12-r02/release/submit.json) · [启动读回](../../../local_artifacts/20261122-phase1-lora-full-external-m12-r02/launch_readback.txt)

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

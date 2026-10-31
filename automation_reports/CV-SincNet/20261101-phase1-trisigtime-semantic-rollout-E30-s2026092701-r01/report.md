# Tri-SIG-TIME 分类分布引导的生成轨迹恢复

- run_id：`20261101-phase1-trisigtime-semantic-rollout-E30-s2026092701-r01`
- group_id：`phase1-trisigtime-semantic-flow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与固定原候选相同的 Phase1 源训练和双轨生成架构，新增冻结 Tri-SIG-TIME 对配对 clean 与两条恢复轨迹的分类分布 KL 指导。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 已核实状态（2026-11-01）

候选在现有双轨生成模型的源 IQ／基座特征损失上，增加固定权重 0.1 的 `KL(p_F(clean) || p_F(restored))`，两条轨迹取平均。`F` 是冻结的 Phase1 Tri-SIG-TIME SSL；目标分布由源 clean 推断产生，训练不读取 TX 真值。

实际 release 代码版本 `87a574801ca8250844df0a8464e0515e933271e6`，归档两端 SHA-256 一致。真实基座和两条源样本测试通过：语义 KL 单独的生成网络梯度范数 39.1355，IQ 梯度范数 4.3879，基座冻结、检查点重载成功，未读 TX 标签或测试 IQ。

控制器 PID 2186572 已独立核实，正等待前序 Phase1 模型的控制器和 GPU worker 均结束，尚未开始本候选训练或测试。该等待只依赖进程和终止状态，不依据前序测试分数调整候选。随后自动执行 E30、七视图预测和独立 truth-last 评分；当前尚无本候选测试性能。

2026-11-01 更新：前序 Phase1 controller 及 GPU worker 已退出。本候选按原预登记固定配置启动，worker PID 2217601，GPU 1；PID、工作目录及 GPU 归属已独立读回。当前属于源训练阶段，尚无该候选测试结果。

## Phase1 七视图完整结果与独立核验

状态 VERIFIED／ANALYZED，固定 E30 模型，七视图各 168000 个 Phase1 测试物理记录；全部 14 份预测完成后才连接真值。冻结基座为原始 Tri-SIG-TIME SSL，原始 clean 准确率 86.5387%。差距为原始 clean 减恢复准确率，单位为百分点。

| 视图 | 原始准确率 | 恢复准确率 | 与原始 clean 差距 | 恢复减 DURA |
|---|---:|---:|---:|---:|
| clean | 86.5387% | 86.6607% | -0.1220 | +0.0452 |
| practical_high | 80.6155% | 76.2173% | 10.3214 | -8.1262 |
| practical_mid | 78.1077% | 74.9256% | 11.6131 | -8.0905 |
| practical_low_suburban | 75.7446% | 73.6679% | 12.8708 | -7.6542 |
| practical_high_urban | 77.0500% | 73.5036% | 13.0351 | -9.5458 |
| practical_mid_urban | 62.7958% | 62.9321% | 23.6065 | -12.7738 |
| practical_low_urban | 51.6048% | 54.2667% | 32.2720 | -14.5345 |

六种信道均未达到差距不超过 3 个百分点或相对损失不超过 3% 的目标，且均低于同一基座下的 DURA 对照。clean 满足预登记的损伤保护线。独立 CPU 进程按物理 ID 复算总体、TX、RX、day 和 Macro-F1，均与原评分一致；全部七个视图的 168000 条原始预测与基座记录逐项一致。DURA 配对比较也核对了同序 ID、原始预测与修复／损伤计数。

最终模型文件 SHA-256 为 `321b1e0266ee8086e478895acd3905b5da4106f93ccd9e3a73c10960007d3689`；独立加载的两条源样本推理具有确定性。单个模型 seed；此前开发测试结果已被观察，该测试集不是独立盲测。困难样本加权路线在上述结果出现前已登记固定，现已启动；锚定路线并行训练。

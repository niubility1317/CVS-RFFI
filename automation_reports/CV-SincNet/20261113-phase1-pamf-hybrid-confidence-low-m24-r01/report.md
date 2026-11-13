# Hybrid 全置信样本低辅助权重对照

- run_id：`20261113-phase1-pamf-hybrid-confidence-low-m24-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（以远端状态和 events.jsonl 为准）

## 目的与对照

接收机划分 2、10 类、1% 标签和 6 源接收机，各对 0.05 与 0.10 辅助权重使用三个配对种子，共 24 行。所有伪标签置信度至少 0.95 的 U 样本均进入 Hybrid 辅助损失；无分支投票和原型。仅改变额外融合／分支损失权重，物理视图路由及其 taper 沿用此前全置信实验。

## 数据、seed与模型来源

保持原配置完整 L/U/V，无源域留出；从同配置同种子原 Tri E20 权重独立追加 E10＋E20，固定末轮 E50。复用普通 PL、原 Hybrid、全置信权重 0.25、0.5 和 1.0 的既有同种子结果。训练不读取 U 真值或目标标签。

## 执行与存储

N607 独立 run 和输出目录；GPU 7 每卡总进程上限 4，其余卡上限 6，并检查剩余显存。真实权重无目标查询 smoke 后自动训练，每行训练完成即自动预测 clean 和六种 practical 场景，七景预测完成后独立 truth-last 评分。系统性技术失败停止本队列新派发，保留既有结果及其他 owner。具体命令、环境、路径见 experiment.json。

执行 commit `b3f6a04ac1792f0f39dd1dc1a3a1829037c0a893`；发布包传输校验和远端编译通过。远端 owner PID 2270706 已启动真实 checkpoint smoke，等待通过后自动派发训练。

## 结果与覆盖

尚无结果。前次测试显示 0.25 在部分配置优于 0.5 和 1.0，因此探索更小权重；目标测试集此前已经曝光，本轮比较为探索性，按固定矩阵完整报告，不依据目标分数选择 checkpoint。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

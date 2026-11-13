# Hybrid 融合与五分支匹配对照

- run_id：`20261113-phase1-pamf-hybrid-fullbranch-match-m4-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（以远端状态和 events.jsonl 为准）

## 目的与对照

E 条件与已发布 C 融合层条件严格配对：Main 的 2026092701 和 10 类的 2026092701–03。保持同一 Tri E20、完整 L/U/V、基础主路由、独立物理视图、0.25 taper、0.1 固定辅助权重和训练预算，仅增加五分支平均辅助项。既有目标测试已曝光并用于选择种子，结果为探索性。

## 数据、seed与模型来源

同配置同 seed 原 Tri E20 后续训 10＋20 轮，固定 E50。训练不读取 U 真值或目标标签。匹配的 C 是融合层单独辅助；普通 PL 与原 Hybrid 原有结果复用。

## 执行与存储

逐样本记录源 U 置信度、伪类别、RX/day、两次增强视图预测及辅助损失；每 50 步同步，记录独立辅助梯度和同状态 AdamW 更新差量。统一 owner 保留其他健康 worker 并按每 GPU 6、GPU 7 为 4 的总进程上限与显存余量调度。每行自动预测 raw 和 DURA-LU 的 clean＋六种场景并独立评分；C 与 E 的预测冻结后独立计算各 TX/RX/day 错变对、对变错和净变化。

执行 commit `aa3abebe712a2ec478318df5c391646e8c807bf4`；发布传输和远端编译已核实。统一 owner PID 2369027 保留既有健康 worker；已逐项核对四组 C/E resolved config，仅实验身份与是否增加分支辅助不同。真实权重 smoke 正等待 GPU 容量，后续训练和测试自动接续。

## 结果与覆盖

尚无结果。逐 seed、逐读出及逐场景比较 E 与 C，Main 只有一个目标选定 seed，不代表完整三种子结果。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

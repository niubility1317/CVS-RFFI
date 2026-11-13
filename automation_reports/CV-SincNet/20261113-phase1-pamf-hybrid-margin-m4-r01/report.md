# Hybrid 有上限分类间隔替换对照

- run_id：`20261113-phase1-pamf-hybrid-margin-m4-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

固定 Main seed 01 和 10 类三个 seed，以分类间隔替换匹配 C 条件中的融合层辅助对比；不叠加五分支辅助或训练专用投影头。测试集此前曝光且参与 seed 选择，比较为探索性。

## 数据、seed与模型来源

保持原完整 L/U/V，不划留源域开发集；每行使用同数据契约、同 seed 的 Tri E20，续训 E10＋E20，固定 E50。分类间隔上限在首个 U 更新前，用当前模型对完整 L 的正确预测间隔中位数计算一次：参数来自 E20，首次有标签前向已更新 BN 统计；后 20 轮复用该上限，不用目标分数设定。U 真实类别不读取。

## 执行与存储

使用与 C 匹配的固定输入路由、物理扰动日程及辅助权重，不开展调参；原始参照与分类器权重仅在新增损失中停止梯度。逐样本记录实际扰动范数、置信度、伪标签和间隔违约，周期性记录同状态 AdamW 更新差量。每行训练完成自动冻结 raw 与 DURA-LU 的 clean 和六个 practical 场景预测，再进行独立评分和 C/M 配对错变对／对变错分析。其他命令、环境、CWD、launch owner、资源限制、输出和 stop rule 见 experiment.json。

发布 commit `233085d1202fea21e49004c182a13c308148662b`；本地针对性检查通过，发布包传输与远端编译已核实。统一 owner PID 2456596 已接续本任务原队列并登记本实验；真实 checkpoint 的无查询 smoke 等待 GPU 容量，四行训练尚未启动。对照配置已逐行比较，仅预定目标函数相关字段不同。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

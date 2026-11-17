# 十类浅层与六源 KL 双方向软协调及损失剂量

- run_id：`20261118-phase1-pamf-repl-dual-clean-m16-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：**VERIFIED / RUNNING**（2026-11-18 01:02 香港时间；以远端实际状态为准）

## 目的与对照

十类和六源各八行；组合参考和监督方向软协调，匹配测试 U 权重与 KL 剂量，固定 E50 clean。

上一轮十类保留浅层与 KL 分别有正向条件收益，但二者梯度时常相反；六源 KL 剂量从 0 到 2 的匹配比较为正而边际递减。更低的 L/U 冲突并不保证 clean 更高。本轮先软化其他辅助项相对参考项的负分量，再软化全部 U 辅助项相对 L 的负分量；监督和基础 U CE 在全局裁剪前不变。两个操作顺序及强度均写入逐行配置，不保证 AdamW 后同时满足两个方向。

十类四行探索 alpha/beta 组合，另外两行比较 U 总权重 0.5、1，两行比较 KL 0.5、1.5；六源四行探索组合，两行比较 KL 1、2，两行比较 U 总权重 0.5、1。各剂量行以本批相同其他条件、或上一批同规则已冻结行为对照；跨训练轨迹和历史行不能排除随机波动。全部沿用同一合法源 Tri E20，独立继续到固定 E50，使用已曝光种子及测试，结果为探索，不作独立确认性声明。

每行记录全程 L/U 参数组范数、冲突和限幅、参考／其他辅助项关系、L CE、源验证与 U 接受；每 100 步记录同状态 AdamW 反事实。固定模型后自动 clean 三划分预测、独立 truth-last 评分，并独立离线评分 U 伪标签；U 真值不用于训练或选模。主对照为同物理 ID 的 cross_rx 和 cross_both，无星地、weak 或恢复器测试。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

发布代码 commit `eaa864df5ca5017b870d1aeba75f2bddf6c73a4c` 与远端分支独立读回一致；发布归档传输核验和远端编译通过。本地相关测试 15 项通过，独立 P0/P1 审查未发现直接阻断项。实际 E20 无查询 smoke 16/16 PASS。

01:02 只读核验：控制器 RUNNING，16 个训练 worker 在 GPU 0–7 各 2 个，全部 PID 存活，工作目录与命令行匹配，逐步指标文件在两次读取之间全部增长；失败 0。其余 48 项预测、clean 评分和冻结后 U 评分已登记并等待对应训练完成。可使用 `E:/type10-7/local_artifacts/pamf_repl_dual_20261118/monitor.py` 和 `verify_workers.py` 只读检查；控制器状态和实际产物为准。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

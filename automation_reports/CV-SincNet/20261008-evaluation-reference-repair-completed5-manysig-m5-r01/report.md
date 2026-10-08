# 修复后首批5个E200模型：立即执行clean及六星地场景独立测试

- run_id：`20261008-evaluation-reference-repair-completed5-manysig-m5-r01`
- group_id：`cvs-reference-repair-completed-evaluation`；类别：`cvs`；阶段：`evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户要求立即给测试结果；固定2026-10-08 15:13已完成的5模型，非按性能筛选，单seed描述性测试。原32行训练与后续固定矩阵不变。

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

## 用户要求立即测试

固定15:13核实已完成的5行E200模型，seed2026092701，非按分数选择。用户“我要测试结果”将这5行的测试时间提前；独立输出root，原32行训练与其测试队列不改。预测共享原VALIDATED_ONCE的168000物理query×7视图，5×7全部预测固定后由独立scorer连接truth。E200来源/角色/原生args逐模型复核，无query真实checkpoint smoke在预测前执行。所有目标分数只作本次报告，不回流27行训练、选模或重排。

独立repair_review P0/P1审查PASS；固定矩阵/target标志拒绝测试PASS。发布以文件入口运行并提供release_commit.txt；一个串行推理worker，不增加训练进程、不停止健康任务。

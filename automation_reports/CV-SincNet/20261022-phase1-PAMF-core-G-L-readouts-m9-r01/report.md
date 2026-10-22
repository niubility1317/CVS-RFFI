# PAMF 核心机制：九模型补同契约 DURA-L 读出

- run_id：`20261022-phase1-PAMF-core-G-L-readouts-m9-r01`
- group_id：`phase1-pamf-three-core-functional-factorial`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Two supervised objectives and one complete native selective-SSL group; fixed CE scaffold/backbone/head. Full-minus-group main table plus full factorial, without score-dependent regrouping.

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

## 固定分组和本地验证

详细方案见 [三个核心机制](../../../docs/PAMF_THREE_CORE_ABLATION_20261022.md)。主文仅完整、去 A、去 B、去 C 四行；八组合完整预登记，三种子内先求配对和交互，再汇总均值及样本标准差。分组不使用目标成绩。

12 新 F、12 既有训练来源（含排队项）复用，21 新预测／21 独立评分／1 CPU 汇总，共 55 新阶段任务。G-L／G-LU 均使用同契约既有谱系，raw 与已有读出按来源复用，不新增 G 或部署计时。

10 项聚焦测试和一次独立 P0/P1 审查 PASS；其中 grouped111/011/101/110 与原 full/F16/F04/F20 的损失和梯度逐元素一致，C=0 在读取数据前关闭 U。N607 只读 preflight PASS。当前仍为 LOCAL_VERIFIED，实际 checkpoint smoke、发布和接管待核实。

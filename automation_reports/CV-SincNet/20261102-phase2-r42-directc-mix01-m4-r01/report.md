# R42 派生零目标旧类标签直接注册：冻结头与源原型

- run_id：`20261102-phase2-r42-directc-mix01-m4-r01`
- group_id：`phase2-r42-direct-c-zero-target-old`；类别：`cvs`；阶段：`Phase2-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：REPLACED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

MIX01 16 KN; two old-class sources, raw/DURA support and query; direct C with no B and no target old support.

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

## 启动读回（2026-11-02）

N607 controller 1508816，四个 seed 真实源模型无 query smoke 和源 L_s 原型准备完成；首批 8 个预测 worker 运行中。预测全部固定后独立评分，再由独立 readout PID 1511884 聚合。当前尚无评分结果。预测代码 commit 74906ce48061439df59327d0af6d20124241acfd；readout commit 1ee70003958197064c437d918747e414fc23bf81。详细运行证据见本地私有 r42_direct_c_launch.json。

用户明确指定 DURA 四格仅用于 PAMF 两种派生变体；MoPC、Orthogonal、LoRa 复用已完成的 raw/raw 结果，不补跑 DURA。结果如实报告；R42 的非 raw/raw 与原生 raw/raw 属于跨输入比较。

2026-11-02：队列已交接至 20261102-phase2-r42-directc-fast-m4-r02。16 个已完成单元和 8 个在途 worker 由新队列保留复用；旧输出不覆盖。

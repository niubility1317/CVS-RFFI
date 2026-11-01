# R42 全 KN CUDA 核判决无中断接管

- run_id：`20261101-phase2-r42-group1-m4-r03`
- group_id：`phase2-frozen-R42-group1-completion`；类别：`cvs`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen R42 and original LocalRidge fixed matrix; retain live predecessor workers and reuse completed predictions, accelerate only original LocalRidge FP64 query scoring.

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

独立 P0/P1 审查完成，修复了 smoke 期间旧 worker 失败可能被重跑的问题，并完成唯一一次定向复核；24 项定向测试通过。新 run 实际启动前依次对四个真实源 checkpoint 做无 query CPU/CUDA 判决比较，失败恢复原控制器；成功后仅精确接管原调度器，旧 worker 自然完成。新旧预测全矩阵冻结后才独立评分。局部合成基准不代表正式吞吐。

2026-11-01 交接核验：VERIFIED TRANSFERRED：真实源 checkpoint 四种子无 query smoke 全 PASS，CPU/CUDA 最大分数差 6.94e-17；前代 controller 已退出，9 个旧 worker 自然运行；180 个闭合旧单元已纳入复用索引，新 root 已完成复用单元。全预测完成后独立评分。 代码版本 4b2fcc7600f45ffbe810d51c734a900d47ba2cae；归档传输校验与远端编译通过。

初步实际耗时观测：完整 MIX 单元、208 行（22 复用/186 新预测），r02 CPU 40 单元总耗时中位数 4560.54 秒，r03 CUDA 25 单元 616.47 秒。两组非配对且共享资源；范围与限制见 [耗时观测](performance_observation_20261101.md)。

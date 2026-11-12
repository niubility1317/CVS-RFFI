# 完整源数据 PAMF 原型参数三种子七景对照

- run_id：`20261112-phase1-pamf-prototype-fullsource-tuning-m33-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（以远端状态和 events.jsonl 为准）

## 目的与对照

固定比较三种数据配置的 11 个新参数，每个种子继承匹配原 E20，逐行自动预测并独立评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
2026-11-12：N607 发布，代码 commit `17c363ee19adb2696d6516c96dcc0db059c1c564`，压缩包传输校验及远端编译通过。真实 checkpoint 无目标查询 smoke 通过；控制器 PID 1843699，已核实 8 个本 run 训练子进程及 GPU 0–7 资源。其他 owner 保持运行。每行完成训练后自动进入七景预测，预测完成后独立 truth-last 评分。

2026-11-13：按用户新的资源限制，调度代码 commit `b8834897eab1f7dd513b8bec2ad3a9f2a87e6356` 接管两个本任务队列，控制器 PID 1950148，既有训练进程及原科学代码、权重和输出不变。GPU 7 最多 4 个进程，其余各卡最多 6 个；均计入其他用户和初始化中的进程，并检查实际剩余显存。已完成训练的预测优先于新增训练。

2026-11-13：用户新增仅置信度筛选实验后，统一调度器更新为 PID 1959602（调度代码 commit `0f0cb6dde56f203e96e415d1692b681e3eb8c4a9`），原健康 worker 接管保留，数据和模型配方不变，维持每卡进程上限及自动测试。

不设源域开发集；三种设置各自沿用原始 L/U/V 角色和同设置同种子 Tri E20 权重，固定新增 33 行。普通 PL 和既有原型对照复用，不重训。测试集此前已曝光，此矩阵为探索性比较，结果不反馈到队列或参数选择。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

# 双骨干互补证据：冻结16模型立即全量测试

- run_id：`20261009-phase1-dual-evidence-test-now-manysig-m16-r02`
- group_id：`cvs-dual-complementary-evidence`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户在GPU每卡4训练满载后要求直接启动；每卡临时增加1推理，总<=5，保留健康训练与冻结测试矩阵。

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

## 本次授权与执行边界

用户在已获知8卡每卡4个训练进程满载后指令“直接启动”。本次将测试调度临时改为每GPU额外最多1个推理进程，总实验进程最多5个，空闲显存至少3000MiB。保持所有健康训练不动，仅接管原双骨干已完成源训练、尚无预测子进程/产物的空闲控制器。接管时持有原共享调度锁，核对PID2074109/start_ticks76356526/CWD/argv、空队列状态、无子进程与无预测产物，然后终止该空闲调度器；新测试使用独立r02目录。

原16行E200/44400更新、合法源域物理ID契约、scratch继承来源、冻结矩阵和所有权重保持不变。推理模型/runtime/evaluate及其依赖与原发布a759963aa没有代码差异；原同包batch256、完整FP32、clean及六种practical视图、168000query/视图均不变。独立评分必须等待全部112组预测固定后才连接truth。此处既有代理基准不作为新盲测或真实在轨验证。

本地`python -m experiments.cvs_dual_test_now.checks`通过7个拒绝接管/重复预测/PID复用负测、源checkpoint路径不变及新输出隔离检查；compileall通过。真实16个checkpoint无query来源检查在控制器接管前执行，各预测仍先运行原真实checkpoint无query前向smoke。新run未重新训练。统一使用原已验证评分与独立复算器，源训练日志从原run只读收取。

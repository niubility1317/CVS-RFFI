# 耦合后验低秩等价求解：源端数值、梯度与微批耗时

- run_id：`20261107-diagnostic-Restore-CoupledLowRank-sourceLU-s2026092701-r01`
- group_id：`source-only-equivalent-posterior-cost`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只读固定Coupled E20与原F0，比较原完整后验和等价低秩求解，不修改任何训练或部署版本。

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

## 发布与实际GPU3等待状态

VERIFIED / QUEUED。等价低秩后验的源端诊断已发布，归档SHA对比与远端编译通过；实际CPU控制器PID/CWD/命令、CUDA为空、无子进程、无GPU占用及等待日志增长已核实。GPU3的进程数已达到原正常上限，故按原总数最多3、已有最多2且剩余至少12000 MiB等待；没有增加并发上限或重启任何任务。7项数值/梯度/低噪声/协议负检查与一次独立P0/P1审查通过。真实原G/F0源检查与14格GPU数值/耗时实测尚待启动，不能据单元检查宣布训练提速。该诊断不更新权重、不访问target/TX/V/OE，不改变任何已发布训练或部署版本。 发布后证据

## 进度持久化故障与技术修复

FAILED。真实原G/F0源检查通过，首个global clean单元保存；第二个单元保存进度时复用了独占创建文件的写入函数，触发FileExistsError。终止产物exit1和当前控制器/worker均已退出已独立核实。未完成14格数值/耗时矩阵，缺少worker存活期独立观测，不宣称提速或全部数值验证通过。原输出保留且不重启r01；r02仅改为逐单元独立文件，其他科学与资源设置保持。8项针对性测试及一次P1修复复核通过。 故障核实

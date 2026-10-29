# PAMF第三阶段：RX24-1跨接收机clean配对控制

- run_id：`20261029-phase3-development-clean-rx24-1-r01`
- group_id：`pamf-phase3-development-clean-data`；类别：`data`；阶段：`Phase3-development-clean-cross-receiver-control`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Restore exact same1248 parent development physical records/IDs as clean controls, no resampling/channel/model/formal target.

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

## 真实构建前固定

实际代码/config commit `959a1a68e76a7e2cfba6700c2a251da8330e59b9`。CWD 为 Git-backed mirror；8 tests＋29 subtests，邻近16 tests＋32 subtests PASS，一次独立 P0/P1 PASS。只制作 RX24-1 的 U384/V864，1248 个物理记录，零模型／GPU／正式 RX24-18 读取。只复用 Full 已验证索引并读两成员；原始数据不重验、不加 hash 链。不可覆盖 pool/build.log；真实 CLI 首行 PID/CWD/argv 与原生进程信息及完整日志/产物独立核对。缺成员/形状数量或角色冲突/非有限/输出碰撞即保留部分产物，不自动重建。

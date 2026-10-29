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

## 实际构建完成（2026-10-29）

run `20261029-phase3-development-clean-rx24-1-r01` 已VERIFIED/VALIDATED_ONCE/ANALYZED_DATA_ONLY，code `959a1a68e76a7e2cfba6700c2a251da8330e59b9`。实际U384/V864，沿用parent同序ID和1248条原物理记录，两个既定Full成员、零新采样/信道/F/G/预测/正式target。实际PID43708的原生命令、CLI CWD及起始live进程已核实，日志541→838B、5行完整读取无异常；完成标记和公共三文件齐全，进程已退出。

专项8 tests/29 subtests及相邻16 tests/32 subtests PASS的原始stdout只留在实现worker工具回合，未保存磁盘测试日志；登记已将误指的focused_tests.log设null，并保留这一观察限制，不为补日志重复测试。一次独立P0/P1 PASS。真实构建日志与post/complete/analysis_summary有独立保存；本构建没有科学性能结果。

|角色|capsule_id|split_id|
|---|---|---|
|discovery|p3dev-clean-ef10219d27274e04e83408d1|64274719133a83590352add5|
|evaluation|p3dev-clean-20e01e9ae54a4bc3ae503a3f|347fd2cb757a37f7f162223a|

真实IQ和truth保持本机local_artifacts，不提交Git；clean与star模型输入隔离，目标95%/90%尚未证明。

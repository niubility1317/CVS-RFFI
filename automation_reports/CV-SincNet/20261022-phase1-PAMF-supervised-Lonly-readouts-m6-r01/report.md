# PAMF严格L系统：复用CE及T1监督权重的DURA-L读出

- run_id：`20261022-phase1-PAMF-supervised-Lonly-readouts-m6-r01`
- group_id：`phase1-pamf-supervised-physics`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

补齐严格不使用U的完整系统、当前预算单路SIG_TIME与物理归纳偏置对照；不改变最终native T1。

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

## 固定设计与当前状态

详细设计：[监督与物理消融](../../../docs/PAMF_SUPERVISED_PHYSICS_ABLATION_20261022.md)。

LOCAL_VERIFIED_NOT_PUBLISHED：15项聚焦回归与独立P0/P1审查通过；本地SHUFFLE_TIME CUDA图捕获与eager输出逐元素一致。此项不代表N607实际时延或目标结果。

新增24训练／30预测／30独立评分／1时延，6个读出复用权重和raw。其余51主监督F及G-L链复用。远端发布未核实，原owner不作任何停止或改动。

实际代码／配置提交 `250457d0793a4dba8bca553e056f8171ed5f4154` 已保留本地；Git 推送因网络不可达失败，remoteOID 未核实。新增实验未发布。独立审查 PASS，严格结构时延仅18case×5batch×5轮＝450case-round，不重复统一面板。

用户已授权本次有界 shell 测试和 Python 发布调用，解释器测试及上游推送/OID核对已 VERIFIED。远端预检当前25条健康worker、输出根均未创建、磁盘约1703GiB；发现旧修复release缺默认拼接模块。新包包含同Git版本原增强模块，并只迁移9条未启动修复F及9条未启动预测的执行release，科学配置不变。

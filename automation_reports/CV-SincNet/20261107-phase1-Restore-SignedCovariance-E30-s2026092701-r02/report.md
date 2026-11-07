# 双向源协方差谱适配，常规GPU4空位或GPU6继任

- run_id：`20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r02`
- group_id：`source-signed-spectral-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SignedCovariance r01原模型/数据/单阶段30轮不变，只增加已授权GPU4常规容量路径，保留GPU6原继任路径。

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

## GPU4 常规容量与 GPU6 继任双路径队列已核实

VERIFIED / QUEUED。只替换尚未训练的Signed r01队列，旧PID/CWD/完整命令/commit/CUDA空/无孩子/无GPU阶段均核实后，在新归档校验和编译完成后精确停止旧控制器，再启动新控制器。独立读回旧控制器退出、新控制器在线、CUDA空、孩子0、GPU阶段0。5项资源测试和一次资源审查通过；模型配置除run_id外一致、每个scratch张量相等，原模型审查复用，数据/源权限/seed/30轮预算及Main矩阵不变。GPU4按原总数3上限、剩余显存至少12000MiB准入；GPU4不可用时，GPU6仍须等当前ConditionedCovariance最后GPU阶段并按总数5上限准入。读回GPU4总数3、GPU6总数5，因此当前仍在等待，新增GPU训练进程0。源2真实检查仍待首个GPU训练阶段执行。两路健康训练和其他队列均在线，旧产物保留。新路径可利用未来GPU4正常空位，实际提前量未作承诺。

[队列验证摘要](queue_evidence.json)

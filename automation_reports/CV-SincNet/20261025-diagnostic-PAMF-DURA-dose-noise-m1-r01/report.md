# 恢复域监督剂量与恢复后轻噪声的源域可行性诊断

- run_id：`20261025-diagnostic-PAMF-DURA-dose-noise-m1-r01`
- group_id：`phase1-pamf-restored-IQ-ssl-source-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

一个E30原普通PL学生在12个L/U源批次求完整原L+普通raw PL和新增raw/G软监督梯度；不更新参数。随后三冻结教师、完整源V检验40dB恢复后复噪声。

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

具体源梯度/后噪声矩阵、停止规则和发布见 [预登记](preregistration.md)。无 optimizer、无新的 F/G 更新。

## 首批技术失败

FAILED。真实冻结F/G L2 smoke通过，首个源L/U批的梯度计算后，在恢复固定非持久滤波bank时KeyError；原state_dict快照遗漏persistent=False缓冲区。没有完整梯度矩阵或V噪声结果，不据此确定剂量；未创建optimizer、未更新参数、未访问target。旧release/run/log保持。新r02仅修全buffer快照并使用独立root，科学矩阵和阈值保持。

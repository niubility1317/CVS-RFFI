# 固定识别器源域一致性诊断

- run_id：`20261108-phase1-frozen-source-proxy-agreement-s2026092701-r01`
- group_id：`phase1-frozen-source-proxy-agreement`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在源域非真值 clean 伪类别参照下比较三个已冻结独立恢复器，不能视为身份准确率。

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

## 技术失败记录

远端进程启动首批源数据时，PyTorch 与环境中载入的 NumPy 数组类型不一致，数组比较报错；进程已退出，运行目录为空，未保存预测或统计。未读取 TX 真值、V 或目标。隔离复现确认，经 DLPack 转换后真实源样本前 8 条 clean/raw/DURA 预测与固定参照一致；修复后将用新的不可覆盖运行目录重新执行，原记录保留。

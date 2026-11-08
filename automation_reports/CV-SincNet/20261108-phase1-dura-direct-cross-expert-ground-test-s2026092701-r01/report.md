# 时频跨恢复器七景测试

- run_id：`20261108-phase1-dura-direct-cross-expert-ground-test-s2026092701-r01`
- group_id：`phase1-dura-direct-cross-expert-ground`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 DURA 与独立直接恢复器，使用源域配对 IQ 训练 3576004 参数时频联合残差网络，自动七景测试。

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

## 后台启动

N607 GPU7 训练前可用显存 18501 MiB，后台队列 PID 1107558、控制器 1107569、训练 worker 1107572。代码 commit `f89c6c645d3ff0423be20b584e8bfa7c85dc3531`；五项针对性测试和独立 P0/P1 检查通过，发布包两端一致，远端编译通过。32 条真实源样本的 DURA/direct 权重、时频网络前向及双分支梯度检查通过。

训练固定使用 17632 条父模型未见的合法源 IQ、3 组模拟视图、20 轮及最终权重；时频重建分支可输出完整 IQ 修正。恢复器训练不使用 F、TX 真值、V 或目标；冻结后自动执行 112 条源信号 smoke、Phase1 七场景预测及全部预测完成后的独立 truth-last 评分。

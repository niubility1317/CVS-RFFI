# 源域均衡融合训练

- run_id：`20261108-phase1-dura-direct-balanced-mix-source-s2026092701-r01`
- group_id：`phase1-dura-direct-balanced-mix-source`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 DURA 与大容量直接恢复器，以源接收机均衡和源干净信号逐条融合比例监督训练门控，训练后按固定七景独立评分。

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

## 首次启动失败

N607 控制器显示 FAILED：训练在构造源接收机权重时，因服务器 PyTorch/NumPy 直接转换不兼容而退出，未进入 GPU 训练，未创建源权重或正式测试输出。保留原运行和日志；修复采用已在该服务器实测可用的 DLPack 转换，使用新的运行编号重试。

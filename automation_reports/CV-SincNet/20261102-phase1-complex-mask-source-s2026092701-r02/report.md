# 无标签多尺度复数频谱恢复与 Tri-SIG-TIME SSL 配对验证

- run_id：`20261102-phase1-complex-mask-source-s2026092701-r02`
- group_id：`phase1-complex-multires-f-free-restoration`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 DURA，使用配对源域 IQ、双时频分辨率复数乘性滤波和低信噪比保护门；不使用发射机标签或分类器训练/选模，源波形选择后接冻结 Tri-SIG-TIME SSL。

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

## 启动读回

N607 CPU owner PID 687305，源训练子进程 PID 687410；工作目录为已核对的既有 Phase1 运行环境。真实源 smoke 已完成，当前执行完整源训练。固定代码提交 9160406b331bad9495e005afc898335d3c14214e，发布包 SHA-256 2e02406b529e4c8b663110b4ceb7d8ec8d5d482cd45e47a7517caa2c9229c。训练不使用 GPU。

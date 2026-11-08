# 源域均衡融合训练

- run_id：`20261108-phase1-dura-direct-balanced-mix-source-s2026092701-r02`
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

## 后台启动与兼容性修复

首次运行 r01 因服务器 PyTorch/NumPy 直接转换不兼容在训练前失败，无权重和目标输出。本次 r02 仅改为服务器已验证的 DLPack 转换，训练方法、固定预算与测试矩阵不变。代码 commit `597b621aa76558c4ba6fdea3487c7f89eb904639`；8 项针对性测试、独立 P0/P1 检查及定点修复复查通过，发布包两端匹配且远端编译通过。

N607 GPU0 启动前显存余量 18501 MiB，后台队列 PID 1067973、控制器 1067986、训练 worker 1067989；源域 32 条真实样本的固定权重检查通过，进入源训练。训练使用 17632 条合法源 IQ、3 种固定视图、20 轮，源接收机仅用于均衡训练损失；推理不读接收机、干净信号或 TX 标签。训练后自动按预登记执行七场景预测及独立 truth-last 评分。

## 训练完成与正式预测

已完成固定 20 轮并冻结最终权重，source_summary.json 为 SOURCE_TRAINED；112 条源信号识别链路检查通过，控制器状态 RUNNING_GROUND_PREDICT。七个视图全部预测完成后才进行独立评分；目前无正式成绩。

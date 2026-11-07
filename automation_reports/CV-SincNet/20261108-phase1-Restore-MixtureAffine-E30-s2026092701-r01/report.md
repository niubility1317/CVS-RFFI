# 源波形多假设抗饱和融合，单阶段30轮

- run_id：`20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r01`
- group_id：`source-mixture-affine-scratch30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与已登记后验 softmax 融合版匹配，仅将融合调整改成中心化仿射 IQ 组合，避免后验概率饱和使源梯度消失。

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

## 运行状态（2026-11-08）

VERIFIED：远端控制器 PID 1014456 按登记命令启动，CUDA 为空，等待原融合版完成 GPU 阶段并等待 GPU 3 空出预留余量。程序包传输核对和远端编译通过；真实源样本/原 F0/双分支梯度及严格重载尚待执行。原候选不停止、不改变，两个版本各自自动完成固定 30 轮和原 Main 两总体七景、独立评分。

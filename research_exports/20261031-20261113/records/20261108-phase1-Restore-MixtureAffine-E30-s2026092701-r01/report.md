# 源波形多假设抗饱和融合，单阶段30轮

- run_id：`20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r01`
- group_id：`source-mixture-affine-scratch30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
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


## 可用时切换到 GPU 0 的接管预登记


接管后 GPU 0 总进程上限 3，每 2 秒核查；若本任务占卡且总数超过 3，或连续 3 次无法核实，仅停止本任务子进程并保留产物。独立于原融合版的 GPU 3 运行；两版比较只在各自完整冻结预测后独立 truth-last。真实源 2 条、原 F0 与梯度、严格 E0 重载检查仍是训练首步。

实际接管命令（在 capacity3-gpu0-dispatch release 下，控制器 CUDA 为空）：

`[已省略路径] -u -m experiments.pamf_phase3_restoration.mixture_affine_capacity_dispatch --root [已省略路径] --legacy-release [已省略路径] --cache [已省略路径] --commit 5f5ca0859a8f4508f3f8b09171b7f818f049300f`

当前 GPU 0 已有 3 条其他计算进程，接管尚未执行；既有 owner 和等待队列继续。预期产物为逐阶段提交／退出状态、源检查、训练日志、完整预测和独立评分。

## GPU 0 后台训练已启动

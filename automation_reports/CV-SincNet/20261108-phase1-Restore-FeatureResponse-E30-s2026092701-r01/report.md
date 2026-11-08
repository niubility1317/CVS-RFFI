# 源连续特征响应保持恢复器，从零单阶段 30 轮

- run_id：`20261108-phase1-Restore-FeatureResponse-E30-s2026092701-r01`
- group_id：`source-continuous-feature-response-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原 dense 响应恢复器增加源端 F0 连续特征差分响应损失，训练部署仍使用相同后验。

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

## 后台队列已核实

VERIFIED / QUEUED。远端控制器 PID 904896 的命令、工作目录和空 CUDA 环境已核实；一次发布包 SHA 比对一致，远端编译通过。GPU5 准入要求原有进程最多 1 个且剩余显存不少于 12288 MiB，运行时总数上限 3；资源不满足时等待。尚无训练子进程，真实源样本、原 F0、两分支特征响应梯度和 E0 重载检查均待首步执行。[排队证据](queue_evidence.json)

## 已授权 GPU 容量下的排队接管预登记

原 owner PID 904896；在操作前再次核对完整命令、工作目录和 GPU 5 容量。仅在确认尚无本版训练目录、提交标记及子进程时暂停并退出原 owner；若已开始训练或状态不明则不中断、不接管。接管仅修改资源调度：原科学 commit 6a54c65a2b54886191f44b6a4d6d94530fcb217d、模型、数据、F0、30 轮预算、输出根及完整七景测试保持。

接管 commit 50abe8303c7e6475032aa82f3422ca5f4b01d65a。GPU 5 准入时至多已有两个计算进程、至少 12288 MiB 空闲，总进程上限 3；每 2 秒核查，若本任务占卡且总数超过 3，或连续 3 次无法核实，只停止本任务子进程并保留输出，不影响其他 owner。真实源 2 条与 F0、响应梯度及严格重载检查仍为训练首步，之后自动完成 30 轮、两个总体七景与独立评分。

实际接管命令（在 capacity3-dispatch release 下执行）：

`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.feature_response_capacity_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261108-phase1-Restore-FeatureResponse-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit 50abe8303c7e6475032aa82f3422ca5f4b01d65a`

控制器 CUDA 为空；预期产物为逐阶段提交／退出状态、真实源检查、训练日志、完整预测与评分。当前为接管前状态。

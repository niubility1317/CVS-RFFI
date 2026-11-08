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

## 安全接管与真实源检查（2026-11-08）

VERIFIED：旧 owner PID 904896 在确认尚未训练且暂停后无子进程时退出；首次尝试启动新控制器时发现远端日志目录不存在，尚未启动新控制器或训练。独立确认无旧／新 owner、无训练目录和提交标记后，建立日志目录并启动新 owner PID 1058139。训练子进程 PID 1058325 使用原科学 commit 6a54c65a2b54886191f44b6a4d6d94530fcb217d 及原 release；接管 commit 为 50abe8303c7e6475032aa82f3422ca5f4b01d65a。GPU 5 入场时其他进程 2 条、空闲显存 18501 MiB；实际确认本进程在该卡，总数 3。没有对其他 owner 发信号。

真实源 2 条／原 F0／完整后验首步检查 PASS：F0 冻结、物理与细节梯度隔离、严格 E0 重载、源特征响应梯度及训练／部署后验一致；零 TX/V/OE/目标读取及零优化器更新。检查通过后训练继续，尚无完整训练或正式测试结果。原模型配置中的资源文字描述旧等待条件，实际接管策略以本记录及远端 resource_controller 为准；科学配方、预算和输出根均未改变。自动跟踪进程在旧、新 owner 交接间隙退出，核实新 owner 唯一后已恢复。

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景83.669048/82.648810/81.216667/82.384524/75.313690/67.546429%，1/6；clean两个总体净+59/+33，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整结果](analysis.md)

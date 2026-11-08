# 源波形多假设融合与细节恢复器，单阶段30轮

- run_id：`20261108-phase1-Restore-MixtureFusion-E30-s2026092701-r01`
- group_id：`source-mixture-fusion-scratch30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

单次物理后验解生成四个源波形假设，观测条件融合后接不限幅多尺度 IQ 细节网络；源端连续 F0 指导。

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

VERIFIED：远端控制器 PID 999379 已按登记命令启动，CUDA 为空，GPU 3 尚未满足预留一条进程余量的入场条件，当前等待训练。程序包一次传输核对及远端编译通过。真实源样本、原 F0、双分支梯度与严格重载检查尚未执行；通过后自动完成 30 轮和原 Main 两总体七视图、独立评分。默认 DURA 不变。

真实源统计绑定检查：N607、CPU 两线程、CUDA 为空，读取原 L_s 6300/U_s 56700 训练先验，四组权重之和 1.00000002235，组内协方差 Cholesky 最小对角 0.0425977，参数 5893131。没有载入 checkpoint、执行神经前向、更新参数或读取目标；此检查不替代排队后的原 F0/源 IQ 检查。

## 合法源训练样本上的初始融合梯度

在 N607 CPU 两线程、相同两个原 L_s/U_s 非 clean 源样本、同一 scratch 初始化及相同 source-fitted 先验上，分别计算原后验证据 softmax 与仅将证据 softmax 后的调整改为中心化仿射系数的前向／反向。只计算 IQ 均方损失，初始输出均逐值等于原未调整混合后验；细节输出头梯度绝对值之和均为 20.757282。原融合头梯度绝对值之和为 4.20910e-25，仿射改法为 0.185376，原两样本浮点后验概率最小／最大值为 0／1。两次 posterior 求解相对残差最大均为 2.824e-14。

这是两条源训练样本的局部梯度数值诊断，不含 F0 前向、checkpoint、TX 真值、V/OE/目标数据或参数更新；不能证明全训练分布同样饱和，也不能证明性能会提升。原版已登记，保留原控制器和后续完整正式测试；新融合规则若训练则作为独立版本固定预算和测试。

## 已授权 GPU 容量下的排队接管预登记

原 owner PID 999379；在任何信号前再次核对完整命令、工作目录和 GPU 3 容量，只在确认尚无本版本训练目录、提交标记或子进程时暂停并退出该 owner。保留原 run 根目录、配置和所有已有产物；若原 owner 已经开始训练或状态不明，则不接管。接管程序 commit c498f710e6040b36121760e89ffdbf8df01a3603，科学模型继续使用原 commit fa22a4663b3fe6c5b7ff8b756ea151faab19e2a3 和原 release；参数、源数据、F0、30 轮及完整测试保持。

新 owner 在 GPU 3 至多已有两个计算进程且剩余至少 12288 MiB 时可启动本任务，总数上限 3；每 2 秒核查，若本任务已占用该卡且总数超过 3，或连续 3 次无法核实进程，仅停止本任务子进程并保留结果，不影响其他 owner。若容量不足则继续等待。真实源 2 条及 F0、梯度和严格重载检查仍为训练首步；通过后继续 30 轮、两总体七视图及独立评分。

命令：在远端 capacity3-dispatch 发布目录执行 Python 模块 experiments.pamf_phase3_restoration.mixture_capacity_dispatch，传入原根目录、原科学 release、原缓存及上述接管 commit；CUDA_VISIBLE_DEVICES 为空，训练／预测子进程按 GPU UUID 指定。独立提交和逐阶段状态、训练日志、七视图预测、truth-last 评分及分析为预期产物。接管前状态为待执行，不能视为训练已经启动。

实际接管命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.mixture_capacity_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261108-phase1-Restore-MixtureFusion-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit c498f710e6040b36121760e89ffdbf8df01a3603`

## 安全接管与真实源检查（2026-11-08）

VERIFIED：核实旧 owner PID 999379 在暂停前未提交训练或产生训练目录，暂停后确认无子进程，再结束该 owner；新 owner PID 1035527 使用接管 commit c498f710e6040b36121760e89ffdbf8df01a3603，训练子进程 PID 1035553 使用原科学 commit fa22a4663b3fe6c5b7ff8b756ea151faab19e2a3 及原 release。GPU 3 入场时其他进程 2 条、空闲显存 18501 MiB；实际确认本进程在该卡，总数 3。没有对其他 owner 发信号。

原 F0／两条源 IQ／完整后验首步检查 PASS：F0 冻结、物理与细节梯度隔离、严格 E0 重载，细节输出头梯度绝对值之和 42.57695；零 TX/V/OE/目标读取及零优化器更新。检查通过后训练继续；尚无完整训练或正式测试结果。原模型配置中的资源文字描述旧等待条件，实际接管策略以本记录及远端 resource_controller 为准；科学配方、预算和输出根均未改变。

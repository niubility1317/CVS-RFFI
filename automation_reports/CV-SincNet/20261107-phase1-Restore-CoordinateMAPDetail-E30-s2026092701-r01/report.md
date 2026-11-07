# C4 等变 IQ 恢复器，单次后验、从零 30 轮

- run_id：`20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01`
- group_id：`restoration-c4-operator-posterior-detail`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

共享物理和细节分支覆盖四个直角 I/Q 旋转。合并回原坐标的物理算子，单次 FP64 后验，恢复原坐标 IQ。

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


## 固定比较

四个共享分支把估计的全实数物理算子变换回输入坐标再平均，同时平均正噪声方差。源 63,000 条 IQ 的高斯统计经四个直角旋转做矩匹配，固定零均值及旋转相容协方差。一次 FP64 后验求解后，共享 MSAT 细节分支校正四个坐标视图，输出映回原坐标再平均。该结构和先验共同改变。

训练仍用原 L_s/U_s，固定源模拟配对、连续 F0 特征和独立物理/细节梯度，全部权重从零单阶段 30 轮。真实 source2、梯度隔离、直角等变检查、初始零细节和严格 E0 重载通过后自动训练。最终轮次固定后自动预测原 Main 双总体七景，预测全部固定后独立 truth-last。逐景核对距同一 raw clean 3 个百分点，两总体分别保持 clean。

## N607 排队已核实

VERIFIED / QUEUED。控制器 PID 3798461 在线，实际工作目录、命令、代码 `bc6bcac3bf8463fb3b2770a0ec2a6115ab322ae1` 和 CPU 环境已核对。发布包 SHA 一致，远端编译通过。快照时训练 worker 尚未准入，真实源检查、训练和 Main 结果均待完成。满足 GPU4/6 准入条件后，控制器按登记自动执行。 [排队证据](queue_evidence.json)

## 源先验旋转矩匹配的取舍

对已保存的原 63,000 条源 IQ 统计独立复算。原均值能量占源二阶总能量的 68.5163%。C4 平均保留二阶矩的迹，变换前后的矩阵差在 Frobenius 范数下为原二阶矩的 68.5880%。原协方差前 16 维解释 59.0929% 的方差，新协方差前 16 维解释 85.3059%。新先验将固定方向的相干均值纳入协方差，取消直接的均值锚定。这项变化可能影响噪声大时的恢复，具体方向要看新模型结果。一个后续可检验的方向是根据当前观测和估计的物理算子，为四个源均值方向计算软权重，再进行一次条件后验。本次只读取已存源统计，无新 IQ、TX 真值、目标观测或训练。

## 实际训练与首轮状态已核实

VERIFIED / RUNNING。原C4控制器已按原容量规则自动启动实际训练，未切换或重启。GPU6准入前已有2个计算进程、剩余20474 MiB，启动后总数3，符合其原上限。读回时该卡总数5，其中本目标两路恢复器均有实际GPU进程；当前ConditionedCovariance保持健康运行。PID/CWD/命令/CUDA UUID/实际GPU PID/日志增长已核实。真实源2、物理/细节梯度隔离、C4坐标检查、初始零细节、冻结F0和严格E0重载通过。E01独立状态检查确认全部63000源记录曝光（L_s6300/U_s56700）、247次AdamW更新、两个分支实际权重变化、模型/优化器有限和先验不变。首轮942.405秒，属于该时共享负载下的实际整轮耗时。30轮/7410总预算和final30后的原TriSIGTIME SSL Main双总体七景/独立truth-last保持。尚无本版正式识别成绩。

[真实源检查、准入与首轮证据](first_epoch_evidence.json)

## 前6轮实际更新与全部日志核实

CoordinateMAPDetail：已核实 6/30轮、1482/7410次更新，累计378000次源曝光（L_s 37800、U_s 340200），在线冻结F0连续特征前向378000条。全部已完成轮次的源顺序均为63000条完整排列，保存模型/优化器有限、两分支和输出头实际更新，源先验固定。

[完整源训练分析](training_progress_E06.md)

## 剩余跨接收机且跨日期测试转正常GPU3

已完成原30轮训练与cross_rx126000条×七景；剩余cross_both42000条×七景及独立score使用原科学release/commit，正常GPU3总数最多3个。仅在确认旧控制器CPU空CUDA、无子进程、剩余阶段从未启动时交接。原权重/完整预测/日志保留，不重新训练或预测已完成总体，不读取目标分数排队。

控制器代码：`2d295adde7ef655368545d650551243a5430797b`；科学代码：`bc6bcac3bf8463fb3b2770a0ec2a6115ab322ae1`。命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.coordinate_remaining_dispatch --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit 2d295adde7ef655368545d650551243a5430797b`。

## 剩余预测已在 GPU3 实际运行

VERIFIED / RUNNING。剩余 cross_both 预测已在正常 GPU3 实际运行，准入时已有 2 个进程、17128 MiB 可用显存，总上限 3 个。原科学代码、工作目录、模型与源检查保持，实际进程/GPU及42000条clean预测完成已独立核实。已完成的30轮训练和126000条×七景预测原样复用；仅退下本任务原排队空闲控制器，0重训、0重复预测。剩余全部预测固定后仍由原评分入口独立truth-last，尚无完整Main成绩。 [独立启动核验](resource_started_GPU3.json)

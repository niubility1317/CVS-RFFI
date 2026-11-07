# 物理接收调制估计与分量监督，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01`
- group_id：`receiver-modulation-source-operator-recovery`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

包级主路/镜像复增益，加独立线性与局部相位。用原源模拟真值监督调制、FIR、完整算子与噪声。独立冻结 Main 七景。

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


已读原 L_s/U_s 512 条固定源记录在 20 个缓存模拟中的调制系数。主路包内相对振幅极差最大值 1.19e-07，非零镜像最大值 1.18e-07。这支持包级复增益的源结构约束。只读缓存模拟系数，无新模型、TX、V 或目标读取。 [源系数摘要](source_trace_shape.json)

## N607 接收调制版排队已核实

VERIFIED / QUEUED。实际代码 3fc9201628e7602ea58c6a5fb13d6ec934513e99，控制器 PID 3869088 在线，实际工作目录、命令与 CUDA 空环境已核对。发布包 SHA 一致，远端编译通过。等待整体相位版最后一个 GPU 阶段终止，然后按原 GPU4/6 准入条件开始。当前真实源两条检查、训练和 Main 均待执行。

## GPU5 串行常规资源预登记

资源控制器 `bf6ab827408f1d20f4e8bbcfcf511d1a118cbc41`，全部科学阶段仍为原 `3fc9201628e7602ea58c6a5fb13d6ec934513e99` 与原release。GPU5正常总数最多3个、已有最多2个且剩余至少12000 MiB；两版按modulation、mismatched顺序，每阶段一个本run GPU子进程。mismatched只等待modulation真实最后GPU终态，不读取成绩作排队条件。源L_s/U_s、架构/损失/seed/原顺序及draw、从零30轮/7410、原Tri-SIG-TIME SSL Main双总体七景和独立truth-last保持。只交接经完整身份、CUDA为空、无子进程、无提交阶段和训练产物核实的旧等待控制器，独立确认退出后才发新owner。原输出骨架、launch.json、release及日志保留；新resource_controller.json独占创建。真实源模型检查仍为原训练器首步，PASS后直接继续；技术失败保留，不重启其他健康任务。9项针对性检查与一次资源范围P0/P1通过。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase3_restoration.registered_restoration_dispatch --candidate modulation --root /home/szu2070436088/2510044040/CV-SincNet --legacy-release /home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01-release --cache /home/szu2070436088/2510044040/CV-SincNet/runs/20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01/cache --commit bf6ab827408f1d20f4e8bbcfcf511d1a118cbc41`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01-GPU5-registered-dispatch-release`。

## GPU5 调度实际状态核实

VERIFIED / RUNNING。调制参数估计版已在GPU5实际训练，准入已有2个进程、剩余19829 MiB，符合原总数最多3个。实际GPU子进程继续执行原科学release/commit，父控制器CUDA为空；真实source2、原F0、完整FP64后验、各物理输出头梯度、严格E0重载及源先验绑定通过，日志已增长。G更新仍只使用原源模拟物理量，F0不提供训练梯度；本轮仅资源安排变化，原source-only单阶段30轮/7410和完整Main测试保持。

[实际执行与资源证据](resource_launch_evidence.json)

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.24e-14，baseline判决差0。六景83.657143/82.395238/80.766071/82.386310/75.182738/67.238095%，1/6；clean两个总体净-66/-29，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整结果](analysis.md)

# DURA 单阶段 30 轮物理恢复与受限 IQ 增强

- run_id：`20261031-phase1-DURA-IQenhance-E30-s2026092701-r01`
- group_id：`phase1-dura-recognition-oriented-IQ-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从零E30/7410；物理noVar恢复与detached后小型shape修正同阶段训练，源F只指导修正，同模型双读出各七景正式评分。

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


最小说明：用户 E30 授权；原数据/原F/defaulthead；从零单阶段物理恢复＋受限IQ增强，F梯度只到新增1030参数。同一训练的两读出全部14视图每角色预测固定后独立评分。5% RMS、factor2参考和全部预算事前固定；无目标反馈或中途选模。实际命令/环境/路径/GPU/停止规则和产物见本登记。当前只是待发布，不代表已训练或有效。


## 启动独立读回

VERIFIED / RUNNING。代码 489f339757b7b3fb9214b3341d780ec98bb14e89；9项聚焦检查、唯一P0/P1审查及原入口P1定点复核通过；一次N607preflight、单归档SHA比较、远端compile通过。controller PID1018738 的 /proc CWD/完整argv/空CUDA读回存在，source2Ls真实F冻结、F梯度隔离、两组G梯度、严格重载、非零增强优化路径smoke PASS。 worker/PID/GPU：{'pid': 1024839, 'GPU': 4, 'command': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.pamf_dura_direct_train.enhanced', 'worker', '--root', '/home/szu2070436088/2510044040/CV-SincNet', '--commit', '489f339757b7b3fb9214b3341d780ec98bb14e89', '--arm', 'scratch_joint'], 'admission': {'visible_GPU_compute_PIDs': 2, 'free_memory_MB': 22382, 'maximum_total_after_adding': 3}}。其他owner未改动；排队不等于训练已完成。完整E30双读出七景与实测成本仍待完成。


## 双读出独立分析准备

新增只读CPU复核程序 enhancement_pair_score.py：仅当原run全部结束且两读出14个target输出和两套基准七景完整后连接truth/clean，复算252个官方分组与每景IQ误差，并新增增强相对物理输出的126组净纠错。对实际成对IQ检查5% RMS修正上限与复增益/相位投影；数值容忍分别0.05001和非微小修正的归一化内积0.003。它不改变正在运行的模型、训练配方、预测或冻结选择。两个针对性算术/边界检查通过，实际全量验证待正式产物完成。

结构成本独立列于architecture_cost.json：整网267981参数、FP32参数1071924字节；卷积/线性MAC25.276968M，其中新增0.208896M（约0.833%）。物理图仍2prior/4CG/27FFT，部署只用增强输出时不增加F调用。MAC不包含FFT/CG/归一化/激活/访存，不据其涨幅推定实际延迟；正式配对预测会计算两次F，仅用于比较，不冒充单输出部署成本。

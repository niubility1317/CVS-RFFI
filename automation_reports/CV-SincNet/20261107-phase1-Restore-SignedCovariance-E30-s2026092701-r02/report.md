# 双向源协方差谱适配，常规GPU4空位或GPU6继任

- run_id：`20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r02`
- group_id：`source-signed-spectral-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SignedCovariance r01原模型/数据/单阶段30轮不变，只增加已授权GPU4常规容量路径，保留GPU6原继任路径。

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

## GPU4 常规容量与 GPU6 继任双路径队列已核实

VERIFIED / QUEUED。只替换尚未训练的Signed r01队列，旧PID/CWD/完整命令/commit/CUDA空/无孩子/无GPU阶段均核实后，在新归档校验和编译完成后精确停止旧控制器，再启动新控制器。独立读回旧控制器退出、新控制器在线、CUDA空、孩子0、GPU阶段0。5项资源测试和一次资源审查通过；模型配置除run_id外一致、每个scratch张量相等，原模型审查复用，数据/源权限/seed/30轮预算及Main矩阵不变。GPU4按原总数3上限、剩余显存至少12000MiB准入；GPU4不可用时，GPU6仍须等当前ConditionedCovariance最后GPU阶段并按总数5上限准入。读回GPU4总数3、GPU6总数5，因此当前仍在等待，新增GPU训练进程0。源2真实检查仍待首个GPU训练阶段执行。两路健康训练和其他队列均在线，旧产物保留。新路径可利用未来GPU4正常空位，实际提前量未作承诺。

[队列验证摘要](queue_evidence.json)

## GPU4 实际训练与真实源检查通过

VERIFIED / RUNNING。双向协方差版已按新增的GPU4正常空位路径实际训练，当前前序ConditionedCovariance尚未结束，因此提前进入训练。准入快照为已有2个计算进程、剩余17032 MiB，符合该路径原总数3上限；随后独立读回总数4，其中本run仅1个，其他进程3个。该变化如实记录；未修改并发上限或停止其他健康任务，后续阶段仍重新核实原准入条件。实际PID/CWD/命令/父子/CUDA UUID/计算GPU/日志已核对。真实源2、原F0、完整后验、初始C不变、双向映射零点、梯度隔离、F0冻结和严格E0重载通过。当前尚未写出首个完整epoch；检查时smoke本身更新为0，不将此等同于训练阶段尚未发生任何更新。模型、数据、seed和单阶段30轮/7410更新不变，final30后自动原TriSIGTIME SSL Main双总体七景与独立truth-last。当前三路恢复器均在实际训练，暂无本版正式识别结果。

[启动、准入快照与实际源检查](launch_evidence.json)

## 前1轮实际更新与全部日志核实

SignedCovariance：已核实 1/30轮、247/7410次更新，累计63000次源曝光（L_s 6300、U_s 56700），在线冻结F0连续特征前向63000条。全部已完成轮次的源顺序均为63000条完整排列，保存模型/优化器有限、两分支和输出头实际更新，源先验固定。 首轮负系数比例0.995937，平均绝对变化0.692812，softsign映射平均斜率0.143544。这是源训练记录，不能据此确定Main收益。

[完整源训练分析](training_progress_E01.md)

## 完整30轮与原Main七景独立复算完成

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景83.489286/82.073810/80.443452/82.207738/74.348214/65.957143%，0/6；clean两个总体净+266/-35，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整结果](analysis.md)

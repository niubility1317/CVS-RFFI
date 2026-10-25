# 任务强度缩放参照的无更新整网源剂量校准

- run_id：`20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01`
- group_id：`phase1-pamf-task-scaled-reference-20261026`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（真实源校准完成，尚无新目标性能）

## 目的与对照

seed01 own E30上保留完整T1 L/rawPL，12源L/U批匹配G普通参照/G任务缩放/raw任务缩放；真实默认头可微分路径，整网物理梯度共同剂量，无optimizer/V/target。

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

## 源校准发布前核实

详见 [最小预登记](preregistration.md)。原L/rawPL主项保持，同输入普通G参照/G任务缩放/raw任务缩放三分支完整12批，依据整网物理梯度给出一个共同剂量；没有新训练或目标访问。两项新检查、基础关系/任务缩放九项检查与compile通过；一次独立P0/P1审查PASS。实际L2默认头/特征图smoke通过后自动继续，不把函数检查当作性能结论。

## 完整真实源校准

ANALYZED，独立读回VERIFIED。实际代码 `3e7103fdff6992a2c6aafbe5b1464295ac38ae6b`；L1536/U3072、12完整批、60次整网梯度计算完成，真实L2/defaulthead/可微分图smoke通过。运行中PID/CWD/argv/GPU0与日志增长独立核实，随后正常完成。

预定共同剂量为 **0.032576809922150361**，主项物理梯度范数中位数0.978037；该值来自新真实整网测量，没有沿用旧0.071458。

| 分支 | 额外范数中位数 | 与主项cos中位数 | 加权中位范数/主项 | 接受样本缩放均值（批均值平均） | 触顶率（批平均） |
| --- | ---: | ---: | ---: | ---: | ---: |
| GUniform | 1.348240 | -0.057523 | 4.491% | 1.000000 | 0.000% |
| GTaskScaled | 3.002250 | -0.052184 | 10.000% | 1.477860 | 0.000% |
| RawTaskScaled | 0.235283 | -0.070754 | 0.784% | 0.152658 | 0.000% |

三个分支共用同一系数与接受集合；GTaskScaled为最大中位范数分支，对应10%整网preclip物理范数，其他分支较小。它不是每批上界、AdamW更新保证或性能证明。source固定原L768、AsKL0、原rawPL主项保留，三个新增分支默认头参数梯度均为0。学生和教师全部参数/registered buffers不变，临时buffers复原，optimizer/V/目标/U真值/参照更新/G前向/新头拟合均为0。耗时11.64秒。

权重和及触顶率仅描述额外监督强度，接受数不是U真实正确率。完整 [12批源记录](source_gradient_batches.csv) 与 [源剂量/执行证据](source_decision.json) 均保留。下一步按此共同剂量登记普通G参照、任务缩放G、任务缩放raw的单种子匹配训练，仍使用原E30+20预算及固定源留出规则；此校准本身不创建新训练模型或访问目标。

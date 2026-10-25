# 恢复域监督剂量与恢复后轻噪声的源域可行性诊断

- run_id：`20261025-diagnostic-PAMF-DURA-dose-noise-m1-r02`
- group_id：`phase1-pamf-restored-IQ-ssl-source-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

一个E30原普通PL学生在12个L/U源批次求完整原L+普通raw PL和新增raw/G软监督梯度；不更新参数。随后三冻结教师、完整源V检验40dB恢复后复噪声。

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

固定矩阵见 [预登记](preregistration.md)。r01收尾首批非持久buffer恢复失败；r02仅技术修复，阈值和科学矩阵不变。

## 完整源可行性结果

ANALYZED，外部执行VERIFIED。r02真实F/G sourceL smoke、12个完整T1-L/普通rawPL及额外raw/G监督梯度批、全V27000×3场景×3冻结模型40dB噪声验证均完成；所有参数和持久/非持久注册buffers不变，无optimizer、V梯度、U真值或target读取。原r01失败产物保留。

primary梯度范数中位数0.909856，新增恢复分支1.802683，方向cosine中位数-0.130245；通过预定≥−0.25源条件。按预定公式所得权重为 **0.12618073797690**。梯度对象遵循CENTER_UNITS物理尺度，是preclip诊断，不是AdamW/momentum更新方向。12批单seed测量不代表三seed最优剂量或目标收益。

40dB噪声的准确率变化为−0.062963至+0.007407pp，幅度较小；但urban预测变化率0.514815%至1.003704%，6/9单元超过预定0.5%稳定性上限，仅3个suburban单元通过。因此整组D按原规则不进入首轮训练；不能将这一源稳定性负结果说成目标增强必然无效或身份被不可逆破坏。

下一矩阵保留原普通PL raw主项与原全部L机制，三组各三个学生：A普通PL控制；B额外raw软监督；C将B额外输入替换成同物理G(H(U))。B/C共享原raw三教师伪标签、接受集合、损失剂量、两次前向平均及两轮升温；不要求G输出自身高置信，不强拉波形/特征。新矩阵仍需独立冻结发布及原raw/DURA七景truth-last；源可行性不代表目标达成。

[12批梯度](source_gradient_batches.csv)、[9个噪声单元](source_noise_cells.csv)、[源决策](source_decision.json)。

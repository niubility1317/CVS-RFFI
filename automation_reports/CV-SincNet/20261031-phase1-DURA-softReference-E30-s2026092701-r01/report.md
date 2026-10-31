# 跨接收机软特征参考：从零单阶段30轮

- run_id：`20261031-phase1-DURA-softReference-E30-s2026092701-r01`
- group_id：`phase1-dura-source-soft-crossRX-reference-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同266951G/noVar四物理项/原alpha/逐样本IQ梯度cap1，新增已源域校准的soft跨RX参考偏移；只读原raw J及1.26MB新偏移缓存，训练F前向0；scratch30/7410及全七景。

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


## 固定方法与判定边界

同一266951参数G从零联合30轮/7410步；原四物理项、alpha和逐样本IQ梯度cap1保持，仅将特征目标从原clean改为已校准的跨RX/day软参考，混合比例0.25。J和偏移使用同一原head秩5基底；参考不读取直接TX真值，不拟合新分类头。训练只读旧J与新1.26MB偏移缓存，F模型/前向/VJP0。旧F50和各缓存/诊断成本单独披露，新增偏移生产时间另记。

源几何和IQ局部校准已通过，不能证明从零训练早期或目标识别有效。偏移非零时不再保证附加loss值≤IQ；只保证步内detach加权后的IQ输出附加梯度范数≤IQ项，非G参数梯度界。12聚焦测试、实际入口检查及唯一P0/P1通过；首步真实2Ls检查后立即继续。固定最终30轮冻结，自动source1800×7及formal168000×7，全部预测后独立truth-last，逐景和逐TX/RX/day分析及全成本报告。主对照75da平衡实例特征，次对照8cf纯物理；不按源成绩取消正式测试。默认DURA不变，六星地逐景距rawclean≤3pp且clean保持的目标仍ACTIVE。


## 实际启动证据

VERIFIED / RUNNING。执行 b6136c53106c9e8e6273a606262d5152a4e08d23 已提交、推送并独立核对远端OID；一次N607 preflight/单archive传输SHA/remotecompile通过。真实2Ls G三模块梯度、实际附加IQ梯度cap、F冻结/原默认头/严格重载通过。实际controller 1410473/worker 1410565/GPU 4 的CWD、完整命令、CUDA、GPU和日志增长核实，旧own句柄全部消失，其他owner保持。至少E1完整前缀预算/加权loss/曝光/逐步cap核对通过，最大压力1.000000119。

新增63000×5 FP32参考偏移缓存1.26MB，生产0.566940秒；独立复算63000条/315000分量、30个otherRX软中心与512校准条，最大偏移差0.0。新F前向/VJP0；原rawJ645.12MB只读复用核对0.075864秒，历史F50/11100、rawJ30.008929秒、F1603.782270秒、参考几何0.212823秒及IQ校准3.156706秒单列。训练中不加载F。当前正式成绩待产出，不将源域校准当作性能验收；满30后自动完成全七景预测/独立truth-last。默认DURA保持，目标ACTIVE。

# ISSL修复版新类学习固定预算验证

- run_id：`20260930-diagnostic-issl-wisig-s392005-r04`
- group_id：`newclass-registration-issl-repair`；类别：`diagnostic`；阶段：`external_execution_acceptance`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

修复KD学生断梯度、SSL梯度累积、queue转置/初始化归一化/容量、冻结teacher及EMA BN状态；保留作者架构/损失权重/温度/LR/阶段流程

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

## 修复与预登记科学判定

修复版保留作者网络、logits对比空间、随机裁剪、旧样本重复、Adam/LR、损失权重与温度、base→SSL→transfer及独立incremental流程。KD保留原温度CE标量且恢复学生计算图，未额外乘T²。teacher冻结并eval；key在no_grad下前向，以EMA同步参数与浮点BN状态、复制整数计数。所有queue向量统一归一化、按transpose计算点积并精确保留最后K条。

预算：base=50、SSL=20、transfer=100、独立incremental=100epoch。两个run在任何新query评分前固定，从零训练，r04不加载r03状态。每GPU最多两个训练进程，本次串行；/root唯一launch owner。

执行判定：loss/梯度有限，KD学生梯度实际非零、key/teacher无反向梯度，queue规范、teacher状态不变、新head行更新；四个checkpoint恢复后预测逐条一致。学习效果同时报告训练旧/新类准确率与独立query旧/新类准确率/H/Macro-F1，不把短预算低分当系统故障或用query调参。B及其衍生指标N/A。原数据/全文缺项不阻断用户授权的WiSig修复诊断，也不据此宣称原论文完整复现。

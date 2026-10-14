# 可见周期预测留出残差：完整L6300可预测性与身份关系

- run_id：`20261014-phase1-visible-period-sourceL-manysig-m1-r01`
- group_id：`phase1-visible-period-observation-feasibility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

No learned model: sourceL6300 visible-only phase/scale/public reference calibration, fixedcopy_cycle1/visible_mean/reference baselines, downstream RX/day/TX counts and fixedcrossRX residual24 geometry. No decoder/backbone/head fitting, checkpoint, U/V IQ or target access.

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

## 固定最小诊断

完整原有 L6300，6TX × 5源RX(1/3/4/6/8) × 3day(1/2/3) × 每cell70包。每包只读取一次；无网络、checkpoint、decoder或分类头拟合。留出120:140，只用80:100、100:120、140:160校准尺度/相位/公开reference delay及复增益。固定copy cycle1、visible mean、knownreference三个基线，预测真实留出参考残差24；所有valid/预测先于合法L标签统计。掩码输入留出段及其他位置为零，不填可见均值。

三个基线在完全相同的可见valid集合上报告每包24维实数MSE、全组/有效组分母、poolederror/energy，以及RX/day/TX/RXday。不能依据留出能量筛样本或只报告修剪后的均值；同时披露能量/误差前1%占比。

visible_mean残差24仅做无拟合关系统计，不是f_theta joint160。固定同day、原cell ordinal，anchor与4个其他RX的同TXpositive，以及该reference RX五个异TXnegative，潜在126000条关系；visible-valid掩码后报告cospositive/negative/gap分布、ordering和全部计数。不hardmine、不加margin、不拟合另一个头。同RX异TX相似度补充描述RX关联，不能单凭差值归因于接收机捷径。

在读真实源结果前固定可行性条件：至少95% visible-valid；visible mean pooledMSE比reference改善至少10%，每RX不反转；跨RX身份cosgap至少.02，至少4/5RX为正。失败关闭本固定pretext版本，不扫描遮挡位置、tone、phase或损失权重。即使通过，也仅支持源条件预测及身份关系，不证明decoder胜过visible mean或新训练的泛化收益；另行预登记后才允许训练。

CPU2threads/0GPU。精确命令/CWD/源码Gitcommit在experiment.json及release_commit.txt。首步2个无数据zeroIQfixture后立即继续源数据；本stage不消费checkpoint，所以不加载无关权重制造smoke。一次P0/P1限本次run正确性，单archiveSHA比较/远程compile；启动后独立读PID/CWD/argv、log增长及完整终态。系统技术失败保留partial；低可预测性或低身份gap是科学结果，不能停止无关任务。

只保留接收后缓存表示结论，上游equalization和原始噪声独立性未证明，不称TX硬件参数。历史正收益TIME/LONG_SILU/Sinc保持独立，不合并不同recipe；本次没有目标预测/评分，不把源诊断冒充84%目标完成。

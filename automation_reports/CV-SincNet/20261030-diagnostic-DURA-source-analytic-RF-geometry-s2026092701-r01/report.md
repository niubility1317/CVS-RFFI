# 单阶段恢复输出的解析 RF 梯度与成本校准

- run_id：`20261030-diagnostic-DURA-source-analytic-RF-geometry-s2026092701-r01`
- group_id：`phase1-dura-source-analytic-RF-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定source512×缓存01/20已存noVar原输出，五解析RF损失与IQ目标的输出梯度；单一权重，CPU2，无训练/TX/V/target。

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


## 最小研究说明

固定此前合法source512（L128/U384）和同noVar cf40模型已保存的缓存01/20原输出，只打开source位置、physicalID、role、原clean和nativeIQ。两份source参考逐项与原缓存匹配，不读取同文件中的F特征/输出/选择，也不读取V、TX逐样本标签或target。首步真实noVar/F50各2Ls smoke及参数快照检查通过后立即继续，不更新任何G/F。

PAMFViews使用新建冻结的frequency、PA、默认初始化Sinc、highfreq和公共参考响应输入变换，没有训练好的F权重，也不执行身份嵌入/分类头。五支各按sourceclean特征能量归一，RF为五项之和。对恢复输出IQ求gIQ/gRF；32个B32的范数比中位数给出唯一alpha=.25/median(ratio)，不拟合G参数或挑分支。两个source缓存视图各自要求中位校准压力[.125,.5]、平均余弦>=0、16批中负余弦最多8批；任一失败不立项此配方、不放宽条件。通过只支持另行登记单阶段scratch E20匹配对照，不是正式有效性结论。

保留全1024访问的两类输出梯度，独立NumPy重算全部batch的范数、点积、比例、alpha与资格。CPU2/0GPU，记录每批与总计算成本；不将CPU变换成本当成GPU训练成本或部署延迟。R01已试过解析RF损失且失败，R02/DURA粗恢复祖先没有这些损失，不声称该约束已证明有效。当前是独立机制检查，默认DURA和逐景3pp/clean/训练效率目标保持。

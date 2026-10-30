# 单阶段恢复输出的解析 RF 梯度与成本校准

- run_id：`20261030-diagnostic-DURA-source-analytic-RF-geometry-s2026092701-r01`
- group_id：`phase1-dura-source-analytic-RF-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

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


## 完整源端结果

VERIFIED / ANALYZED_SOURCE_ONLY。执行commit `c785ebd8816b180445cbdfc721c3551bc845ddbc`。10项聚焦检查、唯一独立P0/P1审查、N607预检、单归档两端SHA与远端compile通过。实际PID898340/CWD/命令/空CUDA已独立读回，14742字节日志记录先真实2Ls G/F smoke，再完成32批；最终G/F及解析变换冻结状态不变，优化更新、逐样本TX/V/target读取均0。所有1024访问只来自原512物理记录的两个源视图。

|源缓存视图|中位校准梯度压力|平均梯度余弦|负余弦批数|资格|
|---|---:|---:|---:|---|
|1|0.258124|0.282386|0/16|PASS|
|20|0.237195|0.283367|0/16|PASS|

原规则唯一alpha=0.0095279442618058027。两源视图均通过预定输出梯度可行性条件，32批余弦全部为正。独立NumPy64从完整1048576个IQ/RF梯度值重算32批的范数、比例、点积、两视图压力与资格，另核对IQ梯度闭式公式，最大绝对差8.13957204e-11；320项批统计一致。完整计算循环1.232532秒，CPU2/0GPU，模型加载与smoke另计。

通过只说明这个固定辅助损失有有限且总体不冲突的输出IQ梯度，不代表G参数空间的方向，也不代表TX保真或星地正式收益。不会按分支、源场景或target结果再选权重。下一可执行项为独立单阶段scratchE20匹配对照：保留noVar四项基础损失，额外加该alpha乘五项解析RF误差；单次G恢复输出复用，只附加解析变换，无训练好F加载/身份头前向。需要实际全七景测试和实测训练成本后判断。默认DURA、每景距rawclean<=3pp和clean保持仍未达成。

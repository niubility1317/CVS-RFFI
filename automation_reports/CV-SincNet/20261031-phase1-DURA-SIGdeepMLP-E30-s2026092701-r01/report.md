# 更深特征的非线性SIG预测：从零单阶段30轮

- run_id：`20261031-phase1-DURA-SIGdeepMLP-E30-s2026092701-r01`
- group_id：`phase1-DURA-source-qualified-deep-SIG-predictor-pair-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留main两prior条件结构与原四物理+输出SIG+.01mean2priorSIG辅助；原预测加192维深层特征残差，192→160-SiLU-160→160，末层零初始化。

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


## 固定学习假设与矩阵

更深特征的非线性SIG预测。源固定权重条件干预的两视图主条件已PASS：正常487/500、关闭481/497、本条clean特征498/505、移位480/498。该证据仅支持本次新学习假设，不是新G资格或目标收益；源数据曾用于G训练、clean特征不可部署，不能作为恢复上限。

保留main344295随机架构全部初始参数、浅中层预测与原条件解码，新的深层残差读取已算出的28×4和40×2池化特征，共192维。线性版加192→160零初始化映射；非线性版加192→160-SiLU-160→160，末层零初始化。两者都保持原始SIG预测和初始IQ，新head只在专用CPU随机fork内建立；MLP首层初始梯度为零是预期行为，新增末层要有实际源辅助梯度。没有加载main训练权重或任何G祖先。

原四物理项、输出SIG alpha1/IQ梯度cap.5、prior辅助 .01×两预测相对MSE均值、条件bound.1不改。辅助梯度在输出cap外，全G clip1；新模型预测是浅层与深层残差之和，同一目标只计算一次辅助损失。G推理只读取本条IQ，无TX/RX/scene/F/clean特征输入；不另训分类头。两版均单阶段从零30轮/7410、同L189000/U1701000、原20信道缓存后复用01..10、native全部路径。F-onlySIG40.32MB复用、新缓存F0，在线prefix/VJP各1.89m，F50知识与成本披露/F-guided。

6项聚焦检查和两个actual入口/全模型/别名/原main初值-IQ-SIG-随机流/损失与新梯度/路由-独立输出-共同CPU锁-七景检查PASS。每候选一次P0/P1在发布前完成；首步真实2非clean L_s smoke（G12/Ffull4/prefix2/VJP2）后直接训练。主对照固定main4452，新增两版互比，并列原SIG/pureG/default/R17。自动profile/源1800×7/正式168000×7，不因源表现取消；各角色全七景预测后独立truth-last，不按目标调参、选轮次或重跑。

GPU4/6/7每卡本任务1个、总至多2个，显存余量≥12GB且既有compute≤2新增1总≤3；CPU评分共用线性新run锁、输出根独立。六景逐个≥137216正确、clean≥142256、R17逐景损失≤1680，low urban70%另列。完整参数、实际速度、总训练与测试成本一起报告，不由更深/更多参数推断有效。默认DURA和其他owner保持，完整目标ACTIVE。

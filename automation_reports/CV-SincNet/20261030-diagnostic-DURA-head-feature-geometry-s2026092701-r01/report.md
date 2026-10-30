# DURA源原头可见特征方向与梯度校准

- run_id：`20261030-diagnostic-DURA-head-feature-geometry-s2026092701-r01`
- group_id：`phase1-dura-source-head-feature-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

FrozennoVarE20/source512x2views;fixed originalhead centered normalized rowspace;4component gradients;deterministic alpha calibration;0updates/TX/V/target

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

## 完整源投影诊断与固定系数

VERIFIED/ANALYZED_SOURCE_ONLY。CPU2/0GPU共38.891秒，固定noVarAux源finalE20、512源位置（L128/U384）×cached01/20，32个B32批次/四种梯度完整。原头rank5行空间由冻结权重计算、无拟合/新增参数，实际原头scale30/normalize eps1e-4公式逐批核对。G/F全部state不变、p.gradNone、0优化器/直接TX/V/target。一个独立P0/P1审查PASS，5项聚焦代数/隔离/资格检查PASS，真实source2Ls G/F checkpoint smoke PASS；一次N607 preflight/归档两端SHA/compile通过，执行commit 486c498c28abcbc87cef195c485181d962f1ae90。

PID4021843实际存活、CWD/完整命令/空CUDA binding及日志387bytes/前3批已读回；后续自然结束，日志增至3198bytes/32批，完整结果回收。独立核对两个视图同512唯一位置/各128L384U，并从原始范数/点积复算余弦/比值共2560标量及所有资格条件。

|cached epoch|校准后投影/base梯度中位比|base平均余弦|负余弦批数|投影/task对齐|旧cos/task对齐|
|---|---:|---:|---:|---:|---:|
|1|0.309277|0.154384|0/16|0.991924|0.716185|
|20|0.230957|0.125238|1/16|0.989151|0.612586|

按预定唯一公式alpha=.25/全32批unweighted projected/base梯度范数中位数，得到0.006607891036090233；两视图资格全部PASS，允许另登记从零单阶段E20投影候选。不做权重网格、不放宽条件、不从目标分数挑系数；该原头可见方向在首个Fguide目标分数读取前由源task诊断提出，当前诊断与系数只用源记录。源梯度对齐不能证明真实TX准确率、波形真实性或目标泛化，新训练仍须完整七景。

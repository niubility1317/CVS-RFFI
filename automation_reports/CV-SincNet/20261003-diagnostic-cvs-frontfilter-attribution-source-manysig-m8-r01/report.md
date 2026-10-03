# CVS全主干前置滤波：逐包/训练源均值/恒等输入的固定权重诊断

- run_id：`20261003-diagnostic-cvs-frontfilter-attribution-source-manysig-m8-r01`
- group_id：`cvs-frontfilter-frozen-source-attribution`；类别：`diagnostic`；阶段：`Phase1-source-frozen-frontfilter-attribution`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

定位逐包系数与全局滤波的实际分类作用；每模型先仅用L_s6300包计算固定系数，再对同V27000包三条件配对；不训练、不新增候选、不改变冻结选择。

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

## 固定权重诊断设计

源训练实际提交`dd5518d1493f2f79fb4213e4731e9f88cb28a349`，8份自己的scratch E200 checkpoint已完成并冻结。本诊断保持权重、参数、buffer和原FP32设置，真实checkpoint先做无query公开输入检查，再读取源数据。禁止新增训练、目标访问或改变已冻结的源选择。

每模型先用全部6300条L_s的无标签IQ输出系数，以float64累积均值并转回实际FP32前向系数；保存系数及物理ID后才创建V loader。L与V严格不交，不用V估计替代系数，不读取L标签。固定系数仅用于反事实诊断，不作为新训练策略、方法候选或校准模型。

同一27000条V分别执行：all_on原逐包滤波；mean_L固定L均值系数、学习基与主干不变；g_identity原IQ进入同一训练后主干。2结构×4seed×3条件共24行648000次预测。全开须复现原E200源准确率/最差RX。每包logits与metadata、源L系数均保存至N607原路径；collector独立复算L均值、CE、argmax、混淆矩阵、RX/TX/day单元、预测变化及帮助/损害数。

本实验为source-only机制诊断，无新候选晋级，因此不新增目标测试；原源赢家和其clean完成证据保持不变。分支替代可能偏离训练分布；L均值与V分布不同也会影响差值，不能把结果当作重训练的因果效应或信道恢复证据。

## 本地验证

运行器58项与独立评分器21项，共79个不同检查通过；远端模板编译通过。仅使用合成metadata及公开/随机tensor，未读取真实数据。独立P0/P1审查通过。真实checkpoint无query检查在远端执行器读取源IQ之前完成。

[本地验证](evidence/local_validation.json) · [独立审查](evidence/p0_p1_review.json)。

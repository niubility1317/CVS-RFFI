# DURA单阶段普通与恒等初始化的源梯度规模

- run_id：`20261030-diagnostic-DURA-initial-condition-s2026092701-m2-r01`
- group_id：`phase1-dura-initial-conditioning`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

2untrainedinitialstatesx512sourcex2cachedviews;5componentderivatives/4fixedsums;0optimizer/F/TX/V/target

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

## 完整源结果

VERIFIED/ANALYZED_SOURCE_ONLY。CPU2/0GPU约65.177秒，64个B32批次、16个组合聚合单元；同512源位置128Ls＋384Us×cached01/20×两种未训练初始化。这里epoch指观测缓存，不是模型训练年龄。0优化器、实际clip、F/TX/V/target；所有参数/缓冲区不变且p.grad未积累，没有新权重或部署预测。

唯一P0/P1审查PASS，geometry与checkpoint负测2 PASS；一个方法commit cdcd61f512f297b59116d6af405252ae5b231ddd、一次N607 preflight、单归档两端SHA/compile通过。真实fastLR源checkpoint两Ls noquery smoke通过后删除，未继承到两初始骨架。实际PID3837542存活、CWD/完整命令/空CUDA、CPU2配置与日志712bytes已读回；完成时所属进程不存在，日志7951bytes、64条批事件及complete已回收。

独立读取完整64批记录，核对每组512位置唯一性/四组相同位置、L/U计数、四组合系数重算loss、范数有限/理论clip/clipscale/余弦边界共1024项，并重新聚合16单元。它是独立算术复核，不是再次计算源导数。0IQ/F/目标读取。公开聚合保留null的有效cos数量；原始批记录/opaque位置仅留私有artifact。

|初始化|缓存epoch|组合目标|梯度范数中位|理论裁剪批次比例|平均目标|
|---|---:|---|---:|---:|---:|
|ordinary|1|base|9.770175|100.0%|2.265491|
|ordinary|1|strong_action|9.809444|100.0%|2.561461|
|ordinary|1|no_variance|9.772913|100.0%|2.239231|
|ordinary|1|no_variance_strong_action|9.818529|100.0%|2.535201|
|ordinary|20|base|10.505288|100.0%|2.295513|
|ordinary|20|strong_action|10.613402|100.0%|2.585265|
|ordinary|20|no_variance|10.508661|100.0%|2.271328|
|ordinary|20|no_variance_strong_action|10.616556|100.0%|2.561080|
|identity|1|base|8.235593|100.0%|1.976390|
|identity|1|strong_action|8.198325|100.0%|2.244841|
|identity|1|no_variance|8.235568|100.0%|1.950512|
|identity|1|no_variance_strong_action|8.198300|100.0%|2.218963|
|identity|20|base|8.741077|100.0%|1.985297|
|identity|20|strong_action|8.671889|100.0%|2.249602|
|identity|20|no_variance|8.741069|100.0%|1.961645|
|identity|20|no_variance_strong_action|8.671865|100.0%|2.225950|

预定noVar条件未通过：identity/ordinary梯度范数中位比为0.84269/0.83180，只降低约15.73%/16.82%，未达到两组均至少减半；平均目标虽降低约12.89%/13.63%，不能替代梯度门槛。所有B32初始批次在四种目标下的理论rawnorm均>1。本次不据该诊断发布identity＋noVar组合，不降低门槛。

原始普通初始化中，strongA与base梯度范数中位只相差约0.4%/1.0%，组合方向相对IQ仍高度一致。故不能把当前strongA训练的后期高裁剪简单归因于“初始action梯度占主导”；本诊断未确定多步AdamW轨迹退化的原因。B32理论裁剪与真实B256训练频率不同，初始条件不证明收敛或TX保真。

本初始化诊断在严格F-free范围完成。用户随后已明确授权独立F指导分支，其真实训练与科学权限另行记录；本诊断不因此改变为F指导实验。默认DURA保持，六景各距rawclean3pp目标ACTIVE。

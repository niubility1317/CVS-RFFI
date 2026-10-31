# 物理损失与识别签名在恢复器参数上的梯度分解

- run_id：`20261031-diagnostic-DURA-source-SIG-parameter-s2026092701-r01`
- group_id：`phase1-dura-source-SIG-parameter-contribution`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定原SIG-only G/source47591与原F50，复用原RX均衡840观测/477物理记录；物理4项与cappedSIG参数梯度分解、重放旧总梯度，0优化/TX/V/target。

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


## 固定物理与签名参数梯度分解假设

固定原SIG-only G/source47591 E30/7410、原F50/source4df7/defaulthead及F-only source SIG缓存。完全复用已完成RX-gradient诊断的source_plan与原总梯度参考：两缓存01/20、5RX×7景×12=840次观测，477条物理记录/363重合，L86/U754。旧RX冲突主FAIL保持；无新G父模型、样本选择、训练、TX/V/target或默认版本变化。

同一次G forward产生物理4项与原cappedSIG surrogate，分别对唯一G参数求均值梯度。物理=IQ+0.1prior+0.05STFT+0.05action，variance仍不加权；两分量相加须复现原总梯度，relativeL2≤2e−5。每个RX和raw等RX汇总均报告物理/SIG/total范数、SIG/physical比、分量夹角及总方向与物理方向夹角，并报告跨RX抵消。raw汇总先在float64平均分量，不平均比例或单位方向。

主假设“SIG参数贡献很小”仅在两个视图都满足以下条件时通过：完整G的5RX及等RX汇总物理范数定义；至少4/5 RX的SIG/physical范数比<0.1，且汇总比<0.1。coarse/action/prior仅描述，不改主模块、阈值、视图或父模型。通过只支持以后另外登记参数平衡学习假设，不自动启动G。批均值小梯度可以同时来自Jacobian映射与批内抵消；不能全部归因网络过滤，也不能用范数代替任务重要性或据此断言放大指导会提高TX识别。原SIG六景真实收益保持。

源参数梯度固定当前cap权重，不是AdamW预条件步、全训练轨迹、真实TX/硬件或物理恢复上限。0优化，G/F全state逐张量保持、param.grad为空。主G840/Fprefix840/FinputVJP840/fullF0/GparameterVJP样本口径1680；真实2非clean L_s smoke G2/fullF4/prefix2/FVJP2/GparamVJP4，严格旧G、实际F clean SIG与cache/原头及双分量非零有限检查，PASS立即继续。上游G30/F50/11100与40.32MB缓存原生产63000prefix/0.847077秒成本保留，F-guided，不称F-free。

3项聚焦检查及实际入口/CPU于读数据前拒绝/help/collector物理4权重与SIG拆分、unused补零及不积累grad通过，唯一P0/P1 PASS。一次N607 preflight、单归档SHA比较/远端编译/不可覆盖root/实际PID-CWD-argv-空CUDA-日志增长；独立NumPy重放20个分量向量、旧总梯度、8组含5RX与汇总的统计（48比值/96夹角/16跨RX保留率）和主判定，不新增NN。默认DURA及完整六星地3pp/clean/同质量效率目标保持。

# 原头保护 RX 特征参考的源输出梯度证伪

- run_id：`20261031-diagnostic-DURA-source-F-RXgradient-s2026092701-r01`
- group_id：`phase1-dura-source-RX-canonical-feature-gradient`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定noVar source512×缓存01/20，比较IQ/旧Fcos/保护RX特征目标梯度；唯一权重及实质差异门槛，CPU2/无训练/TX/V/target。

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

固定已通过的源RX几何（aee80ff、fit62488不含512 probe），.25特征参考收缩不变，不是波形相位规范化或部署校正。原noVar cf40源final20仅作2Ls G/F真实smoke；主计算复用其512条source缓存01/20已保存IQ。训练好F冻结，计算输入IQ对IQNMSE、旧cleanF参考cos损失和新RX参考cos损失的三条梯度，参考全部detach、无F/G更新，不读TX/V/target。

唯一lossweight=.25/全32批norm(gCanonical)/norm(gIQ)中位数。两个view各需中位校准压力[.125,.5]、与IQ平均cos>=0、负cos最多8/16；同时新旧Fcos梯度差范数相对旧梯度范数的中位数各至少10%。这项预定实质差异条件用于避免重跑近似旧失败项；任何失败不加大.25特征收缩、不放宽门槛或重选源父模型。通过仅允许后续独立登记单阶段scratchE20训练，不代表参数梯度方向、身份或目标收益。

记录完整1572864个三类梯度值供独立32批重算，并核对IQ闭式梯度。原F50知识和成本如实继承，非F-free，不产生硬伪TX标签或另训头。CPU2/0GPU，F1024主分析输入、G/F各2Ls smoke。默认DURA和逐景3pp/clean/效率目标未完成。


## 完整源端结果

VERIFIED / ANALYZED_SOURCE_ONLY。执行commit `c5a6f6cbb21fa541d5e46f48e1f510875097975f`，8项聚焦检查与本候选唯一P0/P1审查PASS；一次预检、单归档两端SHA与远端compile通过。实际PID947671/CWD/argv/空CUDA已独立读回，17183字节日志先实际2Ls G/F smoke，再完整32批，正常退出。G/F状态不变、F参数无梯度，神经更新/逐样本TX/V/target读取0；1024源访问来自同512物理记录的两个固定视图。

|源缓存视图|中位梯度压力|与IQ平均cos|负cos批数|相对旧Fcos梯度差中位数|资格|
|---|---:|---:|---:|---:|---|
|1|0.239969|0.045729|0/16|42.323090%|PASS|
|20|0.266674|0.042799|0/16|47.811848%|PASS|

唯一lossweight=0.019528750294704578，.25特征收缩未改变。两源视图均过原可行性和至少10%新旧梯度差条件，不是把旧Fcos重命名。完整1572864个梯度值/32批的范数、点积、比例、梯度差、320项统计与IQ闭式公式已独立NumPy64重算，最大IQ梯度差8.13957204e-11，资格一致。计算循环4.062794秒，CPU2/0GPU，模型加载/smoke另计。

source输出IQ梯度平均cos约.04，虽全为正，但方向大部分与IQ恢复梯度不同；不能推为G参数梯度或正式收益。几何与本次校准通过，只支持一个另登记的匹配scratchE20/4940试验：原四项基础损失、同F原头不更新、Fcos的source参考改为固定头保护RX特征、权重用本次唯一值。直接TX/V/target训练0，G无祖先，冻结后自动全七景。原F50源监督知识与50轮成本明确继承，非F-free。默认DURA和逐景3pp/clean/效率目标未完成。

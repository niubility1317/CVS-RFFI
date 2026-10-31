# 接收机场景均衡的恢复器源梯度冲突诊断

- run_id：`20261031-diagnostic-DURA-source-RX-gradients-s2026092701-r01`
- group_id：`phase1-dura-source-receiver-gradient-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定原SIG-only G E30/source47591与原F50；5RX×7scene×12×2缓存，实际局部训练代理参数梯度、等RX均值及模块分解，0模型更新/TX/V/target。

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


## 固定接收机场景均衡梯度假设

固定contextFiLM之前已登记为匹配对照的原SIG-only G/source47591（从零E30/7410，266951参数），原F50/source4df7/defaulthead与F-only SIG缓存保持。此处只读冻结snapshot，不作为新G父权重，不挑选新目标表现更好的父模型。源63000的L/U角色不变；从一份NumPy2026103111排列，在cache01/20各按5RX×7scene×12取420条，共840次访问，实际独立物理数和L/U数报告。不按隐藏TX平衡，不能隔离纯RX硬件因果。

每RX84条，用原IQ1/priorIQ0.1/STFT0.05/actionprobe0.05+SIGalpha1/输出梯度cap0.5的实际训练代理，对唯一G参数计算raw均值梯度。同两个complex probe按CPU2026103112在每RX/视图重置。主比较为全266951参数；coarse/action/prior分项只解释。两个视图均须全部方向定义、10个RX对至少3个cos<−0.05，且至少1个RX与未单位化等RX均值梯度cos<−0.05。不得换主模块、阈值、视图或checkpoint；零主方向记科学未定义/未通过，不当作技术错误。

本次是固定本步cap权重的局部训练代理导数，不是重新计算自适应cap后的全局风险导数、AdamW实际步或全训练轨迹，也不是TX准确率、恢复上限或泛化收益。通过只支持后续另外固定接收机均衡学习假设，不自动启动新G。0优化/直接TX/V/target，无新G版本及正式预测；F指导的上游源监督知识、原G30/F50与40.32MB缓存生产63000prefix/0.847077秒成本保留。

3项聚焦检查、实际入口/CPU拒绝于数据读取前/help/collector均值梯度与unused补0、零梯度处理及不积累param.grad通过，唯一P0/P1 PASS。首步真实2非clean L_s严格G加载、实际完整F clean SIG与缓存、原头及有限非零G梯度，G2/fullF4/prefix2/VJP2；PASS立即继续。主G840/Fprefix840/VJP840/完整F0，G/F全state逐张量不变、param.grad=None。CPU2/空CUDA；独立NumPy重放选样与10个完整梯度向量的参数布局、8组几何（80对角度/40均值角度）与判定，不新增NN反向。

一次N607 preflight，实际源元数据RX为int64 1/3/4/6/8、7景unicode，每RX/景至少1711条，固定每格12容量足够。不可覆盖root、单归档SHA比较、远端编译与实际PID/CWD/argv/空CUDA/日志增长核实。旧G与其他owner保持，默认DURA和六景3pp/clean/效率完整目标保持。


## 完整接收机梯度诊断

VERIFIED / ANALYZED，SOURCE_ONLY。固定原SIG-only G E30/source47591与原F50，CPU2主循环5.791993秒自然结束。5RX×7景×12×缓存01/20，共840次观测/477条物理记录/两视图重合363，L86/U754。整体G梯度中cos<−0.05的RX对为3/10、1/10；5RX与未归一化等RX均值梯度全部同向，最小cos分别0.162204、0.286014，两个视图均无达到阈值的均值冲突。固定双视图主条件FAIL，不从action/prior分项换主，不自动增加接收机重加权训练。G/F全state保持、param.grad为空，0优化/TX/V/target；主G840/Fprefix840/VJP840/fullF0，smoke真实2Ls/G2/fullF4/prefix2/VJP2。完整选样与10梯度向量的8组几何/80对夹角/40均值夹角独立复算，0额外NN。仅是最终冻结点、有限源样本、固定cap的局部训练代理梯度，不代表AdamW/全轨迹/真实TX/硬件或恢复极限；默认和完整目标保持。

详见[完整分析](analysis.md)、results.json。

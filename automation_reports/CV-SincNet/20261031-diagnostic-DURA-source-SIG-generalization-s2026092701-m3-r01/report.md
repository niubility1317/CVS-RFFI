# 冻结特征预测器的源训练与源验证泛化诊断：三模型

- run_id：`20261031-diagnostic-DURA-source-SIG-generalization-s2026092701-m3-r01`
- group_id：`phase1-dura-frozen-SIG-train-V-generalization`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

三固定final30 G的实际条件特征预测，对比原训练512×01/20及固定sourceV1800×七景；分解冻结head可见/正交误差，不读TX/V标签或target、不拟合。

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


## 固定源泛化诊断

本次不训练、不评选模型或替换部署版本。原条件版、深层线性版和深层MLP版均严格加载同源 final30/7410 和自身原配置，使用同一冻结原F50。三个独立子进程顺序运行，避免重复配置污染；CPU2、CUDA空，所有权重、buffers及grad保持。每组首步真实2条非clean L_s smoke检查观察hook不改IQ、实际SIG参照、原默认头和严格checkpoint，PASS直接继续。

训练池固定512条（按角色等距128L/384U）及既有缓存01/20；源验证池固定既有1800条、全部七视图。V只读IQ/ID/RX/day，不读标签，不进入优化、阈值、早停或持久模型状态。两池物理互斥；源验证仍是已见源RX，不是新接收机测试，训练与验证腐蚀抽样也不相同，故本诊断不能单独证明过拟合的因果机制。

观察两个条件层实际收到的160维复合预测（包含深层残差），不修改输入或输出。以同条clean SIG为参照，计算同一能量分母的整体误差、冻结分类头对比方向误差及正交余量；后两者在FP32投影基的舍入误差内相加等于整体误差。head方向只由冻结head归一化行的中心化span给出，不用数据拟合分类器；该指标不是完整分类准确率或纯硬件身份信息。另记录恢复IQ的SIG误差和IQ NMSE。

两条事前固定假设分别报告：H1要求两个深层版在两训练缓存各至少4/6景预测误差降≥10%，而源V至少4/6景没有改善；H2要求两个深层版在源V至少4/6景整体误差降≥10%，但head方向误差至少4/6景没有改善。各计数分别适用，完整逐景数值也保留；不把其中一项改成替代主指标，不按结果换样本或自动训练。

每模型13624条观察（512×2+1800×7），共40872 G前向与40872次恢复SIG前缀；另真实smoke总G12/完整F12。训练SIG原40.32MB缓存及其既有独立证明只读复用；源V clean SIG1800条/1.152MB由第一组在smoke后计算一次，后两组同ID复用，标为diagnostic-only、不能作为训练数据。F50/source4df7/50轮11100步监督知识和原缓存成本披露，保持F-guided。0TX/V标签、0target、0优化或另训头。

3项聚焦检查和3种实际模型/入口/CPU边界/观察hook/IQ逐值一致/异常恢复/state-grad/误差分解检查PASS，唯一独立P0/P1 PASS。输出27份私有NPZ，覆盖318分组cell；独立NumPy复算源角色顺序、实际head投影、误差分解、全部指标和两条判定，0新增NN。默认DURA与其他owner保持，完整六景3pp/clean/同质量训练效率目标仍ACTIVE。


## 完整诊断结果

VERIFIED / ANALYZED_SOURCE_ONLY。三固定G全部完成，共40872源观察/2312物理记录；318分组和27份私有NPZ独立复算通过，0新增NN。两个深层版源V六景整体SIG误差均降低约15.8%至19.5%、head方向误差均降低约18.0%至23.6%；H1与H2均FAIL，即两条预设退步解释未获支持。源V仍是已见源RX，不排除未见RX泛化或证明梯度干扰。V仅冻结诊断、标签/target/训练更新0；默认和完整目标保持。详见[完整分析](analysis.md)。

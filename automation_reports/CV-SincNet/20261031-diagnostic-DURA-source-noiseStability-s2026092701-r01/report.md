# 单阶段恢复器源域噪声敏感性：与默认 DURA 配对诊断

- run_id：`20261031-diagnostic-DURA-source-noiseStability-s2026092701-r01`
- group_id：`phase1-dura-source-noise-stability`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定pureG repeatE30与默认R69/source512×cache01/20，按缓存标量噪声方差生成同一正负高斯扰动，clean不扰动；全模型冻结，只读源IQ/Fclean引用，不读TX/V/target。

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


## 固定源域噪声诊断

冻结 pureG repeatViews E30/8cf 与默认 DURA R69，source512（128L/384U）×缓存01/20。按缓存标量噪声方差生成相同的 ±ε，clean 场景 ε=0，clean 参考与原 received 不变。这是可控各向同性扰动，不是重新调用原生信道或精确复现其完整噪声分布，不据此声称 SNR 精确降低3dB。

固定资格：两个缓存的非clean总体均满足 pureG/default 输出噪声增益比≥1.25，且 pureG 扰动配对平均MSE/native MSE≥1.10，才支持另登记 scratch 单阶段E30噪声增强对照；不自动训练，失败不放宽。噪声增益采用平均输出变化/平均输入扰动的能量比，不用不稳定逐条小分母。F选择与特征指标仅描述，不参与门槛或权重选择。

CPU2、CUDA空；6144 G/F分析输入和首步4 G/F真实2Ls检查，0优化/TX/V/target。源Fclean缓存只读复用，历史F50监督知识与缓存成本披露；两模型状态冻结。全部4份NPZ及role/RX/day/scene/state/lock聚合保存并独立复算。这只是源训练池噪声稳定性，不是真实TX准确率、目标收益、过拟合的因果证明或恢复极限。默认保持，目标ACTIVE。

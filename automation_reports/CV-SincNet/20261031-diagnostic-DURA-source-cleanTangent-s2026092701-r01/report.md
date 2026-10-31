# 源clean局部身份敏感度与缓存训练可行性诊断

- run_id：`20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01`
- group_id：`phase1-dura-source-clean-tangent`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定pure repeat E30源误差，原F50冻结，在源clean计算原头可见5D特征Jacobian，检验固定局部误差梯度是否比普通IQ更贴近exact特征误差；CPU2/0G新前向/0训练TX-V-target。

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


## 固定研究方案

只复用纯repeat E30的512源native IQ×缓存01/20。原F50/defaulthead冻结，源clean处计算原头可见5维归一化特征对IQ的Jacobian，固定二次型惩罚Jclean·(restored−clean)，与restored处exact特征误差梯度及普通IQ梯度比较。无新G加载/前向，无训练TX/V/target。首步2Ls真实F核实非零J/有限输出/batch与逐条J一致，随后立即继续。CPU2/CUDA空，主F输入1536、smoke6，cleanVJP2560、smoke20、restored1024。

事前门槛：缓存01/20都必须512个cos可定义，tangent/exact平均cos≥0.5且比IQ/exact高≥0.1，至少384条非负，各五源RX均值非负。只有全部满足才支持另登记单阶段scratch30，权重固定为0.25除以两view tangent/IQ输入梯度范数比中位数的较大值；失败不放宽，无自动训练。局部近似误差描述性报告，不能作为真实身份准确率或目标收益。估计全63000源J为645.12MB、30次读取19.3536GB，尚无GPU加速结论。

相关[Jacobian知识迁移研究](https://proceedings.mlr.press/v80/srinivas18a.html)仅支持研究导数信息，不证明本IQ度量有效，也不将本方法称为该文复现。4聚焦测试与唯一审查后发布，默认DURA与完整目标保持。

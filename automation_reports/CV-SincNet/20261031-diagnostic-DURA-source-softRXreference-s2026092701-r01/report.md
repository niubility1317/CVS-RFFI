# 冻结 F 源域软坐标跨接收机参考：身份方向几何检查

- run_id：`20261031-diagnostic-DURA-source-softRXreference-s2026092701-r01`
- group_id：`phase1-dura-source-soft-crossRX-reference`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原source63000clean F特征，fit62488排除probe512，按其他RX/day软6坐标等权构造参考，原头与F冻结；只检查几何，不加载G或读TX/V/target。

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


## 固定源特征参考几何检查

只用原F50/defaulthead与source63000clean特征缓存，fit62488排除固定probe512（128L/384U）；不加载G，不读TX/V/target。以原F实际软6维输出为权重，按RX/day/软坐标求均值，其他4RX×3day等权平均，参考为normalize(0.75原unitfeature+0.25软参考)，beta固定。不是硬伪TX或真实TX类别均值，不另训/替换分类头。F虽冻结，但其50轮监督知识及缓存历史成本明确继承。

事先资格：90个fit单元有效N≥32；probe512原头选择全部保留；原头可见空间内RX关联离散度降至≤80%，软坐标间离散度保留≥90%。失败不改beta/门槛；通过只支持另登记源IQ梯度与可达性校准，不自动训练或正式预测。probe排除于参考拟合，但原F曾训练整个源池，不能称F未见样本泛化。

CPU2/OMP-MKL-OPENBLAS2/CUDA空，新增F backbone仅2Ls检查；head-only63000+512，另smokehead2。4几何检查、实际模块入口、真实原头导出回归通过；唯一P0P1的scale字段问题已在远端启动前修复并定点复审。所有软坐标/均值/参考/几何保存私有NPZ并独立复算。默认DURA及完整3pp/clean/效率目标保持。

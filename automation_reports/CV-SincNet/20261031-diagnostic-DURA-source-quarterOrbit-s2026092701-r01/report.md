# 源域四相位恢复与逆旋转 IQ 合成检验

- run_id：`20261031-diagnostic-DURA-source-quarterOrbit-s2026092701-r01`
- group_id：`phase1-dura-source-quarter-orbit`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

默认R69主对照、pureG E30描述控制；source512×clean/01/20，四个直角旋转经同一G后逆旋转IQ平均，原F只读native或合成一个IQ；0TX/V/target/更新。

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


## 固定四相位源诊断

主对象默认DURA R69，pureG repeatE30只作不可替代主对象的描述性对照。固定source512（128L/384U）×fullclean/cache01/20，四个精确直角相位旋转经同一G后逐路逆旋转，再做IQ平均，F只接收native或一个合成IQ。这不去除原始相位，不要求F对相位不敏感，也不放宽旧phaseanchor失败规则。

两个源缓存均要求IQNMSE≤native的0.9、clean-F选择一致数净增≥6/512；各非clean景最多少1条，clean景不损失；完整512条clean输入也不损失。失败不放宽、不改主对象。通过仅支持另登记完整七景正式runtime测试，不自动部署或训练。源F选择不是TX真值准确率。

代数上平均路由误差=合成误差+路由分歧，只能保证优于四路的平均波形误差，不能保证优于原native或改善身份。4倍G输入/共享266951权重/总8prior16CG108FFT成本披露，不能预先称加速。CPU2/CUDA空，12288G+6144F主输入、smoke36G+8F，0TX/V/target/优化；原sourceF缓存与上游监督成本另计。完整六份NPZ和所有分组独立复算，默认与完整目标保持。

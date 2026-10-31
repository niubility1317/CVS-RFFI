# 软跨接收机特征参考的源 IQ 可达性与缓存梯度校准

- run_id：`20261031-diagnostic-DURA-source-softReferenceIQ-s2026092701-r01`
- group_id：`phase1-dura-source-soft-reference-IQ-calibration`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定beta.25软RX参考与alpha.003313，原source512/63k rawJ缓存、pureG8cf源IQ；最小IQ位移真实F检验及两源view梯度对照，0G/训练/TX-V-target。

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


## 固定源 IQ 校准

复用已通过softRX参考（beta0.25）、原raw source J及pureG8cf source512×缓存01/20 IQ，不加载G，不读TX/V/target。alpha保持0.0033130025035573727。参考偏移投影到原J的同一head basis，J本身不旋转或改写。

最小IQ位移由FP64 Gram特征分解的1e-10相对截断伪逆给出，实际FP32可表示位移再检查。固定线性相对残差max≤1e-3，IQ NMSE均值≤0.001/P95≤0.005，真实F相对参考偏移误差中位≤0.5/P95≤1，512个原clean选择保持。两源缓存都要求512个prototype/IQ梯度cos定义，prototype/exact平均≥0.5且比IQ平均高≥0.1、至少384非负、每RX均值非负、实际梯度压力max≤1+2e-5。失败不调beta/alpha/门槛，通过也仅支持另登记scratchjoint30及完整七景。

偏移损失为alpha*s*||Jdelta-offset||²，s每步detach以限制附加IQ输出梯度。它不再具有guide损失值≤IQ的旧性质，也不保证G参数梯度或全局固定能量；会报告损失值超出IQ的记录。源clean最小位移只是可达性参照，不是可部署算法、真实TX准确率或恢复极限。Fsource1536+smoke2、inputVJP1024+smoke2，0G/新完整J/更新；历史F50与缓存成本披露，默认与完整目标保持。

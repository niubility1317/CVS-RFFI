# 中层签名辅助监督：首步梯度检查修正，原30轮七景

- run_id：`20261031-phase1-DURA-midSIG-E30-s2026092701-r02`
- group_id：`phase1-dura-middle-signature-aux-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原midSIG网络/损失/数据/预算完全不变，仅修正smoke非零检查为零初始化频谱输出层及原时域浅中层/投影；r01未开始训练，产物保留。scratch单阶段E30/完整七景。

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


## 唯一技术修正与原矩阵保持

r01在CPU真实2Ls首步检查中断，未创建GPU worker、训练更新或正式预测；controller1726563已终止，全部产物和FAILED登记保留。本地实际G及新增回归均复现：原频谱输出层零初始化，其上游encoder初始梯度为0，spectral.out和时域浅中层/投影已有aux梯度。错误来自首步非零检查点，不来自源数据或训练方法。

本r02仅检查真正首步接收梯度的spectral.out.weight，保留时域浅层/中层/投影检查；失败报告各norm。先红后绿回归＋7项聚焦检查PASS，原唯一审查后对此问题的一次定向复审PASS。网络、权重0.01、全部初值、源缓存、F50、种子、学习率、30轮/7410及七景完整保持；不在旧目录覆盖或重跑。同指纹经此次一次修复若重现，保留并停止盲目重发。

原主体266951＋训练专用连续SIG投影20640，完整G287591；投影挂在实际时域encoder第2层B16×128，经8格池化/Linear128→160回归同源clean F签名，不是分类头，推理不调用但参数/存储计入。保留原四项物理损失及onlineSIG输入梯度cap0.5；aux普通参数梯度不受该在线输出空间cap约束，全G仍clip1。缓存40.32MB只读复用，历史63000F前缀/0.847077秒与F50/11100知识披露，无新缓存或旧G祖先，单阶段从零联合30轮。

首步真实2Ls完成实际源cache/F/aux梯度/onlinecap/全部G模块/原主体初值/严格E0/无推理aux/原生优化输出/原F原头state检查后立即继续。固定最后30轮，自动同卡profile、source1800×7、formal168000×7，每role全部预测固定后独立truth-last；源差也不取消新G正式测试。默认DURA保持，六景各≥137216正确及clean≥142256、R17逐景1pp和低城区70%继续分别报告，目标ACTIVE。

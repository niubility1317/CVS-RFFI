# 保留源稳定正交特征：10维软参考IQ与梯度校准

- run_id：`20261031-diagnostic-DURA-source-softReference10IQ-s2026092701-r01`
- group_id：`phase1-dura-source-extended-reference-IQ-calibration`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原head5/rawJ5不变，从90源RX/day软均值追加5个稳定正交坐标；source512仅新增5维J，再10维IQ可达性及combined/新增五维独立方向门槛。0G/TX-V-target/更新。

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


## 固定源域10维校准

原head5/J5不改，追加90个source RX/day软坐标均值中心化、去head投影后的完整秩5方向。该均值拟合排除512探针；探索算术显示参考位移能量覆盖从19.4824%增至46.6467%，不是新盲测或IQ可达性证明。混合0.25、原alpha与cap1保持。额外5维必须自身通过方向门槛，不由combined平均掩盖。

固定512源样本（128L/384U）、原pureG8cf缓存01/20；新增仅512×5 cleanJ。rank10最小IQ位移的线性相对残差max≤1e-3，IQNMSE均值≤0.001/P95≤0.005，真实F相对偏移误差中位≤0.5/P95≤1，512原clean判决保持。两个view的combined及额外5维各自要求512个prototype/IQ cosine定义、mean≥0.5且≥IQ+0.1、非负≥384、每源RX均值非负；实际combined压力max≤1+2e-5。辅助部分自身不宣称cap；非零偏移不宣称loss值≤IQ。

新F2048+2smoke、输入VJP4608+2smoke，0G加载/前向/优化/TX/V/target、0新完整源J。7测试/实际入口/唯一P0P1通过；首步真实2Ls原F检查后立即继续。失败不搜维数/beta/alpha/门槛，通过仅支持另登记从零单阶段30轮和全七景。默认DURA/当前F及目标保持。

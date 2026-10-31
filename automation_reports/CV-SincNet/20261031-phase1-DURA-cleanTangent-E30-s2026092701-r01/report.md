# 源clean局部身份敏感度缓存：从零单阶段30轮

- run_id：`20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01`
- group_id：`phase1-dura-source-clean-tangent-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原266951恢复器/noVar四损失＋固定alpha.0033130025035573727的clean处head-visible Jacobian二次型；G从零E30/7410，F仅一次source63000缓存，之后0F训练前向；完整七景truth-last。

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


## 最小方法与验收

独立新G从零、单阶段联合30epoch/7410，原266951参数、两prior/四CG/27FFT执行保持。原noVar的IQ1/prior0.1/STFT0.05/action0.05加固定clean局部身份敏感度二次型，alpha=0.0033130025035573727，来自独立源诊断固定公式，不按目标结果改权重。原F50/defaulthead只在源clean处生成5维特征Jacobian，完整source63000缓存645.12MB/315000VJP，F全state保持，随后释放；学生训练不执行F，仍明确F-guided和F既有50轮知识/成本。参考IQ/J detach，不直接读取TX/V/target。

同纯repeat E30对照的源20缓存后01..10、G初始化/CUDAprobe2026092701/npshuffle2026103004、LR/AdamW/B256/clip全部匹配。原F50不是当前Tri-SIG-TIME，默认DURA不改。源判据通过仅支持研究，局部近似误差中位约0.53，不宣称全局准确。7聚焦检查与唯一P0P1后发布，首步2Ls真实G/F梯度、冻结、strictreload smoke PASS后立即继续。GPU4/6/7自动择容量，每卡新增1/总数≤3、显存≥12GB，其他owner不动。

固定final30后自动source1800×7、target168000×7、全部七景预测固定后独立truth-last，逐TX/RX/day及净纠错；源差也不取消正式测试。逐六景≥81.6761904762%且clean保持，原R17每景≤1pp损失与lowurban70%单列。缓存/G训练/全流程/时延显存及完整网络计算共同报告，不能只凭源fit或训练更快晋级。当前PLANNED，完整目标ACTIVE。

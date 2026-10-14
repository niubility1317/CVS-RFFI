# 受控加性证据候选集合：完整源 V 可行性

- run_id：`20261014-phase1-evidence-set-sourceV-manysig-s2026092701-r01`
- group_id：`phase1-evidence-set-source-feasibility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen own final student/EMA, actual additive joint160 and original defaulthead; fixed beta=.5/pi=.9 sourceV27000 proxy with PAIR/oldhardgate, inspect complement sets and S0/S1 half-PA mass signal; no training, U forward, head fitting or target access.

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

## 固定机制与先行否证

实际教师 joint160 为 a+b+c。只在已经学习的 PA/参考增量上考察 z(u,v)=a+(1-u)b+(1-v)c，u,v∈[0,.5]，不称物理接收机反事实。使用原默认 W 的归一化类别差，逐对最大值 d0+.5 relu(-db)+.5 relu(-dc)。所有对手都存在可不落后证据点的 TX 才进入候选集合。各对手的证据点可以不同，因此集合保守且不保证真实 TX。

学生集合损失为 .16×sum_{全 U batch}(eligible×relu(log(.9)-log P(C)))/N；eligible 仅为原硬伪标签接受集合的补集，没有新增学生同意要求。S0 使用原强视图实际特征，S1 在同一次前向上固定削弱一半 PA 增量。辅助 W 与融合 gain detach，内部特征保留图；主 CE 的原默认头/融合 gain 继续联合训练。本诊断冻结所有模型，只测 source V 和 detached 特征副本梯度。

只分析预定 PA ZERO seed01 的最终第50轮学生及自身当前最终EMA，绝不以最终教师回填早期。完整源 V27000 按原物理 PAIR 元数据排序；强视图 noise=.015/Dropout，原 confidence/temporal/strong gate 不改，真 TX 只在集合/门控/质量固定后的统计中使用。无 U IQ、U TX 或 target 访问，无拟合头、模型优化和 EMA 更新。

在读取源结果前固定科学条件：原硬门控补集真实 TX 包含率至少98%，每RX至少95%；两个学生臂各有至少100个未满足的多类集合，所有未满足集合真实 TX 包含率各至少95%；补集至少一半不是全6类。失败则关闭本固定 beta/pi 版本，不通过扫描 beta、pi 或集合损失权重制造梯度。通过也只支持可行性，不能证明隐藏U精度或跨RX收益。

预计额外54,000次V IQ前向，4个无查询零输入 smoke；集合与S0/S1均复用相同前向。首行 smoke 通过立即继续完整诊断。参数/buffer/梯度、V优化步、默认头结构等边界均读回。候选矩阵 B0/B1/S0/S1 尚未启动；对熵移除、集合本身和证据变化分别做配对归因，不追加ORDER或残差SSL训练矩阵。

## 创新边界

[Credal Self-Supervised Learning](https://arxiv.org/abs/2106.11853) 已研究以候选概率分布集合表达伪监督不确定性。集合/不精确监督本身不作为新发明；本研究假设是由 CVS 实际加性证据的受控变化产生类别集合，并检验学生证据变化是否比普通集合监督额外改善泛化。反事实物理意义和真类保障都必须由证据证明。

[Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html) 提供噪声观察之间学习恢复的先例，不证明射频指纹可由重复残差预测获得。观察/周期预测仅保留为独立源可行性支线，先检验遮挡前处理与copy/mean基线，暂不并入本训练。

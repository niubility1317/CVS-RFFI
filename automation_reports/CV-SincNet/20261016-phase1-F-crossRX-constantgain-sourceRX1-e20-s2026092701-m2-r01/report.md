# F＋跨 RX 同 TX：常数观测增益输入核对与 E20 源对照

状态：RUNNING／VERIFIED。真实冻结CP工程输入核对通过，两组scratchE20已启动，尚未完成训练或源身份评分。PA旁路E20已完整源证伪关闭，旧ordinary含PA保留，不改seed/幅度救跑。此为新的包内常数线性观测增益，不沿用已失败的包内缓变斜坡或公共相位。

## 唯一假设及启动首步

唯一训练假设：保留全部旧ordinary与原U，在独立监督槽加入温和常数接收增益x→a·x，能改善未见RX clean并保护practical。范围固定沿用实际旧ordinary amp_min=0.9/amp_max=1.1；不从本次practical或target分数挑幅度。常数正增益是对已接收IQ的线性可逆缩放，不改相对相位/窗/频率，不等于TX PA改变或真实RX替换；不能因此宣称域泛化有效。

实际代码审计：数据loader的RMS在增强前，普通增强出口仅nan_to_num，模型入口pad_crop之后无统一IQ RMS；参考响应、GroupNorm和cosine头可能消除或弱化该变化，不能仅据默认参数判定。因此将输入作用核对并入launcher第一步的真实CP无query smoke，不另建审批。先严格加载自有匹配B00/fixedfinal20，2个合成包检查，然后固定32个合成包（Gaussian/ARcolored/QPSKsmoothed/chirp+noise各8，seed20261016321）×5视图identity/gain.9/gain1.1/两个gain后RMS恢复control，共162实际NN包，0真实sourceIQ/标签/query/更新。捕获实际encoder/Sinc输入、time_down/t2.pool/reference、最终joint160和checkpoint默认头logits，NoExplicitCFO坐标27保持0，参数和buffer不变；不计算合成身份准确率。

固定工程判断：实际Sinc输入功率应为原功率a²（恢复control为1），相对误差≤1e−6；恢复control输入相对L2≤1e−6、normalizedjoint最大相对变化≤1e−4、logits最大绝对变化≤1e−3。至少一个未恢复端点同时满足normalizedjoint变化中位数≥1e−3且默认logits最大变化≥0.01，才认为对当前完整函数有非平凡作用。任一失败写CLOSED_INPUT_EFFECT_ONLY_NO_TRAINING，0真实sourceIQ/0训练，不改范围或合成seed救跑；全部通过则同进程立即进入下述已登记E20，不再等待审批。该非平凡作用只是工程检查，不是源身份保持、Ag收益或模型晋级证据。

真实CP仅来自自有20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01的B00：scratch20/4440、无祖先/父权重或target接触，fixedfinal20，实际四RX L5040/U45360/V21600，已核实来源结论复用，逐字段/strictload检查保留。不会加载该权重初始化后续训练或EMA；训练worker独立从零开始。未用E50/E200或不匹配权重。工程核对只读取checkpoint/来源角色元数据，不读取sourceIQ缓存。

## 固定最小 E20 矩阵（工程通过才运行）

| row | 原始监督 | 共有ordinary | 第三槽 | 预算/自动读出 |
|---|---|---|---|---|
| B00-RX1-s2026092701 | raw.5 | 旧完整ordinary.25，PA保留 | raw.25 | scratchE20/4440，final源clean+6practical |
| B10-RX1-s2026092701 | raw.5 | 完全相同oldordinary.25 | gain(raw).25 | 同预算/同声明视图 |

固定fit RX3/4/6/8×day1/2/3×全部6TX，L5040/U45360/V21600，heldRX1原L1260只在final20读出。U真值隐藏，13L＋7U oldgate/自有EMA、V健康不选checkpoint、最终20轮冻结均保持；不冒充完整L6300或L5580/Ldev720。复用原carrier/VALIDATED_ONCE相应角色过滤，不重建数据。source-only实验不构造target。

F三路SIG_TIME、统一Sinc、joint160、NoExplicitCFO与必要逐包补偿、默认cosine头联合训练不变。三对RDrop alpha1/LS.01、每包六L前向、原raw.5与ordinary.25份额固定，每对用同一已生成IQ；不以新增视图隐式稀释raw。CROSS_RX仍只在原ordinary两次前向time_down/t2.pool，温度.2、最大权重.1、5轮warm/10轮ramp，无新增raw/gain/U对比或一致性损失。新gain对全部128合法L监督，不用置信度过滤。

新增gain为每包/step独立U[.9,1.1]，私有CPU generator seed20261016301+step，native Python/Torch随机流不前进；无TX/RX依赖、RMS重归一化、相位/CFO/噪声/multipath/PA、边界拼接或As组合。原ordinary原生min(epoch,label_epochs13)日程及U完全不改。训练后两row用相同held物理样本和已有固定source practical抽样，完整逐场景报告。

## 固定科学判断、成本及停止

只有全部同时通过才支持独立源确认：B10−B00 heldRX1 clean准确率≥+1pp、clean MacroF1≥0、每个六practical准确率≥−.5pp、seenV≥−.5pp；每row E20/4440更新/3409920LNN、3330非零SupCon/13320层位前向项、最终LCE≤.5、B10 gain输出568320/对照0、增益a²功率关系误差≤1e−6、0As训练/头拟合/target。任一失败关闭固定gain训练候选，不改range/seed/RX/阈值救跑。工程作用、源fit敏感性、loss下降或健康不等于真实身份收益；单seed源PASS不等于Ag泛化或联合目标。

首步162合成NN单列；工程失败仅162NN/0真实IQ/0更新。若工程通过，两模型共40模型轮/8880更新，各3409920LNN另实际原U/teacher/V/2smoke/8820held。此前同配方每模型4686678NN，当前只用实际计数；若完全一致，总NN含工程162为9373518。B10生成568320gain IQ，B00第三槽复用raw；两组额外NN视图数相同，旧ordinary/PA工作相同。局部验证2474合成NN/2合成更新/0真实CP和IQ，与正式预算分开。

唯一owner codex/root/F-CROSS_RX-constantgain-E20-20261016，普通N607账号，CPUdispatcher2线程；source worker由既有共享容量锁分配，所有owner每GPU最多3进程、启动前≥12000MiB空闲，健康任务不改/不停/不重启。CWD为PROJECT/releases/cvs_F_crossRX_constantgain_sourceRX1_e20_20261016_r01，命令CVS-RFFI/bin/python -u experiments/cvs_constant_gain20/runner.py；worker带固定--worker row_id。精确逐行config/六类seed/路径见experiment.json，Git实际commit及release_commit/resolved标识代码。一个不可覆盖root、一次N607 preflight/归档单次SHA比对/远端compile；启动后独立PID/CWD/argv/GPU/log增长读回。

角色/来源/初始化/非有限/实际预算或合法prediction失败为技术失败，保留产物、停止pending、不干预健康worker，无自动重试；科学输入无效或源条件失败按上文关闭，不延长预算。工程结果或全部源预测完成后独立复算相应完整产物及原判断；hard source predictions独立复算accuracy/F1/CM/repair-break，运行时meanCE仅引用，不冒称从hard predictions重算CE。源码/配置/登记/小汇总入Git，IQ/CP/逐包预测、features/traces/原始日志私有。

一次限定P0/P1审查PASS，14负例及B00损失/头梯度精确对照、同初始化/六前向/同IQ/SupCon/默认头联合更新/export160、私有gain随机流与实际untrainedF输入路径和无效果/恢复反例检查PASS。真实冻结CP工程判断仍待运行，更无训练/正式识别收益。


## 实际发布与输入作用结果

执行commit `1d29887ef68319fe9a75b2c82fca30a7860dbf8e`；本地与远端分支OID独立一致。archive单次比对、远端compile及真实匹配CP smoke PASS，dispatcher和两worker的PID/CWD/argv/GPU与日志增长VERIFIED。实际seed2026092701、20/13/7、sourcefold RX1、无baseline/teacher来源均一致；Ag仅B10，0As。

162合成NN/0真实sourceIQ/0更新输入核对及独立NumPy完整5×32 joint/logits/power复算PASS。gain0.9/1.1的normalizedjoint变化中位数分别0.089504101/0.076122214，默认logits最大绝对变化2.043222874/1.380674869；RMS恢复control在固定容差内回到identity。该结果只证实实际模型对常数增益有作用，不证明标签保持、跨RX识别收益或Ag有效。详情见input_function_summary.json。

- B00-RX1-s2026092701：GPU0，已观察完整step427/4440、E2/20；日志11693字节。
- B10-RX1-s2026092701：GPU1，已观察完整step439/4440、E2/20；日志11693字节。

上述数字是运行进度；全部final源预测完成后再独立评分，不改变已发布配方。联合目标未完成。

# 单 As E30 技术补跑：准备状态

状态：DRAFT_NOT_LAUNCHED。额外训练预算问题仍待用户明确回复；没有上传归档、创建远端补跑输出根或启动训练。自动 goal continuation 不视为预算答复。当前目标仍按2026-10-18最新“星地优先、clean下降≤0.1个百分点”执行，不恢复旧Ag独立资格要求。

原矩阵的B00已完成、B01技术失败，当前没有新As配对成绩。原完整三seed As结果及阶段判断保持：urban均值+22.307937pp、low+17.284921pp、clean−0.321032pp；clean上限不满足，不能声称稳定双提升。

## 唯一待检验假设与精确范围

只补齐原连续E30优化轨迹假设，不改变物理增强、比例、加入时机、CE/RDrop/SupCon或旧U机制。新拟议run为 20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01；拟议release为 cvs_FcrossRX_continuous_recovery_E30_20261018_r01。实际部署代码仍使用现有 experiments/cvs_continuous52/runner.py 的默认dispatcher，不能绕开smoke直接调用worker。预算获准后才适配并发布design/dispatch/report/publish的最小路由变更；当前草案不是已可启动的新release。

新增训练仅B01-s2026092701：scratch E30、13L＋17U、6660更新、epoch/step offset0、parent=None、checkpoint_sources=[]、segment20仅legacy scratch附着键。原连续AdamW/cosine30、学习率2e−4至1e−6、WD1e−4不变。L=0.5RD(ordinary128)+0.5RD(mixed128)+0.25RD(raw128)，原SupCon另计；实际6×128前向/步、5114880 L前向。全部合法As参加标签CE，无新增clean→satellite对齐、特征保护或U星地损失。

固定复用No-As控制：20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01/B00-s2026092701。其原config/run_id、checkpoint、源读出、七视图prediction和独立scoring保留原根，不能复制到新根伪装新行，不再训练或选择性重测。旧staged P43 B01参照和P52原source判据保持；新0.1pp只补充冻结后报告，不用No-As的已知正式成绩改参数。

## 已完成的准备与验证

只读核实control completion/source_frozen/initialization/result/prediction/scoring和既有source_contract句柄：scratch、无checkpoint来源/祖先/target contact，FINAL30/6660；role_ids与原source相同，L6300/U56700 TX隐藏/V27000，全部5源RX；channel声明与失败As相同。未读取checkpoint、IQ或新的正式分数。

11项stdlib检查PASS（含E30信道边界、NoAs条件安装、tensor_iq原桥、动态单/双行smoke成本）。将既有合成梯度smoke从E7改到声明末步E30/6660，不增加每行前向数。N607实际执行两行合成hook PASS：同206项初始state、参数900722、每行768NN，B00 As0、B01 As64，head/Sinc有限非零梯度、六前向与1.25损失分解一致，0优化/头拟合更新。此次额外验证成本单列为1536合成NN、0真实IQ、0checkpoint、0正式分数；它不属于已结束原训练的NN，也不是补跑获准或训练已启动的证据。

smoke成本由实际records派生：两行1536合成＋2真实checkpoint=1538；将来若只探测单As，则768＋2=770。当前仍保留两行匹配初始化探测。实际checkpoint smoke只在未来获得预算、发布后的launcher首步运行，PASS立即进入已获准训练；本轮没有另做真实checkpoint smoke。

只读代码定位确认：保留rows为配对矩阵，train_rows仅B01，row_root明确路由原B00/新B01；只允许新B01 worker/predict/score；报告按style选连续解析器、按reused统计成本，不因复用B00误用staged解析器。旧四行分析常量不能直接指向新run，必须在新分析输出根处理；不会热改已失败分析。以上实现仅在预算明确后落地，不额外铺发布或权限系统。

## 成本与未完成项

额外需要一个从零As E30，不是补10轮。若获准并完成：匹配对照为30＋30=60模型epoch；原失败20轮另计，实际总支出80个完整模型epoch及17760次成功更新，失败步部分成本仍单列。不能丢弃失败成本、把旧No-As算新训练，或将四行当成四seed。

原失败As没有中间权重，不能精确续训；不加载No-As权重来改做微调。当前未完成额外预算确认、补跑调度路由实现、发布、训练、As冻结后七视图评分、匹配完整分层和0.1pp判断。源诊断/科学失败不授权改阈值救援，正式成绩不回流晋级。

公开配置草案：configs/cvs_continuous52_recovery_draft_20261018.json；实际完整候选配置草案和只读audit保存在本地private artifacts，不作为授权、签名或启动gate。新增前向验证及原失败完整曲线均保留。

## 另一对话的当前实际路线

再次只读核实“审计并设计CVS联合增强”及其 satelliteDropout E30 登记：当前新增AS_CE归因控制、AS_DET（只绕过实际satellite行的dropout，保留ordinary/raw/U，仍生成原mask保持RNG），均scratch own20＋10、原P43 staged NoAs/As-RD只读复用。实际run为20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01。它们不是连续E30 As-RDrop，不能代替本补跑的同轨迹As，也不应在本对话重复启动。当前只有训练/分析进展，未读取新正式分数或宣称增益。

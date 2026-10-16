# 普通 AWGN／multipath：完整源因果对照

**ANALYZED／VERIFIED，主候选未通过固定 raw 身份保护条件，关闭。** 移除旧通用信道损伤可以恢复部分默认头错误，尚不足以提供安全的新增强视图，不进入此候选 E20；两个单项移除只用于归因，不择优晋级。

[原始预登记](preregistration.md)与[唯一矩阵／来源](experiment.json)保持。两个既有合法 scratch E20 final20 F＋跨 RX 同 TX 模型，3 SIG_TIME、统一 Sinc、joint160、NoExplicitCFO、原 cosine30 默认头冻结，无另训头。只用其原训练角色 fitL5040，RX3/4/6/8×day1/2/3×6TX、72格各70包。raw 99.841%是 source-fit，不能称跨 RX clean 成绩；四抽样重复同一5040物理包，不当作独立新数据。

## 实际识别与固定判定

| 模型 seed | raw % | full % | no_AWGN % | no_MP % | no_both % |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.841270 | 92.450397 | 92.495040 | 93.928571 | 93.973214 |
| 2026092702 | 99.841270 | 92.390873 | 92.420635 | 93.849206 | 93.878968 |

| seed | 联合移除−full accuracy pp | macro-F1 pp | 联合移除−raw accuracy pp | raw正确保留 % | 相对full修复／新错 |
|---|---:|---:|---:|---:|---|
| 2026092701 | +1.522817 | +1.522024 | -5.868056 | 94.102742 | 334/27 |
| 2026092702 | +1.488095 | +1.476099 | -5.962302 | 94.023251 | 335/35 |

两个模型四次抽样的联合移除增益均为正，full相对修复和分层保护通过；raw相对accuracy/F1≥−0.5pp、raw正确保留≥99%以及每RX/TX/day相对raw≥−1pp均未通过。两模型并非只因微小边界超限而失败。冻结默认头下的raw保护失败不能证明该视图一定没有训练价值，但足以按预登记停止本固定路线，不能事后移除保护条件。

| 固定条件 | seed01 | seed02 |
|---|---|---|
| repair_full_accuracy | PASS | PASS |
| repair_full_macroF1 | PASS | PASS |
| positive_draws | PASS | PASS |
| every_draw_protected | PASS | PASS |
| raw_accuracy | FAIL | FAIL |
| raw_macroF1 | FAIL | FAIL |
| raw_correct_retention | FAIL | FAIL |
| every_RX_vs_raw | FAIL | FAIL |
| every_RX_vs_full | PASS | PASS |
| every_TX_vs_raw | FAIL | FAIL |
| every_TX_vs_full | PASS | PASS |
| every_day_vs_raw | FAIL | FAIL |
| every_day_vs_full | PASS | PASS |

| seed | draw1 pp | draw2 pp | draw3 pp | draw4 pp |
|---|---:|---:|---:|---:|
| 2026092701 | +1.488095 | +3.849206 | +0.138889 | +0.615079 |
| 2026092702 | +1.388889 | +3.769841 | +0.138889 | +0.654762 |

| seed | no_AWGN−full pp | no_MP−full pp | no_both−full pp | 源四格accuracy交互pp | macro-F1交互pp |
|---|---:|---:|---:|---:|---:|
| 2026092701 | +0.044643 | +1.478175 | +1.522817 | +0.000000000 | -0.000258211 |
| 2026092702 | +0.029762 | +1.458333 | +1.488095 | +0.000000000 | +0.000027854 |

在此固定观察中，多径移除恢复约1.46至1.48pp，噪声移除只有0.030至0.045pp；这是源默认头的既定链路总效应，不是 E20 训练收益。每模型160个逻辑batch，MP命中6、AWGN命中17、同时命中0、两者都未命中137。实际MP涉及768观察、AWGN2096观察，门不是逐包独立抽样。**没有同batch双门覆盖**，因此约0的accuracy交互项不能证明物理上无交互或可交换；macro-F1的微小交互还包含度量非线性。不为补齐交互追加抽样或改命中概率。本源四格也不是最终A_g／A_s四格。

## 真实接线与因果范围

native普通顺序：anti-shortcut／移位、增益、相位、CFO、Wiener相位噪声／multipath／AWGN／PA。使用实际checkpoint args和各自原不可变后端，configure_augmentor_for_epoch有效E13、原min(epoch,13)日程；MP门0.0451080601、2至4 taps、整数延时0至1、左reflect后conv1d，AWGN门0.1002401336、各包SNR20至36dB，PA逐包门0.03508404675。保留PA和其他全部旧变换、DAC/class-signature关闭，不恢复RMS、不夹增益、不按RX/TX/day/置信度分支。

每格重置同一原生起始Python／Torch随机状态；被移除原语仍完整执行以消耗相同抽样，随后返回其输入。原语参数、前后RNG、PA掩码与强度一致，结束仅推进一次full的随机流。私有generator从原语起始状态重建MP taps/delays/组合核供记录，不消耗增强流；原模块函数逐格恢复。全样本记录了IQ／功率／参考窗变化，不要求所有特征或参考质量都等于raw。

移除MP会改变后续AWGN按输入功率计算的sigma，并改变后续PA输入。因此设计测量总链效应；没有固定加性噪声或固定PA输出的直接效应。实际源随机流未同batch双命中，功率依赖由本地强制双门的合成检查验证，本轮源数据不能再提供联合触发的独立实证。

既有As直接读取raw，不接ordinary；源码已经否定同一视图把ordinary AWGN/MP再叠完整practical的猜测。本轮没有As生成、target预测或正式truth评分，不改测试received IQ、信道或SNR。

## 完整复算与成本

独立reader没有调用本候选的score/qualify/paired/model：复算171360硬判决及logit argmax、标量logsumexp CE、accuracy、六类macro-F1和混淆矩阵、RX/TX/day、72格、四抽样、raw正确集合、相对full和raw的修复／新错、全部原条件和源四格交互项。320逻辑增强batch完整（每draw39×128＋48），每格原语参数/门/核归一化与合并、返回功率、后续噪声功率、PA、IQ与参考变化和实际计数核对一致。两个模型的全部batch记录相同。

raw与full的全部logits和hard预测还与上一RMS研究中的对应未恢复原ordinary结果逐项完全一致；最大logit差为0，确认新观察包装没有改变full。没有重新执行旧研究、传回原IQ或权重；模型参数/buffer保持和grad为空由原运行断言记录，不冒称另一次权重字节比较。

实际171362 NN＝171360源默认头＋首模型一次真实CP合成2包smoke；161280源ordinary生成、另320合成canonical ordinary、2 checkpoint加载。源MP原语生成6144包／实际施加3072包，源AWGN原语生成16768包／实际施加8384包，丢弃的影子计算没有被写成零成本；私有kernel原语重放456随机标量。源PA每模型每格710观察，四格两模型共5680次PA包处理；不是新增独立物理样本。新epoch／优化器／教师／另训头／target均0。两个既有参照历史40model-E／8880更新／9373356 NN单列，未续训。

原缓存L6300、未用源V／历史特征反序列化；RX1原L1260在推断前排除，U只有角色ID，没有IQ/TX标签。没有heldRX1／U／V推断。L5040原训练契约不冒充完整L6300或L_train5580/L_dev720。

本地17输入/配置负例＋7科学反例、480合成ordinary／72kernel随机标量通过，0源IQ/CP/NN/更新；旧Torch NumPy桥未使用、不装包。预检缺字段的P1在发布前发现、复现、修复并定向复审PASS，未形成真实失败run。真实CP smoke为首worker第一项，PASS立即继续；160合成ordinary/model的canonical输出/RNG对照通过。

一次归档传输SHA比较、remote compile均VERIFIED；每个worker均独立观察到真实PID/CWD/argv和CUDA隐藏，日志在live至终态观察间增长，不声称持续live采样。最终dispatcher与两worker自然退出，无技术failure、重启或其他进程干预。worker主体56.122879／60.657599秒，顺序主体合计116.780477秒；总调度墙钟未单独测量。分析／绘图0源IQ/NN/更新；PNG/PDF已生成且PNG实际查看。

执行commit 3c76c7ebdbad7f2ea6fa2daa6ac87a2aa108f759。聚合见[source_analysis.json](source_analysis.json)，执行范围见[execution_summary.json](execution_summary.json)。图：[源因果对照与保护条件](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_ordinary_channel_source_20261017/final/figures/ordinary_channel_source.png)；同目录PDF保留。原始logits、逐包观察和物理ID／完整日志仅保留在 local_artifacts/F_ordinary_channel_source_20261017/final，未提交Git。

## 判定与下一唯一假设

FAIL_FIXED_JOINT_ORDINARY_CHANNEL_REMOVAL_RAW_IDENTITY_GUARD_CLOSED。关闭固定联合移除旧ordinary AWGN/MP方案，不选单项、PA条件、系数、clip、抽样、RX或seed救跑，不启动该方案E20，不复活RMS。证实部分源身份错误与旧MP处理有关，否定联合移除就足够满足源身份保护；没有证明所有MP训练无价值或普通增强整体应被删除。

下一单一候选是保留公共旧ordinary，在raw上额外加入孤立轻度白噪声观察，沿用既有20至36dB名义SNR；不叠MP/PA/时移等变化，不修改As。源先验证身份保持，不能只凭随机扰动或文献总增益称DG有效。候选与放弃条件见[下一说明](../../../analysis/cvs_isolated_awgn_candidate_20261017.md)，尚未实现／登记／启动。

[OpenRFI（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/download/32003/34158)的预训练采用轻度AWGN并组合片段重排；其总成绩不能代替本协议下孤立AWGN的跨RX证据。本次不采用其重排、原型或损失。最终仍须同预算E20匹配控制、独立确认及完整契约Ag/As四格。

联合目标ACTIVE：有效新增Ag、稳定As、完整契约四格和交互项、跨seed、正式clean＋六practical、lowurban>60%和suburban距自身clean≤2pp探索目标尚未完成。基线F＋跨RX同TX／原默认头联合训练／E20保持。

# 四周期IQ重排：完整源域反证

状态：ANALYZED／VERIFIED。两个冻结学生默认头、全部25204NN和25200个IQ视图自然完成；接收后重排 Ag(As(x)) 未通过urban保护，固定后置候选关闭。没有新训练、正式测试、模型更新或分类头拟合。

| 源视图 | B00原输入 | B00重排 | B01原输入 | B01重排 | B01差值 |
|---|---:|---:|---:|---:|---:|
| clean | 98.388889% | 98.277778% | 97.722222% | 97.611111% | -0.111111 pp |
| practical_low_suburban | 53.638889% | 51.305556% | 90.444444% | 85.972222% | -4.472222 pp |
| practical_mid_urban | 46.333333% | 45.638889% | 76.305556% | 72.666667% | -3.638889 pp |
| practical_low_urban | 38.111111% | 37.194444% | 62.555556% | 59.916667% | -2.638889 pp |

本表是源V1800，satellite合并两个信道draw共3600观察；一个modelseed2026092701，不是正式目标成绩或跨seed。B01源low59.916667%不替代正式B01 low53.137500%/clean82.797024%，本轮没有正式测试或训练增益。

B00与B01的clean均−0.111111pp，固定clean正确保留99.717674%/99.658897%；两个模型均通过clean/RX/TX/周期相干/非平凡变化等源保护。B01四个urban单元（draw0mid/low；draw1mid/low）为−5.055556/−2.500000；−2.222222/−2.777778pp，全部违反≥−.5pp的保护线。因此不晋级该后置方法，不以B00通过或clean保持替代联合条件，不启动E30训练或扩seed。

## 实际损伤与解释边界

low LOS/locked的1060观察：96.037736%→91.792453%，−4.245283pp，修复7/新增52；BLOCKED/unlocked的1565观察：34.952077%→33.993610%，−0.958466pp，修复97/新增112；新增噪声比<3dB的1127观察：31.499556%→30.789707%，−0.709849pp。没有哪类困难退化得到本候选的有效缓解，较好观测亦受损。新增噪声比不是总SNR，状态与质量沿用原完整native观测，未改SNR或删难包。

实际输入变化：clean相干度q10=.844132、median=.994608，IQ相对L2 median=4.616507%；low suburban median变化28.319904/28.628357%，midurban31.803551/31.474307%，lowurban37.333910/37.489012%。功率最大相对误差2.384186e−7，所有相位均可定义，fallback0；全部24排列及完整物理ID/命名流重放一致，两个模型看到的所有IQ变换记录逐项相同。

五个拼接位置的平均跳变能量after/before：clean1.019125；low suburban1.651932/1.688637；midurban1.352664/1.375752；lowurban1.240323/1.237164。原窗口外样点精确不变和总功率保持，并不能保证参考窗内的时序关系、周期交界或默认头判断不受损。不能仅由这些关联分离相位估计、周期次序、TX动态或接缝各自的因果作用。

当前唯一值得独立核查的后续假设是算子顺序：固定同一个Ag、同一排列和同一潜在practical随机流，As(Ag(x))是否能保持原As(x)，而Ag(As(x))已明确失败。现有native残余核62taps/数值延迟24及新增噪声/接收处理意味着同索引窗口不应被假定为同一底层周期观察，这是一项代码与源变化支持的推断，尚未做前置实验。最小后续仅源V1800/两draw/三场景，原生原输入重放对齐后比较两种顺序；不改窗口、周期、排列群、phase估计、概率、物理尺度、seed或预算。不根据本轮目标成绩选方案，本轮target访问0。若前置仍在任一urban单元下降>.5pp，放弃该组合路径；通过也只支持下一匹配E30效果检验，不能证明Ag泛化。该后续尚未登记或启动，不与本次失败产物混并。

## 实现、测试与成本

实际执行commit `7e8dc09d47571e3e99a241d1ba39672d5bba70f6`；实现experiments/cvs_fullsource_cycleorder30/，不修改原F/三SIG_TIME/Sinc/joint160/默认cos头/NoExplicitCFO或内部补偿。新算子只用于源反证。原全5RX、L6300/U56700hidden/V27000合同保持，模型由同契约scratchFINAL30严格加载，V1800用于当前计算；缓存可能反序列化未用L/旧特征，U IQ/真值及target不消费。

4项本地合成测试和语法编译PASS，0真实IQ/CP/NN；全部24排列的已知逆/理想周期/功率/支持外/相位及正增益等变/私有RNG/非法权限和数值门槛反例通过。一次限定P0/P1静态审查PASS。普通账户CPU2预检、单归档单次传输比较、remotecompile、2真实CP/4合成包no-query smoke、PID/startticks/CWD/argv/空CUDA/CPU环境及stdout1068→2137字节增长VERIFIED；同一PID自然结束、completion完整、stderr空，0技术重试或健康任务修改。

实际新增每模型12602NN/12600IQ视图，总25204NN/25200IQ视图，墙钟16.265193s；训练、optimizer/EMA更新、teacher、新channeldraw、头拟合均0。两父各scratchE30/6660/5407922NN另列，合计60modelE/13320更新；本轮预算不冒充新训练或收敛。

独立分析analysis/cvs_cycle_order_source_E30_20261017.py从保存logits/源labels/physics/操作trace重算220个CM、96条分层、25200个私有排列与成本/源判据，并核对两模型实际波形条件完全相同；0新IQ/CP/NN/更新，分析墙钟1.569539s。图表已取回并目视检查PNG。图清单原先沿用旧PNG名，已仅修正metadata指向实际PNG/PDF，0重推理/重评分，原数据不变。

本地产物目录：`E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_sourceV_E30_20261017/analysis_r01`，实际PNG为cycle_order_source_comparison.png，PDF为permuted_source_comparison.pdf，另含summary/completion、pooled_results.csv、source_strata.csv。根目录有实际run completion/smoke/data_roles、单归档发布及两次进程读回；全部logits/逐包q与排列保留在不可变远端run，未入Git。无新训练曲线，原父E30曲线继续保留。

本轮完成真实IQ候选的源反证，否定固定后置四周期重排；没有证明新Ag、crossRX clean净收益、正式low>60%、suburban接近clean或跨seed联合提升，整体目标ACTIVE。源码与配置已本地提交；GitHub连接失败，未由push退出码推定成功，最终远端OID读回另记。

## 原预登记

# 完整源E30：保留线性相位进展的四周期IQ重排

状态：REGISTERED，未启动。上一轮FINAL30自身EMA没有恢复clean和urban的共同收益；目标仍ACTIVE。本轮回到真实IQ候选，不增加loss、门控或部署分支。唯一矩阵、成本和路径由experiment.json维护。

## 假设、来源与反例

唯一假设：当前CVS可能依赖参考窗中偶然的逐周期观测次序；只重排同一物理包四个已观测周期、保留线性相位进展和总功率，可生成非平凡且仍可辨识的IQ视图。它是待证伪的观测增强，不等于更换真实RX或已证明A_g。

先检索FcrossRX和F-crossRX两种命名：旧镜像子空间是架构分支，并非整体IQ反射。原始[RFFI增强论文III-A](https://arxiv.org/html/2304.14795)明确警告共轭翻转会改变硬件相位偏差，因此整体反射候选放弃，0IQ/NN/训练。论文III-B讨论片段重排，但也说明效果依赖符号随机性，连续重复片段可能使其很弱；不能以论文当作本数据标签保持或性能证据。

历史四fitRX/L5040的源repeatcorr（不等于本轮五RX/V1800）固定[80:160]四个20点周期的q10相干度0.803835，支持该窗口具有重复性；RX3残差功率中位数10.275181%，原有色噪声先验因此FAIL且保持关闭。本候选不提取/拟合/注入该残差，只对原始观测作已知重排；TX瞬态和残余相位仍可能被扰动，必须做真实默认头保护检验，不先称其为纯噪声。

## 唯一IQ操作与源判据

对复IQ的[80:160]四个20点周期，用全部三个lag20对得到单位相位q；对四周期的完整24排列均匀私有抽样pi，输出第k周期为原第pi(k)周期乘q^(k-pi(k))。其它176点精确不变；没有时间插值、循环包移位、裁切、遮挡、噪声、PA、DAC、RMS或功率恢复。q相位不可定义时保留该包并记录identity fallback，不删包。只用opaqueID与视图命名空间/seed202610173001，不使用TX/RX/day/置信度定义操作。保留未知TX硬件效应，不拟合或反演TX硬件；已知观测排列的逆仅供合成验证。

两个既有全源scratchE30/6660的B00/B01终轮学生默认头冻结，原F/CROSS_RX/三SIG_TIME/统一Sinc/joint160/cos30/NoExplicitCFO和内部补偿不变；不加载EMA或另训头。严格own_model来源/原L6300/U56700hidden/V27000角色检查后，先各2合成包真实CP no-query smoke，再打开原源缓存。全部RX1/3/4/6/8和day1/2/3保留；只消费既有V1800和P24两draw×low_suburban/mid_urban/low_urban的full condition0，所有观察均参与。模型seed只有2026092701。

该操作放在已接收IQ之后；完整native residual/post_sync/noEQ即G1/ref30/reflect/fs25MHz/fc2.45GHz/quality/lock/phase/IQcomp都已按原依赖生成，不重新抽物理状态或改SNR。源和正式测试随机流分离；本轮不打开target。源缓存反序列化会带入未用L/旧特征，仅V1800参与，U IQ和TX真值不打开。

两个模型的clean都要求：accuracy相对原学生≥−.5pp，每个5RX和6TX各≥−1pp，固定原clean正确集合保留≥99%，clean周期相干q10≥.8，IQ相对L2变化median≥.01，全部clean相位定义有效。B01还要求两个urban×两个draw每单元accuracy≥原学生−.5pp。数值容差1e−10pp，不改变科学阈值。源保护失败即关闭固定窗口/周期/重排群/phasefit/概率/RX/seed/门槛/预算救跑；源FAIL也完成全部模型和视图。PASS只能支持后续另行登记的匹配E30效果检验，不是A_g成立，不自动访问target或扩seed。

保存每包排列、q、相干度、L2、功率误差、所有5个拼接位置(80/100/120/140/160)的跳变能量前后，报告所有状态/质量层；功率守恒不意味着接缝无损。默认头accuracy/macroF1/CM、margin÷30、repair/newerror全部保留。学生参照复用P26同ID/draw/场景/CPU2/batch90缓存，无额外学生参照NN。

## 执行与成本

每模型12602NN（2smoke＋1800clean＋10800sat）和12600个IQ视图，总25204NN/25200视图；0训练/头拟合/EMA更新/新channeldraw/teacher。父两个模型各E30/6660/5407922NN单列，不能把原训练成本当作新增预算或忽略。普通N607账户、CPU2、空CUDA、一个不可覆盖run/output root；命令/CWD/精确CP/句柄见experiment.json。一次归档单次传输SHA对比、一次remotecompile、首步实际CP smoke、一次独立PID/CWD/argv/CPU/loggrowth或自然终态读回。只停本run的来源/角色/非有限/support/power/state/成本/输出冲突等技术错误，保留产物不自动重试，不动健康旧任务。

本地4项合成测试PASS：全部24排列的已知逆/功率/支持外、理想周期及相位进展、全局相位/正增益等变、私有RNG与非法source/判据反例。0真实IQ/CP/NN/训练。Torch旧环境的NumPy桥ABI警告保留，测试不使用该桥，未装包；远端接口通过frombuffer/tolist。预检普通账户CPU2/输出无冲突VERIFIED；唯一限定P0/P1静态审查PASS，无直接P0/P1；未读取IQ/CP/target或执行远端，不加额外gate。

整体目标未完成：有效跨RX A_g、正式lowurban>60%、clean净正收益、suburban接近clean和跨seed联合证据均未证明。不会把source保护或变换非零当作识别收益。

Git交付读回：此前推送失败已恢复。提交1e3065bfa52e557c4fe3241bcd0500d13d986403已由GitHub独立compare确认与远端分支identical，包含本轮7e8dc09及上轮317b5d75/14ffb1c80。源engine配置引用已校正为原P24 observations/actual_engine.json并只读核实存在；0IQ/NN/模型重跑。

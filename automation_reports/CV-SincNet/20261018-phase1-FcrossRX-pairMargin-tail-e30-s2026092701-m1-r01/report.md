# P61：去课程后独立检验单侧默认头间隔

状态：**ANALYZED／VERIFIED，missing=[]**。训练、全部七环境预测、独立评分和全量分析已完成。源判据FAIL，关闭本复验，不扫系数、不扩种或延E50。完整结论见 [最终分析](final_analysis.md)。以下预登记及执行阶段记录保留其当时口径。唯一 run 为 `20261018-phase1-FcrossRX-pairMargin-tail-e30-s2026092701-m1-r01`；前一目标轮属于 PROGRESS，已完成 P62 两行 E50、全部评价与负结果交付。目标仍未完成。

当前最值得保留的是完整 practical、充足 clean 监督和默认头联合更新。P62 的 E50 源判据 FAIL，本轮不延续 DET、不使用其正式成绩选择新系数。另一聊天“审计并设计 CVS 联合增强”已在做 Asweight .125／.375 的 E50 与02／03复现，本线程不重复该路线，不干预其健康任务。

唯一假设：旧 C3 的阴性结果是否受状态课程和旧监督组织混杂；在当前 F＋跨 RX、nominal practical、保留 clean 份额的配方上，单独加入同一单侧间隔约束能否提高 urban。它是历史机制受控复验，不是新 IQ 增强，也不是新损失创新声明。

## 源证据与历史去重

旧 C3−C2 的独立 V27000 clean 三 seed 变化为 −.011111／−.003704／−.022222 个百分点，均值 −.012346；最差源 RX 均值 −.049383。相对无课程 C1，C3 的 V clean 均值 −.065432。旧首seed训练 L720 的同诊断 draw，low urban C1／C2／C3=63.750／63.333／63.611%，不能当作独立V或跨seed阳性。见 [旧源后续报告](<E:/type10-7/automation_reports/CV-SincNet/20261014-phase1-sigtime-urban-paired-e30-manysig-m18-r02/source_followup.md>) 与 [源指标](<E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/source_health_followup.json>)。

本次保留原 rho=.5、mmax=1、lambda=.5、logits/30 和 urban连续q，不按目标成绩调整参数。关键改变是删除课程混杂、使用当前6L clean保留式监督，并复用RDrop已有两次raw／satellite平均logits。旧“梯度约为CE的1.9%”只说明约束曾激活，不是增大系数的理由。

状态课程、等剂量晚加入、MC2／噪声配对、共享传播、batch分层、循环场景、stepALT、urban先验、PA尾段与Dropout等已实际训练；本轮不换名称重开。当前As已覆盖6TX×5RX×3scene，不把未证明的覆盖缺失作为瓶颈。完整源RX训练，无留一域。

## 唯一最小矩阵

| 逻辑行 | 来源 | 本次成本 |
| --- | --- | --- |
| NOAS30 | 原 P43 B00 FINAL30 | 复用source／prediction／score，不重跑 |
| AS30 | 原 P43 B01 FINAL30 | 复用source／prediction／score，不重跑 |
| AS_MARGIN | 原 P43 B01 自身 scratchAs20 student＋ownEMA，再追加10 | 新2220更新；累计E30／6660更新 |

seed2026092701、split392005；原完整 L6300／U56700（TX真值始终隐藏）／V27000、源RX1/3/4/6/8、day1/2/3、6TX不变。默认 F/SIG_TIME_X3／统一Sinc／joint160／cos30 分类头仍按原联合流程更新，无新分支、教师或头。

按现有真实前向，ordinary128×2＋mixed128×2＋raw128×2，共768L前向／步。原L损失尺度1.25和CROSS_RX／旧U保持，label_smoothing=.01继承。mixed含64As＋64ordinary；As从raw进完整practical，不叠ordinary PA/AWGN；low suburban／mid urban／low urban均匀各1/3，硬件按RX/day session稳定，physical/epoch/step命名流刷新。

仅新增 L_pair=.5 mean_i q_i mean_(c≠y) ReLU[.5 stopgrad(clamp(m_clean,0,1))−m_sat]。clean／sat logits先除固定30，再以已有两次前向均值计算；仅当前raw默认头正确的参考激活附加约束，q=urban×sigmoid(clip((signal_to_added_noise_db−5)/5,−60,60))，显式None→0。q不影响satellite CE，不新增置信度阈值／样本丢弃，不要求全部特征或门控相等。不强迫satellite保持全部clean置信度或全部间隔。新增前向0、额外U前向0。

原父20的来源与当前契约已核实，发布时只核对自身 P43 B01 scratch20 的元数据／角色和实际CP，不回退不匹配权重。重置AdamW／cos10／cache，strict ownEMA不startup average；ordinary参数来自scratch20的E13，累计epoch/step+20/+4440。

实际导入必须记录__file__，继承P43 pinned runtime中的 experiments/adv3b02_xuc/code/leo_practical。fs25MHz、IQ256、complexRMS center=false在normalize_input；训练fc2.462GHz／正式固定2.45GHz。residual/post_sync、EQ=false即G=1、refSNR30、质量适配＋相位跟踪＋IQ补偿开启、reflect均不变。实际值由运行审计确认，不用默认值冒充。

## 预先固定的判定与关闭条件

在读取新正式分数之前保存源判定：fullV27000 clean AS_MARGIN−NOAS30≥−.1绝对百分点；V1800×2draw pooledmid/low AS_MARGIN−AS30≥+.5个百分点；low≥AS30。任一FAIL关闭这个固定复验，不扫lambda/rho/q、不救活种子或延长。两个draw仍是同1800物理样本，不是两个模型seed；source只有三训练场景，没有source all7。

源科学FAIL也自动完成本行clean＋六固定practical预测，在该行全部prediction固定后独立truth-last评分。CPU分析先读取已保存source决定，再核实三逻辑行全部7视图闭合后才连接truth；不重新训练、预测或launch评分参考。报告all7 accuracy/F1/最差RX、RX/day/TX、修复和新增错误、固定NOAS30 clean正确集合保持率、margin/30、source状态／质量／新增噪声比分层（不是总SNR）、完整训练曲线和成本。正式结果不决定候选、系数、晋级、种子或重跑。

正式阶段要求相对同预算同seed NOAS30 clean下降≤.1绝对百分点；这不保证最终可重复clean正收益、low>60%或suburban接近clean。单seed／小于.1pp正值只作初步观察。无额外泛化晋级或每样本身份保护硬门槛。

## 执行、成本与产物

CWD=PROJECT/releases/cvs_FcrossRX_pair_margin_tail_E30_20261018_r01；命令 `CVS-RFFI/python -u experiments/cvs_pair_margin61/runner.py`；输出=PROJECT/runs/20261018-phase1-FcrossRX-pairMargin-tail-e30-s2026092701-m1-r01；owner=codex/root/pair-margin61。CPU自动分析 `analysis/cvs_pair_margin61_analysis_20261018.py --project PROJECT --out BASE/analysis_r01`，没有模型／IQ／CP访问。

新增10轮／2220更新／1704960 L前向／142080生成As；父20轮／4440更新／3409920L前向／284160As单列。预期新增nativeNN3162962（student train2271960、student eval324002、teacher eval567000），source读出12600、formal1176002、真实父smoke2另计。预期总远端4351566NN；本地唯一F合成768NN另列。全部实际计数／墙钟以完成产物为准，预算耗尽不等于收敛。

每GPU最多2个总PID，计入其他owner，free≥12000MB；只用普通账号。协议／来源／运行错配、非有限、输出冲突或系统技术失败仅停止自身run并保留产物，不自动重训，不干预健康任务。source低分不是技术停止理由。

执行最小流程：一次聚焦checks＋一个真实父无query烟测；一次独立P0/P1；一个实际code/configcommit；最小登记；独占root／一次N607预检／单归档SHA比较／一次remotecompile；启动后一次PID/CWD/argv/GPU/log增长核实；自动truth-last与同row分析。代码、配置、验证、真实GPU／PID／时间在证据产生后填入。REJECTED_EXTRA_GATE：generic brainstorming的设计审批／固定spec／重复全审不在项目最小白名单，现有授权内继续，不追加审批。

实际修改范围：experiments/cvs_pair_margin61/**、analysis/cvs_pair_margin61_analysis_20261018.py 和对应toy检查；原P43与其他健康任务不修改。Git不提交IQ／checkpoint／原日志／预测。源与正式未完成前不报告新增准确率。

本地执行证据：27项协议负例及margin闭式/stopgrad/固定尺度/连续q/梯度与capture RNG检查PASS（0NN）；4项流程检查PASS（0NN）；唯一实际F合成768NN、0真实IQ/CP/更新，探针wall1.8225325秒、进程wall3.988秒，原6×128与As64/全部CE保持，新增参数/前向/Uhook均0，原默认头/Sinc联合梯度finite非零。19py编译PASS。分析12项纯toy1.020秒PASS，含完整2220步日志fixture/源PARTIAL/三行all7 truth闭合/成本去重；2py编译PASS，0NN/IQ/CP/真实truth/remote。尚无新训练或正式分数。

一次普通账号N607预检已核实ownAs20 scratch/final_only/no-target元数据、精确role_ids与P43 B00/B01原七预测/独立评分闭合；新run/release/zip均不存在，pinned runtime正确。预检没有加载CP或IQ，NN0。dispatcher发布时仍按实际总PID和显存选择容量，不干预其他owner。唯一P0/P1尚在进行。

唯一独立P0/P1审查PASS，P0=0/P1=0；审查者运行相同12项CPU toy1.040秒，0NN/真实IQ/CP/truth/更新/remote，未重复F。核对实际五hook顺序与64物理对应、严格自身As20/EMA、全部CE保留、q/stopgrad/scale、源FAIL仍all7/独立评分/source-first三row闭合和单owner/cap2。代码/config即将一次提交后发布，无须新的全审或报告审批。CPU自动分析同时记录实际codecommit与已测本地768NN。

实际发布：code/config commit 5bc8ccff40c7e1873c261eb37f8b9b444283871e 已push，独立GitHub compare identical；单release archive SHA 0e4ff49b8af910c51f03fe7086b595e58a1703847b62714150624bc9797da127 本远端一次比较VERIFIED，远端21py compile PASS。首步真实自身As20 CP无query smoke PASS（2NN／1CP／0IQ／0更新）后直接训练，未重复远端合成。dispatcher1312196、worker1312286／GPU0，CWD/argv/CUDA及nvidia PID独立核实；同一start_ticks进程日志64064→290042字节，failure为空。

实际运行：source_roles EXACT_MATCH／target_contact false，严格own stage20 EMA、额外teacher=false、优化器空state/cos10/lr.0002/WD.0001与cache重置。actual_channel configurations与原As30完全一致，真正leo_practical导入来自原evidence_responsibility release内experiments/adv3b02_xuc/code；新margin来自本release。完整practical测试IQ／SNR／边界没有修改。实际训练前向与源IQ读取正在发生，完成后报告实际成本，不能记零；新增source／all7／独立score／CPU分析由同一dispatcher自动衔接，当前未完成。


最终结果：AS_MARGIN clean85.273214%、mid urban62.883333%、low urban51.354762%、urban三均值64.019246%。相对As30 clean−.067857pp、mid+.076786、low+.110714、urban+.079960；相对NoAs30 clean+.025595pp，单seed的.1pp阶段约束通过，但cleanF1−.005988pp，无跨seed稳定双提升。源clean−NoAs30 −.118519pp、midlow−As30+.069444pp、low0，原FAIL未重算。默认头、全source及测试IQ/信道不变；low<60%，目标未完成。新增10轮/2220更新、远端4351566NN、本地768NN另计，训练1186.548547秒、source1.829463秒、formal48.443742秒、CPU分析22.385836秒/0NN；两原进程自然退出。全部五图已实际查看。详见final_analysis.md及source_decision.json。

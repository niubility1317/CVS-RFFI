# 完整 clean 保留配方 E30：真实星地 U 教学的最终结果

本轮状态：ANALYZED／VERIFIED，科学判定为 SOURCE_NEGATIVE。三组、一个模型 seed 2026092701，均独立从零训练20+10累计E30；所有clean及6个practical已固定预测、独立truth-last评分并重算。默认分类头与网络结构不变，未留出训练RX。相对B0，UAS的urban均值64.1319%→64.9230%（+0.7911pp）、mid62.9643%→63.8571%（+0.8929pp）、low51.6000%→52.4613%（+0.8613pp）；clean85.5113%→85.2452%（−0.2661pp）。这是小幅星地增益，未达到low>60%或clean净正收益。所有正式RX的clean本轮均下降，RX8-8为−1.1833pp；不以总体平均掩盖该代价。

固定C4在读取正式分数前已因源门槛未通过而关闭，提交b0dcc80bdcbc79a5a234733d5518cb78b7e7afe1；正式结果不改变不扩seed、不调参救跑的决策。三组完整NN相同，URAW的额外raw-U KL相对B0没有可观收益；UAS相对URAW的urban均值+0.7960pp、low+0.8768pp、clean−0.2565pp。模型seed只有1，两个源信道draw不是两个模型seed，不能证明跨seed复现。

## 逐 row／seed 正式指标

B0：原完整L配方＋一次额外rawU零loss前向；URAW：同成本rawU软KL；UAS：同成本full practical U软KL。共同L有效ordinary:satellite=.75:.25、原SupCon／RDrop／旧U门控不变。所有数值仅对应本run、上述seed、相同168000个测试physical样本和固定7视图；不是普通无As基线或新的2×2 Ag因子矩阵。

| 环境 | B0 准确率% | URAW 准确率% | UAS 准确率% | UAS−B0 pp | UAS−URAW pp |
|---|---:|---:|---:|---:|---:|
| clean | 85.5113 | 85.5018 | 85.2452 | -0.2661 | -0.2565 |
| practical_high | 81.3571 | 81.3881 | 81.9613 | 0.6042 | 0.5732 |
| practical_mid | 78.5863 | 78.6333 | 79.3083 | 0.7220 | 0.6750 |
| practical_low_suburban | 76.0030 | 76.0286 | 76.7042 | 0.7012 | 0.6756 |
| practical_high_urban | 77.8315 | 77.8661 | 78.4506 | 0.6190 | 0.5845 |
| practical_mid_urban | 62.9643 | 62.9304 | 63.8571 | 0.8929 | 0.9268 |
| practical_low_urban | 51.6000 | 51.5845 | 52.4613 | 0.8613 | 0.8768 |

| 环境 | UAS Macro-F1% | UAS 最差RX准确率% | UAS clean−本环境 pp |
|---|---:|---:|---:|
| clean | 84.9785 | 79.0333 | 0.0000 |
| practical_high | 81.6417 | 72.8333 | 3.2839 |
| practical_mid | 78.9635 | 71.4208 | 5.9369 |
| practical_low_suburban | 76.3354 | 69.4625 | 8.5411 |
| practical_high_urban | 78.0677 | 70.2708 | 6.7946 |
| practical_mid_urban | 63.3419 | 58.0167 | 21.3881 |
| practical_low_urban | 51.8158 | 48.1208 | 32.7839 |

UAS最差low RX为14-7，46.7000%→48.1208%（相对B0+1.4208pp）。low Macro-F1为51.8158%，B0为51.1258%（+0.6900pp）。suburban三个绝对准确率确有提高，但相对clean仍差3.2839／5.9369／8.5411pp；差距缩小也包含clean下降，不能把它当成保持clean达标。完整每RX/day/TX/view及该seed记录见[per_row_seed.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/per_row_seed.csv)，全部378个正式混淆矩阵已与原scorer一致。

## 收益归因与未解决退化

同预算raw控制非常接近B0：clean−0.0095pp、urban均值−0.0050pp。三组6660步ordinary输入MSE、L/U采样顺序均完全匹配，额外学生NN每组963900、额外教师NN0，本轮对照支持把有限星地增量归于完整practical U教学这一已登记差异，跨seed尚未验证；不归功于新骨干、换头、扩大L或另训诊断头。

源V1800／两次固定draw的low PARTIAL/degraded：URAW77.8656%→UAS81.0277%（+3.1621pp，n253）；PARTIAL/locked88.5993%→90.8795%（+2.2801pp，n307）；BLOCKED/unlocked29.5847%→31.6933%（+2.1086pp，n1565），后者仍很差。新增噪声比8至15dB分层+2.9703pp（n303）、<3dB分层+2.3957pp（n1127）。这些是同源物理分层的描述性改善，不是移除噪声／多径后重算或固定接收状态的因果消融；正式target缺少逐包状态标签，不能把源分层数字冒充正式状态成绩。signal/added-noise不是原录音总SNR。

正式low相对B0修复3113、引入1666个新错，净+1447；clean修复429、新错876，净−447。相对URAW，low修复3146、新错1673，净+1473。固定新B0 clean正确集合143659上，low保持率56.5408%→57.5488%（+1.0079pp）；该集合是本轮As训练控制，不是历史未增强父模型。默认头margin除以固定cos尺度30，low均值.0291499→.0333674；尺度变化没有被当作改善。详细配对见[paired_repairs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/paired_repairs.csv)和[fixed_B0_retention.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/fixed_B0_retention.csv)。

原门控最终接受率约1.47%，额外连续teacher q对约88.67%的U为正；U真值始终隐藏，不能把该比率称作U伪标签precision。全30轮曲线显示常规L CE／RDrop和源V走势接近，UAS额外KL持续非零；辅助loss、q覆盖或梯度健康不替代识别收益，固定预算结束不证明收敛。未保存逐包内部门控激活、逐分支训练梯度轨迹和L逐包TX×状态侧表，不编造这些机制结论；已有源/默认头和不拟合参数的分析见实际产物。

## 覆盖、成本与验证

每组L6300／U56700全部出现，完整5个源RX与3个day保留，无L_dev或训练域留出。每组L默认头前向3409920、L完整As426240。UAS完整额外As(U)963900／3774批，15个RX/day/eq会话硬件跨段稳定，记录变化0；全部residual/post_sync/noEQ（G=1），新增噪声、遮挡与补偿依赖保留，synthetic_reflect边界，AGC限幅67个。原formal received IQ、SNR、信道和样本集合没有改变。

| Row | native训练/验证秒 | 正式预测秒 | 全部源NN | 正式预测NN | 总NN |
|---|---:|---:|---:|---:|---:|
| B0 | 2761.9712 | 47.3221 | 7246624 | 1176002 | 8422626 |
| URAW | 2776.9127 | 58.5044 | 7246624 | 1176002 | 8422626 |
| UAS | 4300.2098 | 47.1424 | 7246624 | 1176002 | 8422626 |

共享预烟测50NN仅计一次，三组总25267928NN；共90新modelE／19980更新，每模型预算E30。父成本为各自已计入的E20，不加载E200或历史目标选出的权重。首次源teacher资格14400NN已在native计数中，冻结后独立源读出12600NN另计；后者没有单独墙钟记录，保留null。源独立分析16.3690s、最终独立分析90.7226s，均0模型NN／0IQ／0checkpoint／0更新；最终分析在源结论提交且全部预测完成后才读正式truth。相同NN的UAS native墙钟约为B0的1.557倍，主要额外工作含真实信道生成和物理记录；不把NN相等误写成墙钟相同。

训练聚焦检查6PASS／1本地CUDA不可用SKIP，实际三个GPU私有RNG回归均PASS；一次独立P0/P1及3处修复的定点复核PASS；真实checkpoint零query smoke2NN、真实F合成梯度路由48NN PASS。独立分析7项检查及编译PASS，全部3×6660完整step／物理采样及378正式、831源混淆矩阵重算PASS。外部发布／运行／结果关闭以实际产物独立核对为VERIFIED。所有8组原PNG已逐张查看；覆盖热图文字对比度单独修正并查看，原16图文件保留，修订不改指标／不重新训练或预测。

## 实际文件与复现入口

训练实现：[runner.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/runner.py)、[model.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/model.py)、[observer.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/observer.py)、[channel.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/channel.py)、[source.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/source.py)、[qualification.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/qualification.py)、[criteria.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/criteria.py)、[design.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/design.py)。执行身份91bbfcdc026b48abfeaa576e14bc4b7f44d3d64a；数据／模型／seed／阶段配置由design及本目录experiment.json记录，各run保留resolved config与actual_modules。实际leo_practical来自原pinned release的experiments/adv3b02_xuc/code，非顶层code；完整__file__审计与命令在下方历史执行记录。

独立分析：[结果程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_Uurban_E30_results_20261018.py)、[7项检查](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/check_Uurban_E30_analysis.py)，执行身份34b02721c60e4b7a7dffa2d313f2190409509b4c；图修订程序[coverage figure](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_Uurban_coverage_figure_20261018.py)，身份26aa4ffe7a805dbd0ba5e9b2598d041564889926。真实命令为固定N607 CVS-RFFI Python执行/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01/experiments/cvs_Uurban39/runner.py；最终分析执行/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_Uurban_analysis_20261018_r03/cvs_Uurban_E30_results_20261018.py --mode final --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/final_analysis_r01 --source-analysis /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/source_analysis_r01 --truth 固定既有truth路径。图修订以已提交脚本内容经SSH UTF-8 stdin执行，参数--analysis /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/final_analysis_r01 --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/figure_revision_r01。未重启、停止、热改其他健康实验。

实际小型汇总见[results_summary.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/results_summary.json)，源闭合决策见[source_development_decision.json](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/source_development_decision.json)，原始源分层见[source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/source_analysis_r01/source_strata.csv)，全量U物理统计见[U_physics_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/U_physics_summary.json)，质量代理敏感性见[quality_state_sensitivity.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/quality_state_sensitivity.csv)。全部963900记录的替代代理阈值交叉为21535／18609，仅元数据反事实；实际状态、波形及补偿未改，不测分类因果效应。

8组图表均实际存在，PNG供查看、PDF供导出：

- [formal_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/formal_accuracy.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/formal_accuracy.pdf)
- [training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/training_curves.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/training_curves.pdf)
- [receiver_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/receiver_accuracy.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/receiver_accuracy.pdf)
- [extra_U_training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/extra_U_training_curves.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/extra_U_training_curves.pdf)
- [training_L_support PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/figure_revision_r01/training_L_support.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/figure_revision_r01/training_L_support.pdf)
- [source_added_noise_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/source_added_noise_accuracy.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/source_added_noise_accuracy.pdf)
- [source_state_lock_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/source_state_lock_accuracy.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/source_state_lock_accuracy.pdf)
- [training_U_physical_support PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/training_U_physical_support.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/final_analysis_r01/training_U_physical_support.pdf)

未完成项：新增Ag的独立有效性、匹配的Ag×As交互项、clean净正收益、low>60%、预先记录的suburban距clean≤2pp探索目标及跨模型seed复现。2pp为探索标准，不另立用户未指定的硬门槛。固定C4不继续扩种子；下一候选尚未选定。后续研究只能基于合法源证据重新提出IQ／训练组织假设，不能从本轮正式分数定标或挑选重跑。整体目标保持active。

---

下面保留启动前预登记、完整审计和执行历史；历史RUNNING／待完成文字仅描述当时状态，以本页最终结果和登记为准。

# 完整 clean 保留配方：独立 U 星地教学 E30

当前状态：RUNNING／VERIFIED，累计 E30、冻结源判定和正式评价均未完成。下述方法与门槛在启动前固定。唯一假设：在完整原参考配方上，复用自身clean EMA的可信分布监督同包完整practical U，能优于同成本的额外raw U监督，并保住clean。P38固定源面板已显示teacher置信precision99.88%至100%、coverage89%至92.56%、low加权教学机会38%至40%；这些是源域资格，不是U真值精度或训练增益。旧SIG_TIME E50的SSL增量很小，不能直接复用其结论或叠加收益。

固定3组、1模型seed2026092701，全部独立从零开始，各自20+10累计E30/6660更新（总90modelE/19980更新）。B0为原参考L配方加一次rawU零损失学生前向；URAW以相同rawU学生前向计算额外软KL；UAS将该额外前向换成完整As(rawU)，其余规则相同。每个row的第一阶段13L+7U/cos20，第二阶段严格加载自身固定FINAL20及自身EMA、重置AdamW/cos10/0L+10U；原L在全部30epoch参与。未用旧目标暴露checkpoint初始化；真实旧E20只做兼容性smoke，随后丢弃。

L严格沿原完整参考：普通128两次、混合128两次；混合64ordinary+64完整As(raw)，有效CE监督ordinary.75/satellite.25；RDrop两次IQ相同、原SupCon仅普通pair。原ordinary冻结E13实际强度，保持原L信道逐包3场景名义采样与stream20261016421。F/SIG_TIME_X3/统一Sinc CENTER_25K与CENTER_UNITS/NoExplicitCFO/原joint160默认cos6×160 scale30，骨干和头联合训练；900722参数206state，不增加部署模块、头、诊断分类器、特征全对齐或新的Ag。

源角色精确复用split392005、7/63/30、L6300/U56700隐藏/V27000，RX1/3/4/6/8×day1/2/3×6TX；无训练RX留出、无L_dev、无V梯度。额外U复用原ownEMA out_w，0额外U教师前向；旧confidence/temporal/strong门控、旧U CE/entropy、EMA更新不变。原native在两个新钩子前均断言U标签=-1，新代码只读eq/rx/day/sig/base_index元数据，不读TX。

每个row全部17个Uepoch都执行额外963900学生包前向，含q=0包；B0附加loss为0。URAW/UAS用T1原scale30 softmax，q=clip((maxp_teacher-.95)/.05,0,1)，teacher stop-gradient，loss=batchmean(q KL)，系数0.16×min(1,(累计epoch−13)/5)。不按入选数重新归一化；不要求urban学生高置信；不从困难urban反向监督clean。该系数、ramp、tau不扫描。新增前向用private dropout stream20261018442+step并恢复CPU/CUDA RNG，三个row同流，避免改动后续原训练随机序列。

UAS调用原真实residual/post_sync/noEQ(G1)/reference30/reflect，fs25MHz/fc2.462GHz、质量适配/phase/IQ补偿和传播遮挡噪声依赖不拆散；新U stream20261018441与L/验证/测试分离，按epoch/step/physicalID刷新；硬件stream20261016423和原L一致，按RX/day/eq会话稳定，禁止TX定义session。dataset RMS后As自身unitRMS，无post-As RMS，不反演未知TX。保留三训练场景、全部LOS/PARTIAL/BLOCKED及锁定状态支持，不拒绝困难U。

首次U前，各row仅用固定合法V1800及既有2draw×3scene full源观测作一次自身teacher资格核查，14400NN（student12600/teacher1800），0更新/状态不变；全部条件固定为precision≥99%、coverage≥75%、每RXprecision≥98%、两draw mid/low q加权teacher正确/student错误机会≥10%。失败则本候选后续额外loss永久关闭，仍生成登记视图、完成预算及全部正式评价，不重新校准或救跑。后段只继承本row自身E20已有资格状态和EMA，不加载旧诊断教师。

最终源科学条件：UAS相对URAW在每个draw的mid/low均≥+1个百分点；fullVclean UAS≥B0−0.25个百分点且≥URAW−0.25个百分点，所有own teacher资格通过。未通过则不扩seed、不调整系数/扰动/阈值/epoch救跑。该门槛只评估固定C4，不代表整体Ag或联合目标完成。目标探索条件仍为clean净正收益、low>60%、suburban各场景距自身clean≤2个百分点，均不保证结果。

每row固定最终checkpoint后立即独立预测clean及既有全部6practical；每环境168000、全部7预测写完后才exec独立truth-last scorer。正式received IQ、测试SNR、信道、边界及样本集合不动。源矩阵判定仅dispatcher写，不读取formal；主agent在源判定固定前不读取正式成绩，不据目标结果选择或重跑。某row完成不等待整批才进测试。

执行CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01；环境：N607普通uid1000、CVS-RFFI Python/Torch2.6、FP32、CPU2threads。命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01/experiments/cvs_Uurban39/runner.py
```

唯一路径/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01，日志/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/logs及同名-launch，owner codex/root/Uurban39；每GPU最多3总进程且计入其他owner，不干预健康旧任务。停止仅限错误角色/来源、越权query/truth、输出碰撞、非有限loss/grad、无法执行或无法生成合法预测等系统技术失败；保留partial，不自动重启。不以低分或科学失败停止。

代码与配置提交身份待实际commit写入；Git先提交push并独立比远端，单release归档一次SHA比较及远端compile。launcher首步真实E20 checkpoint零query smoke，PASS立即继续real-F三组额外U合成梯度烟测；synthetic输出独立不占stage20/source路径。worker分配GPU后先作本次CUDA RNG问题定点回归，0NN/0更新。一次独立P0/P1审查发现3个未定义变量，均已定点修复并PASS；本地6项检查PASS/1项CUDA不可用SKIP、compilePASS。

成本计划：每row新30E，父成本是本row已计入的20E，不另借父预算；LNN3409920、原U学生与教师各963900、额外U学生963900、首次U源检查14400、冻结源读出12600、正式预测1176002及原native验证/初始smoke，实际NN和墙钟按完整日志逐阶段累计。总预算耗尽不等于收敛。预期产物见experiment.json：真实配置/导入文件、source角色、ownparent/EMA/optimizer、完整epoch/step/L-U-sequence、U信道元数据、覆盖、资格、默认头源预测、正式逐row/seed/RX/day/TX七环境scores、独立报告/分层图/训练曲线/成本/负结果。

## 实际发布状态

RUNNING／VERIFIED。执行代码提交 91bbfcdc026b48abfeaa576e14bc4b7f44d3d64a 已独立核实GitHub远端一致，一次release归档SHA比较与远端compile PASS。dispatcher PID776627；B0 PID776653/GPU0，URAW PID776734/GPU1，UAS PID776817/GPU2，实际CWD/argv/物理GPU/日志及自身run根均独立核对。3个GPU的私有RNG定点回归PASS。

真实旧E20兼容性CP零query smoke2NN PASS；真实F合成额外U路由48NN PASS，原Umask保持，CPU RNG恢复，URAW/UAS原默认头和Sinc梯度有限非零。合成使用独立目录，实际stage20/source未被占用；该证据不表示教学资格或训练性能通过。

最新源进度三个row均已完成epoch1／222更新，各自LNN113664、完整L As14208，额外U尚未启用（预计E14首次）。实际初始化均scratch、checkpoint_sources/ancestors为空、角色EXACT_MATCH；真实总参数900722、13L7U、原FP32策略；无traceback。worker日志从约6.4KB增长至约11.5KB，step与sequence文件持续增长。

真实导入leo_practical为不可变依赖release内experiments/adv3b02_xuc/code/leo_practical/channel.py与residual.py，并非仓库顶层code；DataAugmentation与SSDG.train_ssdg同属该adv3b02_xuc实现。具体__file__保存在各stage20/actual_modules.json。训练／测试未完成，首次自身teacher资格尚未执行，当前没有新的准确率、图或跨seed结论。

本地执行证据：E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/verified_running_r01.json 与 source_progress_r01.json。后续只读核对原PID/进度，健康任务继续；逐row冻结后自动进入既定七环境评价。

## 首次 U 与独立分析准备

三组首次自身teacher资格均PASS，源V覆盖B0=80.5556%、URAW/UAS=80.6667%，覆盖内precision均100%；该数字仅来自合法V1800，不是U真值精度。实际E14三个row均完成56700额外U学生前向，UAS完整As(U)也为56700，物理日志约28.9MB并继续增长。URAW首次U更新step2887的权重0.032、q均值0.61072、原mask未改、pseudo_truth_available=0、无非有限skip；仅说明机制实际执行，不说明识别改善。

独立分析程序analysis/cvs_Uurban_E30_results_20261018.py已实现并通过4项聚焦检查及编译。source模式只读已冻结源预测和完整日志，重算6660更新、原L目标分解、oldU掩码/hidden标志、额外963900U前向、Sinc物理步进、原始增强和采样序列及2draw×3scene源分层；final模式必须先有独立源判定，再检查全部七视图固定及独立评分后才连接truth。所有物理统计区分白功率代理、signal/added-noise与未知总SNR，U真值不解封。

该分析不修改正在运行的训练release，不增加NN或更新。门控逐包激活及逐分支梯度未在当前run保存，不从标量loss或准确率反推；L joint TX×state逐包侧表亦未保存，报告将如实区分已测的RX/day/scene/state聚合覆盖和缺项。源/正式图表尚未生成，不提供虚构图路径。

独立源分析等待进程已实际启动并核实：PID791020、start_ticks150569691、CUDA_VISIBLE_DEVICES空、CWD为单独analysis release，argv仅包含source模式及原三个worker PID，不传truth路径。分析代码身份8e9e15c093472bb722915117d3682745260d1394，独立GitHub读回一致、单analysis归档比较与远端compile PASS；训练release未修改。当前等待全3源冻结/源判定文件，stdout/stderr均0，尚未运行完整分析或生成图。最新已观测B0/URAW E19、UAS E16；UAS累计170100个完整额外U观测，3个原worker均活跃且无traceback。

## 自身 E20 到 E10 的实际过渡

已独立核实 B0 与 URAW 完成自身 E20／4440 更新，并仅加载各自固定 FINAL20 和自身 EMA 进入原定 10 轮续训。没有旧目标暴露权重初始化、额外预算、训练 RX 留出或分类头重训。最近完整 epoch 为 B0 E23、URAW E23、UAS E18；仅按完整 epoch 计算的更新下界分别为 5106、5106、3996，活动半轮不计入。原 dispatcher、三个 worker 和 CPU 源分析等待进程的 PID/start_ticks/CWD/argv/GPU 均再次 VERIFIED；没有 traceback。

完整读取已结束的 B0/URAW E20 段各 4440 个 step 和 1554 条额外 U 记录：每组 396900 个 U 观测，旧门控入选分别 5630／5648，额外教师连续权重为正的观测分别 341142／341716；U 真值持续隐藏，额外教师前向为 0，未跳过非有限更新。此为已完成训练段的实际覆盖，不是完整 E30 结果、U 标签精度或分类收益。额外 raw KL 步均加权损失为 0.00035928，B0 附加损失为 0；不得把损失尺度当成有效性结论。

源分析仍只等待所有冻结源产物，正式成绩未读取。证据位于 local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/own20_to_10_verified_r01.json 与 completed_stage20_B0_URAW_r01.json。

## UAS 已封存 E20 的完整物理记录

UAS 第一段已结束，实测 4440 更新、LNN2273280、LAs284160、额外完整 As(U)396900；学生训练NN3067080、学生评估NN606602、EMA评估NN398700、EMA训练NN0。已核实后段仅继承本row自身 FINAL20，角色 EXACT_MATCH，target_contact=false。累计 E30 尚未完成。

完整读取 1554 批／396900 个 U 物理记录，5个源 RX 各 79380，3个day各132300；low suburban132057、mid urban132580、low urban132263；LOS200889、PARTIAL62657、BLOCKED133354，locked239810、degraded66628、unlocked90462。全部为原 residual/post_sync/noEQ/G1、synthetic_reflect；残余核62、数值延迟24，AGC限幅30个。上述是训练覆盖，不是U分类准确率。

实际 signal_to_added_noise_db 与 white-input quality proxy 差值的中位数0.0269dB、q01−1.3015dB、q99+1.2296dB；两者均不是原始录音总SNR，且差值混合波形频谱、RX残差与有限噪声实现效应，不宜直接归因于多径。实际运行质量阈值为5／15dB、quality_reference20dB、max_residual_scale10，来自actual_U_channel.json而非默认值。

独立分析补充元数据阈值敏感性：复算实际记录的接收状态和残差尺度；另统计signal/added-noise与推断的post-comp signal/reference-noise替代代理是否跨阈值。原波形、receiver和补偿均不重新计算或改动，不测分类因果效应。6项检查及编译PASS；该版本尚待单独发布，现有source-only等待进程及训练release不变。

## 功率代理阈值敏感性实际结果

分析代码82e8a7815979f35f16e6920cb4849195eacbc0cf已单独发布r02，独立GitHub读回一致、单归档SHA比较及远端compile VERIFIED。原训练release和source-only等待进程未修改或重启。6项本地检查PASS；用全部396900个已封存E20物理记录复算原状态与residual_scale，均与实际配置吻合。

将同一记录的signal/added-noise比值作为替代质量代理时，有8866/396900（2.2338%）跨过锁定阈值；用推断的post-comp signal/reference-noise去除有限新增噪声功率波动后，有7664/396900（1.9310%）跨阈值。后一比值相对白功率代理的中位差−0.0121dB，q01−1.2072dB、q99+1.0686dB。这里没有重新生成信号、接收状态或补偿，不是总SNR，不证明阈值偏差导致了哪些识别错误；RX/image残差仍包含在推断信号功率中。

局部证据local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/UAS_E20_quality_state_sensitivity_r01.json保留完整分scene转移计数、统计、代码身份和0NN/0IQ/0checkpoint/0Utruth/0target标志。最新原worker为E29/E29/E21且仍健康，源矩阵判定与完整正式分析尚未完成。r02将用于后续既定final_analysis_r01，source_analysis_r01仍由原r01等待进程生成；正式成绩在源判定独立固定并记录前保持未读。

## 前两组完整 E30 日志复算与成本边界

B0与URAW均已完成自身20+10并冻结默认头，原PID自动exec为predict；UAS最近完成E22继续原训练。单独early_source_recount_r01完整读取前两组全部日志，各6660更新、6660条L目标、6660条Sinc物理步进、3774条U额外记录、10434条L/U采样记录；原L6300/U56700全部出现，SupCon5550个有效步，最大L损失分解误差分别2.38e−7／3.58e−7。活动训练字段有限且无非有限跳步或错误标记；原stdout另有300次nan文本提及；逐步活动loss及非有限跳步按完整结构化日志独立核查，不用文本计数判定训练失败。

每组实际native计数：student_train5337720、student_eval930604、teacher_train0、teacher_eval965700，共7234024NN；native段墙钟B0=2761.9712s、URAW=2776.9127s。初次U资格14400NN已包含在native计数中，不再相加。冻结后另行source_qualification的12600NN必须再单列，每组完整源训练及源读出合计7246624NN；该读出没有单独计时，不推算其墙钟。独立分析已补齐此项及后续正式prediction和全矩阵共享烟测的一次计费，7项检查和编译PASS。原r01源等待进程不变，其旧source_identity_NN字段仅包含native部分；后续最终分析将提供明确分项。

该修正只涉及分析成本，不改变训练、源科学阈值、默认头或任何prediction。前两组源训练完整不等于三组矩阵完成；正式成绩依旧未读取，研究目标未完成。

最终分析版本现已固定为r03／34b02721c60e4b7a7dffa2d313f2190409509b4c，7项检查PASS、单归档比较与远端编译VERIFIED；对实际B0/URAW全计数和独立冻结读出再核对，各7246624源NN，共享预烟测50NN仅计一次。r02保留，r03不覆盖任何既有release或输出。最终分析尚未启动；先等待原r01独立source_analysis及源科学判定固定、记录后，再用r03读取既定正式产物。

## 前两组七环境产物的独立关闭核对

B0／URAW的completed检查均PASS：各168000个physical sample、全部7个既定视图、prediction与自身固定FINAL30配置相同、所有参数冻结、head_fit_updates=0、query_fit=false；评分状态SCORED_COMPLETE且truth_last=true。此处只读取状态和成本元数据，未打开scores.json或正式truth／预测内容，不报告正式准确率。两组正式预测NN均1176002，耗时分别47.3221s／58.5044s；各自源训练和源读出7246624NN，连同正式预测各合计8422626NN。矩阵共享预烟测50NN另计一次。

B0与URAW原进程完成后正常退出，有完整row完成产物；UAS原worker PID776817/start_ticks150415404继续训练，最近累计E24。原source-only等待进程PID791020仍存活。整个3row矩阵尚未完成，源判定未写入，正式分数保持未读。

## 本轮实际普通 IQ 增强审计

实际DataAugmentation与cvsrffi.schedule来自原pinned adv3b02_xuc实现。读取ordinary_augmentation_actual.json及实际native args，调用同一configure_augmentor_for_epoch离线重建，逐项对照B0/URAW全30轮、UAS已结束26轮telemetry，全部MATCH。E13冻结scale=0.2506003339349741，此后没有在own10重新ramp。下表是配置的随机事件概率，不是记录过的实际事件次数。

| 顺序／操作 | E13后事件概率 | 实际范围与采样 |
|---|---:|---|
| 1 DC offset | 7.5180% | 每batch开关，每包I/Q独立偏移，上限0.0050120 |
| 2 Tukey taper | 6.2650% | 每batch开关与共享alpha，alpha=0.0050120至0.0250600 |
| 3 循环移位 | 8.7710% | 每batch开关，每包整数−8至8样点，包含0 |
| 4 幅度缩放 | 11.2770% | 每batch开关，每包0.9至1.1；幅度范围没有随scale缩小 |
| 5 全局相位 | 11.2770% | 每batch开关，每包0至2π；角度范围没有缩小 |
| 6 CFO | 8.7710% | 每包±0.000100240134 cycles/sample，25MHz下约±2506Hz |
| 7 相位噪声 | 7.5180% | 每包sigma≤0.0015036020 |
| 8 普通多径 | 4.5108% | batch共享随机2至4tap，时延0至1样点，reflect左边界 |
| 9 普通AWGN | 10.0240% | 每包20至36dB，相对当前增强波形功率；不是录音总SNR |
| 10 普通PA | 3.5084% | 每包mask；mp_sigma0.0125300、mem_sigma0.0100240、ampm_max0.0501201 |
| DAC／class signature | 关闭 | p_dac=0、enable_class_signature=false |

循环移位通过模L gather实现，会把截断端点接到包内；256点IQ上最多8点发生wrap，是结构上限而非实测发生比例。taper先于移位，只有被抽到时才可能减轻端点差异，不能声称循环移位无接缝。普通多径的源码注释含circular-ish，但实际F.pad为reflect，应以执行语句为准。

数据集输入已RMS；普通增强末尾只有有限值清理，没有再作全IQ RMS。实际SIG_TIME/Reference/IdentityOnly/CVSincNet入口把同一增强IQ传入原网络；reference相关、GroupNorm及LayerNorm等分支内部仍有各自归一化，因此不能把增益扰动视为必然有效或必然在全模型完全抵消。该零前向代码审计读取自身固定B0 checkpoint两次，没有读IQ、改变模型或新增训练／预测NN；不把代码观察冒充增益单因素性能试验。

L的普通对使用上述一次已生成IQ作两次RDrop前向；混合对仅64个ordinary加64个完整As(raw)，完整As前不先加普通AWGN、多径、PA或CFO，As后不再RMS。UAS同样直接As(rawU)，原weak teacher及旧strong/noise规则保留。原NoExplicitCFO指身份表征约束，不等于旧普通增强的CFO事件为0。

全部操作是三组共有的旧配方，没有新增Ag；尤其旧随机PA并未由本轮证明身份保持或跨RX泛化有效，不能据配置概率宣称Ag成立，也不据此热改当前实验。标签相关signature分支实际关闭，PA不会反演录音中未知TX硬件。真实operator事件计数与逐包谱变换未保存，报告保留该边界。

## 正式分数读取前固定的源科学判定

SOURCE_NEGATIVE／VERIFIED。三组固定20+10累计E30均已结束，每组6660更新，共90modelE／19980更新。原source-only独立分析完成，0NN、0IQ、0checkpoint、0target，16.369s；831个源分层混淆矩阵已生成。三组全6660步ordinary输入MSE完全匹配，L/U物理采样序列在两个阶段均0mismatch；因此该比较保持原完整L组织和既有U流程，差异为已登记额外U视图及其一个KL。

UAS相对同成本URAW的源增益依次为draw0 mid+0.1111pp、draw0 low+1.5000pp、draw1 mid+1.0556pp、draw1 low+1.2778pp。fullV clean分别为B0 98.0556%、URAW 98.0889%、UAS 98.0778%；UAS相对B0+0.0222pp、相对URAW−0.0111pp，满足clean保持但不是有意义的跨RX clean增益证据。自身teacher资格均通过；每draw mid/low都至少+1pp的条件未通过，独立重算与dispatcher完全一致。

依据启动前规则，固定C4（全名义practical U＋teacher连续q＋0.16单KL）不扩模型seed，不改系数、阈值、信道或预算救跑。保留两次源low增益及mid不稳定的事实，不把局部low改善改写成整候选通过，也不外推所有SSL无效。source_development_decision.json在主agent未读正式分数时固定，后续正式评价仍完成全部3row／7视图。下一候选尚未选择；目标成绩不回流决定候选、晋级或选择性重跑。Ag、low>60%、clean净收益和跨seed复现的联合目标仍未完成。

源分析文件实际保存在local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/source_analysis_r01/。旧r01 source_identity_NN仅为native7234024；最终r03另加已完成冻结读出12600，每组源成本7246624NN，避免漏项。UAS native墙钟4300.2098s，B0 2761.9712s、URAW 2776.9127s；相同NN不等于相同信道生成墙钟。

# P76 共享传播与星地判别四格：完成，固定候选关闭

**C11 没有取得机制增量，且不满足 clean 下降不超过 0.1 个百分点的阶段目标。** 四行 E30、clean＋六种 practical、独立 truth-last 评分和完整日志分析均已完成，ANALYZED，无技术缺项。模型 seed 只有 2026092701，不构成跨 seed 复现；不扩 E50，不按正式分数改参数或选择其他行补跑。

本轮依据源域预登记判定先关闭方案，再读取正式指标。源 fullV clean 相对 NoAs 为 −0.048148 个百分点，通过健康界限；C11 的 pooled mid/low 为 66.2500%，与最佳单因素 C01 持平，较 C00 仅 +0.236111 个百分点，交互 −0.208333 个百分点。两个要求至少 +0.5 个百分点的条件均失败。

四行同 SAT18 AS_CE20 student/ownEMA 父状态，各追加 E10，总 E30／6660 更新。C00=独立 H、辅助 As 对比关；C10=共享 H、关；C01=独立 H、开；C11=共享 H、开。所有格均按 RX/day/eq 共同抽场景，避免场景成组与 H 因素混淆。共同父前 E20 已关闭 As KL。P43 NoAs30/As30 是同 seed、同总预算的整配方参照，历史父状态不同，不冒称同父 NoAs 尾段。

固定 F＋跨 RX 同 TX、SIG_TIME_X3、统一 Sinc、joint160、默认 cosine30 头，正常联合训练；完整 L6300/U56700 隐藏 TX/V27000，五个源 RX1/3/4/6/8 与三日全部使用。没有留出训练 RX、增加骨干或分类头。IQ2×256/fs25MHz、RMS center=false，原 normalize_input 后完整 practical residual；post_sync，EQ off（G=1），reference_snr_db30，原质量适配、相位跟踪、IQ 补偿、AGC、reflect 边界。训练 fc2.462GHz 与正式固定 fc2.45GHz 均保持原值。实际模块 __file__ 指向 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py；实际导入记录已收取。

共享的只是 geometry/probs/state/z/ray_hz/ray_coef/f_doppler；每包保留独立 noise/phase/sync/tracking/equalizer 随机流与稳定 session RX 硬件，分别重算接收状态、补偿和 AGC。第二 As IQ 直接复制；全部 As CE 保留，ordinary/raw 目标与 SupCon0.1、原 U 规则保持。ON 只复用已有时序映射加入 0.05 的 As 对比：同 TX 不同 RX 为正对，不同 TX 同 RX/day/eq 为负对，无新模型前向或部署参数。该辅助项不是额外分类头。

| 正式环境 | NoAs30 % | 原 As30 % | C00 % | C10 % | C01 % | C11 % | C11−C00 pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.200000 | 85.060714 | 85.134524 | 84.960119 | -0.239881 |
| practical_high | 42.685119 | 81.405357 | 81.279762 | 81.239881 | 81.017857 | 80.955952 | -0.323810 |
| practical_mid | 40.995238 | 78.661310 | 78.586310 | 78.495833 | 78.329167 | 78.242857 | -0.343452 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.829762 | 75.763690 | 75.584524 | 75.543452 | -0.286310 |
| practical_high_urban | 41.439881 | 77.767262 | 77.700595 | 77.644048 | 77.386905 | 77.370238 | -0.330357 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.791071 | 62.777976 | 62.610119 | 62.629762 | -0.161310 |
| practical_low_urban | 30.139881 | 51.244048 | 51.266667 | 51.273810 | 51.167262 | 51.215476 | -0.051190 |

C11 的 urban 三场景均值为 63.738492%，较 C00 下降 0.180952 个百分点，较原 As30 下降 0.200794 个百分点；较 NoAs30 提高 28.167659 个百分点。low urban 为 51.215476%，较 NoAs30 提高 21.075595 个百分点，但这些大幅收益主要来自既有完整 As，不能归因于新增共享传播／辅助对比。low urban 未达 60%。

C11 clean84.960119%，较 NoAs30 下降 0.287500 个百分点，超过 0.1 个百分点；较同父 C00 下降 0.239881 个百分点。没有单种子双提升，更没有可重复双提升。三个 suburban 相对自身 clean 差距为 4.004167／6.717262／9.416667 个百分点，不能通过降低 clean 宣称缩小差距。

| 环境 | C00 macro-F1 % | C11 macro-F1 % | C00 最差 RX % | C11 最差 RX % | 修复 | 新增错误 | 净正确 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 84.924993 | 84.658138 | 78.845833 | 78.016667 | 712 | 1115 | -403 |
| practical_mid_urban | 62.438589 | 62.246170 | 57.450000 | 57.433333 | 2155 | 2426 | -271 |
| practical_low_urban | 50.818555 | 50.750283 | 47.029167 | 47.108333 | 2771 | 2857 | -86 |

每环境 168000 条固定正式 IQ，最差 RX 分组沿数据集原 receiver 字段（如14-7）统计。所有六行、七环境的 macro-F1、逐 RX/day/TX、配对变化与间隔在实际 [formal_metrics.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv) 和 [formal_pair_summary.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)。原两行预测和评分复用，未重复测试。

源 low BLOCKED 中 C11 比 C00 净修复 7/1979（+0.353714 个百分点），但比 C10 多错5、比 C01 多错1；mid BLOCKED 比 C00 净修复12/1334，但交互为 −0.524738 个百分点，low BLOCKED 交互 −0.656897 个百分点。low 新增噪声比<3dB 中比 C00 净修复11/1127，却比 C10 多错4；相对 C01 在3–8dB又净多错7/794。这些局部变化没有形成稳定的低质量机制收益。正式包缺少逐包 state/lock 标签，不能把源状态归因直接套到正式测试。新增噪声比不是原录音总 SNR，也没有证明物理信息不可恢复。

C11 相对 C00 的默认头 margin/30：clean −0.000706002、mid −0.002415769、low −0.001461394。固定 P43 NoAs30 clean 正确集合的 low 保持率55.565021%→55.552452%，下降0.012568个百分点；该集合不是共同 E20 父模型的正确集合。本轮未采集新的内部 gate 因果轨迹，不能把失败解释成门控、CFO或某个分支导致。

训练完整序列 L/U、ordinary 随机流匹配；共180条累计 epoch 曲线，所有新尾段各2220更新。C11 实际 As 辅助 raw loss 从1.138317降至0.922505，加权从0.056916降至0.046125；平均有效anchor62.279730/64（潜在合法总138261/142080），辅助目标确实被优化。训练目标下降没有带来识别改善。图 interaction_and_coverage 右上画的是未乘0.05的 raw 辅助项，实际加权值以CSV为准。预算结束不等于收敛。

新增4×E10=40 model-epochs／8880更新。每行 nativeNN3162962、源诊断12600、正式1176002；共享 smoke3076，总新样本前向17409332，本地NN0。共同父E20训练成本4797722NN／1827.538秒，只复用一次，未重复训练；原NoAs/As参考历史成本另列。每行新尾段训练墙钟 C00=1279.180秒、C10=1316.064秒、C01=1334.675秒、C11=1380.535秒；C11较C00+7.923%，四训练进程秒数合计5310.455。四行并行启动至分析完成约1621.135秒（远端产物mtime区间，不含此前本地开发与发布smoke）。同NN不等于同墙钟。

原生信道process每行142080次、As逻辑前向槽284160；每行personal构造142080、未process的environment构造32863。四行新训练实际波形568320，不把environment构造或第二As复制算成独立波形。合成smoke另256次process。

聚焦验证已执行：协议6组/31负例/5源阈值；信道5组/29负例和四格×边界512次fake process；模型7负例，ON真实hook图/梯度、OFF同scene loss/gradient/RNG逐位对照；分析14 fixtures/2AST及语法检查。唯一独立P0/P1 PASS，单归档传输比较和25文件远端compile PASS。真实父权重smoke3076NN PASS；其合成ON样例无合法负对导致aux=0，未把该smoke称作新aux活跃证据；实际源训练首批与全尾段非零loss另有记录。

独立结果核算：756份正式混淆矩阵accuracy/F1、42组最差RX、126组分区相加、1764条配对净数、98条总体配对摘要全部吻合，最大accuracy/F1差0。四张PNG已打开核对，对应PDF文件存在。所有worker与owner已退出，保留全部产物；没有修改其他owner实验。

实际代码 commit：32355ff0e6d73a7305e2fb8ce332651c9d1a7cbe。命令：在N607不可覆盖release CWD中运行 `CVS-RFFI/python -u experiments/cvs_prop_contrast76/runner.py`；launcher自动smoke→训练→源冻结→逐row七环境预测→独立评分→CPU全日志分析。实现为 experiments/cvs_prop_contrast76/ 的18文件及 analysis/cvs_prop_contrast76_analysis_20261019.py、对应 checks；实际配置与命令保存在登记及每row resolved_config/actual_native_args。

实际完整产物：[E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01)。
- [training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.pdf)
- [source_quality_state_lock_noise.png](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_state_lock_noise.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_state_lock_noise.pdf)
- [formal_paired.png](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.pdf)
- [interaction_and_coverage.png](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/interaction_and_coverage.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/retrieved_r01/analysis_r01/interaction_and_coverage.pdf)

技术缺项：无。科学目标未完成：low urban仍低于60%；clean阶段要求未满足；没有多种子双提升；共同E20父clean正确集合和新增gate因果诊断本轮未采集。此前三种子基础As相对NoAs仍是可重复的星地适应证据，但clean配对均值下降0.3210个百分点，不能称联合目标完成。

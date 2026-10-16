# F＋跨 RX：星地训练后新增噪声的固定接收状态诊断

状态：**ANALYZED；执行 VERIFIED；预设科学判据 FAIL。** run：20261017-diagnostic-FcrossRX-postAs-noise-control-fitL-s2026092701-m2-r01。新增训练、默认头拟合、EMA 更新和目标访问均为 0。

增强前基线依最新要求固定为 **F＋跨 RX 同 TX 对比学习**。F 保留三路 CVS／SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160 和既有 cosine30 默认头。训练预算每模型从零 E20，仍按原流程联合更新骨干和默认头。当前没有验证通过的新增泛化 IQ 增强 A_g，也没有完成 B00／B10／B01／B11 的联合收益与交互项证明。

本轮实测否定了“在现有星地训练模型中，只解决新增噪声遮蔽，就足以恢复困难状态大部分默认头身份判别”的固定假设。去除新增噪声确有净收益，但主模型 ANTI 的 BLOCKED／unlocked 仅恢复 full→clean 正确数差距的 17.6018%／18.3118%，均未达到预登记 50%。因此关闭当前假设，不改阈值、追加抽样或晋级旧 ANTI 训练采样方案。结果不能证明所有去噪方法无效，也不能给出信息论上限。

本轮只消费 **seen-fit L_s 5,040 条**，是原模型训练见过的源数据。以下 60.8135% 是这些源训练样本的 full low urban 诊断准确率，**不是跨 RX／正式目标 low urban 超过 60% 的证据**。

| 模型 | clean | full low urban | 去新增噪声、固定 RX | full 宏 F1 |
|---|---:|---:|---:|---:|
| RAW10 | 99.9405% | 42.4206% | 48.9881% | 40.9635% |
| ANTI | 99.7421% | 60.8135% | 68.1349% | 60.6380% |

| 模型／分层 | 样本数 | clean 正确 | full 正确 | 控制正确 | 恢复／差距 | 恢复比例 | 50% 门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| RAW10／BLOCKED | 2739 | 2737 | 669 | 896 | 227/2068 | 10.9768% | FAIL |
| RAW10／unlocked | 2140 | 2138 | 498 | 675 | 177/1640 | 10.7927% | FAIL |
| ANTI／BLOCKED | 2739 | 2730 | 912 | 1232 | 320/1818 | 17.6018% | FAIL |
| ANTI／unlocked | 2140 | 2134 | 665 | 934 | 269/1469 | 18.3118% | FAIL |

两分层存在重叠，不能当独立重复试验。预设主检验为 ANTI，RAW10 仅描述对照；没有事后改选更优控制。每个物理样本三个视图、两个模型均使用相同命名信道／噪声流。

| ANTI 分层 | 数量 | full 正确率 | 固定 RX 去新增噪声正确率 |
|---|---:|---:|---:|
| RX=3 | 1260 | 59.5238% | 66.3492% |
| RX=4 | 1260 | 62.5397% | 69.5238% |
| RX=6 | 1260 | 58.9683% | 66.2698% |
| RX=8 | 1260 | 62.2222% | 70.3968% |
| lock=degraded | 922 | 54.5553% | 62.6898% |
| lock=locked | 1978 | 95.9050% | 97.1689% |
| lock=unlocked | 2140 | 31.0748% | 43.6449% |
| state=BLOCKED | 2739 | 33.2968% | 44.9799% |
| state=LOS | 1543 | 97.9261% | 98.5094% |
| state=PARTIAL | 758 | 84.6966% | 89.9736% |

ANTI 全体 full→控制：修复 675 条、破坏 306 条，净增 369 条（+7.3214 个百分点）；BLOCKED 修复 600、破坏 280，净增 320（+11.6831 个百分点）；unlocked 修复 491、破坏 222，净增 269（+12.5701 个百分点）。噪声移除不是逐样本单调改善，不能只报修复数。RAW10 对应全体修复 680、破坏 349，净增 331。

固定 RAW10 clean 正确集合共有 5,037 条；ANTI 的 full／控制保留率为 60.8497%／68.1556%。这是本轮 seen-fit 集合，不得与 P8 留出 RX1 的固定 clean 正确集合 1,004 条混用。ANTI 的最差 seen RX 为 RX6，full 58.9683%；各 TX full 召回率为 59.0476%、54.0476%、60.9524%、51.9048%、59.1667%、79.7619%。没有全 RX 或全 TX 恢复的结论。

沿真实前向的观察结果：ANTI 全体默认头间隔／30 为 clean 0.237120、full 0.059124、控制 0.076131；BLOCKED full −0.014995、控制 −0.000085；unlocked full −0.017468、控制 −0.002080。主模型困难状态在控制后仍没有稳定正间隔。

BLOCKED 的 SIG_TIME 与自身 clean 的平均余弦仅从 0.083324 升至 0.154191，unlocked 从 0.081489 升至 0.156801；joint160 对应 0.871666→0.890969 和 0.870090→0.890144。时域／频域投影仍有较高整体余弦，不能据此宣称身份线索完整，余弦和梯度不替代识别收益。七处分支及两个既有门控的 CE 中间导数全部有限、均有实际梯度，未发现未用输出或零导数；这是 eval CE 的局部导数，不是训练梯度健康或收敛证明。

ANTI 频域原门控在 BLOCKED 的均值从 full 0.061947 变至控制 0.221386（clean 0.169858），PA 原门控仅从 0.452543 变至 0.459113。该共变没有隔离门控因果效应，不能把失败归咎于门控或据此改门控。没有新增分类头、分支或部署模块，没有把任何 branch／embedding／gate 强制对齐。

接收状态与物理审计：实际导入 channel 为 /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py；residual 为 /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/residual.py，来自 experiments/adv3b02_xuc/code/leo_practical，不能用另一 code/ 下默认值替代。25 MHz，IQ 256，fc 2462 MHz；residual、post_sync、EQ=false（G=1）、reference_snr_db=30、reflect；质量适配开启，锁定阈值 15／5 dB，相位跟踪及 IQ 补偿配置均开启。实际相位跟踪 2,900／5,040 条，unlocked 全部关闭；IQ 补偿 5,040 条均实际应用。AGC 配置上限 30 dB，本轮 0 条达到限幅；这是原增益标量限幅，不是波形削顶。FIR 数值延迟均为 24，残余核长度均为 62，边界 reflect，保留原全 256 点，不裁切以提分。

全体接收质量代理均值 10.8521 dB，实际 signal_to_added_noise_db 均值 10.8860 dB；二者均值相近但逐样本差异为 −4.9811 至 +2.4082 dB。按相同 5／15 dB 阈值描述性重分层会有 122／5,040 条不同，其中原 unlocked 有 46 条实际新增噪声 SNR>=5 dB。原接收状态从未改写。这不是总 SNR，也不能用此差异更改测试接收机。BLOCKED／unlocked 的实际新增噪声 SNR 均值为 2.1507／0.8391 dB。原始录音中的未知 TX／RX 失真保留，未做反演。

控制效应严格定义为：原 full 一次生成，trace 中 signal+noise 乘原 AGC 后与原输出逐位完全重建；仅移除该 trace 的新增噪声，再乘同一个原 AGC 标量。原接收质量、lock、相位／IQ 补偿参数、传播核／残余核、边界、数值延迟均固定，没有重算接收状态或 AGC。控制中实际信号功率变化是该干预的一部分，不能冒充“去噪并重新运行接收机”的总效应。本轮未测后者，不将受控效应推成总效应，也不把人工无噪控制当可部署去噪器或正式测试结果。

合法输入与来源：两模型来自 P8 同一 fold、同一 seed 2026092701，从零 E20／4,440 更新、FINAL20 student，默认头原联合训练，0 上游模型／优化器／EMA 继承、0 目标训练或目标选模。逐项核验原 config／实际 args／初始化／角色／冻结记录后严格加载。fit RX3/4/6/8，day1/2/3，六 TX，72 个 TX×RX×day 单元各70条，源 carrier split392005／VALIDATED_ONCE 复用。缓存实际物化原 L6300、未使用 V 和旧特征，但只有精确 fitL5040 IQ／可见标签／角色字段被消费；U、V 和旧特征数值不入前向。RX1 留出数据及任何目标数据不在本轮消费范围。

验证：47 个协议负例、96 个合成 NN（0 真实 IQ／checkpoint、0 更新）通过；原前向与观察挂点 logits 逐位一致，full 重建精确、控制 AGC 不重算、参数缓冲不变、.grad 无累积、NoExplicitCFO 第27坐标为零、临时钩子移除，固定整数半差距边界核对通过。唯一独立 P0/P1 审查 PASS，无发现。新增离线评分／登记校验及编译通过，0 模型前向。远端一次归档传输比较 VERIFIED、compile PASS；两份真实 checkpoint 各2条零输入 no-query smoke PASS，发生在任何源 IQ 之前，随后立即执行。独立读回 PID／CWD／argv／GPU／日志增长和两 row 完整结束，均 VERIFIED。没有重启或修改其他健康任务。

独立评分消费全部 30,240 个源决策和两份完整物理记录，检查全部物理 ID 配对、跨模型物理观测逐条相同、六类 logits／argmax／CE／间隔尺度、角色／覆盖、冻结状态和预算。重算的整数判据与在线计数一致。新增 NN=30,244（含4条真实 checkpoint smoke），240 次 autograd.grad、30,240 包等效中间导数，**0 新 epoch／0 优化器／0 EMA／0 另训头更新**。RAW10／ANTI probe 实测墙钟 16.5594／15.2269 s，进程启动到 summary 约18.3907／17.1197 s；并行启动到 dispatcher completion 约21.9927 s，boot 时间戳精度1 s。离线重算、绘图不增加模型前向。

每个被诊断父模型的既有成本单列为 E20／4,440 更新／6,959,958 NN；父 P8 的原训练墙钟 RAW10 1,563.7967 s、ANTI 3,359.4108 s，父全矩阵三 row E60／13,320 更新／20,879,876 NN 已在原报告核算，本轮不重复计为新训练。没有本轮训练曲线；父完整曲线仍在 [P8 报告](cvs_FcrossRX_noise_pair20_20261017.md)，未用诊断曲线替代训练曲线。预算耗尽不是收敛声明。

本轮实现位于 experiments/cvs_f_postas_noise_probe/：固定 design／channel／原前向 observer／probe／runner／dispatch／publish；checks 为本地最小验证，analyze 为独立全量评分，harvest 仅取预测与小型报告，plot 仅读离线汇总。实际发布代码 commit 为 eacbeb92ec4547671ffdb4ef92a43f433a0473a0，已 push 并独立 GitHub OID 核对 identical（ahead0／behind0）。后续交付 commit 另记 git_delivery.json；仅修正图例位置和归档分析，无真实推理重跑。完整命令、CWD、seed、逐 row 矩阵、checkpoint 及输入输出位置见同 run experiment.json；发布前冻结文本保留 preregistration.md。

本地复现独立分析命令：

```text
F:/App/miniconda3/python.exe -m experiments.cvs_f_postas_noise_probe.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_postAs_noise_control_20261017/final
```

CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014。原始 IQ／权重／完整预测／源 ID／日志不入 Git，仅引用原存储。

实际产物：E:/type10-7/local_artifacts/cvs_FcrossRX_postAs_noise_control_20261017/final/analysis_summary.json、source_strata.json、mechanism_strata.json、mechanism_details.json、cost.json；26 个完整回收文件（0 IQ／权重），8 个 PNG/PDF 图（4 张图，每图两格式）在 final/plots_visual_r02，已逐图查看。首版图保留 final/plots；仅第二版修正遮挡图例，新增 NN=0。以下为已核实图路径：

- [全部／困难状态](<E:/type10-7/local_artifacts/cvs_FcrossRX_postAs_noise_control_20261017/final/plots_visual_r02/accuracy_and_hard_states.png>)
- [ANTI 状态与 lock](<E:/type10-7/local_artifacts/cvs_FcrossRX_postAs_noise_control_20261017/final/plots_visual_r02/ANTI_state_and_lock.png>)
- [原默认头与现有分支](<E:/type10-7/local_artifacts/cvs_FcrossRX_postAs_noise_control_20261017/final/plots_visual_r02/default_head_and_existing_branches.png>)
- [既有门控](<E:/type10-7/local_artifacts/cvs_FcrossRX_postAs_noise_control_20261017/final/plots_visual_r02/existing_gates.png>)

回答最终性能问题仍以 P8 留出源 RX1 证据为限：相对同预算匹配 RAW10，ANTI 的收益集中于 LOS／PARTIAL 和 locked，BLOCKED／unlocked 未改善；默认头 urban 均值 +10.3175、low +6.0317、clean +6.1111 个百分点，low 仍仅33.7302%。相对 INDEP 的额外收益为负，故主采样方案已关闭。该来源不是正式目标结果，只有一个模型 seed，无法声称跨 seed 重现。P9 人工控制本身不产生训练收益或 clean 代价。

未完成：有效 A_g、联合因子矩阵及交互项、正式 clean＋全部六 practical 的新联合模型 truth-last 评分、跨 seed 复现、low urban>60% 与 suburban 距 clean<=2 个百分点的联合目标。下一步不能继续调噪声相关性或按当前辅助统计堆一致性损失；需找到能独立改善合法留出源 RX 身份识别的 IQ 机制，再重新登记受控训练。当前目标仍 ACTIVE。

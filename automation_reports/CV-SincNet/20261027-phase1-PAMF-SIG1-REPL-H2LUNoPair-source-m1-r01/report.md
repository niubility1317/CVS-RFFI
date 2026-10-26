# 单视图 REPL：双 L/U 信道增强，关闭低置信同包一致性

当前结果：唯一 seed2026092701 已完成 E50 和源四景，raw 98.340741%、H3 76.725926%，未通过原固定源门槛。这是单种子源域结果，目标双提升仍未达成；后续仅按两项预登记的共同源选择计划执行。

以下为初次登记与逐阶段记录。

本试验补齐一项现有源消融：保留 H2LU 的全部 L/U 双信道视图、全 L 真标签 CE/As KL0、原 ordinary 浅中/joint 跨 RX、高置信 raw95 PL、weight6/cap1/style0，只关闭低置信同包 Ha/Hb 互概率一致性。不增加 L-anchor 项、网络或前向；被拒绝 U 的计数继续记录，但该项 loss/weight 为 0，不能称其已得到低置信优化。

依据来自此前完整源结果：U 双 H 的 raw/H3 正确数为 26549/62255，增加低置信 pair 后是 26547/62199。源 V 的拒绝子集中存在两 H 同错、甚至三 H 同错情况。这不足以证明 pair 有害，却说明不能先假定该项有独立收益；原矩阵缺少“L 也双 H、但无 pair”的单因素。此处补齐它，正在运行的 L-anchor 学生完全保持原配置。

仅 seed2026092701，从自己的同契约普通 PL 单 SIG_TIME scratchE20 分叉 +10+20 到 E50/11100（13L37LU）；不继承 H2LU E50 或 L-anchor 学生。完整原 PAMF 和默认 6×160 联合头保持。训练只读 clean/直接 practical residual/post_sync/noEQ IQ，U TX 隐藏、RX/day 可见。V27000×4 冻结评价，已有 H2LU 参照不重训、不重复预测；无 G/恢复 IQ、目标或新诊断头。

固定源门槛 raw≥26568/27000、H3≥62109/81000，与 L-anchor 源试验相同。两个独立源试验都完成后，在过门槛者中按 raw 正确数、H3 正确数依次排序；完全同分选无 pair 以减少训练损失计算。没有通过者则保留负结果，不扩种子或目标。只允许另行登记选中方法的三 F seed 正式确认，源通过不是目标成功。源选择计划在两个新源结果产生前固定，见 [source_selection_plan.json](source_selection_plan.json)。

此前主目标结果已公开，本试验不宣称新的独立盲测；不根据目标弱类、RX、分数调整损失、权重、门槛、候选顺序或选择性重跑。原目标 raw clean≥85.98174603174603%、六景>72.0961970899471%，且较匹配监督/普通PL clean≥+1pp、六景>+1.5pp 不下调。当前是未启动的新源小试验，不能写成已经有效。

本地 23 项 focused 检查及 compile PASS，其中新增回归验证禁用项不执行 peer 计算且原 U 梯度保持；独立 P0/P1 审查尚待完成。真实 E20 联合 backward smoke 是 launcher 第一步，PASS 后立即训练；源 E50 后自动冻结并评价四景。GPU0–7 按全球 CUDA+pending<2/free≥10000MiB 调度，计入其他 owner、保护健康任务；无空位仅 CPU 控制器排队。独占输出、唯一 owner、系统失败停止新派发且不自动重跑。

代码、环境/命令/CWD、祖先与数据句柄、精确矩阵和产物见 [experiment.json](experiment.json)。当前目标 ACTIVE。


本地验证收尾：23 项 focused 和 compile PASS；本候选唯一独立 P0/P1 审查 PASS（0 P0、0 P1）。真实 E20 smoke 及新学生尚未启动。见 [验证范围](local_verification.json)。


## 实际发布、检查通过与训练启动

VERIFIED_RUNNING。执行 commit `99550ef10a6cfcffa5e6fd2cfd204f45c2e11d47`，发布前本地/上游 OID 一致。第一次入口 CWD 误指 carrier，读取本地 SSH 配置即退出；独立核实远端 run/release/archive 均不存在后，改从项目根目录复用同一冻结 archive 继续，单 archive 传输比较 VERIFIED、远端 compile PASS。没有更改模型、预算或现有进程，不能把最初提交失败写成成功启动。

真实 E20 联合 backward smoke PASS：L768、U256×6，L/U 两 H 实际不同；H2 state 的 dual_L=true、pair_enabled=false、pair_weight=0，无低置信 peer 或 L-anchor 单项梯度。原完整 U 目标对 SIG_TIME/head/other 的梯度有限且非零。检查没有 optimizer 更新、U 真值、目标或恢复器读取。

唯一学生已在 GPU0 实际启动。控制器/worker 的 OS 身份、CWD/完整命令及显卡归属独立核实，控制器不占 GPU。先核算当时全部完整可读 24 条训练记录，L768/U6、双 H、peer loss/weight=0、cap1/weight6/style0/无非有限跳步均符合；后一次观察完整记录增长至 93 条。第一轮尚未结束，stdout 字节暂未变化，增长证据是压缩逐步训练日志，不能把 24 条审计扩大称为 93 条或全 E50 审计。

正在运行的 GPU7 L-anchor 学生保持原配置。两学生都完成 E50 和四景源评估后，继续按先前固定 source raw/H3 门槛与排序决定是否进入另行登记的同一方法三种子确认；目标未达成。证据：[实际检查、进程和日志增长](launch_evidence.json)。


## 单种子 E50 源试验完整结果

VERIFIED_COMPLETE_OWNED_SOURCE_PILOT。执行 commit `99550ef10a6cfcffa5e6fd2cfd204f45c2e11d47`。同一方法唯一 F seed2026092701 完成原完整 PAMF/单 SIG_TIME/原 6×160 联合头、累计 E50/11100（原同契约普通 PL E20＋自身10＋自身20，本批新增6660步）；实际最终模型、父链、有限值和全部新阶段计数已核算。本次新读闭合记录6660条，复用已闭合 E30 审计0条，共6660；没有重训对照或额外 F/IQ/U真值/目标/G读取。

冻结源V27000包×raw/三种直接 practical H 共108000个新类别决策完整重算，60个RX/day/TX/overall单元及4个混淆/macroF1符合原评估表。相同物理ID、真V标签和视图的既有H2LU参照108000决策复用，配对纠错与新增错误均逐包计算，不是仅用混淆矩阵的边际差。

| 源V指标 | 本方法正确数 | 准确率 | 减固定H2LU参照（pp） | 原门槛 |
| --- | ---: | ---: | ---: | --- |
| raw | 26552/27000 | 98.340741% | -0.007407 | ≥26568 |
| 三H合计/均值 | 62148/81000 | 76.725926% | -0.051852 | ≥62109 |

本行未通过固定源门槛。参照始终raw26554、H362190，源V共享训练源RX，源通过不能保证未知RX的clean或目标星地收益；不得按本行单种子分数宣布三种子方法有效。两项源候选都完成后，仅在过门槛者按raw正确数、H3正确数排序，完全同分NoPair；无通过者不扩种子或目标。

| 源视图 | 本方法准确率 | 减H2LU（pp） | 相对H2LU纠错/新增错误 |
| --- | ---: | ---: | ---: |
| raw | 98.340741% | -0.007407 | 7/9 |
| practical_low_suburban | 91.218519% | -0.029630 | 78/86 |
| practical_mid_urban | 75.944444% | -0.066667 | 160/178 |
| practical_low_urban | 63.014815% | -0.059259 | 251/267 |

本批新增训练耗时合计6929.93秒，原E20成本另计；这是各实际GPU/并发下的墙时，不能当作硬件无关加速比。新增U曝光1701000、学生视图10206000；同一物理记录跨epoch/视图重复曝光，不能称为独立样本数。NoPair记录的拒绝数仅作诊断、peer优化曝光为0；L-anchor的valid查询是双H曝光，也不能当作独立包数或U真值准确率。

完整U梯度保护统计及每阶段曝光在CSV中保留。范数保留比例不等于样本利用率或AdamW参数位移；原other组包括SIG_TIME等多种参数，不能写成SIG_TIME专属保留率。source raw95接受/拒绝精度仅为冻结V代理，不是实际隐藏U精度。训练及评估始终只有clean/直接星地IQ，无恢复器协同。

原正式目标仍为同一方法三F seed的主168000 raw clean≥85.98174603174603%、六景>72.0961970899471%，且较匹配监督/普通PL clean≥+1pp、六景>+1.5pp。源单seed达标不改变原目标；目标分数不回流选模/调参/候选重排，后续正式确认须独立登记、预测固定后独立truth-last。

证据：[四个源视图](source_per_scene.csv)、[完整源分层](source_strata.csv)、[同包配对差](source_paired_strata.csv)、[源置信代理](source_confidence_proxy.csv)、[U曝光](source_U_exposure.csv)、[整体梯度保护](source_gradient_summary.csv)、[完整核算范围](source_complete_evidence.json)。


## 两项预登记源试验的共同选择

两项均未通过原固定源门槛，不扩种子或目标，保留全部负结果。 配方、门槛、raw/H3排序与同分规则未更改，未使用目标反馈。见 [完整固定规则结果](source_selection_result.json)。

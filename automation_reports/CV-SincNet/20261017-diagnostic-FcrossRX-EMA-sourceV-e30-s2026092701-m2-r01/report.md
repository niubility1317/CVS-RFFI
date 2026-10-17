# 完整源 E30 EMA 读出：源域负结果

状态：ANALYZED／VERIFIED。两个既有 FINAL30 默认头的 EMA 只读比较已自然完成；固定源假设失败，关闭这条读出路线。没有新训练、正式测试、参数或 EMA 更新。

| 源视图 | B00 学生 | B00 EMA | B01 学生 | B01 EMA | B01 EMA−学生 |
|---|---:|---:|---:|---:|---:|
| clean | 98.388889% | 98.388889% | 97.722222% | 97.666667% | -0.055556 pp |
| practical_low_suburban | 53.638889% | 53.861111% | 90.444444% | 90.333333% | -0.111111 pp |
| practical_mid_urban | 46.333333% | 46.666667% | 76.305556% | 76.305556% | 0.000000 pp |
| practical_low_urban | 38.111111% | 37.944444% | 62.555556% | 62.416667% | -0.138889 pp |

这是同一源 V1800 面板，satellite 数字合并两个信道 draw（3600观察），不是正式目标成绩，也不是两个模型seed。B01 EMA源low62.416667%不能写成正式low>60%；原scratchE30正式B01 low仍53.137500%、clean82.797024%，保持原独立truth-last产物，不重测或回流设计。

B01固定四单元增量（draw0 mid/low；draw1 mid/low）为−0.111111/−0.055556；+0.111111/−0.222222 pp，0/4达到+1pp。clean−0.055556pp，原“不降”条件也失败；RX3 clean−0.277778pp，其余四RX不变，RX保护条件通过不抵消整体失败。mid pooled修复5/新增5，low修复9/新增14；源low宏F1由62.216949%降至62.087753%。

low的BLOCKED/unlocked为1565观察，学生与EMA均34.952077%（修复6/新错6）；新增噪声比<3dB为1127观察，31.499556%→31.410825%（修复3/新错4）。新增噪声比不是总SNR。没有已证实被EMA缓解的困难退化；源分层亦不构成正式接收状态因果结论。固定FINAL30 EMA假设关闭，不改衰减、epoch、RX、seed、门槛或预算救跑；不推广为所有EMA方法无效。

## 实现、验证与实际成本

实际执行commit `317b5d75d319f8f4b5cba03791014c703d160749`；实现为experiments/cvs_fullsource_ema29/（runner/probe/stats/publish/checks），独立复算为analysis/cvs_EMA_source_E30_20261017.py。保持整个既有模型和默认头；无新IQ算子、新loss或部署参数。此轮排除读出稳定性这一具体解释，没有证明新A_g或联合收益。

本地3项unittest和语法编译PASS。一次独立限定P0/P1审查发现浮点门槛问题，最小修复及恰好+1pp/略低阈值反例通过，针对性复核PASS。普通账户CPU2预检、单归档单次传输比较、远端compile、真实两CP/4合成包no-query smoke、独立PID/start_ticks/CWD/argv/空CUDA/CPU环境及日志601→1277字节增长均VERIFIED；同一PID自然退出、completion齐全、stderr为空。无重复启动、重试、健康任务修改。

新增每模型12602NN、总25204NN；源训练/EMA更新/头拟合/教师训练/新物理draw均0，墙钟14.769812s。原两个父模型各scratchE30/6660/5407922NN分别保留，合计60modelE/13320更新，不混为本轮新增训练。实际缓存反序列化V27000和未用L/旧特征，仅V1800及既有full信道观察参与计算，U IQ/真值和target未打开。

独立分析只从保存logits/可见源标签及原源physics复算220个CM、96条状态/质量/RX分层和原固定判定；与运行结果一致，0IQ/CP/NN/更新。分析墙钟1.058210s。PNG/PDF均已取回，PNG已目视检查。无新训练曲线，原完整E30曲线保留在父报告，不伪造新训练。

本地产物：`E:/type10-7/local_artifacts/cvs_FcrossRX_EMA_sourceV_E30_20261017/analysis_r01`，含summary.json、pooled_results.csv、source_strata.csv、EMA_source_comparison.png/.pdf、completion.json；根目录另有实际completion/smoke/data_roles及发布和两次runtime读回。完整logits保留不可变远端原run，不入Git。

## 当前研究判断

本轮完成了一个源域反证：终轮自身EMA没有恢复clean与urban的共同收益。普通A_s的适应证据仍保留；有效A_g、同一模型的clean净正收益/正式low>60%/suburban接近clean和跨seed复现仍未证明，整体目标ACTIVE。下一训练候选必须有与已关闭路线不同的源依据，不能把换读出、局部指标或已失败算子重包装为新增强。

发布命令/CWD/唯一矩阵/科学判据/技术停止条件见experiment.json。代码commit首轮push连接reset，独立读回远端仍旧；第二次push连接443失败，本地commit保留。最终交付时另行记录独立远端OID核对状态，不能从push退出码推定成功。

## 原预登记

# 完整源 E30：自身 EMA 默认头冻结比较

状态：REGISTERED，未启动。上一轮补齐历史关闭路线，但没有性能新证据。本次仅检验现有训练的终轮学生读出是否掩盖了自身EMA的同时保留能力；不把读出稳定性称为A_g。

唯一假设：同一个完整源、scratchE30/6660 checkpoint中的完整EMA，在固定源V观测上能保持clean并改善mid/low urban。模型仍为F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、原cos30默认头、NoExplicitCFO与内部补偿；整个EMA包含原联合训练头，没有另训或替换头。原E30训练仍为20L＋10U，全部RX1/3/4/6/8、day1/2/3、L6300/U56700隐藏TX/V27000；无训练RX留出、无额外L_dev。

## 固定矩阵与判据

B00-EMA、B01-EMA分别读取原全源E30 B00/B01的own ema_model；严格使用既有own_model核对scratch来源、固定终轮、6660更新、原始物理role_ids和无目标选模。新训练和初始化均0。各模型2个合成包真实CP no-query smoke通过后，才打开原源缓存。保留两父各30轮/6660/5407922NN的历史成本；不能写成新训练或零累计成本。

只复用P24既有V1800（90个TX×RX×day单元各20包）、两draw、low_suburban/mid_urban/low_urban的完整practical condition0，原reflect/residual/post_sync/EQoff即G1/ref30/25MHz/2.45GHz及接收依赖完全不变；不新增物理draw或滤波。源缓存可能反序列化未用L、旧特征，只有V1800进入计算，U IQ/真值和target不打开。学生参照复用P26中同一物理ID、draw、场景、CPU2/batch90的已保存logits，不重做其前向或正式测试。

B01固定源规则：clean accuracy≥学生；全部5RX的clean各≥学生−.5pp；两个urban×两个draw的accuracy均≥学生+1pp。比较采用1e−10pp浮点容差，不调整科学阈值。全部四个urban单元必须通过；B00完整结果作读出控制，不能借B01结果改选B00。失败关闭固定FINAL30 EMA读出假设，不改EMA衰减、epoch、源RX、seed、门槛或添加训练救跑。通过也只支持读出层面的后续论证，不能称A_g、正式部署收益或跨seed复现，不自动访问target或扩seed。

每模型12602NN（2smoke＋1800clean＋2×3×1800sat），合计25204；0训练/优化器/EMA更新/头拟合/教师训练/新信道。每行保存全部logits、默认头accuracy/macroF1/CM、margin÷30、repair/newerror、源RXclean差及判据，独立程序从保存预测重新计算CM与差值。两信道draw不是两模型seed。

## 执行与保护

唯一owner /root，普通N607账号、CPU2、CUDA_VISIBLE_DEVICES空。命令和CWD、不可覆盖run/output root、精确CP和数据句柄见experiment.json；运行入口experiments/cvs_fullsource_ema29/runner.py。一次预检、一个归档单次本地/远端SHA对比、一次远端compile；首步实际CP smoke，PASS立即继续；一次独立PID/CWD/argv/CPU/log-growth或自然终态核对。不修改任何健康旧任务。角色/来源/非有限/模型状态改变/NN计数/输出冲突/系统错误保留产物并停本run，不自动重试；科学负结果也完成两个模型。

本地3项focused tests通过：5类metrics非法输入、6类判据负例/浮点边界、14类源角色/观察越权负例；0真实IQ/CP/NN/更新。语法编译通过。唯一P0/P1审查发现恰好+1pp被浮点舍入误拒绝，已加1e−10pp容差并新增1224→1242/1800边界及低1e−8pp反例；修复针对性核对后记录结论。无重复全审或额外gate。

当前不是实验完成或性能成功。正式lowurban>60%、clean净正收益、suburban接近clean、有效Ag和跨seed联合证据仍未证明。

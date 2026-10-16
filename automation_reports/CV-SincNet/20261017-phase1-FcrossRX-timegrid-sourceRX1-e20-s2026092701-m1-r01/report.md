# F＋跨 RX：有界包内时间栅格收缩

状态 RUNNING；实际发布、归档传输、远端编译及首步真实checkpoint无query smoke均VERIFIED／PASS。已读回唯一worker的PID／CWD／argv／GPU／日志增长，尚无FINAL20评价结果。run=20261017-phase1-FcrossRX-timegrid-sourceRX1-e20-s2026092701-m1-r01。起点 commit 56cf99125fc954e4c2edb4132afe30deff752dd5；实际发布 commit 在 release_commit.txt／同 run 登记记录。

前轮 P9 完整诊断为真实进展：两个冻结 own-default-head 模型、30244NN、0更新；主模型 BLOCKED／unlocked 去新增噪声仅恢复17.6018%／18.3118%的full→clean正确数差距，固定50%门槛FAIL，关闭当前噪声遮蔽理由。P8正负噪声配对未胜独立噪声，固定采样方案已关闭；相干IQ混合的跨seed clean收益未重复，LA16／相位／RXMIX等各自原失败保留。本轮不重新解释这些失败为成功，不重新扫其参数。

当前唯一新假设：既有普通增强不覆盖包内采样时间尺度的小幅变化；一个保留原样本信息、无人工边界的轻度时间栅格变换，是否能提高固定 F＋跨 RX 同 TX SupCon 的未见源 RX clean 识别。代码检索未发现 time_stretch／clock_skew／time_warp／resample_poly 的现有实现，登记查重为0。它与整包时移、固定分数延迟、幅度衰减、同TX波形混合不同；不继承它们的有效性假设。

研究动机仅参考 RIEI 对不同接收采样率的独立实验（[原论文 V-C3](https://arxiv.org/html/2411.03636v1#S5.SS3.SSS3)）；该研究用0.6至0.9倍采样率与样条处理，不是本候选的配方，也不能证明本算子的身份保持或CVS收益。当前256点自身不足以识别真实RX时钟；以下rho来自数学变化界，不是估计出来的硬件漂移或真实卫星范围。

固定算子 t[n]=127.5+(1−rho)(n−127.5)，n=0..255；只对原256个观测值做相邻两点的凸线性插值，I/Q同坐标，无padding、反射、循环、裁切、外插、RMS、AGC、CFO、相位噪声、噪声、多径、PA、DAC或拟合参数。rho∈[1/8192,1/4096]，真实RX/day/eq定义session、全部TX共享；每epoch私有流20261017701刷新，不推进native/loader/dropout/U流。不将TX标签用于时钟参数；TX仅原CE和覆盖统计。所有合法有标签L仍受监督，不做置信度过滤。

最大位移127.5/4096=0.03112793采样点，标称25MHz下约1.245ns；这只是数字坐标位移。线性插值矩阵非负、行和1、对角≥0.968872、严格对角占优，实数精确算术可逆；||M−I||2≤255/4096=0.06225586，因此全包相对L2变化≤6.2256%，不依赖标签或逐包强度适配。端点取样也始终在原[0,255]内。数值逆仅验证已知M，不是反演未知TX硬件；数值可逆不等于身份保持或有效Ag。

最小矩阵1新训练＋1结果复用：TIMEGRID-RX1-s2026092701从零E20=13L+7U，4440更新；匹配D_raw-RX1-s2026092701既有E20/8L/defaulthead结果和完整序列只读复用，0状态继承。原共同目标raw.75＋ordinary.25＋原time_down/t2.pool跨RX同TX SupCon完整保留；第四对原.25RD(raw)改为.25RD(timegrid(raw))，同IQ两次dropout。均8个L前向/包，总pair系数1.25，不以视图数重新归一化，不稀释共同clean绝对监督。旧ordinary仍只在共同支路生成一次；新增算子直接作用raw，不叠在ordinary后，不调用As。原LS=.01／RDrop alpha1／AdamW／cos／ownEMA／旧门控／13L+7U保持，0新增一致性损失、头、分支或参数。

模型固定F三路/SIG_TIME_X3、统一Sinc CENTER_UNITS、NoExplicitCFO、真实joint160、默认cosine30；骨干与默认头原流程联合更新，无另训诊断头。部署导出结构完全相同。新模型/optimizer/EMA全部scratch，不载E50/E200或任何父状态。独立真实CP兼容smoke只加载来源已知同合同D_raw FINAL20，2个合成no-query前向后丢弃；不是初始化。原初始化/祖先/选模/角色配置在smoke前核对，新worker再核对有效角色。

源权限：原ManySig split392005 VALIDATED_ONCE carrier，fitRX3/4/6/8、day1/2/3、全部6TX，L5040/U45360(hiddenTX)/V21600；heldRX1 L1260只FINAL20评价，heldL/U不训练。复用角色句柄，不重验数据、不增加标签，不冒充fullL6300或内部5580/720。已有fold准备的1404额外L读取（held1260＋144 unused anchors）照实保留，unused anchors丢弃；不新增伙伴bank或IQ缓存。V仅健康，不更新或选epoch。源RX1是已暴露研发折，不称新盲测。

事前源判据：TIMEGRID相对精确D_raw，held clean accuracy≥+1pp，clean macroF1不降，fitV均值下降≤.5pp；完整4440更新、原默认头联合更新、原L/U全5994批顺序相同、72个TX×fitRX×day和240个epoch×RX/day session覆盖，实际变化界与有限状态符合，最终共同base objective≤.5。任一失败关闭固定timegrid，不改rho／插值／概率／权重／RX／seed／epoch／门槛救援。独立Ag资格按clean端点；全部practical仍完整报告，不能用clean收益冒充星地保持。通过仅支持另登记第二modelseed源确认，不自动进入target或联合矩阵。

FINAL20固定学生默认头自动生成源clean＋六原practical：high、mid、low_suburban、high_urban、mid_urban、low_urban。实际25MHz/IQ256/fc2462MHz/residual/post_sync/EQoffG1/reference30/reflect、原接收质量/相位与IQ补偿/困难包和source_eval命名流20261016103均保持；signal_to_added_noise_db不是总SNR。0训练As/0正式target访问，不能从源指标宣称正式low>60或跨seed成立。最终目标仍clean净正／low>60／各suburban距自身clean≤2pp的探索目标。

预算1新E20/4440更新/4546560LNN；按现有原U/EMA/V/held计数预计每row5823318NN，dispatcher真实CPsmoke另2。568320个新timegrid视图，无额外U或teacher前向；240个session，0新独立物理L。D_raw历史E20/4440/5823318NN成本单列，不是本次新训练或免费父模型。2060个本地合成NN/2合成更新、0真实IQ/CP单列。实际墙钟/构造成本/更新/NN以最终记录为准；预算用尽不是收敛。

本地测试PASS：23协议负例；2060合成NN/2更新；矩阵非负行和/严格对角占优/逆重建误差4.44e−16和变化界；无padding/RMS/零包保留；session与TX无关、epoch刷新；新旧206初态、共同损失和全部共同参数梯度逐位一致、RNG结束相同；实际8L前向中第四对同IQ、默认头联合更新、导出160精确。已有NumPy2 ABI提示非fatal，未安装包。唯一独立静态P0/P1 PASS，无发现。一次纯配置准备误用不含Torch的Python13，仅导入失败0IQ/CP/NN，已切回已验证SSRPy10路线，不是科学运行失败。

环境/完整命令/逐row配置见experiment.json。普通N607 CVS-RFFI Python/fullFP32，CWD releases/cvs_FcrossRX_timegrid_sourceRX1_E20_20261017_r01；一个owner codex/root/FcrossRX-timegrid-20261017，一个worker，所有owner/preCUDA每GPU≤3且free≥12000MiB，CPU2。唯一不可覆盖runs/20261017-phase1-FcrossRX-timegrid-sourceRX1-e20-s2026092701-m1-r01。一次N607预检、一个release归档一次本远SHA比较、remotecompile；launcher首先真实CP no-query smoke PASS即继续。独立PID/CWD/argv/GPU/日志增长确认后为VERIFIED。

stop rule只针对本run非法输入/来源、非有限数、错误预算/冻结/输出冲突/不可启动等系统技术失败；保留partial，无自动重试，不停止或热改任何其他健康进程。低分不停止健康训练。旧额外设计审批、重复review/configSHA/seal等记REJECTED_EXTRA_GATE，不阻塞。

预期：逐row实际args/modulefile/contract/init/defaulthead/sinc/channel；全20轮/4440step/5994主L-U批次；完整568320timegrid trace/rho/功率/变化界；7源视图默认头预测、间隔/30、physics与独立评分；class/day/state/lock/quality/noiseSNR/CFO/kernel/AGC/boundary分层，固定D_raw clean正确集保持与修复/新错，训练曲线、覆盖、实际成本、全部负结果和commit。无IQ/权重/原始日志/物理ID入Git。当前B00/B10/B01/B11与交互项、合格Ag和正式跨seed联合突破未完成。

当前实际读回：E8／20、1776更新、227328个新视图、96个epoch/session、72个TX×RX×day单元；最大相对L2=0.03012570。worker仍存活、同日志持续增长，没有技术failure。最新原始状态在本地artifact monitor_latest.json，不能把此段当后续实时状态。

实际args：seed2026092701、20/13/7、LS.01、baseline_ckpt和teacher_ckpt空、source_fold_held_rx=1、use_aug=true、phase1_lr_schedule=cosine、lr=.0002/min=.000001。新旧科学数据契约逐字段无差异（仅忽略三项说明性metadata），有效L/U/V为5040/45360/21600。actual_modules记录真实leo_practical为adv3b02_xuc路径；ordinary_augmentation_actual在final阶段才写出，当前未生成，不据默认值假定它已记录。

训练发布commit fee77907549a89722dd20736498308fa640dd5f3已push并经独立GitHub OID核对identical；本地后续analyze/harvest/plot属于离线工具，不热修改远端健康release。离线读取器已解析匹配D_raw完整20epochs/4440steps/7sourceviews，六类oracle及4负例PASS，0新NN/IQ/CP。最终比较仍须等新row完整冻结和七视图结束，再独立全记录核对。

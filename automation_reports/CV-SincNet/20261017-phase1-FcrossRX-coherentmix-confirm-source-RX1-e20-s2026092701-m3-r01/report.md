# 真实 IQ 相干插值：seed01 独立确认

LOCAL_VERIFIED，尚未启动。一次限定 profile/launcher P0/P1 审查 PASS，N607 普通账户预检 VERIFIED。起点 commit 444f19e6937112a85b16b5ad4177c1a4af8fc05d；实际 release commit 在发布时记录。本轮唯一假设：已在 seed02 源 RX1 获得的 clean 独立收益能在 seed01 的匹配三行矩阵中保留，不改变算子、强度、参考窗、阶段或 U 规则。

前轮已完成：seed02 raw81.190476%、SAME81.269841%、CROSS83.492063%；CROSS 相对 raw clean+2.301587pp/F1+2.308524pp，相对 SAME clean+2.222222pp；fit V−.050926pp，全部预算/默认头/覆盖健康项通过。urban 均值−.978836pp，low−.238095pp，BLOCKED/unlocked 下降，未得到联合收益。对照实际扰动 L2 均值 SAME .07360/CROSS .09492，未逐包等强；不能唯一归因 RX 内容。本轮只确认固定配方的识别收益，不将单 seed 结果视为有效 Ag，也不把不同实验增益相加。

固定 F＋跨 RX 同 TX SupCon、三路 SIG_TIME_X3、统一 Sinc CENTER_UNITS（Fs25MHz、中心单位25kHz）、NoExplicitCFO、joint160、默认 cosine30；900722 参数，骨干及既有默认头正常联合训练，不加部署分支、头或新损失。每新行从零 E20＝13L＋7L/U；原 RDrop LS.01/KL1、old gate/own EMA、AdamW/cos 与普通增强原样保留。原 ordinary 含 PA/shift 等共享操作，不称为完全无增强；新增 IQmix 不施加普通增强/完整信道，不引入 PA/DAC、额外噪声、多径、时移或 CFO 身份输入。

数据复用 VALIDATED_ONCE carrier392005。fitRX3/4/6/8、day1/2/3、6TX：L5040/隐藏U45360/V21600；held sourceRX1 原 L1260 仅供 FINAL20 默认头读出，不作训练、校准、EMA、donor或缓存特征。不是 L6300 或 5580/720 契约；U 真值隐藏、V不拆分。配对仅 filtered 可见 L，同 TX/day、不同物理包；SAME 同 RX，CROSS 均匀取其他3个fitRX。session不由TX定义。实际角色ID与 seed01 raw 对照严格相同；训练、教师、bank均不使用 heldRX1。

矩阵仅3行：reused_D_raw-RX1-s2026092701（既有匹配默认头结果/轨迹只读复用，不启动、不加载为初始化）；SAME_RX-RX1-s2026092701、CROSS_RX-RX1-s2026092701（两条新 scratchE20）。model/global ordinary seed2026092701；split/data392005、伙伴20261016961+step、evaluation20261016103、原source session流均不变。不是只改 args.seed：runner 固定 backend7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b，显式激活真实 F 构造配置为 seed01。未知 profile、其他 row、预算或配方覆盖拒绝。R2 实际发布归档和全部旧产物保持不可变。

算子原样复用：参考窗[80,160)估计 donor→anchor 复 LS 标量 a；v=.75x+.25ay，恢复 anchor 全包能量，再在实512维球面上限制弧长为2asin(.1/2)，确保相对 IQ L2≤.10。退化返回原 x 并记录，不重抽、不删样本或设学生置信门控；配对标签只选合法同TX伙伴，全部系数/公式统一。不是同步同一次发射的两RX，也不反演/替换真实 RX/TX硬件。独立生成一次 IQ 后用于该对两次随机子网络前向。

三行同8L前向、总 pair 系数1.25。共同 base=.50R(raw)+.25R(ordinary)+.25R(raw第三对)+原 ordinary-only SupCon；第四对 .25R(raw/SAME/CROSS)。共同 raw绝对.75与ordinary.25不变，新增view不稀释base，物理样本逐view均值独立归一化。无跨view一致性、As训练或新增U视图；原13L+7U自身EMA和门控保持。伙伴私有随机流不推进原初始化、loader、ordinary、U/dropout流；完整主L/U序列与原reference全量逐条核对。

固定主资格与 seed02 相同：CROSS 相对raw clean≥+1pp、相对SAME≥+.5pp；clean宏F1相对两者非负；fitV相对两者下降≤.5pp；完整20/4440/8L/3330SupCon、base最终≤.5、每行568320mix/5040伙伴/完整72或216单元、10%界、无额外teacher/U/As/headfit/target或继承。仅1e−10pp浮点容差。单独Ag各practical变化全部报告，−.5pp保持标记仍不阻断clean资格，不把此资格当联合或部署成功。两seed方向/均值/逐TX/day/状态/quality均保留，未按目标结果调参。

确认FAIL关闭当前固定相干插值配方，不换seed03/RX、幅度、参考窗、阶段或预算救援，不事后晋级SAME。PASS仅支持源域可重复clean效应；仍须其他源证据及 B00/B10/B01/B11、交互与固定正式测试验证，不自动加载target或宣称low>60%。suburban距clean≤2pp保持探索目标，不能靠降低clean缩差距。

一次真实CP no-query smoke：只读 20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01/D_raw-RX1-s2026092701/source/final_ssdg.pth；来源scratch20/4440/final-only/同角色/无祖先和target，严格load后2合成包无更新，丢弃该模型。两学生、EMA、optimizer从零，不继承 seed02 或 E50/E200。兼容性 smoke 与正式训练来源区分，新增训练无父权重成本；复用raw历史20E/4440/5823318NN单列。

新增预算40modelE/8880updates；每条4546560L NN、原U/EMA/V/8820held，预计5823318总NN但以实际计数为准；每行568320伙伴曝光/混合、bank5040读取，唯一可见L仍5040。另 dispatcher smoke2；本地本轮0NN/更新，仅3个模型初态比较；原3090合成NN/3更新/20负例与16scope检查作为已验证未变代码证据，不重复。E20不是收敛证明。

每行完成后自动 source clean+六种原practical、各1260，完整默认头间隔/30、physics和分层保存。25MHz/IQ256、residual/post_sync、EQoff G=1、reference_snr_db30、接收质量/补偿/边界/SNR全部按原引擎。记录实际模块 __file__；不能将默认值冒充运行值，也不能将新增噪声SNR当总SNR。0正式target及truth服务；源域结论与正式测试严格区分。

命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_coherentmix_confirm_source_RX1_E20_20261017_r01/experiments/cvs_f_coherentmix20/runner.py。CWD同release，普通N607账户。owner codex/root/FcrossRX-coherentmix-confirm-20261017；每GPU全部owner≤3、free≥12000MiB，2worker/CPU2/loader0。独占 run root PROJECT/runs/20261017-phase1-FcrossRX-coherentmix-confirm-source-RX1-e20-s2026092701-m3-r01，禁止覆盖/重复启动，不动其他健康任务。

技术stop仅本run协议/角色/来源、输出碰撞、非有限、越界或不能合法预测的系统失败；停止pending、保留partial和健康worker，不自动重试或因科学分数停训练。一次限定P0/P1、一次N607preflight/单归档本远比较/remotecompile、smokePASS即继续、独立PID/CWD/argv/GPU/log增长和自然终态，再独立全量读回/图/登记/Git。REJECTED_EXTRA_GATE：无单独设计审批/报告hash/新数据复验或重复不变代码审查。

本地25个定向负例通过：实际构造与native seed一致、SAME/CROSS/同seed raw初态完全一致、全部科学配置等于seed02；直接file入口到达argv保护。仅初态构造、0真实IQ/CP/NN/更新；本地已知NumPy2/Torch2.1警告不影响退出0与PASS，不改环境。

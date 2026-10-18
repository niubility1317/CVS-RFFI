# 连续 E30 单 As 技术恢复：最小预登记

状态：VERIFIED_RUNNING。发布、远端 compile/bootstrap 和实际启动均已完成；唯一 As 训练进程独立核验存活、CWD/argv/CUDA 正确、日志增长。用户明确允许补跑单个As E30。原B01完成20轮后因旧E20guard退出，无可续训模型；只修执行边界及恢复路由，不改变训练机制。原失败不是科学负结果。

唯一新训练B01-s2026092701，scratch E30/13L17U/6660updates/6L前向，L总尺度1.25，原ordinary/mixed/raw RD、原跨RX同TX SupCon、U/EMA/门控、固定F/默认cos头均不变；segment20仅scratch附着键，0offset、无父权重。角色L6300/U56700隐藏/V27000、源RX1/3/4/6/8、day1/2/3、split392005、eq1/IQ256/25MHz/RMS centerfalse原样。不留训练域，不训练诊断头。

完整As保留原训练三场景low_suburban/mid_urban/low_urban，residual/post_sync/EQoff G1/reference30/fc2.462GHz/reflect及原质量、相位、IQ、AGC处理。真实模块为旧pinned release的adv3b02_xuc/code/leo_practical；正式receivedIQ、六practical及SNR不改。

匹配控制只读复用 20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01/B00-s2026092701：固定FINAL30/6660、原source与全部七视图prediction/scoring；不加载它初始化As，不重训重测，不改run_id。实际旧控制配置逐项相同；新As较失败行仅run_id/output_root/launch_owner三项变化，已在真实依赖中0NN核对。

原source判据不变：As完整源V clean≥匹配NoAs；较旧P43 stagedAs clean≥−.25pp、每源RX≥−.5pp；原固定1800×2draw mid/low四项均≥旧As+1pp。任一失败关闭固定连续轨迹，不用正式分数救援；固定注册As仍全部七视图预测与独立truth-last。新阶段额外报告matched clean下降≤.1绝对百分点；该报告不改source资格，不以单seed宣称复现。

每rowFINAL30冻结后自动12600NN源panel、clean及六practical各168000固定IQ预测，全七prediction完成后独立CPU score，报告准确率/F1/最差RX/urban均值/paired修复新增错/source状态与质量/完整曲线和成本。旧stagedB00/B01仅固定参照，不成为额外seed。

代码/config commit：6f94261642df1fbeba1aa69bae3350259b59aa17。CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_continuous_recovery_E30_20261018_r01。命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/experiments/cvs_continuous52/runner.py。输出：/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01；日志/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01/logs/B01-s2026092701.log，dispatcher /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01-launch。owner codex/root/continuous52-technical-recovery；按原全owner每GPU≤3且free≥12GB容量调度，健康其他任务不动。

原候选一次P0/P1与E30 guard/NoAs直接缺陷定向复核保留，补跑路由有持久化负例，不新增runner/full重复review。14项stdlib检查PASS；E30/6660真实F合成hook两行1536NN通过，同初始state、实际6L、As0/64、默认头/Sinc有限非零梯度，0更新/真实IQ/CP。原tensor_iq桥已保留。真实checkpoint2NN无query已在本launcher第一步PASS；两行合成验证仍1536NN，PASS立即进入唯一获准As训练，不另设token/授权artifact。

一次N607preflight已通过；新run/release均不存在，旧控制完成和旧失败保留。单archive传输一次SHA比较与远端compile/bootstrap均VERIFIED，实际dispatcher1130384、唯一worker1130449（GPU0）已独立核实PID/CWD/argv/GPU/log增长。技术stop为本run角色/来源/非有限值/冲突/非法预测/启动失败：保存产物，不自动重试，不停止健康其他owner；科学FAIL不停止已登记正式测试。

成本分列：新30E/6660更新/5114880LNN；复用控制30E/6660；失败As20E/4440及失败步部分成本保留。成功补齐后实际累计80模型epoch/17760成功更新，不把比较60E当全部实际投入；学生/教师/验证/源panel/正式prediction、1538共同smoke和墙钟按实际产物计数。失败As总NN未知，只保留已知下界。

用户另允许表现好的方案总预算E50：仅合法源域资格支持后、正式分数不回流地另行预登记匹配延长；当前仍固定E30，不热改。若后续30＋20，明确父/增量/总成本与重置策略，不冒称scratchE50。当前连续As尚无新准确率，也无跨seed双提升证据。另一对话已实现固定原As与NoAs的E30+20匹配延长（run20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01，code115dc1c9），本owner复用其结果，不重复启动E50。

独立后处理计划：新CPU-only release cvs_FcrossRX_continuous_recovery_analysis_20261018_r01；输出本run/analysis_auto_r01。复用旧NoAs与P43两行，对新As完整30轮日志、源状态/质量/覆盖与固定七视图保存预测做四行配对分析；先源分析、再等全部正式prediction/scoring completion，最后独立truth-last读取保存预测评分。0新IQ/CP/模型前向/训练更新。原失败run和其analysis_auto_r01不改。代码与命令随后记录到同报告；技术失败保存而不自动重试。

后处理实际代码 `d51c6068e4d94a83a26914a92fc107bc3f4850fc` 已提交并独立核对远端分支 OID。单归档传输比较、远端compile/import PASS，11项合成检查远端PASS（0.008s）。CPU分析PID1139252已独立核验CWD/argv/start_ticks/GPUdisabled；当前WAIT_SOURCE，观察到本run dispatcher和唯一As worker，0NN/IQ/CP/updates，尚无正式结果。完整命令及路径见analysis_plan.json。

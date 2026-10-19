# 完整 As 后前缀屏蔽（预登记）

REGISTERED；owner codex/root/as-prefix81。起点 2467287bec9a9d9d078fafd30df526ce3bade930；实际发布代码 commit 写入 release_commit.txt，启动后回填登记。

保留完整 practical(raw)、普通/raw 监督及固定 CVS 默认头。P79 的边界敏感性不是屏蔽有效的证据。唯一假设是：终端观测屏蔽能否减少人工前缀依赖并改善原 reflect 识别。等长随机非参考区段控制一般信息损失；P26 padding混合失败不等于本假设已经检验。原始记录只有256点，无真实前历史可用。

| 条件 | 来源及区别 | 新预算 |
|---|---|---|
| NOAS30 复用 | 原 P43 B00 自身 NoAs20＋10，父轨迹不同 | 0新训练/预测/评分 |
| AS30 复用 | 原 P43 B01 自身 As20＋10 | 0新训练/预测/评分 |
| RANDOM30 | 原 B01 As20 student＋ownEMA；完整As后随机非参考段屏蔽 | E10/2220更新 |
| PREFIX30 | 同一个原 B01 As20 独立分叉；完整As后前缀屏蔽 | E10/2220更新 |

两条新行累计 E30/6660更新，不加载 LRaw/P78 训练权重。F＋跨RX同TX SupCon、SIG_TIME_X3、统一Sinc、真实joint160、默认cos6×160 scale30，900722参数/206state。正常联合训练，AdamW重置、原cos10/lr0.0002/weightdecay/Sinc单位缩放。六次L前向768包/step、L总系数1.25、AsCE名义.25，原U student/自身EMA不变，0新损失/参数/NN。新预算合计20模型epoch/4440更新；历史父成本与新模拟、NN、墙钟分别报告。

屏蔽概率.5，长度由 actual residual_kernel_length−1 获取并检查61，不使用24点延迟替代。PREFIX[0,61)；RANDOM起点均匀取整数[0,19]∪[160,195]。两者共享命名启用流，RX/day/eq/sig和累计epoch/step刷新，不按TX定义。完整物理传播、质量、补偿和AGC先执行；随后两路IQ归零，不重算状态、不归一化，不原位修改；记录移除能量，不声称能量匹配。一次屏蔽供原RDrop两前向复用。参考[80,160)直接样本保留，GroupNorm及功率比例仍会变化。正式完全不屏蔽。

数据复用 split392005/VALIDATED_ONCE：L6300、U56700 TX隐藏、V27000，源RX1/3/4/6/8全部参与、day1/2/3，无Ldev/留RX；源panel1800×两draw，只有一个模型seed2026092701。TX六类原映射、目标RX0/2/5/7/9/10/11 day0..3。Phase1 capsule/support/K不适用。原As20完整源域FINAL来源、角色和ownEMA由landing preflight＋strict load核实。

源单包256、Fs25MHz，eq1原dataset逐包RMS；As从raw经native normalize_input，不先经过ordinary。实际导入 __file__ 写actual_modules；leo_practical来自 experiments/adv3b02_xuc/code。训练fc2.462GHz，完整residual/post_sync/EQfalse→G1/reflect/referenceSNR30。质量适配、phase tracking、IQ compensation、AGC实际值随actual_channel完整保存；不用默认值替代运行值。训练环境low suburban及mid/low urban；源诊断cache fc2.45GHz差异保留。正式IQ/信道/SNR/样本不改。

固定FINAL30；源判定在读取新正式成绩前保存：PREFIX fullV clean−NOAS≥−.1绝对百分点，panel两draw midlow−AS≥+.5百分点，midlow−RANDOM≥+.25百分点。不通过则结束本固定概率/长度配方，不换seed或微调救援。没有自动E50/扩seed/重试。两新行无论源通过与否都自动完成clean＋六practical；每row七视图prediction全部固定后独立exec scorer连接truth。正式分数不回流调参/晋级/重跑。

命令：CVS-RFFI/python -u experiments/cvs_as_prefix81/runner.py
CWD：PROJECT/releases/cvs_FcrossRX_asPrefix_E30_20261019_r01
独占输出：PROJECT/runs/20261019-phase1-FcrossRX-asPrefix-e30-s2026092701-m2-r01
本地：local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01
N607非特权现有CVS-RFFI/FP32/CPU2，dispatcher容量锁每GPU最多2总进程含其他owner。固定代码依赖LRaw runtime d14450bf1b0d12cc281df1f4d0e79e8760e48282，仅复用代码。

停止仅限本run错误权限/父来源/头/预算/非有限数/输出碰撞/无法合法预测或系统技术失败；保留产物，不干预健康任务，不因低性能停止。一次本地focused negative tests、一个真实CP no-query smoke、唯一P0/P1运行缺陷审查；一次N607预检、archive SHA传输比较、remotecompile、PID/CWD/argv/GPU/log增长核验。Launcher先smoke，PASS立即训练，覆盖E21和末E30实际信道调用。

本地mask首跑7PASS及1项test-only NumPy桥接错误；改项目既有frombuffer后3项相关检查PASS，保留原错误，不安装依赖。17协议负测试/18Python AST PASS，0NN/CP/IQ；实际smoke计划1540NN/0更新。预期全量epoch/step曲线、mask步骤、物理覆盖、实际modules/config/初始化/EMA/冻结/成本、源分层及固定NoAs clean正确集合保持率、七视图accuracy/F1/最差RX/错误修复与新增、PNG/PDF、负结果和缺项。IQ/CP/原始log/预测不进Git。单seed不能证明重复双提升。

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

独立P0/P1结论：PASS，唯一浮点边界P1已针对性核对修复，无未解决P0/P1。普通账户/CPU2/依赖及独占输出preflight VERIFIED；0源IQ/NN/CP读取。

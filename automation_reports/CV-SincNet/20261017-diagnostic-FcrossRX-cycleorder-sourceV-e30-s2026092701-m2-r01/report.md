# 完整源E30：保留线性相位进展的四周期IQ重排

状态：REGISTERED，未启动。上一轮FINAL30自身EMA没有恢复clean和urban的共同收益；目标仍ACTIVE。本轮回到真实IQ候选，不增加loss、门控或部署分支。唯一矩阵、成本和路径由experiment.json维护。

## 假设、来源与反例

唯一假设：当前CVS可能依赖参考窗中偶然的逐周期观测次序；只重排同一物理包四个已观测周期、保留线性相位进展和总功率，可生成非平凡且仍可辨识的IQ视图。它是待证伪的观测增强，不等于更换真实RX或已证明A_g。

先检索FcrossRX和F-crossRX两种命名：旧镜像子空间是架构分支，并非整体IQ反射。原始[RFFI增强论文III-A](https://arxiv.org/html/2304.14795)明确警告共轭翻转会改变硬件相位偏差，因此整体反射候选放弃，0IQ/NN/训练。论文III-B讨论片段重排，但也说明效果依赖符号随机性，连续重复片段可能使其很弱；不能以论文当作本数据标签保持或性能证据。

历史四fitRX/L5040的源repeatcorr（不等于本轮五RX/V1800）固定[80:160]四个20点周期的q10相干度0.803835，支持该窗口具有重复性；RX3残差功率中位数10.275181%，原有色噪声先验因此FAIL且保持关闭。本候选不提取/拟合/注入该残差，只对原始观测作已知重排；TX瞬态和残余相位仍可能被扰动，必须做真实默认头保护检验，不先称其为纯噪声。

## 唯一IQ操作与源判据

对复IQ的[80:160]四个20点周期，用全部三个lag20对得到单位相位q；对四周期的完整24排列均匀私有抽样pi，输出第k周期为原第pi(k)周期乘q^(k-pi(k))。其它176点精确不变；没有时间插值、循环包移位、裁切、遮挡、噪声、PA、DAC、RMS或功率恢复。q相位不可定义时保留该包并记录identity fallback，不删包。只用opaqueID与视图命名空间/seed202610173001，不使用TX/RX/day/置信度定义操作。保留未知TX硬件效应，不拟合或反演TX硬件；已知观测排列的逆仅供合成验证。

两个既有全源scratchE30/6660的B00/B01终轮学生默认头冻结，原F/CROSS_RX/三SIG_TIME/统一Sinc/joint160/cos30/NoExplicitCFO和内部补偿不变；不加载EMA或另训头。严格own_model来源/原L6300/U56700hidden/V27000角色检查后，先各2合成包真实CP no-query smoke，再打开原源缓存。全部RX1/3/4/6/8和day1/2/3保留；只消费既有V1800和P24两draw×low_suburban/mid_urban/low_urban的full condition0，所有观察均参与。模型seed只有2026092701。

该操作放在已接收IQ之后；完整native residual/post_sync/noEQ即G1/ref30/reflect/fs25MHz/fc2.45GHz/quality/lock/phase/IQcomp都已按原依赖生成，不重新抽物理状态或改SNR。源和正式测试随机流分离；本轮不打开target。源缓存反序列化会带入未用L/旧特征，仅V1800参与，U IQ和TX真值不打开。

两个模型的clean都要求：accuracy相对原学生≥−.5pp，每个5RX和6TX各≥−1pp，固定原clean正确集合保留≥99%，clean周期相干q10≥.8，IQ相对L2变化median≥.01，全部clean相位定义有效。B01还要求两个urban×两个draw每单元accuracy≥原学生−.5pp。数值容差1e−10pp，不改变科学阈值。源保护失败即关闭固定窗口/周期/重排群/phasefit/概率/RX/seed/门槛/预算救跑；源FAIL也完成全部模型和视图。PASS只能支持后续另行登记的匹配E30效果检验，不是A_g成立，不自动访问target或扩seed。

保存每包排列、q、相干度、L2、功率误差、所有5个拼接位置(80/100/120/140/160)的跳变能量前后，报告所有状态/质量层；功率守恒不意味着接缝无损。默认头accuracy/macroF1/CM、margin÷30、repair/newerror全部保留。学生参照复用P26同ID/draw/场景/CPU2/batch90缓存，无额外学生参照NN。

## 执行与成本

每模型12602NN（2smoke＋1800clean＋10800sat）和12600个IQ视图，总25204NN/25200视图；0训练/头拟合/EMA更新/新channeldraw/teacher。父两个模型各E30/6660/5407922NN单列，不能把原训练成本当作新增预算或忽略。普通N607账户、CPU2、空CUDA、一个不可覆盖run/output root；命令/CWD/精确CP/句柄见experiment.json。一次归档单次传输SHA对比、一次remotecompile、首步实际CP smoke、一次独立PID/CWD/argv/CPU/loggrowth或自然终态读回。只停本run的来源/角色/非有限/support/power/state/成本/输出冲突等技术错误，保留产物不自动重试，不动健康旧任务。

本地4项合成测试PASS：全部24排列的已知逆/功率/支持外、理想周期及相位进展、全局相位/正增益等变、私有RNG与非法source/判据反例。0真实IQ/CP/NN/训练。Torch旧环境的NumPy桥ABI警告保留，测试不使用该桥，未装包；远端接口通过frombuffer/tolist。预检普通账户CPU2/输出无冲突VERIFIED；唯一限定P0/P1静态审查PASS，无直接P0/P1；未读取IQ/CP/target或执行远端，不加额外gate。

整体目标未完成：有效跨RX A_g、正式lowurban>60%、clean净正收益、suburban接近clean和跨seed联合证据均未证明。不会把source保护或变换非零当作识别收益。

# 本轮共有普通增强的实际作用

适用运行：20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02，F＋CROSS_RX，全源L6300/U56700/V27000，模型seed2026092701。执行代码b94076b191648103daeb8a9cb1c17bdcc9c0e231。本文记录实际调用，不把共有旧增强当作新增Ag或已验证的身份保持机制。

已读取实际release调用的DataAugmentation.RFFIAugmentor、cvsrffi.schedule.build_aug_base_cfg/configure_augmentor_for_epoch、SSDG.train_ssdg中的L/U入口、dataset_wisig、CVSincNet.forward及频域压缩函数；并核对actual_native_args、完整20轮aug telemetry、ordinary_augmentation_actual及4440条reference_steps。不是只查类默认值。PA内部常量来自本次factory确实未传入自定义PaParams、随后epoch函数只重设部分幅度的实际路径。

## 对象和次序

原L：原数据取样／必要裁切→原逐包RMS归一化→一次ordinary→默认模型前向。ordinary内部次序：DC偏置→时域Tukey边缘窗→循环移位→幅度→常相位→CFO→Wiener相位噪声→通用多径→AWGN→可选PA。代码中的bandedge_taper实际是乘时域窗，不是频域带边滤波。

每个RDrop对复用同一个已经生成的IQ；REPEAT的两个对也复用该IQ，不重新抽ordinary。SupCon只使用共有第一对浅／中层特征。V不调用训练增强、不更新模型或EMA。CONCAT的第二视图从ordinary之前的原规范化L生成完整practical一次；不从带通用AWGN、多径或PA的ordinary继续传播。所有合法As均监督，不作置信度筛除。

原U不是调用这个RFFIAugmentor：MUSE关闭，教师读原x_u，学生读原强视图x_u+N(0,0.015²)，每个实分量独立加噪，末端nan_to_num；原EMA、lambda_u=.16、强预测一致门控等保留，U TX真值始终隐藏。native concat开关关闭，无新增U星地视图。本文不改变其配方。

## 实际概率与范围

以下p是本操作的门概率，不是“每次一定施加”的幅度。除PA逐包mask外，通用操作先作全批一次Bernoulli门，再抽表中粒度参数。E13至E20值保持相同：训练入口调用configure_augmentor_for_epoch(...,min(epoch,label_epochs),...)，label_epochs=13；因此不能把配置的aug_scale_max=.35误写为E20已达到.35。

|操作|E1门概率|E13至E20门概率|条件范围／参数粒度|标签依赖与边界|
|---|---:|---:|---|---|
|DC|.03000|.07518|每包每IQ分量均匀±.002至±.005012|无标签依赖；改变偏置与窗口功率|
|Tukey窗|.02500|.06265|全批共享alpha，E1 .002至.01；E13 .005012至.025060|时域边缘乘窗；不是新真实RX|
|循环移位|.03500|.08771|每包整数，E1±3点，E13±8点|模256索引；产生切片首尾接续并移动固定80:160参考窗口|
|幅度|.04500|.11277|每包0.9至1.1；范围不随scale缩放|无标签依赖；后续没有统一RMS抵消|
|常相位|.04500|.11277|每包0至2π|无标签依赖；各支路不保证整体等变／不变|
|CFO|.03500|.08771|每包均匀±.000040至±.000100240 cycles/sample|仅扰动观测；显式CFO身份坐标仍屏蔽，内部必要补偿保留|
|相位噪声|.03000|.07518|每包sigma_step为0至.000600／.0015036 rad，独立高斯增量累加|无标签依赖；256点内有记忆|
|通用多径|.01800|.04511|2至4抽头；E1最大延迟0、E13最大1点；延迟和复系数全批共享|抽头能量归一化，reflect左填充；无固定LOS，抽头可合并抵消|
|通用AWGN|.04000|.10024|每包SNR20至36dB；噪声功率按此前处理后本包功率算|不在后续As之前叠加；不改变正式测试SNR|
|PA normal|.01400|.03508|逐包mask；强度sqrt(U)，链内随机门见下文|会改变非线性身份特征；低门概率不等于弱失真或已验证保身份|
|DAC|0|0|本轮不执行|量化、DAC IQ镜像、抖动等参数存在不代表实际启用|
|class signature|关闭|关闭|enable_class_signature=false|不使用TX类签名；不因此证明PA安全|

PA链内：memory polynomial门.9、最高7阶／3记忆位；FIR记忆门.4；Rapp门.9，sat_min=.6、sat_max=1.1、p=2至5；AM/PM门.6；IQ镜像门.05。epoch函数缩放mp_sigma、mem_sigma、ampm_max、iq_img_max；Rapp饱和区间、sqrt强度分布、链内门概率以及memory-polynomial部分增益／归一化不会整体乘当前scale。因此本轮只能称低概率旧PA，不能称已证实的轻度标签保持增强。没有按这项代码风险热改健康模型。

## 增益和归一化

RMS在ordinary之前。RFFIAugmentor.__call__末尾只有nan_to_num／dtype转换；CVSincNet.forward从pad_crop进入Sinc、时域、频域和PA路径，没有统一输入RMS。频域log1p功率保留幅度作用；PA的多项式和固定clip也不能由最终LayerNorm／cosine归一化简单抵消。参考响应的归一化互谱／平坦复增益商会消去对应尺度，但这是局部支路性质，不能外推为整个模型精确增益不变。数字增益也不等于真实替换接收机。

## 实际输入与证据边界

两条已完成行各保存4440条实际ordinary与原IQ的MSE及RDrop复用记录，并确认20轮／4440更新、所有6300 L与56700 U物理ID覆盖。输入MSE说明变换确实发生，不能证明身份收益。具体随机算子没有逐包mask日志，不能从配置概率推造实际触发次数；不补造历史输入。CONCAT真实As计数和覆盖由独立实际计数器记录，完成后与本表区分报告。全部正式接收IQ、场景、SNR和168000个测试样本保持原样。

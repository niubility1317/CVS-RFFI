# 有限窗酉相位观测：源 V 身份保护检验

状态：IMPLEMENTED／LOCAL_CHECKS_PASS／P0P1_PASS_AFTER_TARGETED_FIX。尚未启动或读取真实源IQ，0新CP／NN／更新。不是新Ag有效性或完整联合目标证明。

唯一假设：一个轻度、复线性、近似保能量、无循环接缝的有限窗频率相位变化，能够在原F默认头下保留源身份；若此必要条件成立，才有理由独立研究它的训练价值。这里不声称真实EQ1已经观测到该RX扰动，也不声称数字变化替换真实接收机。

## 物理依据、旧负结果与未知

[Analog Devices接收滤波说明](https://www.analog.com/en/resources/analog-dialogue/articles/2022/05/18/02/02/how-digitally-tunable-filters-enable-wideband-receiver-applications.html)指出频率相关群延迟会改变宽带接收波形；这只支持观察相位色散这个问题，不能证明WiSig残留或当前幅度真实存在。[UNM一阶全通讲义](https://ece-research.unm.edu/bsanthan/ece539/note7.pdf)给出单位幅频与非平坦群延迟的基本性质。下面采用自己的有限窗酉离散算子，未使用器件数值冒充实际RX校准范围。

旧幅度谱FIR改变增益，旧时间相位在时间点乘相位，本算子是有限窗余弦谱上的单位模相位旋转。旧频域相位闭合量估计器因必要频点缺失失败，本项不修改其窗口、频点、质量阈值或判定，也不使用其结果作为RX相位已识别的证据。它仍失败关闭。这里提出独立的、标签无关的数学观测变化，不拟合RX或TX相位；若源身份受损，固定算子同样关闭。

[随机重排原论文](https://arxiv.org/pdf/2304.14795)的增强依赖符号随机性，连续重复片段会削弱其作用。WiSig EQ1的重采样、同步、均衡及短窗口不能直接替代其前提，本项不做前导交换／任意切片／窗拉伸，也不重复另一对话R25监督配比或其sourceV星地诊断。

## 固定算子

输入仅现有x∈R^(N×2×256)。定义T为邻点平均，端点采用半采样对称边界：首点使用(x0+x1)/2，末点使用(x254+x255)/2；T为实对称矩阵。L=(T−I/256)/rho，rho=cos(pi/256)+1/256，谱范数1、trace0。以复数IQ施加U_alpha=exp(i alpha L)。等价余弦谱相位为alpha*(cos(pi*k/256)−1/256)/rho；相位绝对值≤.1rad，幅值为1。没有新增随机噪声、PA、TX IQ失衡、量化或剪裁。

|固定项|值与边界|
|---|---|
|alpha|仅0／−.1／+.1，全部样本全保留，不按标签或置信选择|
|实现|Taylor0至8阶，精确算术余项界3.046e−15|
|边界|固定半采样对称，不循环拼接、不读取窗外或长IQ|
|RMS/漂移|无额外RMS归一化；unitary精确理论，浮点实际误差单列|
|时频含义|不做CFO平移或时间移位；有限窗余弦谱相位变化，不冒称逐个DFT幅值严格不变|
|参考窗口|原索引和长度不改；值的轻度变化由源默认头保护条件检验|
|随机性|本检验算子端点确定；只有旧ordinary使用固定seed20261018061|

L保持复线性，I/Q由同一个算子耦合，未引入共轭镜像或TX专属参数。U为完整有限向量的相位变化，并不证明所有身份证据都能保留。合成证据：18输入/权限负例；独立DCT稠密对照误差1.19e−13；FP32能量误差9.57e−8、反变换误差1.79e−7；零强度逐位恒等、无首尾循环泄漏、全局RNG不变。只说明数学实现，不能替代真实身份检验。

## 两个固定默认头与原数据权限

原完整源REPEAT-s2026092701和CONCAT-s2026092701，均固定scratch-final20／4440更新，来自同一fullsource run，按预算冻结、不按target排名。加载前使用原source proof并核对完整物理角色、scratch来源、默认头及checkpoint身份；不加载其EMA或派生失败模型。它们只是冻结readout，本轮不训练或换头。

F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／默认6×160 cosine scale30及NoExplicitCFO、内部必要补偿保持。原L6300／U56700隐藏TX／V27000、source RX1/3/4/6/8、day1/2/3、split392005。只从原V按opaque ID固定取每TX×RX×day前20个，90格共1800个物理样本；没有留出训练RX。V只eval/评分，不更新optimizer、教师、归一化状态或prototype；U不参与，不拟合额外诊断分类器。

原source_cache复用已验证raw数据和角色；只取validation的x/y/rx/day/ids，忽略旧模型feature/logit字段。raw与once-generated ordinary分别比较alpha0／±.1。ordinary按原父args、实际E13日程，Python 与 Torch RNG 均按固定 seed 设置并在结束／异常时恢复；只生成一次并供全部模型/alpha复用，不能把它称为完全无增强；旧PA等开关据实记录，不把其标签保持视为已证明。

## 最小矩阵与固定反证

2冻结默认头×2base(raw/ordinary)×3alpha×1800＝21600个源预测，另2CP×2合成noquery smoke＝4，总21604 NN；0反向／optimizer／teacher／头拟合／target。所有视图的默认logits固定后，独立CPU读出accuracy、macroF1、混淆矩阵、RX/TX/day及全部90格、相对alpha0的修复/新错和正确保持率。源V标签原本允许读出，不冒称target truth-last。

必要条件全部固定：每模型×base×非零端点，相对0的accuracy≥−.5pp、原正确保持率≥99.5%、每RX／每TX的accuracy≥−1pp；全部包能量相对误差≤1e−5、正反变换相对误差≤1e−5。任一科学条件失败，关闭本固定profile／.1bound／Taylor8／边界，不换幅度、profile、seed或父模型救跑。PASS仅是源保护，不是RX域不变、Ag/As识别收益或自动进入E30训练、扩种子/target的依据。

本轮完整记录1800次旧ordinary生成、10800次正向IQ算子＋10800次逆检查；额外NN=0不等于算子零成本。两个原父的E20成本复用单列，不计为本轮训练；每读出CP只加载一次。测试IQ/SNR/信道和困难样本不变。

## 发布与产物

唯一owner codex/root/F-unitary-dispersion-source；普通N607账户，CPU2，沿既有shared capacity锁每GPU最多3个全owner实验进程且至少12GB空闲。CWD PROJECT/releases/cvs_F_unitary_dispersion_source_20261018_r01；命令CVS-RFFI/bin/python -u experiments/cvs_unitary_dispersion18/runner.py；不可覆盖根PROJECT/runs/20261018-diagnostic-F-unitaryDispersion-sourceV-s2026092701-m1-r01。精确[登记矩阵](experiment.json)、[配置](../../../configs/cvs_unitary_dispersion_source_20261018.json)、[本地检查](verification.json)。

一份代码配置Git提交、聚焦负例和真实CP无query smoke、一次限定P0/P1、一个release archive/一次传输比较/远端编译、一次PID-CWD-argv-GPU-log增长核对。来源/角色/状态/非有限/输出冲突等技术失败保留产物，不自动重试，不改其他健康任务。期待resolved、全源固定logits、源完成/完整计分和成本；数据/权重/预测/日志保持私有存储，只有汇总进入Git。

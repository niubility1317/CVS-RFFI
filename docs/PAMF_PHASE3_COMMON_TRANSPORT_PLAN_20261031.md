# Phase3 公共类传输对齐（2026-10-31）

Run `20261031-phase3-common-transport-s2026092701-r01`。本轮在已完成冻结Sinc的开放骨干PCD机制上增加公共类检测与对齐（CCD），原F0/defaulthead、DURA与所选4af基线保持。只训练一个新TARGET_CCD学生444步；精确复用body-prototype r01/a336的SOURCE和TARGET_PCD权重与r02/4273的六列冻结预测、密度和阈值，不重复训练对照、不继承U适配权重作为新源初始化。

## 固定机制与对照

新学生仍独立从原source-only r03初始化，Sinc两24×1叶和所有buffers固定，body231985+projection102688更新，普通AdamW lr1e-4/wd1e-4/eps1e-8/clip5。每步192L/64OE、source crossset+OE_KL+10*parent-z锚点、U384弱全量及两份固定phase/gain强view、K16临时原型、global/local各权重.1均与原PCD控制相同。源和U采样索引逐步核对原SOURCE/TARGET_PCD的完整444步；源seed2026103017/18、Ubatch2026103112、原型2026103113/25Lloyd、augmentation2026103114不改。原PCD control已在任何新CCD训练之前完成冻结。

新增六个固定公共原型为原r03源L的parent_z按真实源类等权均值后单位化，尺寸6×160，仅训练使用，不是新身份头，不替换默认头。前100步只有原source+PCD目标；第101至444步另加`.1*CCD_CE`。CCD使用当前学生停止梯度弱U与这六原型的cosine相似度，原有两strong输出通过cosine/.1的六类CE学习所选伪标签，不新增真实IQ前向。未选行不参与CCD，选中集为空时CCD为可微0，保留PCD，不凭空生成伪标签。

CCD参考[UniOT作者lib.py](https://github.com/changwxx/UniOT-for-UniDA/blob/main/utils/lib.py)、[main.py](https://github.com/changwxx/UniOT-for-UniDA/blob/main/main.py)及[Office31配置](https://github.com/changwxx/UniOT-for-UniDA/blob/main/config/office31-config.yaml)。取其reg=.01、reg_m=.5、gamma=.7、beta初始六类均匀、mu=.7；本实现固定400轮FP64 log域generalized Sinkhorn，不依赖新增POT包，不使用目标真实K/未知比例。每次运输归一化总质量为1，以每行最大质量>1/N且对应列质量≥1/6选择公共行；N包含人工填充行而非真实类别数。beta按`.7*beta+.3*列质量`更新，停止梯度，仅训练状态，不部署。

自适应填充的正率由cosine最大值>.7计：≤.5时以初次UOT筛中的公共行有放回填充`N-2npos`行，初筛为空则不填；>.5时将U与其最远源原型的等权单位化混合生成人工负特征，填充`2npos-N`行。填充随机流独立seed2026103120；人工行只参与分配统计，最终只保留原384条U的标签/选择掩码，绝不计为新增物理数据。所有分配、选择和beta停止梯度；输出只用于U训练，不改部署拒识方式或阈值。

这是PAMF固定源原型CCD变体：保留原PCD而不复现UniOT完整memoryqueue/学习分类头/最终聚类；gamma、mu和本地预算不由Q成绩选择。原型16是辅助容量，不是未知类数。本轮不开展未知发现或匿名注册，A_N/H_OW=N/A，完整goal ACTIVE。若复用实质作者代码，保留原MIT版权与许可。

## 数据与冻结测试

源与U/Q、星地设置完全不变：Phase1 L6300/隐藏U_s56700；Phase3只星地P1 L6300/OE6134训练，OE仅成员监督与公共RX/day，五源RX三日期；V27000额外源标签校准、NV3150只评分。新CCD可读原r01训练完成元数据和采样索引作配对核对，不能读旧Q预测作为训练输入。原r03及F0/DURA匹配392005来源，SOURCE无U，PCD/CCD及其源拟合聚合均如实标目标U祖先。

U384为RX24-1/2021-03-08的既有固定G10星地观测，与源及未来Q物理互斥。Q为同RX/03-15同864条物理记录（已知6×96=576，未注册3×96=288），clean/star都先DURA；clean仅冻结测试。G10 MIX01/600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect及receiver2027/scene2026103001/channel2026104001保持，不重抽received；phase/gain是同观测计算视图。新R_dev3-2仅容量元数据候选，未建池/不进入本轮；正式RX24-18不访问。

新CCD冻结后仅以完整源L/OE拟合一份known90/OE32/normalizer，源V拟合两组六阈值共12新值；原六列36阈值精确复用，总8列48值384B。三模型E0/SOURCE/PCD原六列与新CCD两列共同组成八列，主唯一`CCD_predicted_ratio`（index7），新增pooled列为对照。源seenRX星地DURA、crossRX cleanDURA、crossRX starDURA三条件全预测固定后，独立CPU分别连接三truth一次、Utruth0。新Q通过同G和F0验证旧控制defaultlogits/closed不变，六列既有readout精确复用；不能因源科学失败省略Q或按目标结果换列。

源known12通过及known42诊断、必需场景覆盖保持。最新人类v3：clean宏R_U≥95%、star宏R_U≥90%、C_K≥90%，A_K较同DURA原F0闭集下降≤2个百分点，主报宏A_K/R_U/H。适应另要求star A_K严格高于E0、SOURCE、PCD三控制，R_U均不下降；任一失败不晋级、不扩seed、不改变原基线。历史源V校准复用/NV与Q曝光、单seed单RX、日期重合、高仰角缺19-15导致宏R_U/H=N/A等限制不改写，不宣称新独立盲测或严格未见全部未知。

## 执行与成本

唯一owner root；N607实际空余显存≥4096MiB的一张GPU，训练/预测单worker顺序运行，每卡本任务新增≤2，CPUscore CUDAempty，不停改他人。新不可覆盖root/release；一次源码commit、聚焦负测与真实checkpoint noquery smoke、一次独立P0/P1、最小登记、一次preflight、单archive比较/compile及启动后PID/CWD/argv/GPU/log增长核验。无额外设计审批或重复数据验证，属REJECTED_EXTRA_GATE。

新训练真实父signal12818（源12434+弱U384）、学生340992（源113664+弱170496+两strong56832）、G384/F0real0；新增body/projection各444、临时prototype2560，六公共质心/六beta仅训练状态。新评估CCD source42584+Q1728＝44312 signal，原F0/G各1728，新增链合计signal398122/G2112/F01728；六控制已有预测/GMM/τ及原训练成本单列继承，不重复记新增。合成smoke/CCD梯度probe按实际记录；若实现确需额外前向必须如实记账。

保存全部444步损失、CCD启用/选择数/伪类直方图/正率/填充分支与数量/beta、UOT调用和数值残差、clip前norm/首2步及首CCD步分模块梯度、完整索引匹配/曝光；选择数和质量不是伪标签真准确率，不连接U真值。记录全部24点、三配对、TX/RX/day/scene/coverage、实际成本与科学判断，提交公开结果并push/OID读回。仅祖先/角色/冻结状态错误、nonfinite、碰撞、不能产生合法预测等技术失败停止保留；科学低分不停止、不覆盖原root重跑。

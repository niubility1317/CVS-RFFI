# 接收机输入状态无标签适配

训练两臂已完成，预测加载 FAILED_PRESERVED；NO_PERFORMANCE_RESULT，完整目标 ACTIVE。

# Phase3 接收机输入适配：16 参数固定实验（2026-10-31）

Run `20261031-phase3-receiver-input-s2026092701-r01`。前一轮当前 PCD 的相对距离读出增加接收却损失未知拒识。本行检验不同层面的改进：保持 DURA、PAMF/default head 和 PCD 冻结，只更新 P1 后的低维接收机输入状态，不继续拟合距离或目标阈值。

## 方法与两个控制

已有表征微调、统计偏移/缩放、冻结距离重构均有未过门槛的结果。本轮选择输入域的有界滤波，与原 PCD 和等步数纯源输入状态配对；这是 Agent 在已有自主研究授权内固定的新假设，并非用户指定算法或已验证提升。

[DeepFIR 原论文](https://arxiv.org/abs/2005.04226)研究了冻结分类器下的可微 FIR 波形优化，但其滤波位于发射端。本行是接收端、无目标 TX 标签、单 RX 共享参数的独立方案，不宣称论文复现或继承其性能。

模块有一个 FP32 theta[16]，全零初始化。前14个数形成7个复 FIR 校正系数，后2个形成共轭项 gamma：校正均为0.05 tanh(theta)，中心实系数再加1。输出为 sum_lag h_lag x[t+lag]+gamma conj(x[t])，lag为−3至3，reflect pad3。无二次 RMS、额外 IQ、信道重抽或可训练分类头。该变换是待验证的数字校正，不宣称辨识了真实硬件参数或物理逆信道。

SOURCE_INPUT和TARGET_INPUT从同一恒等初值独立优化，各200步，Adam lr0.01/wd0、ordinary normclip5，仅更新16参数。每步L128，np seed2026103160，两个模型逐步索引一致；SOURCE先完成并封存，才打开U。TARGET不从SOURCE终态续训。

源目标为真实标签CE＋原F0同P1缓存教师KD（T=2、KL batchmean×4）＋0.1 mean(tanh(theta)^2)。TARGET再加入U64硬伪标签CE（mask后按完整64平均、权重1）＋10 mean(sum((PCD(A(U))−PCD(U))²,dim1))。U独立采样seed2026103161，不按真类别分组，不增加phase/gain增强。

适配前仅一次对U576计算冻结F0概率、预测类及PCD分数。mask为max softmax≥0.95且原PCD预测类条件margin≥0；伪标签是原F0 argmax。掩码与PCD单位特征教师全程固定，不刷新教师或解封U真值。接受率和预测类直方图不解释为伪标签准确率；即使接受数为0也完成预定步骤并如实报告。PCD特征锚定对全部抽取U计算，不把低置信样本按真实角色送入六类监督。

## 模型与数据权限

冻结原Tri-SIG-TIME SSL F0 E50/11100/default6、fresh f4b5最终TARGET_PCD444、原r03与DURA v1.0 R69（reuse/graph、非native、FP32/TF32off）。PCD已有目标U576祖先，两种输入状态的完整系统均保留这一来源；SOURCE只表示新增theta的拟合输入纯源，不是无目标祖先的模型。原模型参数/buffers/eval保持，原模型梯度应为None。

训练配置仅包含run/code、fresh训练配置、冻结reference prediction root、输出、steps200/lr0.01；只读取其model/density/calibration头信息，不消费历史query行或truth。源L6300的P1/y/原F0 teacher_logits来自原cache的L角色，核对原F0/G身份，不打开V/OE/novel/Us。旧source合同392005、L6300/Us56700/V27000、OE6134训练祖先保持。

SOURCE封存后，仅复用fresh训练根U_features.npz的views[0]恢复P1及其已验证public IDs/manifest，576条无标签源外RX3-2/index34、2021-03-01。parent_z是r03表示，不冒作最终PCD教师；PCD教师由当前冻结最终PCD实际前向。其他派生views不训练，不新执行U恢复。源五RX1/3/4/6/8及03-08/15/23与目标角色、U/未来Q互斥保持。

真正的第一次步骤为实际F0/r03/PCD/G无query合成smoke；再经恒等输入模块用8条合成IQ验证到theta的非零梯度、原模型无梯度/状态不变、无优化更新。PASS后立即继续L，再SOURCE，再U/TARGET；不生成额外许可产物。完整记录两份200步损失、梯度/裁剪、参数变化、采样、实际唯一/曝光与接受统计。

## 固定三类 DURA 测试

接收机状态仅在新目标RX34使用；源RX不应用这两个状态。六方法为OLD_PCD/SOURCE_INPUT/TARGET_INPUT各自pooled_ratio及predicted_ratio，唯一主TARGET_INPUT_predicted_ratio。源V27000/NV3150三臂精确重复原PCD两列，不读取源IQ或重校准。源已知V是校准角色，不是新的独立测试。

新状态全冻结后，future clean/star各10238条同canonical received经过同一DURA一次，每个P1分别经两个输入状态，再由同F0与最终PCD计算。每臂实际F0预测类同时路由旧密度和旧tau；身份预测可能改变，classification_unchanged=False。原F0未输入变换的同DURA logits/closed来自精确匹配的旧记录，作为2pp保护基准，不能换成候选自身较低closed。源known90/OE32、normalizer及原两组六tau均不更新；6方法重复保存36值/288B，唯一旧12值/96B，新校准0。

Q仍为RX3-2的03-15/23共5525/4713条：known4508、unknown5730。旧六TX14-10/14-7/20-15/20-19/6-15/8-20各1491/1182/150/406/1186/93；四未注册TX5-5/10-7/2-19/7-11各1454/1439/1427/1410。7-11从早日U整类排除；03-23缺旧20-19/8-20，完整宏AK/CK/H=N/A，不删类认证。clean只测试，共用star-U状态，不用于拟合或提供逐样本恢复参考。

source G10 MIX01与fresh未来episode保持：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、一次input RMS。U/源P1复用已有恢复，Q从原固定received重算同G，不重抽场景/信道；它不增加独立物理记录。

全部6×3预测固定后独立CPU连接sourceNV/clean/star三truth各一次，Utruth0。按TX宏平均报告候选closed、原F0closed、AK/CK/RU/H、逐RX/day/六景/缺类、身份与接收/未知配对变化。sourceknown12继承并单列；cleanRU95/starRU90/CK90/原F0closed−AK≤2pp及主景覆盖需全过。另要求主starAK严格高于OLD_PCD_pred和SOURCE_INPUT_pred，RU对两者不降。否则NO_PROMOTION，不扩seed/不换主/不按Q调阈值。匿名REG关闭，AN/HOW=N/A，完整goalACTIVE。

## 成本、版本与停止

新增theta优化200×2=400步，每臂16可训练参数，部署主仅16参数/64B；完整主F0 232993＋PCD334721＋G266951＋16=834681。L教师6300为缓存读取，预计新训练F0=SOURCE25600＋TARGET38400＋U教师576=64576，PCD=U训练12800＋教师576=13376，r03/G真实前向0；实际计数与合成smoke分列，不抹去PCD原U576/F011100/r032220/PCD444/G4940祖先。

新Q预测两状态各F0/PCD20476，故F0/PCD各40952，G20476；原PCD控制无新前向。参数、source/U采样曝光、缓存和实际时间/显存记录，不把批量阶段时间当B1。新状态与输出独立保存，不改历史模型/数据。

单seed2026092701，root唯一owner，N607单worker串行训练/预测后CPU评分，实际free≥4096MiB、每卡新增≤2，不干预他人。仅最小八项流程：实际Git版本、聚焦负测/真实CPsmoke、一次P0P1、预登记、单preflight/单包SHA/compile、livePID/CWD/argv/GPU/log增长、全预测后独立评分。非法角色/祖先、原冻结模型变化、非有限、输出碰撞或无法合法预测闭合保留停止；科学低分仍完成预定全部测试。NV/Q已曝光、单RX/seed、global physical history UNKNOWN及正式RX24-18未访问如实保留。


实际code `ac8bbd27277fcb1f16806c7b8b94e8c3e27e390b`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-receiver-input-s2026092701-r01-release`；owner `codex-phase3-receiver-input-20261031`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.receiver_input_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-receiver-input-s2026092701-r01-release/runtime.json`。输入/输出路径见experiment.json。


## 技术中止与只评估恢复

原theta与训练400步保留。CPU重算FIR与CUDA训练报告最大差SOURCE4.656612873077393e−10/TARGET1.1641532182693481e−10，theta本身一致，逐位派生报告比较在任何候选smoke/Q/truth前中止。没有本行性能结果，不改标NO_PROMOTION。修复仅允许派生报告2FP32ULP，theta/arm/来源仍严格；独立r02只补完整预测评分，新优化0。见`docs/PAMF_PHASE3_RECEIVER_INPUT_RECOVERY_20261031.md`。

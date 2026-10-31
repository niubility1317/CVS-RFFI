# Phase3 完整原生身份网络的已知门控自训练

Run `20261031-phase3-native-identity-s2026092701-r01`。当前 PCD 为研究基线。本行检验完整原 F0 身份学生的无标签适应，同时保持 PCD 拒识分支冻结；不是继续调整融合残差、密度距离或目标阈值。

既有 joint-identity 使用 r03 骨干与 F0 头合并初值，并只用 PCD 目标适应共享表征，未证明初始 logits 等于 F0；receiver-input 使用同类置信门控，但只训练16个输入参数。两者均未通过。本轮将学习位置改为完整原生身份网络，并采用目标身份伪标签，不能称为上述方法的单因素严格复现，也不继承其成绩。与本轮 SOURCE_ID 的比较才是同初值、同源目标及同结构的适应对照。

## 模型、权限与固定目标

两个学生独立严格加载同源 Tri-SIG-TIME SSL F0 E50/11100 的完整原生结构与权重，初始化 logits 必须与原 F0 在同合成输入上一致。部署 reader 为 `model.id_backbone.encoder`，232993 参数；冻结 Sinc48、全部 buffers 及 wrapper 的 reader 外667729参数，仅联合训练 reader 内其余232945参数，包括原默认六类头W960。全程eval保留数值/BN/Dropout状态，但学生梯度开启；不调用旧监督训练hook引入原As/KL/CRIC配方。

原 F0/default head、DURA v1R69、r03、最终fresh PCD、源密度和全部历史模型保持。学生只保存自己的完整 reader state，回载须先构造真实原生父，再strict覆盖reader；实际形状/dtype/有限值、Sinc与persistent buffers、body和head改变分别核对，不以源模型loader的target_reads=0限制冒充适配模型loader。

| 臂 | 新身份训练输入 | 目标身份损失 | 固定开放分支 |
|---|---|---|---|
| SOURCE_ID | 源L128，每步真实CE＋原F0 KD | 关闭，U0 | 已有fresh PCD |
| TARGET_ID，唯一主 | 同源采样与完整原F0初值 | 固定已知门控hard PL | 同一fresh PCD |

两臂独立各444步，SOURCE先封存才打开U，TARGET不从SOURCE终态续训。AdamW lr1e−4/wd1e−4/eps1e−8、ordinary normclip5；L采样seed2026103160，两新臂逐步相同。源loss=`CE_L + 4×KL(softmax(F0_L/2) || softmax(student_L/2))`，各权重1，教师来自同P1合法L缓存。

SOURCE完成后，对已有星地U576弱P1仅一次计算原F0预测及最终PCD margin。固定mask为原F0 max-softmax≥0.95且原PCD predicted_ratio margin≥0；伪标签仅原F0 argmax。mask/标签全程不刷新，不使用EMA、真实TX/K/类别配额或目标身份验证。PCD教师已有同U576适配祖先，SOURCE_ID只表示新增身份权重U0，整个系统仍有目标U祖先。

TARGET使用U_rng2026103112，每步抽64条记录，对原缓存的两份同记录phase[-π,π]/gain[0.95,1.05]强视图计算CE。PL为两强CE的均值乘固定mask后按完整64平均，权重1；全mask空批次损失为0，可跳过无效学生前向，实际行数单列。不扩大曝光或改变mask以凑接受数，不把预测类直方图解释为准确率。

## 数据与冻结三条件

沿用合同392005与VALIDATED_ONCE句柄。新源训练只读原星地L6300的P1/y/原F0 teacher_logits，原Us56700与OE6134仅祖先，本行训练新读0；源五RX为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。旧六TX为14-10/14-7/20-15/20-19/6-15/8-20。V27000只在学生冻结后校准阈值，NV3150的19-15/19-16/7-1各1050只冻结预测后独立评分；V不是独立已知测试。

目标开发RX3-2/index34，03-01星地U576；future Q为03-15/23的5525/4713条，共10238条物理记录，旧4508／四未注册5730。旧六类数量1491/1182/150/406/1186/93；5-5/10-7/2-19/7-11为1454/1439/1427/1410。后来专有7-11整类不在U；角色/真实K/配额仅builder/scorer可见。源/U/Q物理互斥，clean/star Q同canonical及顺序；03-23缺旧20-19/8-20，完整AK/CK/H为N/A，不删类认证。

离开Phase1后的学习观测全部是既有固定G10 received及其DURA P1，不重抽信道或新增物理记录。600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、一次input RMS/输出无RMS；源scene/channel2026103001/4001，future2026103133/34，receiver2027。DURA固定R69 E20/4940、266951参数，reuse=True/graph=True/native=False、FP32/TF32off。clean只冻结测试，不建立clean U或使用clean参考。

固定六读出为OLD_PCD/SOURCE_ID/TARGET_ID各pooled_ratio与predicted_ratio，唯一主TARGET_ID_predicted_ratio。两新学生预测同时决定其类条件PCD密度路由与τ索引；源PCD表征和known90/OE32密度精确继承，无新density/normalizer拟合。各新学生两读出用V按原known12规则校准6τ，共24新值；旧PCD两组12值精确复用，总36 FP64/288B。全部神经与密度先于V、全部τ先于NV和Q冻结。

源V/NV从既有P1计算两新身份，固定PCD源特征精确复用。future每条件用同一G输出，共享原F0、固定PCD和两学生前向；原F0 logits/closed与原PCD特征须匹配已有同记录控制。原F0与学生closed均保存，不能用学生较低closed替换原F0保护基准。六读出×源DURA/跨RX clean_DURA/跨RX star_DURA全部预测固定后，独立CPU连接三truth各一次，Utruth0。

成功需sourceknown12与cleanRU≥95%、星地RU≥90%、CK≥90%、同P1原F0闭集−AK≤2pp及必需场景覆盖同时通过；主star AK严格高于SOURCE_ID及OLD_PCD，RU不低两者。报告模型closed、AK/CK/RU/H、AUROC/OSCR、逐TX/RX/day/六scene、源known42诊断、同ID身份纠错与拒识净变化。科学失败仍完成全部固定测试，不扩seed/换主/按Q改温度或mask；AN/HOW=N/A，完整目标ACTIVE。

## 执行、成本与停止

训练新增888 optimizer步、其中TARGET444。每臂源L56832条次、两臂113664；TARGET学生强U最多56832条次，按实际非空mask批次计数。新弱U原F0和PCD教师各576，源L教师6300为缓存继承，训练新G0；已有原F0/r03/PCD/G/cache成本不删除。学生wrapper路径外参数冻结且不部署，不把900722误报为身份网络成本。

评估新学生身份共101252条次（源60300＋未来40952），原F0/PCD/G各20476，旧控制复用。无新源密度，24新τ是源统计校准成本。部署主为新身份232993＋固定PCD334721＋G266951=834665参数，无原型或source/U行bank；没有B1速度承诺。完整记录全部888步CE/KD/PL、mask曝光、梯度/裁剪、初值等价和状态变化，合成计数与真实计数分开。

root为唯一launch owner。默认N607，一个GPU训练进程内两臂顺序执行，随后预测及独立CPU评分；启动核实际free≥4096MiB，每卡新增≤2任务进程，不干预其他用户。只执行最小八项：实际代码配置commit、聚焦负测与第一步骤真实CP无query smoke、一次独立P0/P1、最小登记、不可覆盖root、单preflight/单包SHA/compile、实际PID/CWD/argv/GPU/log增长、全预测后评分。非法祖先/角色、原冻结模型变化、nonfinite、输出冲突或不能产合法prediction时技术失败并保留；不盲目重启。单RX/seed、历史NV/Q已曝光、global physical history UNKNOWN、正式RX24-18未读均保持，不宣称独立盲测。

# 完整原生身份已知门控适应实验

ANALYZED，执行 VERIFIED；科学 NO_PROMOTION，完整目标 ACTIVE。

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


实际code `c5e167a687e915993845056355afeee97a8da8a1`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-native-identity-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.native_identity_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-native-identity-s2026092701-r01-release/runtime.json`。


# Phase3 完整原生身份学生：结果与边界

**本轮执行完成，科学结论为 NO_PROMOTION。** 唯一主方法 `TARGET_ID_predicted_ratio` 在星地开发集上的学生闭集宏准确率由原 F0 的 41.721059% 升至 42.957172%，但开放准确率 A_K 为 24.685490%，低于 OLD_PCD 的 27.951722%；R_U 也由 49.307698% 降至 48.189693%。闭集身份的小幅改善没有转化为联合开放性能改善，原 PCD 研究基线保持，完整目标仍为 ACTIVE。

Run：`20261031-phase3-native-identity-s2026092701-r01`；实际代码：`c5e167a687e915993845056355afeee97a8da8a1`。本报告依据[完整分析](../local_artifacts/pamf_phase3_native_identity_20261031_r01/analysis.json)、[固定评分结果](../local_artifacts/pamf_phase3_native_identity_20261031_r01/score/results.json)和[既定计划](PAMF_PHASE3_NATIVE_IDENTITY_PLAN_20261031.md)，不新增评分、阈值扫描、训练或目标诊断。

## 方法与输入权限

SOURCE_ID 与 TARGET_ID 各自严格复制同源 F0 E50/11100 的完整原生身份 reader 及默认六类头，独立训练 444 步；TARGET 不从 SOURCE 终态继续训练。真实 checkpoint 无 query 合成检查中，学生初始 logits 与原 F0 的最大绝对差为 0。它不同于此前 r03 骨干与 F0 头拼合的 joint 初值。

reader 路径为 `model.id_backbone.encoder`，共 232993 参数，其中 232945 可训练，包含默认头 W 的 960 参数。Sinc 的 48 参数、全部 buffers 与 reader 外 667729 个 wrapper 参数保持冻结；训练使用 eval 数值状态并开启学生梯度。原 F0、DURA、r03、最终 PCD 及其 known90/OE32 密度和归一化统计均不更新。

每步源 L128 的目标为 `CE_L + 4 KL(softmax(F0_L/T) || softmax(student_L/T))`，T=2，CE 与已含 T² 的 KD 各权重 1。AdamW 的 lr=1e−4、wd=1e−4、eps=1e−8，ordinary normclip=5。SOURCE 封存后才打开星地 U，TARGET 从相同原 F0 初值独立训练。

U 的固定教师仅计算一次：原 F0 最大 softmax≥0.95，且冻结 PCD 的 predicted-ratio margin≥0；hard 伪标签为原 F0 argmax。TARGET 每步 U64 的同记录 phase/gain 两强视图 CE 取均值，乘固定 mask 后按完整 64 平均，权重 1；无 EMA、mask 刷新或真实类别路由。290/576（50.347222%）通过门控，接受伪类直方图为 `[36,52,2,49,140,11]`，这是模型分配与覆盖率，**不是伪标签真实精度**。

预测中，两个新学生各自的 `closed` 同时路由冻结 PCD 类条件密度及该学生的源阈值；原 F0 同 DURA 的 logits/closed 独立保存为保护基准。PCD 表征固定，因此本轮 `classification_unchanged=False`，不能用原 F0 的路由替代学生路由，也不能用学生较低或较高的闭集成绩替换原 F0 保护基准。

## 数据、校准与测试

沿用 source contract392005：源 L6300、Us56700、OE6134；本轮身份训练仅新读 L 的 P1、真实旧类 y 与缓存 F0 teacher logits，Us/OE 新训练读取均为 0。源 RX 为 1-19、18-2、19-2、2-19、3-19，日期为 2021-03-08/15/23；旧六 TX 为 14-10、14-7、20-15、20-19、6-15、8-20。

源 V27000 只在网络冻结后校准阈值，仍是复用校准集，不能作为独立已知盲测。源 NV3150 包含 19-15、19-16、7-1 各 1050 条，只有预测完整后独立评分。这三类设备与目标四个未注册设备集合不同。

开发目标为 RX3-2/index34：03-01 星地 U576；03-15/23 的 future Q 分别 5525/4713，共 10238 条，含旧类 4508、未注册 5730。六旧类容量为 1491/1182/150/406/1186/93；5-5、10-7、2-19、7-11 分别为 1454/1439/1427/1410。后来专有 7-11 整类不在 U；真实 TX、K、角色及配额不进入训练或预测。03-23 缺旧 20-19、8-20，完整旧类宏指标保持 N/A。

源、U、future Q 物理互斥；clean/star Q 使用相同 canonical 与顺序。全部三条件均经过冻结 DURA v1.0 R69（E20/4940，266951 参数，FP32/TF32off，reuse/graph 开、native 关）。固定 G10 received 不重抽；clean 只测试，不建立 clean U。信道与归一化细节沿用计划中的 600 km/25 MHz/2.45 GHz、一次 input RMS、输出无 RMS 及固定种子。

PCD 已有同 U576 的适配祖先。因此 SOURCE_ID 的新增身份权重虽然 U0，完整 SOURCE_ID 系统仍不是纯源；TARGET_ID 明确新增目标 U 身份适应。历史 NV/Q 已曝光、单 RX/单 seed、源日期重合、global physical history UNKNOWN 均保留；正式 RX24-18 未读，不声称新盲测或严格未接触未知设备。

## 全部 888 步训练证据

两臂均完整记录 1 至 444 步，888 步连续、有限，源 L 索引逐步完全相同，L/U RNG 重放吻合。日志总损失按 `CE+KD+PL` 重建最大残差分别为 1.49e−8、6.71e−8；未重新读取逐样本 logits 计算 CE/KL。第 1 步 CE/KD 相同，PL/总损失/梯度已分歧；源 CE/KD 自第 2 步分歧，符合两臂目标不同。

下表为全量记录的三等段均值，不是尾部抽样；KD 列已包含 T²=4。

| 臂、步段 | CE | KD | PL | 总损失 | 裁剪前梯度范数 |
|---|---:|---:|---:|---:|---:|
| SOURCE 1–148 | 0.249206 | 0.019095 | 0 | 0.268301 | 2.746255 |
| SOURCE 149–296 | 0.228501 | 0.022752 | 0 | 0.251253 | 2.597039 |
| SOURCE 297–444 | 0.209823 | 0.028302 | 0 | 0.238125 | 2.600745 |
| TARGET 1–148 | 0.257437 | 0.026007 | 0.451228 | 0.734672 | 4.056570 |
| TARGET 149–296 | 0.238633 | 0.035723 | 0.315236 | 0.589591 | 3.849130 |
| TARGET 297–444 | 0.220321 | 0.044832 | 0.221046 | 0.486199 | 3.928819 |

SOURCE/TARGET 总损失首末为 0.310741→0.173622、1.052001→0.361894。TARGET PL 下降，同时 KD 上升，表明学生拟合固定伪标签并偏离源教师；两臂总损失包含不同项，不据绝对值排序模型。SOURCE 首步 KD 的 −4.66e−10 为浮点舍入量级，全部记录有限。

浅层、中层、body、默认 head 在每臂全部 444 步均有非零梯度；SOURCE/TARGET head 梯度范数均值为 0.752605/1.370971，body 为 2.535935/3.687683。Sinc 全程梯度为 0、optimizer 步为 0；buffers、reader 外参数及原模型保持不变，学生 body/head 实际改变。ordinary 裁剪分别触发 20/97 步，最大裁剪前范数 15.697509/11.264811；physical clip 调用为 0。

每臂 L 曝光 56832 次、unique6300（每条 1 至 23 次）。TARGET U 抽样曝光 28416 次、unique576（每条 29 至 70 次），其中通过固定 mask 的曝光 14400 次；其六槽曝光为 `[1853,2572,96,2414,6904,561]`。每批接受 20 至 43 条，**实际非空 PL 为 444/444**，没有跳过强视图前向。TARGET 学生实际前向 113664 行（L56832＋两强 U56832）；SOURCE 为 56832 行。名义 `new_target_UDA_updates=444` 对应 TARGET optimizer 步，不是整个系统更新总数；全系统新增学生 optimizer 步为 888。

## 完整 6×3 冻结结果

以下均为百分比，A_K、C_K、R_U 按所需真实 TX 等权宏平均，H 为 A_K 与 R_U 的调和均值；C_K 是已知接收率。源 R_U 使用三类 NV，目标 R_U 使用四个未注册设备。原闭集来自同 DURA 原 F0，学生闭集不含拒识。各类容量不同，宏变化不能由净正确条数直接推算。

| 方法 | 条件 | 原闭集 | 学生闭集 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|---:|
| OLD_PCD pooled | source | 90.981481 | 90.981481 | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD predicted | source | 90.981481 | 90.981481 | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| SOURCE_ID pooled | source | 90.981481 | 91.488889 | 89.751852 | 96.200000 | 70.380952 | 78.894775 |
| SOURCE_ID predicted | source | 90.981481 | 91.488889 | 89.751852 | 96.037037 | 69.841270 | 78.554555 |
| TARGET_ID pooled | source | 90.981481 | 91.277778 | 89.925926 | 96.770370 | 59.047619 | 71.286641 |
| **TARGET_ID predicted** | source | 90.981481 | 91.277778 | 89.848148 | 96.437037 | 63.809524 | 74.622601 |
| OLD_PCD pooled | clean | 48.567015 | 48.567015 | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD predicted | clean | 48.567015 | 48.567015 | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| SOURCE_ID pooled | clean | 48.567015 | 48.657004 | 27.412021 | 39.173775 | 65.069426 | 38.573887 |
| SOURCE_ID predicted | clean | 48.567015 | 48.657004 | 27.208051 | 38.552381 | 65.540999 | 38.453070 |
| TARGET_ID pooled | clean | 48.567015 | 49.193785 | 28.735082 | 41.423722 | 58.322826 | 38.501067 |
| **TARGET_ID predicted** | clean | 48.567015 | 49.193785 | 27.678544 | 39.367787 | 60.722136 | 38.024602 |
| OLD_PCD pooled | star | 41.721059 | 41.721059 | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD predicted | star | 41.721059 | 41.721059 | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| SOURCE_ID pooled | star | 41.721059 | 42.410687 | 24.025041 | 42.527207 | 53.997757 | 33.254340 |
| SOURCE_ID predicted | star | 41.721059 | 42.410687 | 23.834319 | 42.201815 | 54.025583 | 33.076409 |
| TARGET_ID pooled | star | 41.721059 | 42.957172 | 25.763997 | 46.150744 | 46.473061 | 33.150071 |
| **TARGET_ID predicted** | star | 41.721059 | 42.957172 | 24.685490 | 44.673654 | 48.189693 | 32.647223 |

## 身份纠错与拒识得失

配对均为相同 ID。下表“增/失/净”分别表示新得到、失去、两者相减的记录数；它们不是宏百分点。

| 主方法相对控制 | 条件 | 闭集正确增/失/净 | 正确接收增/失/净 | 接收净数 | 未知拒对增/失/净 | ΔA_K / ΔR_U（pp） |
|---|---|---|---|---:|---|---|
| OLD predicted | source | 313/233/+80 | 255/224/+31 | −84 | 70/85/−15 | +0.114815 / −0.476190 |
| OLD predicted | clean | 152/173/−21 | 40/79/−39 | −59 | 67/138/−71 | −3.983260 / −1.257511 |
| OLD predicted | star | 165/114/+51 | 71/54/+17 | −32 | 74/137/−63 | −3.266233 / −1.118005 |
| SOURCE predicted | source | 160/217/−57 | 185/159/+26 | +108 | 28/218/−190 | +0.096296 / −6.031746 |
| SOURCE predicted | clean | 171/145/+26 | 60/42/+18 | +9 | 48/322/−274 | +0.470493 / −4.818863 |
| SOURCE predicted | star | 139/114/+25 | 81/31/+50 | +74 | 31/363/−332 | +0.851171 / −5.835890 |

星地相对 OLD，闭集正确由 1944 增至 1995，正确接收由 1564 增至 1581，但宏 A_K 反而降低 3.266233 pp。逐类正确接收净变化为 `+13,+12,0,0,+13,−21`；容量仅 93 的 8-20 损失 21 条，对等权宏平均的影响大于几个大类的净增，不能写成“总正确条数下降”。主方法的 1995 条闭集正确中有 414 条被拒，另有 1020 条错误接收；OLD 对应为 380/1069，说明身份变化与拒识变化同时存在。

相对匹配 SOURCE，TARGET 星地宏 A_K 增 0.851171 pp，但 R_U 降 5.835890 pp，H 降 0.429186 pp；相对 OLD，H 降 3.030888 pp。SOURCE 本身相对 OLD 的联合性能也退化，因此不能只选较弱控制解释为成功，更不能用 pooled 的局部成绩替换预定主方法。

## 分组、曲线与保护约束

主方法星地六个场景均具有完整旧六类和未知四类覆盖，但联合约束全部失败；此处不存在用删类后宏平均认证的处理。

| 星地主场景 | 已知/未知数 | 学生闭集 | A_K | C_K | R_U | H |
|---|---:|---:|---:|---:|---:|---:|
| practical_high | 206/256 | 45.814908 | 23.297347 | 35.182821 | 58.422428 | 33.311095 |
| practical_mid | 293/443 | 49.102187 | 24.271024 | 37.634160 | 54.166154 | 33.521554 |
| practical_low_suburban | 1715/2167 | 48.712407 | 26.498297 | 42.493263 | 49.914838 | 34.618608 |
| practical_high_urban | 197/244 | 56.835541 | 38.641498 | 53.358603 | 53.917939 | 45.019071 |
| practical_mid_urban | 315/397 | 45.151968 | 27.319660 | 41.143603 | 49.859534 | 35.298257 |
| practical_low_urban | 1782/2223 | 34.023748 | 21.741021 | 49.222027 | 43.229966 | 28.931794 |

星地逐旧 TX 的 A_K/C_K 为：14-10 45.942321/62.977867，14-7 12.859560/66.920474，20-15 2.666667/26.000000，20-19 2.216749/9.605911，6-15 59.696459/63.827993，8-20 24.731183/38.709677（%）。四未知的 clean/star R_U 分别为：5-5 34.731774/32.530949，10-7 77.275886/62.056984，2-19 68.114926/45.830413，7-11 62.765957/52.340426（%）。这是固定输出的评分细分，不据此改变门控或温度。

目标仅 RX34，不能外推多 RX 稳定性。03-15 的 clean/star A_K 为 27.673233/24.487227%，R_U 为 59.999950/48.664137%；03-23 缺两个旧 TX，A_K、C_K、H 及完整闭集宏值为 N/A，R_U 仍可定义，为 61.458671/47.746037%。源五 RX 主 A_K 为 83.148148% 至 92.722222%，R_U 为 51.904762% 至 74.444444%；源三日 A_K 为 89.755556% 至 89.944444%，R_U 为 61.238095% 至 68.190476%，不是未见 RX 确认。

主方法源 known12 校准通过，整体原 F0−A_K 损伤为 1.133333 pp；源 RX3 保护恰为 2 pp。已存下一 common-tail 候选使该 RX 损伤达 2.018519 pp，被既定保护拒绝。30 个 RX×scene 仅诊断，其中 8 组未过：RX3 的 high、mid、low_suburban、mid_urban、low_urban，RX4 的 high_urban，以及 RX8 的 high_urban、low_urban；故不能称 known42 全通过。

| 条件 | OLD predicted AUROC/OSCR | SOURCE predicted AUROC/OSCR | 主 TARGET predicted AUROC/OSCR |
|---|---|---|---|
| source | 94.730673/87.940433 | 94.667079/88.270190 | 94.182419/87.759347 |
| clean | 54.364758/33.693180 | 51.216600/30.410850 | 49.948653/29.999361 |
| star | 50.997190/27.001261 | 47.891374/24.383352 | 46.712466/23.971172 |

以上曲线为固定 margin 的真实类别宏加权诊断，未选择新运行点。评分同时保存逐真实类 Wilson 95% 区间；例如星地 20-15 的 A_K 为 4/150，区间约 1.041794% 至 6.655389%，不能代替跨 RX/seed 确认。完整 TX/RX/day/day×scene、42 组和配对见评分 artifact，未只挑选有利子组。

主方法 clean/star 的原 F0 闭集−A_K 分别为 20.888471/17.035569 pp，远超 2 pp；C_K 为 39.367787/44.673654%，低于 90%；R_U 为 60.722136/48.189693%，低于 clean95%/star90%。星地 A_K 未同时高于两控制，R_U 未保持两控制，`adaptation_success=False`、`development_gate_passed=False`。源校准通过不能抵消这些失败。

## 冻结顺序、成本与执行证据

网络及原密度先于 V 冻结；新增 SOURCE/TARGET 各两组六阈值，共 24 新值，OLD 两组 12 值精确复用，总 36 个 FP64/288 B，单主六阈值 48 B。新 density/normalizer fit 为 0，不能把新 24 个 V 阈值说成零统计校准。全部 6×3 预测完成后，独立 CPU 才连接源 NV、clean Q、star Q 三份真值各一次；Utruth=0，预测阶段真值读取为 0。[预测完成标记](../local_artifacts/pamf_phase3_native_identity_20261031_r01/predict/complete.json)与[评分完成标记](../local_artifacts/pamf_phase3_native_identity_20261031_r01/score/complete.json)分别记录这些顺序与计数。

| 新增计算 | 训练 | 冻结预测 | 合计 |
|---|---:|---:|---:|
| 原生身份学生真实前向行 | 170496 | 101252 | 271748 |
| 原 F0 真实前向行 | 576 | 20476 | 21052 |
| 固定 PCD 真实前向行 | 576 | 20476 | 21052 |
| DURA 真实前向行 | 0 | 20476 | 20476 |
| 合成 F0 / r03 / PCD / G 行 | 4/2/2/4 | 4/2/4/4 | 8/4/6/8 |
| 合成原生身份学生行 | 2 | 4 | 6 |

训练 F0/PCD 各 576 行只生成固定 U 教师；L6300 的 F0 教师 logits 为已有缓存。预测两个学生在源 V/NV 共 60300 行、两个 future 条件共 40952 行；Q 的 G 输出由原 F0、PCD 与两个学生共享，旧控制精确复用。新物理观测为 0，不将重复前向计为新样本。原 F0 E50/11100、r03/2220、既有 PCD 的 U576 训练、源 L/OE 密度及 DURA E20/4940、缓存构建成本均继承保留。

主部署为身份 reader232993＋固定 PCD334721＋G266951＝834665 神经参数；reader 外 667729 参数不部署，不能把 wrapper900722 当身份网络部署成本。SOURCE_ID/TARGET_ID 各新增 body/default-head 444 步，原 F0 不变；不部署 source/U 行 bank、训练教师或伪标签表。无发现或匿名注册，A_N/H_OW 为 N/A。

实际 train/predict/CPUscore 耗时为 96.677292/43.072292/18.075613 s，pipeline 为 158.513412 s；它们包含不同阶段工作，不是 B1 时延或匹配加速证据。三阶段 return0、终态 ANALYZED_DEVELOPMENT，无真实运行技术失败。6 条观测独立捕获 driver1886998、train1887023、predict1887947 的 CWD/完整 argv；后两者 CUDA5、UUID `ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331`。训练 GPU 样本为 408/704/704 MiB，预测为 518 MiB，均非峰值。CPU1888567 有 pipeline 的 CPU/gpuNone、命令与完成产物，未独立捕获 live，不称全部阶段均有 live 核验。

完整四份 stdout 共 22 行、2088 B，warning/error/traceback 均为 0；训练日志 507→996→1257→1392 B、预测 506→630 B 有实际增长，终态 owned live/GPU 空、foreign 操作为 0。本报告亦完整读取两份 444 行 steps，而非只用 stdout 的稀疏进度。证据见[流水线](../local_artifacts/pamf_phase3_native_identity_20261031_r01/pipeline_state.json)、[观测](../local_artifacts/pamf_phase3_native_identity_20261031_r01/observations.jsonl)、[训练完成](../local_artifacts/pamf_phase3_native_identity_20261031_r01/train/complete.json)及[训练烟测](../local_artifacts/pamf_phase3_native_identity_20261031_r01/train/smoke.json)。

合成验证共 13 个唯一测试、43 个子项，7 AST、4 CLI、实际 runtime 接口检查通过，一次独立 P0/P1 审查 PASS。run 集成初期 fixture 的目录、完成字段及强制测试类问题经定向修正后通过，未声称初轮全绿；其 1056 条旧 `np.trapz` 弃用警告来自合成测试，不是生产告警。测试并未替代真实 checkpoint 首步骤烟测。单包 830 文件/data0，SHA `42e88264827db3b65586d6b0ac6e2964797cba6c2c9cc76fd7a6d6b9260e0a3e`，本地/远端比较、compile/runtime 为 VERIFIED，见[验证记录](../local_artifacts/pamf_phase3_native_identity_20261031_r01/test_evidence.json)、[审查](../local_artifacts/pamf_phase3_native_identity_20261031_r01/review.json)与[发布核实](../local_artifacts/pamf_phase3_native_identity_20261031_r01/release_verification.json)。

本结果只否定当前固定原生身份自训练方案的晋级资格：闭集宏值小增而拒识后的联合指标退化，不能归因于单一 τ、温度、mask 或伪标签污染，因为未做这些因素的分离因果对照，且 U 真值从未读取。目标 Q 不回流改参数，不扩 seed、不换主；原 PCD 研究起点与历史困难结果保留，正式测试、匿名注册及完整目标均未完成。

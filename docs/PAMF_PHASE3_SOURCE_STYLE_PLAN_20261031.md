# Phase3 同 TX 跨 RX 中间统计混合：固定实验

Run `20261031-phase3-source-style-s2026092701-r01`。以 PCD 为固定比较基线，研究源端接收机泛化；同时训练身份与开放表征，不沿用上一轮“只变身份、冻结 PCD 特征”的接法。该实验不新增目标适应或注册，完整目标保持 ACTIVE。

方法参考 [MixStyle 原论文](https://openreview.net/pdf?id=6xHJ37MVxxp)的逐实例特征统计混合。这里的同 TX 跨 RX 配对、DURA 后 RF 输入及共同身份／开放目标是本项目实验变体，不能继承视觉论文效果，也不能说 MixStyle 算子在项目内从未研究过。此前 Phase1 有相关实现，当前没有该完整 Phase3 接法的结果。源-only DG 与未知拒识都要实测，不从损失下降推定有效。

## 两臂与真实层位

SOURCE_NATIVE 与 STYLE_NATIVE 独立严格构造同源原 F0 E50/11100，再提取完整 reader/default6；开放 projection102688仅取合规 pure-source r03 E_SIGNAL2220（code dc01a402582f93e452dbaa3ef59a9fbb44151818）。不使用 target-adapted PCD 参数。初始 native logits 要与原 F0 实际无 query 合成前向一致；由于 body 来自 F0，初始 open z 不宣称等于 r03。

NativeOpen只持有reader232993＋projection102688，共335681参数；冻结Sinc48，训练335633。构造wrapper的reader外667729参数不进入学生，不保存或部署。一次reader.features产生原native身份及四贡献，原默认头得到logits6，原SignalOpen规则由四贡献640及单位raw生成unit z160；二者共享同一次特征计算，不插入新身份头。源CE/KD、crossset、OE拓扑与父z锚定共同训练body/defaulthead/projection。

| 臂 | 源数据及目标 | 浅／中层统计混合 | U 新读 |
|---|---|---|---:|
| SOURCE_NATIVE | Lanchor128＋Lpartner128＋OE64 | 关闭，无临时hook | 0 |
| STYLE_NATIVE，唯一主 | 完全相同的A/B/OE索引和目标 | 固定两层、各自p=0.5 | 0 |

每臂444步，独立原F0reader＋pure-r03 projection初值，AdamW lr1e−4/wd1e−4/eps1e−8、ordinary normclip5。L anchors采样seed2026103160；每条partner从全L中同y、异rx_i候选随机抽取（seed2026103170）。OE复用原DomainSampler的15域IID→域内IID（seed2026103018）。两个新臂完整A/B/OE轨迹相同，但与此前192L/64OE或128L的旧实验不是完全计算预算匹配。

在reader相对路径`core.id_backbone.time_down`（实数[N,48,128]）与`core.id_backbone.t2.pool`（[N,96,32]）混合，沿最后时间维计算population mean/std，eps1e−6。均值／标准差detach、标准化内容保留梯度。L两半互为伙伴，OE valid=False精确透传；每层逐步独立启用p=.5，逐实例lambda~Beta(.1,.1)，独立style RNG2026103171。context只包该次源前向，finally清除hooks；全程eval保留BN/Dropout数值状态，梯度开启。推理无MixStyle、无新模型参数或跨query统计。

源loss为`crossset(L256) + OE_topology_KL64 + 10*anchor + CE_L256 + KD_L256`。anchor沿旧body定义，等权平均L256均值与OE64均值，父z来自冻结pure-r03在原P1上的缓存；KD为原F0同P1缓存logits、T=2、KL batchmean×4。OE只成员监督和公共RX/day，不读OE真实TX或K。源L教师不新增F0真实前向，父r03 z在L/OE共12434条只缓存一次。

## 数据、校准与完整测试

沿用合同392005/VALIDATED_ONCE。新训练只读原星地L6300/OE6134的P1；原Us56700只祖先、本轮0。源五RX为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；旧六TX为14-10/14-7/20-15/20-19/6-15/8-20。V27000只在神经与密度冻结后校准，不是独立已知测试；NV3150的19-15/19-16/7-1各1050只冻结预测和独立评分。

目标开发RX3-2/index34：既有03-01 U576本轮不打开值或U ID数组；未来03-15/23 Q5525/4713共10238，旧4508／四未知5730。旧六类数量1491/1182/150/406/1186/93，未知5-5/10-7/2-19/7-11为1454/1439/1427/1410。7-11整类不在早日U，真实角色/K/配额只builder/scorer可见；03-23缺旧20-19/8-20，完整AK/CK/H保持N/A。源/U/Q原角色互斥复用，clean/star Q同canonical和顺序。既有PCD对照含U576祖先，新两个学生及其新源统计均纯源，不混淆两者。

所有Phase1后学习仍是既定G10 received经DURA的P1；feature混合不增加物理记录。G10 600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，一次input RMS/输出无RMS，receiver2027，源scene/channel2026103001/4001、future2026103133/34。DURA固定R69 E20/4940、266951参数、reuse=True/graph=True/native=False、FP32/TF32off。源、跨RX clean、跨RX星地全部接DURA，clean只冻结测试，无clean U/训练/逐样本恢复参考。

两新学生冻结后，各仅用源L/OE拟合known90/OE32、L残差白化、固定10%收缩及背景32容量/25Lloyd/seed2026103019，新增密度2套。V按原known12规则校准各pooled/predicted两组六tau，共24新值，旧E0/PCD24值精确复用，总48 FP64/384B。自身native预测同时路由自身密度和tau；原F0同DURA闭集独立保存为2pp保护基准，不能用更低学生closed替换。

固定E0/OLD_PCD/SOURCE_NATIVE/STYLE_NATIVE各两读出，共8×3。主唯一STYLE_NATIVE_predicted_ratio。全部模型、密度、tau冻结，所有24个运行点预测完整后，独立CPU才连接sourceNV/cleanQ/starQ三truth各一次，Utruth0。科学成功需sourceknown12、cleanRU≥95%、starRU≥90%、CK≥90%、原F0同Gclosed−AK≤2pp及所需主景覆盖；主starAK高于SOURCE_NATIVE与OLD_PCD，RU不低E0/SOURCE_NATIVE/OLD_PCD全部控制。成功只记source_style_success，非新增目标UDA成功。

报告原F0/student closed、宏AK/CK/RU/H、AUROC/OSCR、逐TX/RX/day/六scene、source42组诊断、同ID身份纠错与拒识变化、源配对/梯度/全888步曲线。单RX/seed、NV/Q已曝光、global physical historyUNKNOWN、正式RX24-18未读均保留；无发现/匿名注册，AN/HOW=N/A。未通过不扩seed/换主/按Q改层位、lambda或阈值。

## 执行、成本与停止

新增学生888 optimizer步；每臂320×444=142080真实学生联合前向，两臂284160，另父r0312434一次，L教师6300为缓存；训练新F0/G真实前向0，U/Q/V/NV/Us训练读取0。L双曝光和临时特征视图不是更多独立样本。评估两新学生源85168＋future40952=126120，原F0/G各20476，旧控制复用；源新密度2、24新tau是统计学习成本。部署主335681＋G266951=602632，无source/U行bank或临时hooks。原F0/r03/PCD/G/cache成本继承，参数量与阶段耗时不当作受控B1加速证据。

root唯一owner；默认N607，一个训练进程内两臂顺序，随后冻结预测与独立CPU评分。每卡新增≤2、本行实际free≥4096MiB，不干预其他用户。最小八项流程沿用：实际code/config commit、聚焦负测/首步骤真实CP无query smoke、一次独立P0P1、最小预登记、独占root、单preflight/单包SHA/compile、实际PID/CWD/argv/GPU/log增长、全预测后评分。真实首步骤还核初始F0 logits、forced双层MixStyle/OE透传、退出恢复与梯度，PASS立即继续。非法祖先/角色、冻结父或Sinc/buffer变化、nonfinite、输出冲突或不能合法预测时技术失败保留；科学低分完成全固定测试，不热改或盲重启。

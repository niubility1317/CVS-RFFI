# Phase3 共享身份与拒识学生结果（2026-10-31）

**主方法 `TARGET_JOINT_predicted_ratio` 不晋级（NO_PROMOTION）。** 匹配SOURCE_JOINT的星地学生闭集与宏A_K略增，但R_U下降；相对OLD_PCD宏A_K下降6.593973个百分点。原F0保护、拒识、CK等绝对条件未过，保留原4af参考和全部旧PCD结果，完整目标ACTIVE。

Run：`20261031-phase3-joint-identity-s2026092701-r01`；实际代码 `f540589c0478bed77a9fb4e664481c80ffe7eecc`。固定设计见[本轮计划](PAMF_PHASE3_JOINT_IDENTITY_PLAN_20261031.md)；合法默认头联合适应与新身份口径见[协议新条款](PROJECT_PROTOCOL.md)。

## 模型、初始化与训练证据

两臂独立复制source-only r03 body232033+projection102688，并恢复同源F0原默认六类W960与原生cosine计算。总学生335681，冻结Sinc48和全部buffers，联合更新body231985/projection102688/head960；SOURCE_JOINT只源训练，TARGET_JOINT加U576 PCD。原F0/defaulthead、r03、G及历史学生不变；不是冻结骨干另训/替换身份头。

源损失=`crossset+OE_KL+10*父z锚点+1*L_CE`，CE仅合法L；TARGET再加`.1*(global+local)`，无目标身份标签。当前学生原生默认头决定自身closed、类条件密度路由和τ索引，不能继续称classification unchanged，也不能用F0预测代替学生路由。16×160辅助原型仅训练容量，不是未知K，不部署。

初始化合规但“r03 body/projection＋F0 W”的完整初始logits并未证明等于原F0，本轮没有预登记该合并初值Q读出。因此SOURCE_JOINT相对F0闭集下降不能唯一归因CE或源微调；同时有body初始化/身份路径差异。TARGET对SOURCE才是匹配的PCD比较，不补Q初值测试或据本次结果选参数。

完整SOURCE_JOINT/TARGET_JOINT各1..444、共888步连续有限；源L/OE索引全444匹配，TARGET的L/OE/U与旧freshPCD全444匹配。各源L85248/OE28416曝光，uniqueL6300/OE5913；TARGET uniqueU576，弱全U255744、两strong56832条次。SOURCE无U、完成后才首次打开U；两臂从相同纯源初值独立训练。

下表为全444记录的连续三段及全程均值，非抽样；anchor未乘10，源CE权重1。

| 臂 | 步段 | total | source | crossset | OE_KL | anchor | L_CE | global / local |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SOURCE_JOINT | 1–148 | 0.459052 | 0.459052 | 0.288443 | 0.063903 | 0.002211 | 0.084601 | 0.000000 / 0.000000 |
| SOURCE_JOINT | 149–296 | 0.447739 | 0.447739 | 0.285128 | 0.062780 | 0.002226 | 0.077574 | 0.000000 / 0.000000 |
| SOURCE_JOINT | 297–444 | 0.430541 | 0.430541 | 0.273045 | 0.061388 | 0.002579 | 0.070317 | 0.000000 / 0.000000 |
| SOURCE_JOINT | 1–444 | 0.445777 | 0.445777 | 0.282205 | 0.062690 | 0.002338 | 0.077497 | 0.000000 / 0.000000 |
| TARGET_JOINT | 1–148 | 1.024528 | 0.475791 | 0.290242 | 0.067079 | 0.003345 | 0.085018 | 2.730410 / 2.756963 |
| TARGET_JOINT | 149–296 | 0.925930 | 0.466029 | 0.287911 | 0.065564 | 0.003417 | 0.078378 | 2.295904 / 2.303113 |
| TARGET_JOINT | 297–444 | 0.878963 | 0.449594 | 0.276540 | 0.064100 | 0.003771 | 0.071241 | 2.148332 / 2.145359 |
| TARGET_JOINT | 1–444 | 0.943140 | 0.463804 | 0.284898 | 0.065581 | 0.003511 | 0.078212 | 2.391549 / 2.401812 |

两臂CE与段均total下降；TARGET末段source.449594、anchor.003771仍高于SOURCE.430541/.002579。不同总目标不能直接排优劣。普通AdamW lr1e−4/wd1e−4/eps1e−8、normclip5各444次；裁剪前最大13.816538/16.221083，346/369步>5。首2步head/body/各分支梯度非零、Sinc0；全程模块梯度未记录。完成标记确认head/body/projection变化，Sinc/buffers/原模型保持，888步是联合优化次数，不加成三倍模块更新。

原生分类和open输出复用同一次学生编码；源CE各85248个L条次不增加IQ前向。复制body特征累计13764、projection2664，学生head新增444、原F0仍11100；TARGET原型不部署。全部step方程/梯度/极值/索引与smoke见analysis。

## 24个运行点：原F0、学生闭集和开放识别

均为真实TX等权宏百分比；F0损失为原F0同DURA闭集−A_K（pp）。新两模型“模型闭集”来自各自学生头；E0/OLD_PCD仍用原F0。C_K是接收率而非正确率，H为宏A_K/R_U调和平均。

| 条件 | 方法 | 原F0闭集 | 模型闭集 | A_K | C_K | R_U | H | F0损失 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source_DURA | E0_pooled_ratio | 90.981481 | 90.981481 | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 |
| source_DURA | E0_predicted_ratio | 90.981481 | 90.981481 | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 |
| source_DURA | OLD_PCD_pooled_ratio | 90.981481 | 90.981481 | 89.770370 | 97.081481 | 63.619048 | 74.465443 | 1.211111 |
| source_DURA | OLD_PCD_predicted_ratio | 90.981481 | 90.981481 | 89.733333 | 96.748148 | 64.285714 | 74.907247 | 1.248148 |
| source_DURA | SOURCE_JOINT_pooled_ratio | 90.981481 | 91.596296 | 89.648148 | 95.707407 | 65.428571 | 75.647077 | 1.333333 |
| source_DURA | SOURCE_JOINT_predicted_ratio | 90.981481 | 91.596296 | 89.618519 | 95.551852 | 64.761905 | 75.189144 | 1.362963 |
| source_DURA | TARGET_JOINT_pooled_ratio | 90.981481 | 91.581481 | 89.725926 | 95.985185 | 68.222222 | 77.510273 | 1.255556 |
| source_DURA | TARGET_JOINT_predicted_ratio | 90.981481 | 91.581481 | 89.718519 | 95.888889 | 67.777778 | 77.219871 | 1.262963 |
| clean_DURA | E0_pooled_ratio | 48.567015 | 48.567015 | 31.187288 | 43.537692 | 62.980616 | 41.716859 | 17.379728 |
| clean_DURA | E0_predicted_ratio | 48.567015 | 48.567015 | 30.999690 | 42.654045 | 64.187968 | 41.808091 | 17.567325 |
| clean_DURA | OLD_PCD_pooled_ratio | 48.567015 | 48.567015 | 32.123647 | 44.791196 | 60.790839 | 42.034854 | 16.443369 |
| clean_DURA | OLD_PCD_predicted_ratio | 48.567015 | 48.567015 | 31.661804 | 43.761391 | 61.979647 | 41.912794 | 16.905211 |
| clean_DURA | SOURCE_JOINT_pooled_ratio | 48.567015 | 41.405030 | 20.458189 | 33.236763 | 66.991864 | 31.344342 | 28.108827 |
| clean_DURA | SOURCE_JOINT_predicted_ratio | 48.567015 | 41.405030 | 20.199283 | 33.007479 | 67.220363 | 31.064027 | 28.367732 |
| clean_DURA | TARGET_JOINT_pooled_ratio | 48.567015 | 42.228981 | 21.341129 | 34.514492 | 63.922079 | 31.999015 | 27.225886 |
| clean_DURA | TARGET_JOINT_predicted_ratio | 48.567015 | 42.228981 | 21.147817 | 34.325475 | 63.883173 | 31.776407 | 27.419198 |
| star_DURA | E0_pooled_ratio | 41.721059 | 41.721059 | 26.867357 | 47.141720 | 51.837083 | 35.391279 | 14.853702 |
| star_DURA | E0_predicted_ratio | 41.721059 | 41.721059 | 26.440597 | 46.197372 | 52.658525 | 35.204508 | 15.280462 |
| star_DURA | OLD_PCD_pooled_ratio | 41.721059 | 41.721059 | 29.175335 | 50.230381 | 48.454372 | 36.420916 | 12.545723 |
| star_DURA | OLD_PCD_predicted_ratio | 41.721059 | 41.721059 | 27.951722 | 48.582412 | 49.307698 | 35.678111 | 13.769336 |
| star_DURA | SOURCE_JOINT_pooled_ratio | 41.721059 | 38.156352 | 20.532053 | 37.769812 | 57.540137 | 30.264737 | 21.189006 |
| star_DURA | SOURCE_JOINT_predicted_ratio | 41.721059 | 38.156352 | 20.693258 | 37.977471 | 57.573809 | 30.444215 | 21.027800 |
| star_DURA | TARGET_JOINT_pooled_ratio | 41.721059 | 38.639796 | 21.497725 | 40.407576 | 52.982125 | 30.585323 | 20.223334 |
| star_DURA | TARGET_JOINT_predicted_ratio | 41.721059 | 38.639796 | 21.357750 | 40.113129 | 52.561046 | 30.373484 | 20.363309 |

主source原F0→学生闭集提升.600000pp，但学生闭集→A_K损1.862963pp，总原F0损1.262963pp。clean闭集下降6.338034pp＋拒识损21.081164pp＝总27.419198pp；star下降3.081262pp＋拒识损17.282046pp＝总20.363309pp。更低学生闭集不能替代原F0的2pp保护，也不能把拒识伤害全写成身份退化。

匹配TARGET对SOURCE_JOINT：star模型闭集38.639796 vs38.156352（+.483444pp）、A_K+.664492pp、R_U−5.012763pp，H−.070731pp。正确接收35增/38失（净−3）、未知拒对44增/329失（−285），总接收+43；closed纠对33/错48（−15）。clean宏A_K+.948534pp，但正确净−22、未知−189。类别容量不同，宏均值和净条数方向可不同，必须分列。

主对OLD_PCD：star正确接收117增/169失（−52）、未知261/76（+185），宏ΔA_K−6.593973pp；clean正确55/238（−183）、未知211/103（+108），宏ΔA_K−10.513987pp。主对E0：star正确121/167（−46）、closed纠对266/错279（−13）、未知143/148（−5），clean正确−192/未知−17。源与目标完整三配对、逐TX事件和ROC/OSCR保留于评分及analysis，不因pooled局部值或源收益换主。

## 覆盖与目标判定

| 主条件 | 日期 | 原F0闭集 | 学生闭集 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|---:|
| clean_DURA | 2021_03_15 | 49.211347 | 45.059718 | 21.255283 | 32.746064 | 62.944143 | 31.779209 |
| clean_DURA | 2021_03_23 | N/A | N/A | N/A | N/A | 64.813884 | N/A |
| star_DURA | 2021_03_15 | 41.194948 | 40.050707 | 21.071605 | 39.366523 | 52.493331 | 30.071901 |
| star_DURA | 2021_03_23 | N/A | N/A | N/A | N/A | 52.647759 | N/A |

主source known12通过、known42未全过，8个RX×scene诊断超限；完整42组、源五RX三日/targetRX34、所有TX/day/scene和CI保留。主未知source旧NV三类拒对700/1050、644/1050、791/1050；target新四类5-5/10-7/2-19/7-11的clean529/1454、1056/1439、1066/1427、1002/1410，star488/1454、870/1439、777/1427、871/1410。两种未知名单不可混用。

主star六景A_K/R_U（%）：high16.769898/58.215389、mid21.409559/57.981468、low_suburban22.727194/54.809008、high_urban37.048046/57.470708、mid_urban21.951900/52.905007、low_urban19.622031/47.981360。03-23缺旧20-19、8-20，完整六旧A_K/C_K/H/闭集宏保持N/A；四未知完整时R_U仍可定义，不缩roster、补零或省略缺类日。合并两日场景覆盖另报，不与旧RX24-1 high缺19-15混淆。

主clean CK34.325475/RU63.883173/原F0损27.419198pp，star CK40.113129/RU52.561046/原F0损20.363309pp，均不满足cleanRU95/starRU90/CK90/损失2pp联合目标。star未胜OLD_PCD、RU低于SOURCE_JOINT及E0，适应与晋级均False。联合学生的源端闭集提升、目标PCD的小幅匹配闭集增益和拒识下降分别呈现，不能只归因某一个损失项或唯一原因。

## 冻结顺序、成本和执行

两新神经状态全冻后各用L6300/OE6134拟合known90/OE32和normalizer，再以源V27000校准四组六τ24新值；旧E0/OLD_PCD四列预测/聚合/24τ精确复用，共48τ384B。每个新模型的own closed索引自己的阈值，原F0独立baseline保留；全三DURA×八读出固定后CPU连接三truth各一次，Utruth0。V为额外源标签校准/重复源诊断，不是独立test。

新训练实际SOURCE113664/TARGET426240、父source/U13010，合计signal552914、G576/F00；888优化联合更新body/projection/head。新评估两学生各63060，共signal126120，F0/G各20476；E0/OLD_PCD新signal0。新增链signal679034/G21052/F020476，旧训练/缓存及F011100/r03原2220/G4940继承。两份源GMM/normalizer与新24τ是真实统计成本，非零拟合。

训练parent/F0/G各2条合成、学生梯度probe48；评估新两学生共4条、F0/G各2条，smoke/probe优化0。单主部署学生335681+G266951＝602632参数，原F0只作评估保护参考，不计单主部署，也不把全控制驻留当主部署。单GMM364340B/单主六τ48B，原型和source/U行bank不部署；TARGET及其源聚合有U祖先、SOURCE_JOINT无U。

三阶段COMPLETE/return0，train117.055821s、predict49.889152s、CPU score22.929024s、流程190.067902s，非B1。6次观测独立核driver1639732/train1639753 CWD/fullargv/CUDA5，GPU UUIDef5f59f2-e03e-fe6e-ea6b-3e03afc9b331采样406/738/1382MiB非峰值；predict1641045只有pipeline GPU5/完整产物，未live捕获。CPU1641570已独立live核CWD/fullargv/CUDAempty；终态owned空、foreign0，无真实技术失败。

模型/train4项+9子项（12.09s）、producer/score6+14（22.20s）、launcher17项（3.83s）PASS，AST10/一次P0P1 PASS；测试保留RED及最终GREEN，最终1476条旧np.trapz警告，四份完整生产stdout无warning/error/nonfinite。751文件/data0发布包SHA `c4d8787d5119539c9da949acdf1095d211a11d5c40fb7abfbd91d6bb2cfb5af5`比较/compile/runtime VERIFIED。

复用fresh f4b5池：sourceL6300/Us56700/OE6134/V27000/NV3150，U576为RX3-2/03-01，Q10238为03-15/23同ID clean/star，known4508/unknown5730；7-11从早期U整类排除，四未知/K/配额仅builder/scorer。全部DURA/G10固定观测，clean只测试，U/Q物理互斥，正式24-18未读。此Q已在前run评分，全历史仍UNKNOWN、单RX/seed、未知集合/日期容量与旧RX24-1不同，非新盲测或单因素RX比较。

后续可在源端诊断合并初值头/骨干衔接与保真约束；当前没有该初值Q读出，不从本次Q即时挑参数/阈值、不补新Q测试，也未声称定位唯一原因。原PCD历史结果及研究起点身份保持；本轮无发现/匿名REG，A_N/H_OW=N/A，完整目标ACTIVE。

复核：`local_artifacts/pamf_phase3_joint_identity_20261031_r01/train/{SOURCE_JOINT,TARGET_JOINT}/steps.jsonl`（888步）、`score/results.json`（24点/三配对/闭集纠错与拒识全分组）、`predict/models/complete.json`/`calibration/complete.json`、`analysis.json`（全曲线/索引/分解/预算/日志）、`observations.jsonl`和`pipeline_state.json`。

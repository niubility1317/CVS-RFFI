# Phase3 公共类传输对齐结果（2026-10-31）

**主方法 `CCD_predicted_ratio` 不晋级（NO_PROMOTION）。** 相对匹配PCD，星地A_K/C_K/R_U/H均下降，正确接收净少4条、未知拒对净少2条；clean正确净少18条。相对较弱E0的局部增益不能替代匹配控制和绝对保护，保留原4af基线及PCD比较，完整目标ACTIVE。

Run：`20261031-phase3-common-transport-s2026092701-r01`，实际代码 `4038105cb7f7c2defe75eb7fbfa825e7f932b40f`。完整机制、角色和信道见[本轮计划](PAMF_PHASE3_COMMON_TRANSPORT_PLAN_20261031.md)；旧body训练a336/r01、冻结评估4273/r02的SOURCE/PCD及六读出精确复用，不重复训练。

## 固定机制与完整444步证据

一个新TARGET_CCD仍从source-only r03独立初始化，冻结Sinc48坐标及buffers，更新body231985+projection102688；源192L/64OE、source crossset+OE_KL+10×父z锚点、U384平衡原型global/local均沿PCD。原F0/defaulthead、G和旧模型冻结；16×160辅助原型与6×160固定源类质心、六beta仅训练使用，不替换身份头或作为真实K。

前100步CCD关闭；101至444步加`.1*CCD_CE`，只用当前弱U与六源质心的停止梯度伪分配监督两个strong view。reg=.01/rho=.5/gamma=.7/mu=.7、每次400轮log-UOT，人工feature正/负填充不产生新IQ；本轮是固定CCD变体，不是UniOT完整memory queue/分类头/聚类复现，无未知发现或匿名REG。

完整444步连续、数值有限；L/OE/U索引与历史PCD全部444步逐值相同，batch192/64/64，unique L6300/OE5913/U384；实际源曝光L85248/OE28416，弱全U170496、两strong56832条次。SOURCE/PCD旧888步只继承；本轮新优化444，普通AdamW lr1e−4/wd1e−4/eps1e−8/clip5。

前100步目标、采样/augmentation/prototype seed固定同构，但标量并非bitexact：|新−PCD|最大/均值为total1.516342e−4/4.387677e−5、source1.558065e−4/4.193783e−5、crossset9.748340e−5/2.293199e−5、OE_KL8.384138e−5/1.538647e−5、anchor7.955357e−6/2.059757e−6、global2.255440e−4/5.779743e−5、local2.048016e−4/5.922794e−5、gradnorm.0319941/.00562176。source首差step2、total step3，早于CCD。新GPU0与旧GPU1的数值路径差异是可能解释，未独立确定；不把之后差异称为纯CCD因果，也不增加重跑门槛。

连续三段全量均值如下，anchor未乘10；不同总目标绝对值不能直接排序。

| 模型 | 步段 | total | source | anchor | global / local | CCD |
|---|---|---:|---:|---:|---:|---:|
| PCD | 1–148 | 0.881109 | 0.388741 | 0.003337 | 2.457973 / 2.465700 | 0 |
| PCD | 149–296 | 0.790940 | 0.387879 | 0.003464 | 2.019242 / 2.011363 | 0 |
| PCD | 297–444 | 0.740752 | 0.375274 | 0.003548 | 1.830134 / 1.824651 | 0 |
| CCD | 1–148 | 0.906263 | 0.389642 | 0.003411 | 2.458267 / 2.465491 | 0.242450 |
| CCD | 149–296 | 0.860601 | 0.394419 | 0.003973 | 2.002565 / 1.994208 | 0.665046 |
| CCD | 297–444 | 0.805647 | 0.383689 | 0.004164 | 1.802840 / 1.796337 | 0.620401 |

CCD启用344步，loss均值.657351；101以后source均值.390381高于PCD.383557，末148步anchor.004164高于PCD.003548，目标global/local略低。全程clip前norm最大10.801229，230步>5（PCD191步），普通clip调用444次；Sinc/physicalclip全0。分模块梯度仅step1/2/101记录，浅中层/四分支/body/projection非零、Sinc0，完成标记确认body/projection变化及原模型/Sinc/buffers保持。

344次CCD全部走人工负特征填充，算法positive rate .807292至.841146，填充236至262行，总N620至646；positive填充实际0。每步全U选112至193条、batch选13至40条，均非空；累计伪槽直方图[31195,0,0,20532,0,2682]，不是去重人数、真实类构成或伪标签准确率。beta按.7/.3更新、分配停止梯度，不部署。

实际344次UOT各400轮（共137600内迭代），保存的是每次末诊断而非全内迭代轨迹；全部有限，末KKT残差最大1.176768e−7、scaling变化最大2.400607e−7。source及总loss方程按全部444步核对，含source+.1(global+local)+.1CCD；完整误差、所有beta/选择量/曲线和极值步在analysis，不用U真值推准确率。

同row只读source几何诊断：六质心off-diagonal cosine最小−.215879/中位.286661/最大.723501；L真实类cos中位.979144、p05 .877424，真实类−最佳其他类中位.451059、p05 .078345；OE最大cos中位.508420、p95 .752503。γ=.7下source L positive99.460317%、OE8.493642%，不支持“源六质心几乎共线或γ在源完全无区分力”的解释；该源域区分不能证明跨RX U选择纯度，也不能据目标退化断言源原型差或伪标签错。诊断无新IQ/NN权重/forward/U/Q/truth/优化。

## 24个冻结运行点

数值为百分比；损失为同DURA原F0闭集−A_K（百分点）。A_K/R_U按真实TX宏平均，H为调和平均，C_K为已知接收率。

| 条件 | 方法 | A_K | C_K | R_U | H | F0损失 | AUROC / OSCR |
|---|---|---:|---:|---:|---:|---:|---:|
| source_DURA | E0_pooled_ratio | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 / 87.944738 |
| source_DURA | E0_predicted_ratio | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 / 87.937558 |
| source_DURA | SOURCE_pooled_ratio | 89.777778 | 97.088889 | 60.063492 | 71.974388 | 1.203704 | 94.963471 / 87.965076 |
| source_DURA | SOURCE_predicted_ratio | 89.740741 | 96.744444 | 61.174603 | 72.754089 | 1.240741 | 94.685262 / 87.940159 |
| source_DURA | PCD_pooled_ratio | 89.718519 | 97.014815 | 65.365079 | 75.629637 | 1.262963 | 95.310346 / 88.179099 |
| source_DURA | PCD_predicted_ratio | 89.740741 | 96.729630 | 65.523810 | 75.743693 | 1.240741 | 94.962969 / 88.096052 |
| source_DURA | CCD_pooled_ratio | 89.766667 | 97.074074 | 65.555556 | 75.774137 | 1.214815 | 95.354974 / 88.210018 |
| source_DURA | CCD_predicted_ratio | 89.755556 | 96.785185 | 65.904762 | 76.002910 | 1.225926 | 94.999594 / 88.117048 |
| clean_DURA | E0_pooled_ratio | 68.923611 | 85.416667 | 99.652778 | 81.487441 | 11.111111 | 95.991874 / 77.463710 |
| clean_DURA | E0_predicted_ratio | 70.833333 | 87.326389 | 100.000000 | 82.926829 | 9.201389 | 96.158251 / 77.770544 |
| clean_DURA | SOURCE_pooled_ratio | 68.750000 | 85.590278 | 99.652778 | 81.365979 | 11.284722 | 95.644049 / 77.158685 |
| clean_DURA | SOURCE_predicted_ratio | 70.833333 | 87.500000 | 100.000000 | 82.926829 | 9.201389 | 95.715181 / 77.326268 |
| clean_DURA | PCD_pooled_ratio | 70.833333 | 87.500000 | 98.958333 | 82.566462 | 9.201389 | 96.682702 / 78.059896 |
| clean_DURA | PCD_predicted_ratio | 73.090278 | 89.756944 | 99.305556 | 84.204711 | 6.944444 | 97.025704 / 78.495732 |
| clean_DURA | CCD_pooled_ratio | 68.055556 | 84.722222 | 99.652778 | 80.877617 | 11.979167 | 95.979818 / 77.391372 |
| clean_DURA | CCD_predicted_ratio | 69.965278 | 86.631944 | 99.652778 | 82.210992 | 10.069444 | 96.556110 / 78.105710 |
| star_DURA | E0_pooled_ratio | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.728250 / 62.604890 |
| star_DURA | E0_predicted_ratio | 64.062500 | 85.590278 | 77.083333 | 69.972325 | 8.854167 | 85.027850 / 62.982253 |
| star_DURA | SOURCE_pooled_ratio | 63.715278 | 85.937500 | 72.916667 | 68.006141 | 9.201389 | 84.578149 / 62.604287 |
| star_DURA | SOURCE_predicted_ratio | 64.409722 | 86.458333 | 75.347222 | 69.450483 | 8.506944 | 84.692684 / 62.856867 |
| star_DURA | PCD_pooled_ratio | 64.062500 | 86.284722 | 76.736111 | 69.828915 | 8.854167 | 86.184052 / 63.673683 |
| star_DURA | PCD_predicted_ratio | 65.277778 | 87.152778 | 78.125000 | 71.125908 | 7.638889 | 86.250362 / 63.941937 |
| star_DURA | CCD_pooled_ratio | 63.888889 | 85.937500 | 76.388889 | 69.581958 | 9.027778 | 85.861545 / 63.497058 |
| star_DURA | CCD_predicted_ratio | 64.583333 | 86.458333 | 77.430556 | 70.425835 | 8.333333 | 86.137635 / 63.886478 |

主对PCD predicted：source正确18增/14失（+4）、未知57增/45失（+12）；clean正确0/18（−18）、接收−18、未知1/0（+1）；star正确1/5（−4）、接收−4、未知0/2（−2）。星地ΔA_K/ΔC_K/ΔR_U均−.694444、ΔH−.700073个百分点，主未胜PCD，不能只挑E0或SOURCE解释成功。

主对E0 predicted：source正确+3/未知+125，clean正确−5/未知−1，star正确7增/4失（+3）、未知5增/4失（+1），星地ΔA_K+.520833/ΔR_U+.347222pp。主对SOURCE predicted：source正确+4/未知+149，clean正确−5/未知−1，star正确3/2（+1）、未知7/1（+6）。三完整pair及分TX/RX/day/scene计数保留，不按Q换主。

同G原F0闭集source/clean/star仍90.981481%/80.034722%/72.916667%，身份默认头不变。主clean接收499/576，正确403、错误96；star接收498，正确372、错误126。接收/正确事件须分开；净正确变化是拒识集合变化，不是默认身份头纠错。

主source known12通过、known42未全过（RX3 high/mid urban/low urban、RX4 high urban、RX8 low urban五交叉组超2pp）。未知19-15/19-16/7-1拒对为source680/1050、661/1050、735/1050；clean96/96、96/96、95/96；star74/96、74/96、75/96。完整TX/RX/day/42组及原有CI保留于评分与analysis。

主star六景A_K/R_U：high67.301587/N/A、mid69.722222/97.222222、low suburban67.670891/90.981607、high urban75.330688/93.333333、mid urban75.509259/75.555556、low urban54.666566/53.938172（%）。high未知0/4/2缺19-15，宏R_U/H=N/A，不删景/补零认证。

绝对目标亦失败：clean R_U99.652778%虽过95%，但CK86.631944%和F0损失10.069444pp未过90%/2pp；star CK86.458333%、RU77.430556%、损失8.333333pp及覆盖不合要求。star未同时严格胜E0/SOURCE/PCD且拒识不降，adaptation_success=False、NO_PROMOTION。

## 冻结顺序、真实成本与执行

旧六方法的全部评分分组与body r02逐值一致，E0/SOURCE/PCD三GMM及旧六行τ精确；旧控制新增真实signal前向0。仅新CCD冻结后用源L6300/OE6134拟合一份known90/OE32/normalizer；V27000仅新拟合两组六τ12值，旧36值复用，共48值384B。所有τ冻后NV/Q，全部24点后独立CPU三truth各一次，Utruth0；神经/密度/校准顺序标记及完整视图均完成。

新训练父signal12818（源12434+弱U384）、学生340992（源113664+弱170496+strong56832）、G384/F00；评估CCD signal44312、F0/G各1728。新增链real signal398122/G2112/F01728，旧SOURCE/PCD888及其训练/评估另列继承。新body/projection各444、原型2560；公共六质心/beta/人工填充只训练使用，不增加独立IQ样本。

合成训练学生64＝原PCD probe48+CCD-only16，parent/F0/G各2；评估parent/CCD各2、F0/G各2，probe优化0。部署主NN834665不变，单GMM364340B/单主τ48B，不部署原型/公共质心/beta/source或U行bank；本轮新1GMM和12τ是真实统计拟合，非零statfit。祖先F011100/r03原2220/G4940与旧缓存成本不抹去。

三阶段均COMPLETE/return0，train111.654490s、predict31.903433s、CPU score24.684922s、全流程168.438859s，非B1时延。4观测独立捕获driver1480122/train1480142 CWD/fullargv/CUDA0、GPU UUID56adac86-77cd-36c9-8770-dbf002650461离散406/940MiB（非峰值），train日志517→852→1058B。predict1483523仅pipeline GPU1/产物、CPU1484878仅gpuNone/代码CUDAempty/产物，未独立live，不能继承上一轮CPUlive证据；终态owned空、foreign0。

实际发布前初P1审查发现真实train complete缺controls_frozen_before_training，会使launcher拒绝；actual train→disk→launcher回归RED→GREEN后仅定点复审PASS，未改变算法。0105草稿archive保留未发布；实际403包729文件/data0、SHA `70c652bba9b8da0b441df15ab2734a373b3f680d573e9985f30b4fa48ca39344`比较/compile/runtime VERIFIED。Win206长payload失败在本地CreateProcess未spawn，stdin修复probe IQ/truth0；不是实际run失败，代码Git并发已由父级post-state核实。

最终25测试项+22子项PASS：训练core5+6（10.85s）、actual handoff定点1（7.53s）、producer/score6+16（16.72s）、launcher13（3.17s）；worker AST8/root AST5/两CLI，初失败保留而不写初轮全绿。最终测试旧np.trapz警告928条，四份完整生产stdout无warning/error/nonfinite信号；实际首步checkpoint/PCD/CCD梯度smoke及冻结保护PASS。

## 数据与结论边界

源L6300/Us56700/OE6134祖先、V27000额外known校准/NV3150、五RX三日；新CCD与PCD均有target-U祖先，SOURCE仍无U。starU384为RX24-1/03-08，与source及未来Q互斥；Q同RX/03-15同864记录，旧576/非旧288、九TX各96，clean/star均DURA、clean不学习，G10 MIX01固定信道/phase-gain派生view不重抽received。历史NV/Q曝光、单seed单RX/源日重合、正式RX24-18未读保持，无REG、A_N/H_OW=N/A。

future_registration_data_audit仅metadata：排除注册/OE/NV/formal共34TX后列出4个未来未知候选，它们早期也有记录，若将来建池须从U全排除；当前未建池、packet级历史排除完整性UNKNOWN，不是本轮预测输入、测试或盲测证据，不改变当前矩阵。

复核：`local_artifacts/pamf_phase3_common_transport_20261031_r01/train/steps.jsonl`（444步完整诊断）、旧body r01的`train/TARGET_PCD/steps.jsonl`（匹配控制）、旧body r02及本run的`score/results.json`（24点/三pair/全分组）、`analysis.json`（前100差幅/所有曲线/344UOT及beta/成本）、`observations.jsonl`/`pipeline_state.json`/`future_registration_data_audit.json`。

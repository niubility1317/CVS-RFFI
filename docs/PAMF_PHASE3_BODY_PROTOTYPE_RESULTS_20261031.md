# Phase3 冻结 Sinc 的开放骨干原型适配结果（2026-10-31）

**主方法 `PCD_predicted_ratio` 有相对改善，仍不晋级（NO_PROMOTION）。** 星地 A_K/R_U 同时高于冻结 E0 和同预算 SOURCE，但绝对已知保护、未知拒识和场景覆盖未全过。保留原 `predicted_ratio` 重点候选及 `pooled_ratio` 冻结对照，完整目标 ACTIVE；不把局部进步写成完整适应成功。

训练：`20261031-phase3-body-prototype-s2026092701-r01` / `a33612e978948da25bd733bcb9edba3ac9d74e9a`；恢复评估：同名前缀 `r02` / `4273e88392abe0717c1bd94fca1370d19c67cbb3`。两份学生权重精确复用 r01，r02 实际优化0。固定配方与恢复边界见[原计划](PAMF_PHASE3_BODY_PROTOTYPE_PLAN_20261031.md)、[恢复说明](PAMF_PHASE3_BODY_PROTOTYPE_RECOVERY_20261031.md)。

## 方法与技术恢复

两臂独立从 source-only r03 初始化；冻结开放编码器 Sinc 的48坐标和全部buffers，训练其余body231985＋投影102688＝334673参数，各444步。原F0/defaulthead、G及原r03不更新；不能将本轮称为“整个编码器冻结”。TARGET另训练16×160＝2560辅助原型，仅训练使用，16是事前容量而非真实K。

源每步192L/64OE，损失为`crossset_metric+OE_KL+10*父源z锚点误差`；TARGET再加`.1*(U_global+U_local)`。同record两份phase/gain强view预测当前弱全U384的停止梯度平衡分配及固定父近邻分配；eps.05/20轮Sinkhorn、温度.1，原型每步单位化。无真实TX/K/比例/配额或query训练；属于固定配方的原型变体，不是完整SwAV/UniOT、未知发现或匿名注册。

r01两臂训练完整结束，第一次学生加载却把全部named_buffers当作state_dict必含键，误拒6个合法nonpersistent runtime buffers（response的bank_imag/bank_real/tone_bins、三view的window_order）。实际只读诊断两权重均无持久buffer或Sinc差异，48个Sinc参数完整；故障发生于评估smoke/密度拟合/V校准/NV/Q预测及独立评分之前，属于技术失败而非科学负结果。

原r01以FAILED_PRESERVED保留。一次定点修复仅比较进入state_dict的持久buffers及Sinc，完整strict加载与祖先检查保持；nonpersistent重载回归RED→GREEN，持久buffer/Sinc篡改仍拒绝。新r02只运行predict→CPU score，真实checkpoint smoke及全流程PASS，无同类故障复现；未在旧root覆盖或重训。

## 完整888步训练分析

完整SOURCE/TARGET各1..444记录连续且有限；源方程最大误差3.92e−8、TARGET总方程8.95e−8。444步L/OE逐索引完全相同；各曝光L85248/OE28416，实际unique L6300/OE5913（候选池6134，未抽到221条），TARGET强批U64×444=28416、unique384、两strong共56832条次，弱全U共170496条次。源输入始终匹配，total/梯度从step1、source分量从step2首次分歧。

下表为全记录连续三段和全程均值，非首尾抽样；anchor未乘10，G/L为目标global/local。

| 臂 | 步段 | total | source | crossset | OE_KL | anchor | G / L |
|---|---|---:|---:|---:|---:|---:|---:|
| SOURCE | 1–148 | 0.368763 | 0.368763 | 0.287152 | 0.063361 | 0.001825 | 0.000000 / 0.000000 |
| SOURCE | 149–296 | 0.364528 | 0.364528 | 0.285097 | 0.061748 | 0.001768 | 0.000000 / 0.000000 |
| SOURCE | 297–444 | 0.353902 | 0.353902 | 0.273840 | 0.059812 | 0.002025 | 0.000000 / 0.000000 |
| SOURCE | 1–444 | 0.362398 | 0.362398 | 0.282029 | 0.061640 | 0.001873 | 0.000000 / 0.000000 |
| TARGET_PCD | 1–148 | 0.881109 | 0.388741 | 0.289554 | 0.065814 | 0.003337 | 2.457973 / 2.465700 |
| TARGET_PCD | 149–296 | 0.790940 | 0.387879 | 0.288858 | 0.064382 | 0.003464 | 2.019242 / 2.011363 |
| TARGET_PCD | 297–444 | 0.740752 | 0.375274 | 0.277129 | 0.062661 | 0.003548 | 1.830134 / 1.824651 |
| TARGET_PCD | 1–444 | 0.804267 | 0.383965 | 0.285180 | 0.064286 | 0.003450 | 2.102450 / 2.100571 |

SOURCE段均total0.368763→0.364528→0.353902；TARGET0.881109→0.790940→0.740752，global/local末段1.830134/1.824651。但TARGET末段source0.375274、anchor0.003548仍高于SOURCE0.353902/0.002025；各步有波动，不能以两个不同目标的总loss绝对大小判断优劣，也不将锚点偏移直接断言为性能变化的唯一原因。

Fresh AdamW lr1e−4/wd1e−4/eps1e−8、普通norm clip5各444次；SOURCE/TARGET裁剪前norm最大10.820683/10.803161，超过5的165/191步。Sinc优化步、physical clip全0。首2步浅/中层、时频/PA/参考/签名与body/projection梯度均非零、Sinc0；模块记录仅首2步，不能宣称全444步逐模块审计。完成标记确认body/projection变化、Sinc/buffers及原模型不变；复制feature累计13764，投影2664，原F0仍11100。

原型初始化25轮后分配数5至60；444步软分配质量范围21.647755至24.733082，接近算法平衡参考384/16＝24。这是训练分配统计，不是真TX纯度、真实设备数或注册证明；原型不部署。全部888步标量曲线、极值步、分模块梯度和曝光直方图保留于analysis。

## 18个运行点与匹配收益

均为百分比；损失为同DURA输出原F0闭集−A_K（百分点）。A_K/R_U为逐真实TX宏平均，H为调和平均，C_K是已知接收率。

| 条件 | 方法 | A_K | C_K | R_U | H | F0损失 | AUROC / OSCR |
|---|---|---:|---:|---:|---:|---:|---:|
| source_DURA | E0_pooled_ratio | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 / 87.944738 |
| source_DURA | E0_predicted_ratio | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 / 87.937558 |
| source_DURA | SOURCE_pooled_ratio | 89.777778 | 97.088889 | 60.063492 | 71.974388 | 1.203704 | 94.963471 / 87.965076 |
| source_DURA | SOURCE_predicted_ratio | 89.740741 | 96.744444 | 61.174603 | 72.754089 | 1.240741 | 94.685262 / 87.940159 |
| source_DURA | PCD_pooled_ratio | 89.718519 | 97.014815 | 65.365079 | 75.629637 | 1.262963 | 95.310346 / 88.179099 |
| source_DURA | PCD_predicted_ratio | 89.740741 | 96.729630 | 65.523810 | 75.743693 | 1.240741 | 94.962969 / 88.096052 |
| clean_DURA | E0_pooled_ratio | 68.923611 | 85.416667 | 99.652778 | 81.487441 | 11.111111 | 95.991874 / 77.463710 |
| clean_DURA | E0_predicted_ratio | 70.833333 | 87.326389 | 100.000000 | 82.926829 | 9.201389 | 96.158251 / 77.770544 |
| clean_DURA | SOURCE_pooled_ratio | 68.750000 | 85.590278 | 99.652778 | 81.365979 | 11.284722 | 95.644049 / 77.158685 |
| clean_DURA | SOURCE_predicted_ratio | 70.833333 | 87.500000 | 100.000000 | 82.926829 | 9.201389 | 95.715181 / 77.326268 |
| clean_DURA | PCD_pooled_ratio | 70.833333 | 87.500000 | 98.958333 | 82.566462 | 9.201389 | 96.682702 / 78.059896 |
| clean_DURA | PCD_predicted_ratio | 73.090278 | 89.756944 | 99.305556 | 84.204711 | 6.944444 | 97.025704 / 78.495732 |
| star_DURA | E0_pooled_ratio | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.728250 / 62.604890 |
| star_DURA | E0_predicted_ratio | 64.062500 | 85.590278 | 77.083333 | 69.972325 | 8.854167 | 85.027850 / 62.982253 |
| star_DURA | SOURCE_pooled_ratio | 63.715278 | 85.937500 | 72.916667 | 68.006141 | 9.201389 | 84.578149 / 62.604287 |
| star_DURA | SOURCE_predicted_ratio | 64.409722 | 86.458333 | 75.347222 | 69.450483 | 8.506944 | 84.692684 / 62.856867 |
| star_DURA | PCD_pooled_ratio | 64.062500 | 86.284722 | 76.736111 | 69.828915 | 8.854167 | 86.184052 / 63.673683 |
| star_DURA | PCD_predicted_ratio | 65.277778 | 87.152778 | 78.125000 | 71.125908 | 7.638889 | 86.250362 / 63.941937 |

主对E0 predicted：source正确34增/35失（净−1）、未知178增/65失（+113）；clean正确14/1（+13）、接收+14、未知0/2（−2）；star正确9/2（+7）、接收+9、未知7/4（+3）。星地ΔA_K+1.215278、ΔR_U+1.041667、ΔH+1.153583个百分点；这是固定同ID运行点的实际相对改善。

主对匹配SOURCE predicted：source正确27增/27失（0）、未知203/66（+137）；clean正确14/1（+13）、接收+13、未知−2；star正确5增/0失（+5）、接收+4、未知9/1（+8）。星地ΔA_K+0.868056、ΔR_U+2.777778、ΔH+1.675425个百分点。strict A_K above both与R_U不低两控制均True；不能沿用上一投影版“未胜SOURCE”的结论。

原F0/defaulthead同G闭集source/clean/star90.981481%/80.034722%/72.916667%逐条保持。主clean接收517/576，其中正确421、错误96；star接收502，正确376、错误126。新增正确接收来自拒识集合变化，不是原身份头纠错；接受不等于身份正确。

源known12通过，known42未全过：RX3 high/mid urban/low urban、RX4 high urban、RX8 low urban五交叉组原F0损失超2个百分点；30个交叉组完整诊断，不能将12组保护标为42组通过。源未知三TX拒对682/1050、650/1050、732/1050；clean96/96、96/96、94/96；star75/96、74/96、76/96（依19-15/19-16/7-1）。全部TX/RX/day、覆盖和原有CI见评分及analysis。

主star六景A_K/R_U：high72.857143/N/A、mid73.055556/97.222222、low suburban68.206605/90.981607、high urban75.330688/93.333333、mid urban77.361111/81.111111、low urban54.666566/54.632616（%）。high未知0/4/2缺19-15，R_U/H=N/A；不能删掉该景或补零认证。

绝对门槛仍失败：clean R_U99.305556%过95%，但C_K89.756944%低于90%、原F0损失6.944444个百分点超2；star C_K87.152778%、R_U78.125000%低于90%，损失7.638889个百分点超2，且必需场景覆盖不完整。因此relative gain不能替代target protection，adaptation_success/eligible=False、NO_PROMOTION，原基线保持。

## 拟合顺序、实际成本与执行

两学生神经状态冻结后，r02各用源L6300/OE6134新拟合known90/OE32及normalizer；两组背景25轮完整有限、SSE非增、空模式0（SOURCE207.833411→183.390755，TARGET212.348413→185.855175）。原E0聚合及十二τ精确复用；V27000只拟合四组六τ共24新值，不训练NN/拟合密度。全36τ冻结后NV及两个Q完整预测，独立CPU最后连接source/clean/star真值各一次，Utruth0。TARGET的源拟合聚合仍有目标U神经祖先。

r01训练实际学生signal454656＋父源/U缓存12818＝467474真实前向，G384、F00；两臂body和projection各444更新，总优化888，TARGET另2560辅助原型。r02新增优化0，评估source角色42584/Q1728条次，新signal90352（E0 1728、两学生各44312）、F0/G各1728；恢复链合计real signal557826、G2112、F01728。r01失败加载的构造/I/O和4.870201s保留，不隐藏为免费，也不冒算成新IQ。

合成成本：r01原F0/r03/G各2条实际checkpoint smoke，学生live-signal梯度probe48条；r02原F0/G各2、三个signal模型共6条。两新GMM与四新阈值集是真实统计成本；三模型密度各364340B、36个FP64τ288B（新24/旧12，单主48B）。单主NN仍F0232993+open334721+G266951＝834665参数，不部署辅助原型或逐行source/U bank。祖先F011100/r03原2220/G4940与缓存成本继承。

r01 train161.361771s、首次加载失败4.870201s；r02 predict38.864556s、CPU score12.976472s、流程51.929297s，均非B1时延。r01独立train GPU1离散258/406/738/940/940MiB；r02独立核driver1415050/predict1415080的CWD/fullargv/CUDA1，GPU UUIDa2d039ff-1792-4222-2b47-140429e10f94采样304/392MiB，均非峰值。CPU1415842已独立live确认CWD/fullargv/CUDAempty；r02预测日志509→1597B，score44B，终态live/ownedGPU空、foreign0。

原r01训练验证/一次P0/P1及技术失败证据保留；r02恢复验证19项+23子项PASS：loader fix1+2（7.12s）、producer/score6+21（11.45s）、launcher12（3.17s），仅定点复审PASS，未重做完整审查。日志保留RED与最终GREEN；最终测试696条旧np.trapz警告，r02三份完整生产stdout无warning/error/traceback。718文件/data0包SHA `e5ab5a85ee97214fa2676992f12600dc1c8c81cd94030ea1671bb6cf3aea22b4`比较、remote compile/runtime VERIFIED。

## 数据范围与后续容量的证据等级

原source L6300/隐藏U_s56700/OE6134、V27000额外校准标签/NV3150，源五RX三日；SOURCE U0，TARGET只用合法G10星地U384（RX24-1/03-08），与source及未来Q物理互斥。future同RX/03-15同864物理记录，旧576/未注册非旧288、九TX各96，clean/star均同DURA且clean不学习，G10 MIX01固定单record信道不重抽。未注册设备可在U被无标签接触，不能宣称严格训练未见。

历史NV/Q曝光、单seed2026092701/单开发RX、源日期重合、high缺类及正式RX24-18未读均保留。无发现/匿名REG、多session注册或正式盲测，A_N/H_OW=N/A，完整目标ACTIVE。

已有只读容量审计仅列出潜在RX3-2：候选早期03-01旧类7072/未知314条、未来03-15与03-23旧类4508/未知501条；尚未完成既有角色排除、未选择physical indices、未建池/新生成信道/预测。它不是本轮测试结果、合法新输入池或独立盲测证据，不扩大当前结论。

复核：`local_artifacts/pamf_phase3_body_prototype_20261031_r01/train/{SOURCE,TARGET_PCD}/steps.jsonl`（888步）、`load_failure_diagnosis.json`/`reload_fix_tests.log`；r02的`score/results.json`（18点/全分组/两配对）、`predict/models/complete.json`、`predict/calibration/complete.json`、`observations.jsonl`、`analysis.json`（合并完整训练与评估），以及r01的`future_pool_capacity_audit.json`（仅容量）。

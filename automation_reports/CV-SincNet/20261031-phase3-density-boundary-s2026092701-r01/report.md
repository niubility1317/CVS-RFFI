# PAMF 固定密度拒识边界配对训练

PLANNED，当前无新性能；run `20261031-phase3-density-boundary-s2026092701-r01`，实际代码commit `e3194bc6ed46ce1580967f036a35fcb138e1c906`，完整Phase3目标ACTIVE。

唯一主BOUNDARY_predicted；E0/REPLAY/BOUNDARY各pooled/predicted共六读出。两个独立r03副本各2220增量，sourceL192/OE64索引逐步匹配；共同crosssetKL+原F0 OE topologyKL+r03 pointanchor，主另加权1的fixedconditionalGMMratio/160 rolebalancedBCE。原F0/default6、DURA、r03本体、固定density11及点teacher保持；没有新分类头或目标U。训练只parentheader/models，禁止旧Vtau/cal/VNVQ数组。方法与方程见 `docs/PAMF_PHASE3_DENSITY_BOUNDARY_PLAN_20261031.md`。

源L6300旧六各1050/每TX×RX×day70，OE6134仅outside-six和公共RX/day，真TX/K隐藏。RX1/3/4/6/8＝1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。原Us56700未读；L占L+Us10%，V额外known标签。V27000旧六各4500，全部NN冻结后四新6tau known12（30cross诊断），E0两旧tau/cal exact；全部六tau先于NV3150/两个Q，sourceNV19-15/19-16/7-1各1050仅truthside。训练/校准IQ只原G10MIX01/DURA P1。

目标clean/star为RX24-1(index26)03-15同864物理记录，旧六/未知三各96，576known/288unknown；两个条件都同DURA。目标实时F0/G+两学生、E0 z/raw/margin exact父缓存，核新F0logits/closed；U384/03-08和正式RX24-18不读。source日已见、NV/Q历史曝光、单模型/RX，high未知0/4/2缺19-15保持NA，不重抽或认证六景。没有目标TX/K/配额/queryfit或未来匿名REG，AN/HOW为NA。

来源class-density4af0668/originalsource-onlyr03dc01a4（原2220增量、完整334721）/Tri-SIG-TIME SSL F0E50/11100/default6/DURAv1R69 E20/4940，sourcecontract392005/VALIDATED_ONCE句柄不变。新sourceNN4440总更新（两臂各2220），各feature15540/MLP4440，F0仍11100；固定GMM/normalizer新fit0、V新阈值24标量。训练student1136640＋frozenparentreference12434真实signal，F0head-only12434单列（sourceF0backbone/G0）；eval63756signal/QF0G1728，E0真实新增0，synthetic trainparent/F/G各2+gradstudent16，evalopen6/F/G各2。部署单主834665活跃NN/固定聚合364340B/六tau288B，teacherbank不部署；原训练/cache成本继承，不冒称只训MLP或B1时延。

主宏AK/RU/H，并列CK/同G原F0closed−AK/AUROC/OSCR、全TX/RX/day/scene/42组、主对E0_pred/REPLAY_pred/BOUNDARY_pooled配对。gate sourceknown12、cleanRU95/starRU90/CK90/损≤2pp/主景完整，starAK/RU同时不低于E0和REPLAY。sourceRU仅诊断，源失败不取消已登记Q，不换主/回流/选择性重跑；完整目标不缩小。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_density_boundary_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-density-boundary-s2026092701-r01-release/runtime.json`。N607环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-density-boundary-s2026092701-r01-release`；唯一owner `codex-phase3-density-boundary-20261031`，不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-density-boundary-s2026092701-r01`、logs同root/logs。一个GPU train/eval worker实际free≥4096MiB/每卡新增本任务≤2，两个训练臂按保存配对trace顺序完成；独立CUDAempty CPU scorer，不停止/修改/重启其他owner。实际GPU/PID启动后核实。

23unique tests+34sub PASS、10AST/11文件镜像、一次直接P0/P1审查PASS；首步actual三CPsynthetic smoke＋source/BCE no-step梯度probe，PASS立即继续。单preflight/archive SHA比较/compile及PID-CWD-argv-GPU-log核实。非法角色/祖先、非有限、输出碰撞或不能合法prediction时保留，不partial评分/自动重启；UNKNOWN核原PID/root，科学负结果不停止健康任务。

预期launch/pipeline/configs/logs；train fixed density/smoke/complete与两arm checkpoint+全steps；predict共享state/四新sourceVtau/两旧tau、sourceknown/novel/两个Q完整pred与smoke/complete；独立score results/complete，observations及完整训练/结果分析。private teacher行cache、IQ、权重、原日志和预测不进Git。

## 最终结果（DENSITY-BOUNDARY，NO_PROMOTION）

- 两臂各2220步/三DURA条件六方法全部COMPLETE，唯一BOUNDARY_predicted不晋级；保留E0_predicted重点候选和E0_pooled冻结对照，全目标ACTIVE。
- 主source A_K/C_K/R_U/H=89.611111/96.696296/62.761905/73.820998%；clean74.652778/91.666667/97.222222/84.455668%；star62.673611/84.027778/78.125000/69.551480%。
- 星地主vsREPLAY正确13增16失/net−3、未知49增8失/net+41；ΔA_K−.520833pp/ΔR_U+14.236111pp/ΔH+6.011710pp。对E0正确−8/未知+3，H−.420845pp，不能以匹配拒识收益晋级。
- clean虽CK/RU单项过，原F0closed80.034722%到AK仍损5.381944pp；star原72.916667%损10.243056pp且CK/RU失败。F0默认身份逐样本不变。
- 全4440步连续/有限/同L192OE64索引，实际unique各L6300/OE6134；module梯度仅首2，global norm/Sinc/clip全程，>clip5步674/863。两目标不同，不按total loss排序。
- 固定r03/GMM/pointteacher/F0/G与studentbuffers保持；新两NN4440更新，特征15540/MLP4440，GMM/normalizer0fit，四新6τ仅V；源AUROC主较E0−5.282118pp。
- 主部署834665NN参数；三open+F0/G架构活跃合计1504107，非实际驻留测量。共享density364340 B/全36τ288 B，无teacherbank部署。
- student1136640+parent12434 signal；F0head-only12434/backbone0/Gtrain0；evalsignal63756、F0/G1728。合成trainparent/F/G各2+student16，evalopen6/F/G各2，祖先成本保留。
- train/predict/score/pipeline462.670735/26.080005/16.561103/505.483873 s。独立仅driver/train GPU4 758MiB离散样本与log增长；predictor/CPUlive未捕获，非peak/B1，终态空/foreign0。
- sourceknown12过/42未过；high未知0/4/2缺19-15，R_U/H=N/A。源seenRX/历史NVQ/单seedRX，U/formal0、无DA/发现REG，AN/HOW N/A。
- 23unique+34sub和一次P0/P1 review PASS，627files/data0单SHA比较/compile/runtime VERIFIED。详见[公报](../../../docs/PAMF_PHASE3_DENSITY_BOUNDARY_RESULTS_20261031.md)；仅既有产物分析，无新fit/score/阈值扫描。

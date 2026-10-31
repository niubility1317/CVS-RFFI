# PAMF 选择性已知密度无标签适应

PLANNED，无新性能。Run `20261031-phase3-selective-uda-s2026092701-r01`，actualcode `0f1fda4c8c543394a938bd75dcb5cadbc29c6491`，完整Phase3目标ACTIVE。唯一主selective_predicted_ratio，原pooled/predicted、上一run SHARED-DIAG两列、新SELECTIVE两列，共六列×三DURA；方法与固定gate见`docs/PAMF_PHASE3_SELECTIVE_UDA_PLAN_20261031.md`，逐row/六类seed/路径只在experiment.json维护。

原F0 TriSIGTIME SSL E50/11100/default6、r03/dc01a402、DURA R69 E20/4940、classdensity4af known90/OE32/normalizer/two6tau冻结。pU=.5pKnown(a*w+b)prod(a)+.5pOE(w)，每U/Q同时评估两分支，无trueknownunknown路由；Mstep只knownsoftresponsibility/Neff，正a/no postunit、newidentitystart20MAPEM/lambda160。Jac在known分量和最终ratio保留不能取消；固定OE是待检验近似，不保证RU保护。不是旧known-only全U拟合，也非λ/迭代扫描。

旧73f DIAG320/20步/λ160只matchedcontrol，不作新初始化或老师；原r03适应前U4公开字段384沿73f继承d72，F0logits/真实TX/K/配额不fit，旧Q/score不入newfit。actual3CPsynthetic2首步PASS→source与control聚合/cache身份→sourcefreeze→U4fields→newSELECTIVE20/freeze→sourceV/NV六列原两列重复、future两Q前4 exact旧[0,1,4,5]/新two重算→全18pred→独立CUDAempty CPU sourceNV/clean/star三truth，Utruth0。新tau/GMM/NNfit0，原sourcecal exact，12uniqueτ96B重复六列288B/单主48B，targetstate非sourceonly。

source contract392005/VALIDATED_ONCE：L6300/Us56700/OE6134祖先本轮0，V27000额外known标签只原cal继承/源诊断，NV3150三个unknown19-15/19-16/7-1各1050；旧六14-10/14-7/20-15/20-19/6-15/8-20，sourceRX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19、2021-03-08/15/23。U RX24-1/index26/03-08、futureQ同RX03-15同864物理old576/nonold288各类96，U/source/Q互斥；三测试均DURA，clean仅测试。G10 MIX01600km/25MHz/2.45GHz residual/post_sync/noEQ/refSNR30dB/reflect、原六景/郊区城区各0.5、receiver2027/scene2026103001/channel2026104001，一record一received不重抽。

gate sourceknown12、cleanRU95/starRU90、CK90、sameDURA原F0闭集损2pp、主景coverage、主starAKstrictgain/RUnotbelow原predicted；H无独立门槛，胜过弱SHARED或U目标提高不替代绝对目标。源失败不取消Q，不按Q调λ/mix/迭代/换主/选择性重跑。历史NV/Q曝、单seed/RX/source日重合、high缺19-15宏RU/H NA、正式RX24-18未读保持；无发现/匿名REG，AN/HOW N/A，全goalACTIVE。

newtarget统计20EM/320参数，U384每步全量曝光另报；旧SHARED20及上轮两arm40研究成本继承不重复新fit。newNN/Gaussian/normalizer/tau/realF0Gsignal/sourceIQ0，actual3CPsynthetic2each另计，原NN834665/density364340B/source训练/cache成本继承，无source/U行bank部署。adapter真实字节实测，过程时间非B1。

N607解释器`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD`/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-selective-uda-s2026092701-r01-release`，唯一owner`codex-phase3-selective-uda-20261031`，不可覆盖root`/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-selective-uda-s2026092701-r01`。命令`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_selective_uda_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-selective-uda-s2026092701-r01-release/runtime.json`；singleproducer GPU显存≥4096MiB/每卡本任务新增≤2，独立CPU，其他owner健康工作保持。

9unique+29sub通过，7+22sub8.17s完整log/696旧trapzwarnings、rootlauncher2+7sub.032s console，AST7/镜像8、唯一直接P0P1 PASS。一次preflight/archiveSHA/compile及postlaunch PID-CWD-argv-GPU-log读回。非法角色/祖先/非有限/EM退步/输出碰撞/无法合法pred保留，无partialtruth/覆盖/自动重发；UNKNOWN核原PID/root，科学低分不停健康任务，缺遥测不补实验重跑。IQ/weights/pred/rawlogs/private runtime不进Git。

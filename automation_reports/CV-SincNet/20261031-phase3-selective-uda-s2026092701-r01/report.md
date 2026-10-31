# PAMF 选择性已知密度无标签适应

PLANNED，无新性能。Run `20261031-phase3-selective-uda-s2026092701-r01`，actualcode `0f1fda4c8c543394a938bd75dcb5cadbc29c6491`，完整Phase3目标ACTIVE。唯一主selective_predicted_ratio，原pooled/predicted、上一run SHARED-DIAG两列、新SELECTIVE两列，共六列×三DURA；方法与固定gate见`docs/PAMF_PHASE3_SELECTIVE_UDA_PLAN_20261031.md`，逐row/六类seed/路径只在experiment.json维护。

原F0 TriSIGTIME SSL E50/11100/default6、r03/dc01a402、DURA R69 E20/4940、classdensity4af known90/OE32/normalizer/two6tau冻结。pU=.5pKnown(a*w+b)prod(a)+.5pOE(w)，每U/Q同时评估两分支，无trueknownunknown路由；Mstep只knownsoftresponsibility/Neff，正a/no postunit、newidentitystart20MAPEM/lambda160。Jac在known分量和最终ratio保留不能取消；固定OE是待检验近似，不保证RU保护。不是旧known-only全U拟合，也非λ/迭代扫描。

旧73f DIAG320/20步/λ160只matchedcontrol，不作新初始化或老师；原r03适应前U4公开字段384沿73f继承d72，F0logits/真实TX/K/配额不fit，旧Q/score不入newfit。actual3CPsynthetic2首步PASS→source与control聚合/cache身份→sourcefreeze→U4fields→newSELECTIVE20/freeze→sourceV/NV六列原两列重复、future两Q前4 exact旧[0,1,4,5]/新two重算→全18pred→独立CUDAempty CPU sourceNV/clean/star三truth，Utruth0。新tau/GMM/NNfit0，原sourcecal exact，12uniqueτ96B重复六列288B/单主48B，targetstate非sourceonly。

source contract392005/VALIDATED_ONCE：L6300/Us56700/OE6134祖先本轮0，V27000额外known标签只原cal继承/源诊断，NV3150三个unknown19-15/19-16/7-1各1050；旧六14-10/14-7/20-15/20-19/6-15/8-20，sourceRX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19、2021-03-08/15/23。U RX24-1/index26/03-08、futureQ同RX03-15同864物理old576/nonold288各类96，U/source/Q互斥；三测试均DURA，clean仅测试。G10 MIX01600km/25MHz/2.45GHz residual/post_sync/noEQ/refSNR30dB/reflect、原六景/郊区城区各0.5、receiver2027/scene2026103001/channel2026104001，一record一received不重抽。

gate sourceknown12、cleanRU95/starRU90、CK90、sameDURA原F0闭集损2pp、主景coverage、主starAKstrictgain/RUnotbelow原predicted；H无独立门槛，胜过弱SHARED或U目标提高不替代绝对目标。源失败不取消Q，不按Q调λ/mix/迭代/换主/选择性重跑。历史NV/Q曝、单seed/RX/source日重合、high缺19-15宏RU/H NA、正式RX24-18未读保持；无发现/匿名REG，AN/HOW N/A，全goalACTIVE。

newtarget统计20EM/320参数，U384每步全量曝光另报；旧SHARED20及上轮两arm40研究成本继承不重复新fit。newNN/Gaussian/normalizer/tau/realF0Gsignal/sourceIQ0，actual3CPsynthetic2each另计，原NN834665/density364340B/source训练/cache成本继承，无source/U行bank部署。adapter真实字节实测，过程时间非B1。

N607解释器`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD`/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-selective-uda-s2026092701-r01-release`，唯一owner`codex-phase3-selective-uda-20261031`，不可覆盖root`/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-selective-uda-s2026092701-r01`。命令`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_selective_uda_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-selective-uda-s2026092701-r01-release/runtime.json`；singleproducer GPU显存≥4096MiB/每卡本任务新增≤2，独立CPU，其他owner健康工作保持。

9unique+29sub通过，7+22sub8.17s完整log/696旧trapzwarnings、rootlauncher2+7sub.032s console，AST7/镜像8、唯一直接P0P1 PASS。一次preflight/archiveSHA/compile及postlaunch PID-CWD-argv-GPU-log读回。非法角色/祖先/非有限/EM退步/输出碰撞/无法合法pred保留，无partialtruth/覆盖/自动重发；UNKNOWN核原PID/root，科学低分不停健康任务，缺遥测不补实验重跑。IQ/weights/pred/rawlogs/private runtime不进Git。

## 最终结果（ANALYZED_DEVELOPMENT / NO_PROMOTION）

唯一主selective_predicted_ratio，clean AK74.826389/CK91.666667/RU98.263889/H84.958347%，star66.319444/88.888889/63.194444/64.719243%；source原predicted两列重复，不称源适应收益。
主−原pred clean正确23增0失/接受+25(2错)/未知−5；star正确13增0失/接受+19(6错)/未知1增41失净−40，AK+2.256944pp/RU−13.888889pp/H−5.253081pp。对SHARED星地正确2增4失净−2、未知19增15失净+4，不能以弱控制改善晋级。
同DURA原F0损clean5.208333/star6.597222pp，starCK88.888889/RU63.194444及主景覆盖失败；sourceknown12通过但42未全过，high缺19-15的RU/H N/A。NO_PROMOTION，原predicted重点/pooled对照保留，目标ACTIVE。
新known-only混合每record同时算两分支，OE(w)固定、known affine Jac不消去，Neff是软责任非真known。新21records和继承SHARED21全有限/非下降；新目标181695.052814→183843.675666，post.50756→.65166非真实比例，不从Q/GT推构成。
新20统计updates/320参数/U384×20=7680条次，SHARED旧20包含于上轮两臂40成本而不重计。新NN/GMM/normalizer/τ/realGFsignal/sourceL-OEIQ0，三CPsynthetic各2；Utruth0、部署state带U祖先非source-only。
SELECTIVE2740B/SHARED2716B，NN834665/density364340B继承；原12uniqueτ96B无新V校准、six重复288B/单主48B，无source/U行bank。
predict/score/pipeline=10.401474/5.277220/15.790489s，三生产stdout无warning/error；独立driver/predictorCUDA1/256与406MiB两样本非peak，CPU仅pipeline/产物未捕live，终态空foreign0。
9unique+29sub、一次review、单包677filesdata0/SHA-compile-runtime VERIFIED。完整18点/三pair见[公报](../../../docs/PAMF_PHASE3_SELECTIVE_UDA_RESULTS_20261031.md)，无发现REG，AN/HOW N/A。

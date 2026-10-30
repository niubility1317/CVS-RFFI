# PAMF 无标签正对角仿射域适应

PLANNED，无新性能。Run `20261031-phase3-diagonal-uda-s2026092701-r01`，actualcode `73f460c2e0c89b192e52af2e1b6ef01a576ca686`，完整 Phase3 目标 ACTIVE。唯一主diagonal_predicted_ratio，原pooled/predicted、SHIFTpooled/predicted、DIAGpooled/predicted共六列×三DURA。固定公式、预算和晋级规则见`docs/PAMF_PHASE3_DIAGONAL_UDA_PLAN_20261031.md`，逐row/六类seed/路径在experiment.json唯一维护。

冻结原F0 Tri-SIG-TIME SSL E50/11100/default6、r03/dc01a402、DURA R69 E20/4940、classdensity4af known90/OE32/whitener及原two6tau。只拟合白化空间h=a*w+b（正a/no postunit），SHIFT a1/b160与主DIAG a/b320分别从identity固定20MAPEM，lambda160、known/OE组先验0.5/0.5，带Nsumloga+NlogdetW和identity ridge，不搜索、不Q早停。与旧rank14/class平移/sourcecov积分不同；F0 c不变。

actual三CP synthetic2首步smokePASS后→原source/Gaussian/tau及Ucacheheader核对→sourcefreeze→旧mixture/d72 adaptation/U_features.npz exact四字段原r03适应前starU384→两新20fit/freeze→源V/NV六列repeat原[1,3]，futureclean/star同一frozenstarUstate→全18pred→独立CUDAempty CPU sourceNV/两Q三truth。旧mixadapter/prior/cal/score不读，不用其错误语义的tau；U TX/K/配额/场景真值隐藏，F0 logits不fit。无新阈值校准，original12 unique/96B六列重复288B，source状态不改，目标adapter新U祖先不能称sourceonly。

source契约392005/VALIDATED_ONCE，原L6300/Us56700/OE6134祖先新读0；V27000额外knownlabels只源诊断/原tau继承，NV3150三个unknown19-15/19-16/7-1各1050。旧六14-10/14-7/20-15/20-19/6-15/8-20，sourceRX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19、2021-03-08/15/23。U RX24-1/index26/03-08，Q同RX/03-15同864物理记录old576/unknown288各类96，与U/source互斥。G10 MIX01600km/25MHz/2.45GHz residual/post_sync/noEQ/refSNR30dB/reflect、原六景/郊区城区各0.5、receiver2027/scene2026103001/channel2026104001，一record一received，三测试均DURA，clean仅冻结测试，无新物理/重抽。

sourceknown12、cleanRU95/starRU90、CK90、sameDURA原F0闭集损2pp、必需主景coverage及主starAK严格高于oldpred/RU不低于oldpred为gate；H无独立阈值，source失败不取消Q，U目标提高不替代future。sourceNV/Q历史曝光、单seed/RX/source日期重合、high缺19-15RU/H NA、正式RX24-18未读保持。无发现/匿名REG，AN/HOW N/A；不按Q调lambda/mix/迭代、换主或选择性重跑。

newNN/Gaussian/normalizer/tau fit和真实F0Gsignal/sourceIQ0；新增U统计拟合40EM（每arm20）、SHIFT160/DIAG320有效参数，U384载入一次、每arm20全量曝光另报。原NN834665/density364340B及训练/cache成本继承，actualCP synthetic2each另计；adapter真实字节实测，无source/U bank部署，完整过程不是B1。

N607解释器`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、CWD`/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-diagonal-uda-s2026092701-r01-release`、owner`codex-phase3-diagonal-uda-20261031`，不可覆盖root`/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-diagonal-uda-s2026092701-r01`。命令`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_diagonal_uda_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-diagonal-uda-s2026092701-r01-release/runtime.json`。singleproducer显存≥4096MiB/每卡本任务新增≤2，CPU评分CUDAempty，保留其他owner健康工作。

9unique+30sub union通过，两个旧fixture各只定向修复/重跑失败项，生产未改、首轮失败保留；AST7/镜像8，唯一直接P0P1 PASS。一次preflight/archiveSHA/compile/启动PID-CWD-argv-GPU-log核实；非法角色/祖先/非有限/EM退步/输出碰撞/无法合法pred保留，无partialtruth/覆盖/自动重发，UNKNOWN先核原PID/root。源科学低分不停健康任务，缺遥测不补实验重跑。IQ/weights/pred/rawlogs/private runtime不进Git。

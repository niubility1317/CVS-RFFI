# 仅适应已知密度的开放集无标签适应

Run `20261031-phase3-selective-uda-s2026092701-r01`，单模型seed2026092701，唯一owner `codex-phase3-selective-uda-20261031`。实际代码、逐行六类seed、命令/解释器/CWD、路径、资源与产物在本run experiment.json维护。原predicted_ratio重点候选和pooled_ratio对照保持，完整Phase3目标ACTIVE。

## 独立假设与固定算法

上一轮 shared DIAG 同时变换known/OE，已知正确接收增加但未知拒识下降。本次仅适应known90，OE32在原白化坐标w上固定；与旧mixture DA_K将全部U交给known拟合不同，本次完整混合的软成员权重仍含OE。冻结原F0 Tri-SIG-TIME SSL E50/11100/default6、r03/dc01a402、DURA v1/R69 E20/4940、class-density/4af known90/OE32/normalizer和两组六τ。F0默认身份c保持，不改神经网络、Gaussian参数或阈值。

拟合模型为 p_U(w)=0.5 p_K(a⊙w+b)prod(a)+0.5 p_OE(w)，a>0，变换后不重新单位化。每条U/Q都同时计算两个分支，不按真实known/unknown路由输入。新SELECTIVE从a=1/b=0独立做20次MAP-EM，λ=160固定单位变换先验；不从旧DIAG初始化，不搜索预算/权重/λ。观察目标为Σlog p_U(w)+Nlog|W|−λ/2Σ[(a−1)²+b²]。OE对新域不变是需检验的统计近似，不是已验证的物理规律，也不保证未知拒识率不降。

E步known的90个组件各包含sum(log a)，OE的32个组件不含该项。M步只用known软责任r_ik：p_ik=r_ik/v_k、t_i=Σp、M_id=Σp μ，N_eff=Σr_known；其余s0/sx/sxx/m0/mx/D/H/g沿上一轮解析式，但正根为a=(g+sqrt(g²+4H N_eff))/(2H)，负g用等价稳定式；b=(m0−a*sx)/D。零known责任自然得到先验单位变换，不加伪样本。N_eff是软责任质量，不是真实known样本数或已知/未知比例。

主分数为ℓ_c(a⊙w+b)+Σlog a−ℓ_OE(w)，pooled诊断替换分子为known90。由于只有known变换，affine Jacobian不能在ratio中相消。source原白化Jacobian仍为两项共有量。保存iteration0至20全21条有限、非下降目标及尺度/偏移/软质量，不按Q早停或选择迭代。

六列固定为pooled_ratio、predicted_ratio、shared_pooled_ratio、shared_predicted_ratio、selective_pooled_ratio、selective_predicted_ratio；唯一主最后。SHARED精确复用上一run73f460c2e的DIAG适配器/20步记录和对应两列预测，仅作同参数320/20步/λ160对照，不作为新source权重、老师或初始化。source旧RX保持单位变换，六列原两列重复；新RX clean/star两Q共用同一个star-U适配器。不按真实TX路由。

## 数据权限与顺序

source契约392005/VALIDATED_ONCE原L6300/U_s56700/OE6134均为祖先、本轮新读0；V27000额外known标签仅复用原校准及源诊断，NV3150为19-15/19-16/7-1各1050、独立评分。旧六14-10/14-7/20-15/20-19/6-15/8-20；source RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。原L/(L+U_s)10%不包含额外V/OE监督。

U384为RX24-1/index26、2021-03-08，来自上一diagonal run的adaptation/U_features.npz，仍是原source-only r03/DURA适应前四公开字段ids/default_logits/closed/signal_features；验证其原d72缓存祖先未变，不读取旧Q/score来拟合。U真实TX/K/配额/场景真值隐藏，F0 logits不参与拟合。U capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`、split `83c5cc1fe2f2a1512dbc1287`保持。

future clean/star Q为同RX、2021-03-15的同864条物理记录，旧六各96=576、未知三各96=288，与U/source互斥。三类测试均接冻结DURA，clean仅冻结测试、不学习。原G10 MIX01/600km/25MHz/2.45GHz residual/post_sync/noEQ/refSNR30dB/reflect、六景原权重/郊区城区各0.5，receiver2027/scene2026103001/channel2026104001，单记录单received，不重抽或新增物理数据。

actual F0/r03/G各synthetic2首步smoke PASS立即继续→核原source与共享控制聚合、U缓存身份→原状态和τ冻结→U公开四字段一次→新SELECTIVE20步从单位变换拟合/冻结→源V/NV及两Q缓存六列预测（旧shared两列逐值复用）→完整18运行点固定→独立CUDA-empty CPU三truth评分。U真值读取0；旧SHARED只作匹配控制，自己的Q/score不进入新拟合。

## 验收、成本与执行

主报逐真TX宏AK/RU/H，附CK、同DURA原F0闭集损、AUROC/OSCR、逐RX/day/scene/TX与配对增失。gate为source known12、cleanRU≥95%/starRU≥90%、CK≥90%、原F0闭集−AK≤2pp、所有主景所需roster完整，以及主star AK严格高于原predicted/RU不低于原predicted。另报对SHARED匹配控制及selective pooled的配对，不能只因胜过较弱控制而晋级。H无独立门槛。

sourceNV/devQ历史曝光、单seed/RX/source日期重合保持；high缺19-15，宏RU/H为N/A，不删景或补样认证。正式RX24-18不读；本轮无发现/匿名注册，AN/HOW为N/A。即使局部适应有益，也不替代完整Phase3目标；不据Q调λ/迭代/权重、换主或选择性重跑。

newNN/Gaussian/normalizer/tau fit和真实F0/G/signal前向0，actualCP synthetic2each另计；新target统计拟合20步/320有效参数，U384每步全量统计曝光另报。复用SHARED本身20步、上轮完整两arm40步及其原source训练/cache成本均保留，新增不重计。原NN834665/density364340B不变；12唯一τ96B原样重复存36值288B、单主48B。adapter实际数组字节另报；source/U逐行bank不部署，进程时间非B1延迟。

N607单producer，显存≥4096MiB/每GPU本任务新增≤2，独立CPU评分，不停止/修改/重启其他owner。一次聚焦协议负测/实际无query smoke、一次直接P0/P1、一次preflight/archive SHA/compile及PID-CWD-argv-GPU-log读回。非法角色/祖先/非有限/EM退步/输出碰撞/无法合法pred保留；UNKNOWN先核原PID/root，不覆盖或自动重发。科学低分不停健康任务，缺遥测不补实验重跑。

区分未知样本与可适应样本的动机可参考[STA原论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.html)。本次是独立选择性Gaussian MAP-EM，不是该论文的网络或训练算法，不继承其性能结论。

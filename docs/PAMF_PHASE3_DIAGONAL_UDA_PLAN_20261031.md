# 无标签星地特征的正对角仿射适应

Run `20261031-phase3-diagonal-uda-s2026092701-r01`；单模型 seed2026092701，唯一 owner `codex-phase3-diagonal-uda-20261031`。实际代码、命令、解释器/CWD、逐行六类 seed、路径、资源与预期产物在本 run experiment.json 维护。完整 Phase3 目标保持 ACTIVE，原 predicted_ratio 重点候选及 pooled_ratio 对照不改。

## 固定问题与方法

旧 mixture-UDA 只学 rank14 公共平移，class-shift 学类条件平移，nuisance-marginal 是源协方差积分；本次独立检验白化空间的全局逐维尺度与平移。冻结原 F0 Tri-SIG-TIME SSL E50/11100/default6、r03/dc01a402、DURA v1/R69 E20/4940、class-density/4af known90/OE32、normalizer 和 pooled/predicted 两组六τ。F0 默认类别 c 不变，不改身份头、神经模型、Gaussian 或阈值。

输入先由原 source whitener 得到 w；适配变换为 h=a⊙w+b，a严格正，之后不重新单位化。源混合 p_s=0.5known90+0.5OE32，权重不读取目标类构成。两独立 arm 从 a=1、b=0 开始：SHIFT 固定 a，只拟合 b160；DIAG 拟合 a/b320。唯一主为 diagonal_predicted_ratio；六列固定 pooled_ratio、predicted_ratio、shift_pooled_ratio、shift_predicted_ratio、diagonal_pooled_ratio、diagonal_predicted_ratio。

每个 arm 固定20次精确 MAP-EM，λ=160 在访问U前固定，不搜索。拟合目标为 Σ_i log p_s(a⊙w_i+b)+NΣ_d log a_d+N log|W|−λ/2Σ_d[(a_d−1)²+b_d²]。Jacobian 必须进入 U 似然；它在 known/OE log-ratio 中抵消。全局类无关仿射只是开放特征的近似域变模型，不宣称物理信道可逆恢复或能自动剥离未知身份。

M步令 p_ik=r_ik/v_k、t_i=Σ_k p_ik、M_id=Σ_k p_ik μ_kd，s0=Σt、sx=Σtw、sxx=Σtw²、m0=ΣM、mx=ΣwM、D=s0+λ、H=sxx+λ−sx²/D、g=mx+λ−sx*m0/D。DIAG解 a=(g+sqrt(g²+4HN))/(2H)，g<0用等价式2N/(sqrt−g)；b=(m0−a*sx)/D。SHIFT令a=1。保存 iteration0至20完整有限目标及尺度/偏移诊断，检查非下降，不用 Q 早停或选择迭代。λ是固定身份锚定先验强度，不声称由源数据调优。

六列只报两个 ratio：pooled known/OE、预测类15等权模态/OE。源接收机不使用新目标接收机适配器，源六列原两列重复；新 RX 的 clean/star 两Q共享同一 star-U 冻结适配器。该条件路由不读取真实 TX。原两组六τ精确复用并重复到六列，不新校准；新 target状态明确有 U 祖先，整个系统不是 source-only。

## 输入与执行顺序

source contract392005/VALIDATED_ONCE 原 L6300/U_s56700/OE6134 为祖先、本轮新读0；额外已知 V27000（旧六类各4500）仅复用原阈值和冻结预测。source NV3150 为19-15/19-16/7-1各1050，仅独立评分。旧六14-10/14-7/20-15/20-19/6-15/8-20；source RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19、2021-03-08/15/23。原L/(L+U_s)10%不包含额外V/OE监督。

只复用原 mixture-UDA/d72a7b7 的 adaptation/U_features.npz：ids/default_logits/closed/signal_features 四字段，特征由适应前 source-only r03＋原 F0＋DURA 产生；不加载旧 prior/adapter/拟合结果/阈值或评分。U384 为 RX24-1/index26、2021-03-08，公共 capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`、split `83c5cc1fe2f2a1512dbc1287`。原 F0 logits 不参与拟合，目标 TX/K/配额/场景真值隐藏。

future clean/star Q 是同RX、2021-03-15的同864条物理记录：旧六各96=576、未知三类各96=288，与U/source互斥。三类测试全部接原冻结 DURA；clean只冻结测试，不参与学习。G10 MIX01/600km/25MHz/2.45GHz residual/post_sync/noEQ/refSNR30dB/reflect、原六景权重/郊区城区各0.5，receiver2027/scene2026103001/channel2026104001，一条记录一份固定received，不重抽、不加独立物理样本。

顺序：actual F0/r03/G各synthetic2首步smoke PASS立即继续→原source状态/τ及U缓存祖先核对→source冻结→U公开四字段一次→两新20步适配器冻结→源V/NV与两个Q六列逐样本预测→完整18运行点固定→独立 CUDA-empty CPU三truth评分。U真值不读，source预测/Q不进入拟合；源/目标Gaussian、神经权重及所有冻结状态保持。输出不可覆盖，无部分预测评分。

## 科学判定与成本

逐真TX宏 AK/RU/H主报，附CK/同DURA原F0闭集损/AUROC/OSCR、逐RX/day/scene/TX及同ID配对。门槛为source known12、cleanRU95/starRU90、CK90、原F0闭集−AK≤2pp、每个主景完整roster，并且主star AK严格高于原predicted_ratio、RU不低于它。与shift_predicted配对隔离尺度作用；不以U目标上升、局部CK提高或较弱控制改善替代综合目标。H无独立门槛。

历史sourceNV/devQ曝光、单seed/RX/source日期重合保持，不称新盲测；high缺19-15，RU/H为N/A，不删景或补样认证。正式RX24-18不读。无发现/匿名注册，AN/HOW为N/A；局部适应成功也不代表Phase3完成，不从Q改λ、mix权重、迭代、主方法或重跑。

newNN/Gaussian/normalizer/阈值fit及真实F0/G/signal前向0，actualCP synthetic2each另计；新U统计拟合两组各20步，SHIFT160/DIAG320有效参数，U384复用且全量迭代曝光另报。NN834665、原density364340B与旧训练/cache成本继承；两组六τ12个唯一值96B，六列存储重复为36值288B，单主48B。source/U逐行bank不部署，新增adapter数组真实字节另报；进程时间非B1延迟。

N607单producer，启动时显存≥4096MiB、每GPU新增本任务≤2，独立CPU评分；不停止/修改/重启其他owner。一次聚焦负测和实际无query smoke、一次直接P0/P1、一次preflight/archive SHA/compile、PID/CWD/argv/GPU/log读回。非法角色/祖先、非有限/EM退步、输出冲突、无法合法prediction则保留；UNKNOWN先核原PID/root，不覆盖或自动重发。低科学性能不停止健康任务，缺遥测不补实验重跑。

统计尺度适应的背景动机可参照[CORAL原论文](https://arxiv.org/abs/1511.05547)；本次是带OE混合和Jacobian的独立对角MAP-EM，不是该论文原算法，也不继承其结果。

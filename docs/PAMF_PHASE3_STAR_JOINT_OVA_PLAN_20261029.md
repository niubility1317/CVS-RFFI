# G10 开放损失联合表征与特征保真

Run `20261029-phase3-star-joint-ova-s2026092701-r01`，唯一主 `joint::ova_max`。固定两组 `detach_control/joint`，均从原 F0 E50/11100 独立初始化并更新2220步；不是继承此前按novel成绩选择的Phase3增量。方法、权重和门槛在本轮任何新IQ访问前固定。

上一冻结特征上的类条件OVA未达到未知拒识要求。本轮检验让开放边界损失参与特征学习是否有益，并用原F0逐记录单位特征约束保护未注册设备的几何结构。该约束不能保证设备纯度或未来识别；结构是否保持由源端保留设备的独立诊断检查，不靠训练loss宣布成功。

## 两组的唯一差别

两组均保留完整三视图网络232993参数和原默认六类身份头，并联合更新该头。增加独立的160→6线性辅助层（966参数、含bias），六个binary旧类存在性logit只用于拒识，不替换原分类头。两组辅助层都更新；control仅将送入OVA的学生单位特征detach，joint允许OVA梯度反馈至F。CE、KD与特征保真在两组中均保留学生梯度。

每步用L128条uniform IID与OE128条按15RX/day域均匀采样的cachedP1，concat256做一次完整F前向。L seed2026092701、OE seed2026102907，两组相同。无随机多视图/额外星地重抽/As/PL/EMA/VICReg，也不对OE的默认身份logits做uniform目标。

总损失为：CE_L＋KD_L(T2 KL×4)＋5×[.5 cosinefeatureloss_L＋.5 cosinefeatureloss_OE]＋OVA。所有teacher均detach。OVA=.5×[.5真L类positive softplus(−g_y)＋.5最难错误旧类softplus(g)]＋.5 OE六类最大softplus(g)。OE不提供TX或真实K。借用源端难负类思想，不执行原版OVANet目标OEM，不声称论文原生复现。

teacher单位rawfused160来自冻结原F0对同一P1记录的前向，只导出L6300/OE6134；L full6 teacher logits复用已有同输入cache。只由source L/OE更新F/aux，V和novel不进入NN训练或teacher特征拟合。P1中的G固定为DURA v1.0/R69，训练本轮不加载或更新G。

F使用原native可微Sinc图及CENTER_UNITS physical AdamW，reset optimizer、lr1e-4、parent weightdecay、physical clip5；aux同seed2026092701 normalstd.01/bias0，独立AdamW lr.001/wd.0001、clip5。model eval但梯度开启，BN buffer和dropout冻结；不是将原frozen-reader的no-grad Sinc包装当可训练骨干。每组F2220步/累计13320，aux2220步；每组两个optimizer共4440次step调用，两组共4440F更新＋4440aux更新＝8880optimizer调用。2220是固定IID更新预算，不是十遍自然数据遍历。

## 数据与权限

真实训练、teacher和评估数据仅用既有G10 P1 cache：source L6300/OE6134/V27000/novel3150，总42584条原物理记录；realP0模型消费0、原U_s新增0、raw读取/真实信道重生0。原F0祖先仍是L6300/U56700/V27000、训练L/(L+U)=10%，本轮L真实旧六标签和OEoutside-six成员监督必须披露，不能称全系统零源标签。V的known标签额外用于阈值校准。

source RX为1-19/18-2/19-2/2-19/3-19（1/3/4/6/8），日期2021-03-08/15/23。source novel为19-15/19-16/7-1各1050，与L/OE/V物理互斥但RX已见、设备已有历史开发曝光；不是目标跨RX确认或新TX盲测。开发target U/Q、正式RX24-18不在输入schema，没有任何新目标句柄或query回收。

G10沿既有MIX01：600km/25MHz/2.45GHz、receiver2027、scene/channel2026103001/2026104001、六分量面积概率、residual/post_sync/noEQ，一次输入RMS、输出不重归一化；同一原记录的received不重抽，P1为其固定DURA输出。合成无query探针是单独的G10 P0输入：teacher exporter2条、每train native梯度smoke2条、每新candidate eval2条，共10条F前向；它们不计物理样本、不充作真实P0对照，也不冒称新G/P1管线smoke。全部新IQ均为星地，没有clean逐条参考。

## 逐行冻结和评估

teacher导出只L/OE originalF0单位特征及opaque IDs，完整后两组可在不同GPU并行；每组固定2220终点即冻结F/aux/两个optimizer状态、保存完整步日志，立即进入自身评估和独立评分，不等待另一组、不用已返回结果改变另一组健康训练。

每组仅三score：candidate默认头energy、aux在candidate原身份预测类的routed、aux最大presence。身份始终由candidate原默认六类头给出，不采用aux获胜类。V27000只校准每score六个predicted-candidate-class阈值，共18/组；42组true-TX宏known风险（overall/5RX/6scene/30RXscene）同时满足C_K≥90%、原F0同P1 closed−A_K≤2pp。原F0 baseline用Vcache[:,P1]，不能重设成新candidate闭集；candidate闭集也单报。若候选闭集本身已损>2pp，零拒识不可行，有限acceptall回退但known失败，仍完成预登记novel评测，不重训或取消。

全部F/aux及18τ固定后才打开novel。保存默认logits、auxlogits和实际margin，独立scorer可重建energy/routed/max与原candidate类路由，不按trueTX/scene路由部署。source novel额外保存新candidate及原F0同P1单位fused160；原F0仅作冻结参考预测，每组额外3150前向，不将novel加入teacher训练cache。

所有该row预测完成后，独立CPU scorer连接source private TX，主报逐TX宏A_K/R_U/H-score，C_K、原F0损失、candidate闭集、宏AUROC/OSCR、逐RX/scene/TX、置信区间和计数。结构诊断为source novel余下RX中的cosine1NN（排除相同RX，opaque ID稳定解ties），报告candidate与原F0配对差。该诊断用truth只在scorer，不产生部署原型或训练标签，不是A_N、聚类回测注册或目标域无标签适应。

## 科学判定与发布

只有主 `joint::ova_max` 的known42保护、unknown总体＋5RX＋6scene宏拒识≥90%且必要覆盖完整，并且source novel跨RX1NN较原F0最多下降2pp，才取得未来另行预登记小目标实验资格；结构guard是本候选的源端保护代理，不替代最终匿名A_N≥80%。controls/energy/routed不能替主晋级。任一失败NO_PROMOTION，不调参、换主或选择性重跑；A_N/H_OW=N/A，完整跨RX星地拒识、零标签适应及未来匿名注册目标仍ACTIVE。当前用户v3要求保持，不把source通过称目标成功。

预期F前向：teacher12434实物＋2合成；两组真实训练1136640＋4合成；两组candidate eval60300＋4合成；原F0 novel参考6300，总1215684行，真实G新前向0。重复IID曝光不是更多独立数据；原F0与历史G训练/缓存成本保留。主部署NN参数F232993＋aux966＋G266951＝500910，训练/诊断额外teacher及控制模型另计，不预先宣称时延改善。

唯一launch owner当前root。N607 teacher单GPU，随后至多2组各占不同实际显存≥4096MB卡，每卡最多新增2本任务进程；不停止/修改/重启其他owner。一步code/config commit、聚焦协议负测、唯一独立P0/P1审查、最小prereg、一次preflight/archiveSHA比较/compile与实际checkpoint first-step smoke，启动后独立PID/CWD/argv/GPU/log增长，truth-last同row分析。具体命令、环境、CWD、输入输出及artifact由本run登记固定。teacher技术失败不启动rows，failedrow保留partial、独立健康row继续；科学失败不会停止、重启或热修改。方法身份以Git和匹配契约为准，不加封印/receipt链或单独审批。

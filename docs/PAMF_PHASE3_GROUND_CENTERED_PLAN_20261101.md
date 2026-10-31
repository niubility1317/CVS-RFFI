# 地面默认头的中心化证据训练

Run `20261101-phase3-ground-centered-s2026092701-r01`，owner `codex-phase3-ground-centered-20261101`。本轮继续用户指定的地面训练优先；完整跨RX星地开放识别、无标签适应、发现及未来匿名注册目标保持，正式RU≥95%、CK≥95%、原同DURA F0闭集−AK≤2pp及主场景95%不降低。原方法、数据与其他owner健康任务保持。

## 假设与固定方法

上一轮源Us子空间数值约束成立但拒识失败，不将其作为初始化。本轮从完整原F0独立训练一个原生学生CPAIR，默认六类头与骨干联合训练，Sinc48和全部buffers固定，232993参数/232945可训练/头960。原F0 Tri-SIG-TIME SSL E50/11100和DURA R69v1.0均不更新。没有投影、另训头、EMA、目标U或新数据增强。

原能量a(z)=logsumexp(z)。本轮证据c(z)=logsumexp(z)−mean(z)−log(6)，等于KL(Uniform6 || softmax(z))。对每条记录同时加同一常数，分类、源CE、teacher KL和c均不变，但a变化。该代数说明原排序有共同平移自由度；它不是既有训练实际走了捷径的证明，也不保证中心化将改善拒识。

参考[能量OOD原论文](https://arxiv.org/abs/2010.03759)及[OE原论文](https://arxiv.org/abs/1812.04606)的能量和均匀分布背景；本轮是自己的公共域内中心化配对排序变体，不声称复现任一论文的完整训练或保证。推理c用FP64 logits计算稳定LSE后减均值与log6，训练用TorchFP32同一公式，保持原六logits和预测类别，不改成中心化分类头。

共同CE为L真实六类交叉熵，KD=4*KL(softmax(original same G(clean) cached logits/2)||softmax(student logits/2))，两者权重1。每个公共RX/day域d，计算全部L/OE配对softplus(c(OE)−c(L))的均值，然后对15域等权；权重1。CPAIR只改变原DMEAN损失中排序使用的标量，不增加OE真实TX、K或未知语义。每step L192/OE64，同SourceSampler种子2026103017/18，逐步比对旧DMEAN全部2220步L/OE索引；同完整原F0初始化和2220更新、AdamW lr/wd1e−4/eps1e−8/clip5，最终步冻结，不选checkpoint。

旧DMEAN为`20261101-phase3-ground-tail-s2026092701-r01` / code `315f8ec1be994c40a97cbede4e27942e14267f12`的匹配控制，原D7 ground r03数据/教师不变。只用旧DMEAN的源训练祖先，不继承其权重进新学生。旧DMEAN真实模型仅在CAL新增前向，所有query列使用其既有完整cache；原能量列与阈值精确复用。

固定四方法：`DMEAN_energy`、`DMEAN_centered`、`CPAIR_energy`、`CPAIR_centered`；主方法预定`CPAIR_centered`。0/1同旧DMEAN身份，2/3同新CPAIR身份。相同模型内差异隔离评分规则，同读出跨模型差异隔离训练目标。保存canonical原F0 baseline_logits/closed；旧tail的col0是SSL，不得拿它充当原F0。原F0基准由旧tail的独立baseline_logits继承。

## 数据、校准与评分

复用VALIDATED_ONCE `20261101-phase3-ground-open-s2026092701-r03-source-clean-v1` / `sourceV-half-y-rx-day-s2026110101`，不新增导出或重验链。新训练只读合法G(clean) L6300及OE6134，L可读六类y与公共RX/day，OE只允许outside-six成员监督及公共RX/day，不读OE TX。源Us新增读取0，原F0/旧控制的Us祖先不清零，不称全系统无标签。

六已知TX14-10/14-7/20-15/20-19/6-15/8-20；源RX index1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23，split392005/eq1/256点。原历史V27000每y×RX×day格300按固定seed2026110101分CAL150/TEST150，共CAL13500/TEST13500，物理互斥。CAL只在模型冻结后保护接收率和原F0身份；TEST标签只在独立CPU评分侧。

源Q16650=knownTEST13500+NV3150，未知19-15/19-16/7-1各1050，public角色隐藏。跨RX仍为开发RX3-2/index34未来03-15/23 Q10238（5525/4713），known4508六类1491/1182/150/406/1186/93，未注册5730四类5-5/10-7/2-19/7-11=1454/1439/1427/1410。03-23缺20-19/8-20，完整已知指标N/A，不能present-only晋级。未注册不等于从未在OE或历史出现，不新增OE真TX去证明互斥。

四条件source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA全部固定DURA v1.0、FP32/TF32off/reuse+graph/nativeoff。源G(clean)/G10P1复用，跨RX两条件逐记录经同G。G10MIX01、600km/25MHz/2.45GHz、post_sync/noEQ/refSNR30dB/reflect、输入RMS一次/输出无RMS、源scene3001/channel4001/future3133/3134/receiver2027保持，不重抽观测、不用clean参考恢复star，目标clean只冻结测试。

四方法用相同CAL known24组规则；旧DMEAN_energy六阈值继承，新三方法18值冻结，合计24个FP64/192B，主6/48B。rank/学习率/损失和主方法不由CAL/NV/target选。零尾仍不满足已知保护时记录失败，继续合法预测，不借测试修复。源NV、TEST、目标U/Q均不参与训练或早停。

4方法×4条件共16点完整固定后，独立CPU连接source_knownTEST/sourceNV/crossclean/crossstar四真值各一次，旧DMEAN_energy四个overall与旧JSON完全一致。CPU重建raw/margin/指标，不加载NN、源训练数组或拟合新状态。主方法与三控制的配对净数和宏变化全部报告；对照原能量与中心化证据的两类输出不混列。

formal各条件CK95/RU95/原F0损≤2pp且完整名单；源clean all24组，跨clean overall+clean，源/跨star overall+六主景。`ground_source_success`、`cross_clean_success`、`ground_stage_success`与`frozen_rejection_success`分开，scientific只在四条件联合通过为True。通过也只完成冻结拒识子阶段；完整Phase3 ACTIVE，目标UDA/发现/注册仍需后续独立验证。历史开发数据已曝光，本轮不是新盲测；正式RX24-18不访问，不按Q结果改参数/主方法/扩seed。

## 成本、执行与产物

新训练2220×256=568320学生前向；新学生CAL13500+两源Q33300+两跨Q20476=67276预测；旧DMEAN只新增CAL13500前向。新真实NN前向总649096（新学生635596+旧CAL13500），原F0真实新前向0，G20476；旧源data G99284/F036450、原F0/G/旧DMEAN训练分别继承。合成smoke单列：train student6/F02/G2；predict student2/legacy2/F02/G2，r03/PCD真实及合成0。主部署仍499944神经参数+6阈值48B，无辅助统计模型/源行bank；全stage墙钟不冒充B1时延。

默认N607，启动free≥4096MiB，每卡至多新增2个本任务worker，不停止他人。单owner、单不可覆盖root，train首真实CP无query smoke PASS即继续→固定2220步→predict首smoke→CAL→四Q→CPUscore→同row全日志分析/报告/Git。具体commit、命令、环境/CWD、远端路径、GPU与期望产物写在同run登记。训练trace全步记录CE/KD/centered域风险/原能量域风险、原能量/中心化证据/共同均值、索引与梯度；不拿实际损失下降当测试成功。

停止只限输入/祖先/协议、非有限、输出冲突或无法合法预测；低分不是技术失败。不覆盖历史输出，不因低分重跑。一次独立P0/P1、聚焦负测、一次预检/发布包SHA比较/远端compile、实际PID/CWD/argv/GPU/log增长。无需另加审批或签名链。

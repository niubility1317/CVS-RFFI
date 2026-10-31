# 源侧无标签子空间与冻结拒识

Run `20261101-phase3-ground-geometry-s2026092701-r01`，owner `codex-phase3-ground-geometry-20261101`。上一轮尾部损失配对实验为实测进展，但源RU仅小幅提高且跨RX退化，不能晋级。本轮在完整原三视图F0上学习源侧拒识几何，神经模型保持冻结；完整Phase3的无标签目标适应、发现与未来匿名注册目标不缩减，当前仍优先地面开放能力。

## 固定方法

参考[ViM原论文](https://arxiv.org/abs/2203.10807)及[官方实现](https://github.com/haoqiwang/vim/blob/master/benchmark.py)：将原生特征残差与六类logits能量结合。标准分支沿用低维160空间的80维残差、uncentered二阶矩和源meanMaxLogit/meanResidual定alpha；源域均值约束分支是本项目变体，不声称官方复现或已保证接收机不变性。

唯一神经识别器为完整原F0 Tri-SIG-TIME SSL E50/11100 reader/default6，232993参数，头960，无bias、Python scale30；DURA固定R69 v1.0、266951参数/E20/4940/reuse=True/graph=True/native=False，FP32/TF32off。F0/r03/G全部frozen eval、参数/buffers不变，无优化器、教师更新、EMA、替代头、源OE或目标U学习。新状态是统计拟合，不称新增神经训练或全系统无标签。

必须取当前F0分类头的实际FP32坐标x=F.normalize(raw160,eps1e-4)，并在同一次forward逐项验证logits=30*F.linear(x,F.normalize(W,eps1e-4))。不复用旧FP64 unit/raw特征冒充此坐标，不再归一化。原head bias0，因此ViM原点为精确0，不作全样本均值减法或协方差shrinkage。

只用旧G(clean)源Us56700的IQ和公共RX/day拟合两套状态，不读Us真实TX、不生成伪标签、不读取L/OE值。Us在原Phase1合同中属于六已注册身份，已知成员资格来自原合法角色，并非混合未知池；这不代表整个系统从未用过标签。原F0及旧SSL对照的源Us祖先保留。

standard：M=X'X/N的最低80特征向量作为残差基R。domain_orthogonal：按公共15个RX/day求源均值μd；14行μd−μ0与居中的15均值张成同一空间，用FP64 fullSVD及eps*max(shape)*smax数值rank得Nuisance基B、秩r≤14；在B的正交补C（160−r维）上作uncentered (X C)'(X C)/N，取最低80向量并映回R=C Vlow。两分支残差均固定80维；后者principal维为160−r−80（r14时66），不是再次折半。约束R'R=I80、B'R=0。若域均值无差异r0，等价标准分支；数学残差退化或alpha非正则明确失败，不静默回退。

每分支alpha=mean(max六logits)/mean(||xR||)，分数s=logsumexp(logits)−alpha||xR||，高分接收。B只作拟合/审计，推理只需R、原点和alpha，不使用query统计、目标RX标签、TX、真实K或配额。该约束只消除选定源均值方向对残差的贡献，原logits路径仍可能受域影响，不能据B'R=0宣称整个系统域不变。

固定四读出F0_energy、OLD_SSL_energy、F0_vim、F0_domain_vim；唯一主F0_domain_vim。前两能量控制逐值复用旧ground r03的col0/2和对应六阈值；标准ViM隔离新残差约束的作用，不依据结果改主。两新读出用同一次当前F0原默认头和坐标；保存freshF0 logits与旧canonical F0独立基准，逐条件报告FP32重放最大logit差和argmax差。小数值差不改标成头重训，也不伪称与旧cache逐值相同；原F0基准不能被fresh候选闭集替代。

## 数据与角色

直接复用`20261101-phase3-ground-open-s2026092701-r03` / code `d7e0f875397b2f9801df4605ba0051d221aa4e2a`的VALIDATED_ONCE胶囊与物理ID，不重建。源split392005、eq1、256点、输入RMS一次；六注册TX14-10/14-7/20-15/20-19/6-15/8-20，五源RX index1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23。Us56700只读四公共metadata字段ids/rx_i/day_i/domain，原合同每公共域3780，不读sig_i/base_index或TX标签。

CAL13500和已知TEST13500来自原V27000每y×RX×day格150/150、seed2026110101，物理互斥。旧builder合法用V已知y分层后将TEST真值导出私有侧；本轮拟合/校准/预测不读TEST标签。仅CAL13500已知y用于阈值身份保护，源NV3150（19-15/19-16/7-1各1050）只评分。TEST加NV为源Q16650，clean/原固定G10星地同序，public不含known/unknown角色。历史V/NV/Q已曝光，本轮不是新的盲测。

跨RX仍为开发RX3-2/index34、未来03-15/23 Q10238（5525/4713），六已知4508（1491/1182/150/406/1186/93），四未注册5730（5-5/10-7/2-19/7-11各1454/1439/1427/1410）。03-23缺20-19/8-20，完整已知指标N/A，保留present-only诊断而不替代完整名单。未知表示未注册，不证明未在OE或历史出现；不新增OE TX读取去核对。正式RX24-18未访问。

全部source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA四条件保持固定DURA。源G(clean)/G10 P1复用；目标clean仅冻结测试，星地为原单物理记录单固定practical residual received：G10 MIX01 600km/25MHz/2.45GHz、post_sync/noEQ/refSNR30dB/reflect、源scene3001/channel4001、future3133/3134、receiver2027、输出不再RMS。跨RX两条件同10238物理记录，逐记录经同G，不重抽信道或用clean参考恢复query。

## 冻结与正式验收

神经原F0/G始终冻结；两个统计模型先于CAL冻结。新两方法只用CAL已知24组（overall/5RX/3day/15RXday）固定公共低尾规则，保护CK≥95%、同输入canonical原F0闭集−AK≤2pp；rank/r/alpha不由CAL/NV/target挑选。无缺预测类或零尾仍无法保护时记录失败，合法预测继续，不以测试修复。新12阈值＋旧12=24个FP64/192B，主方法六值48B。只保存聚合basis/alpha、域方向审计和必要预测坐标，不部署source行bank。

固定4读出×4条件共16点全部完成后，独立CPU连接source_knownTEST/sourceNV/futureclean/futurestar四真值各一次；旧F0/SSL四overall共8项必须精确保持。CPU只重建分数/阈值与指标，不重做PCA/SVD、不读Us数组或运行NN。source三未知字符串与目标四标签使用各自名单，宏AK/CK/RU/H按真实TX等权；报告原与fresh闭集、逐RX/day/scene/day×scene/TX、CI/AUC/OSCR和配对净变化。

正式门槛不变：全部条件宏RU≥95%、每预登记主场景≥95%、CK≥95%、同canonical原F0损≤2pp且所需名单完整。源clean全部24组；crossclean overall＋clean；source/cross星地overall＋六主景。ground_source_success/cross_clean_success/ground_stage_success与四条件frozen_rejection_success分别报告；scientific_success仅四条件全部通过且主CAL保护通过。通过仅是冻结拒识子阶段，后续UDA/发现/未来匿名注册仍未完成；否则NO_PROMOTION。旧90仅历史诊断，不作为晋级；完整Phase3 ACTIVE，AN/H_OW=N/A，不按新Q改主、调参或扩seed。

## 执行与成本

默认N607，卡0至7启动free≥4096MiB，每卡至多新增2本任务worker，不停止或修改他人任务。单owner/不可覆盖run，GPU predict进程顺序实际CP无query smoke→源Us拟合→CAL→四Q，随后独立CPU score。原F0真实前向123976=Us56700＋CAL13500＋两源Q33300＋两跨RXQ20476，G新20476、SSL新0、NN优化0；合成F0/G各2单列。旧数据G99284/F036450、原F0/SSL/G训练及缓存为继承成本，不重计新导出。

主部署神经参数499944，残差读出12961个FP64数字/103688B，加六阈值48B；域方向B审计额外160*r*8B，不将其或原点数据隐藏。完整两模型统计与四读出阈值审计矩阵另报，不把统计值计作神经参数。whole-stage墙钟不是B1时延。

具体commit/精确命令/Python/CWD/路径/GPU/预期产物写在同run登记；一次独立P0/P1、聚焦协议负测和实际CP首smoke PASS即继续，单archive一次本地/远端SHA及compile，实际PID/CWD/argv/GPU/log增长核对。停止仅协议/来源/输出冲突、非有限或无法合法预测；不按低分技术重跑。代码/登记/汇总Git交付，IQ/weights/原日志/运行时和私有进程状态不提交。

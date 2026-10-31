# 地面开放训练：分组平均与困难样本尾部损失

Run `20261101-phase3-ground-tail-s2026092701-r01`，owner `codex-phase3-ground-tail-20261101`。上一轮为已完成实证进展：源 clean 主 SSL 的 AK/CK/RU/H=97.755556/98.303704/69.365079/81.148828%，普通源 PL 的额外 RU 仅0.126984个百分点，AUROC未改善。当前优先强化地面开放训练，保留完整跨RX/星地/无标签适应/发现及匿名注册目标；不把源已知保持当作拒识成功。

## 假设与固定两臂

检验高已知接收率下的拒识是否受容易已知/OE配对主导的平均损失影响。参考[困难异常样本训练研究](https://arxiv.org/abs/2405.08766)和[尾部风险研究](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b691334ccf10d4ab144d672f7783c8a3-Abstract.html)提出本项目变体，不声称论文复现、确定性分类尾部准确率保证或RFFI有效性已经成立。已曝光NV/目标结果不用于选尾分位、margin、权重或采样身份；0.25为本次事前固定的单一尾比例，不作查询扫描。

两臂均从同一完整合规原F0 Tri-SIG-TIME SSL E50/11100独立复制，reader232993、trainable232945、默认六类头960联合训练，Sinc48及全部buffers冻结。原F0/r03/PCD/G不变；不以旧SUP/SSL/目标PCD作初始化或教师。主部署学生＋固定DURA为499944神经参数、六个FP64阈值48B，不另训头、投影、EMA或源行bank。

每步同L192/OE64、同索引：L按六类和15公共RX/day平衡，OE按15公共域抽取；原L2026103017/OE2026103018种子和顺序。共同项CE_L（LS0）＋原F0同G(clean)教师T2 KL×4（权1）；OE项权1，s=logsumexp六logits、T1、高分接收。设P_d为同公共RX/day的L/OE配对，l_ij=softplus(sOE_j−sL_i)。DMEAN对各域所有l求均值后等权平均15域；DTAIL对各域按稳定row-major顺序取损失最大的ceil(|P_d|/4)项均值，再等权平均15域。尾部选中值保留梯度，排序索引不反传；未注册TX真标签不参与分组。非4倍数采用明确ceil离散近似，不将其称为精确连续CVaR。

唯一主方法DTAIL_energy；DMEAN_energy为同预算对照，旧固定主SSL_energy作为OLD_SSL_energy精确复用。相比旧全配对均值，本轮DMEAN与DTAIL共同控制域等权，只改变域内尾部聚合。每臂固定2220步、总4440/1136640真实学生训练行；AdamW lr/wd1e-4、eps1e-8、clip5、FP32/TF32off、eval保留梯度。不读新Us/目标U或其ID，无PL；不称等同上一轮SSL前向预算，也不由未知类结果挑checkpoint。

## 固定数据和四条件

直接复用`20261101-phase3-ground-open-s2026092701-r03`、code `d7e0f875397b2f9801df4605ba0051d221aa4e2a`的VALIDATED_ONCE G(clean)源胶囊和旧物理ID，不重建数据或重跑Full审核。源split392005，eq1/256点/输入RMS一次；六旧TX14-10/14-7/20-15/20-19/6-15/8-20，五源RX index1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，日03-08/15/23。训练仅L6300真实六类y、OE6134 outside-six成员监督/公共RXday，旧原F0同G(clean)L教师缓存。Us56700保留既有祖先，本轮新读0。

旧CAL13500/已知TEST13500按源V27000每y×RX×day格150/150、seed2026110101物理互斥；已知TEST每类2250，NV3150为19-15/19-16/7-1各1050，合成role-blind源Q16650。builder原先合法用V已知y分层，当前训练/阈值/预测不读TEST标签。旧V/NV/开发Q已曝光，不声称新的盲测。

跨RX沿用开发RX3-2/index34未来03-15/23 Q10238（5525/4713），已知4508、四未注册类5730。已知各类1491/1182/150/406/1186/93；未注册5-5/10-7/2-19/7-11各1454/1439/1427/1410，名单仅独立评分/原合法builder。03-23缺20-19/8-20，完整已知指标N/A。目标clean仅测试；正式RX24-18不访问；不由未注册名单推断其TX与OE互斥，不解封OE真TX。

源clean/星地共享Q16650、跨RXclean/星地共享Q10238。全部接固定DURA v1.0 R69 E20/4940、reuse/graph开/native关、FP32/TF32off；源恢复IQ精确复用旧G(clean)/G10 P1，跨RX received再经同G逐记录恢复。G10 MIX01 600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、源scene3001/channel4001/future3133/3134/receiver2027及固定观测保持，输出不再RMS。

## 冻结、指标和晋级

两新模型先冻结，再只用旧CAL已知13500和原F0基准校准各自六个“预测身份”阈值；仍用旧ground已知24组（overall/5RX/3day/15RXday）保护CK≥95%、原F0同G闭集−AK≤2pp，固定公共低尾算法。不读NV/target决定τ；即使零尾或缺预测类不满足也明确失败并完成合法预测，不做测试驱动修复。旧SSL六τ/相应CAL报告精确复用，共新12＋旧6=18个FP64/144B。每行均计算六类，由各自默认头argmax路由；原F0baseline logits独立保存，不能将col0的OLD_SSL当作原F0。

固定3方法×source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA，共12点。旧SSL logits/closed/raw/margin/τ与旧r03 col2完全相同；四条件overall评分必须与旧同角色结果完全相同。新学生预测134552行＝2×CAL13500＋4×sourceQ16650＋4×crossQ10238，G新20476行，F0新0；新学生训练＋预测1271192行。旧胶囊G99284/F036450、旧SSL训练/Us和所有Phase1/G祖先成本另存，不算新物理样本。

全部12点固定后独立CPU仅连接knownTEST/sourceNV/futureclean/futurestar四truth各一次，无GPU/NN/再拟合。逐真实TX宏AK/CK/RU/H、原F0和学生闭集、差值、CI/AUROC/OSCR、逐RX/day/scene/day×scene/TX和配对均报告；源与目标未知分别三字符串TX/四目标标签命名空间，不混宏名单。

**本轮正式验收服从当前完整目标：宏RU≥95%、每预登记主场景RU≥95%、已知宏FRR≤5%（CK≥95%）、含拒识AK较同输入原F0闭集下降≤2pp，且所需名单完整。** sourceclean要求全部24组，crossclean要求overall及clean场景，source/cross星地要求overall及六主场景。旧90%仅历史里程碑，不作为新正式晋级门槛。

ground_source_success、cross_clean_success分别记录；ground_training_success为两者同时通过。frozen_rejection_success要求四条件全部通过且新主CAL保护通过，scientific_success同此。True仅表示冻结拒识候选达到该开发矩阵约束，下一阶段仍是目标无标签适应/发现/未来匿名注册；False为NO_PROMOTION。不据任何新Q结果改主、调参、选模或追加种子。完整Phase3 ACTIVE、adaptation_success=None、AN/H_OW=N/A、formal0。

## 执行

默认N607、卡0至7实际free≥4096MiB，每卡最多新增2本任务worker，其他用户/owner任务不停止/修改。单不可覆盖root，单owner顺序train→predict→CPUscore；当前源胶囊直接复用，无data派生。实际commit/精确命令/Python/CWD/路径/GPU和产物写在同run登记。一次独立P0/P1；launcher首真实checkpoint无query smoke覆盖原F0/native学生新loss、梯度及G，PASS立即训练；一次发布包本地/远端SHA比较及远端compile；启动后PID/CWD/argv/GPU/log增长读回。

停止仅因协议/来源不符、输出冲突、非有限、无法启动或无法产生合法预测；低分不是技术重跑条件。产物为两CP/4440全步日志、已知CAL/18τ/四条件预测、CPUscore、pipeline/实际观测、同row报告与Git推送读回。只发布代码、方法/登记/汇总报告，不提交数据、权重、原日志或私有进程状态。

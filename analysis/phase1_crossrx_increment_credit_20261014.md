# 跨 RX 辅助更新贡献：训练机制假设

下一主线使用 SIG_TIME 实际中层的身份排序信号，并根据真实 AdamW 辅助增量在其他源 RX 上的作用决定是否施加该信号。原始重复观测的24维相对参考与14维正交补版本均已源否证，不接入骨干。候选集合监督的固定版本也已关闭。SIG_TIME 等历史正收益按各自配方保留；本方案尚无新训练收益，84%目标仍 active。

## 信号从哪里来

输入是既有 `sequence_pool.output_norm` 的160维输出，随后仍经原有 bounded gain 加入实际部署 `id_feat_joint`。不新增 embedding、投影、身份/RX头或另训读出。主身份 CE 与原默认分类头正常联合训练；只用真实可见 L_s 标签建立辅助关系，U_s 保持原 oldgate/ownEMA 配方。

同 day、同 TX、不同 RX 的 (a,p) 构成正关系，n 与 p 同 RX/day、异 TX。当前学生正负端点停止梯度，anchor 保留图；单位特征 u 的误差为

`r(a,p,n)=.1 + <u_a,stop(u_n)> - <u_a,stop(u_p)>`。

按 anchor RX 拆五项 `L_d=.1*sum(relu(r))/完整合法triplet数`。满足关系后梯度停止；保留满足项在分母中，不放大少数困难样本，不 mining、不缓存伪真值。五项相加是 ungated ORDER。普通 SupCon 作为独立控制，明确其端点和分母不同。仅有非零 hinge 或非零梯度不等于有效学习；纯 RX 编码虽有 active 关系，也可能因对称抵消而无有用梯度。

## 更新是否值得施加

使用同一真实 optimizer pre-state 的 moments、LR、weight decay 与全局 clip5，分别计算基本梯度的临时步 B 和加入一个 RX 辅助梯度的临时步 B+d。两条都只在临时对象执行，真实训练仍只更新一次。

在同批 L_s 的其他 RX 上，以**同一 pre-step 默认分类权重 W**比较两临时模型 CE。平均 CE 改善超过固定1e−6、且每个其他RX下降不低于−1e−6，才接受这一辅助分项。该数值容差不是调参候选。评价时固定 W 是排除临时头拟合解释的计算对照；原真实默认头正常训练，不冻结骨干、不另训或替换头。

这些其他 RX 仍是主 CE 已见的源训练数据，因此称“同批跨 RX 增量贡献”，不能称真正留出 RX 或泛化验证。最终目标仍必须由冻结后的默认头、独立 truth-last clean 比较证明。评价 mode、buffers和RNG须恢复；辅助 gain 不在损失图中。该机制已接入实际训练入口：评价使用完整 L/U 主损失，辅助项在统一 backward 前只加入一次；四组保持相同的临时评价预算。

## 与已有工作的关系

排序和 SupCon 本身不是新发明。MLDG 已提出源域内模拟域迁移并要求更新改善其他域：[原始论文](https://arxiv.org/abs/1710.03463)。Fish 已研究跨域梯度内积匹配：[原始论文](https://arxiv.org/abs/2104.09937)。本方案待验证的具体差异是：已部署时序分支上停止施压的 anchor-only 关系，在同一**实际 AdamW 状态**中的辅助增量评价，以及同一默认 W 排除读出拟合解释。没有先声称文献首创或泛化有效。

## 当前有界源诊断

只取预声明 SIG_TIME seed2026092701 的 own final50，不依据目标胜者换父来源。完整 source_proof 在加载前检查角色、物理ID契约、scratch ancestry、final-only 选择和0目标训练/选模；严格载入学生和原生一组 optimizer，不能调用 Stage2 两组或补造 moments。

读取完整 L6300，保存同一次 forward 的 pre-gain160 与 actualjoint160，分别分析126,000固定同ordinal关系。四个事先固定 cell ordinal0/17/35/52 各90包，用真实默认 CE、ORDER/SupCon 参数梯度与每RX临时贡献诊断。此处基本梯度只有 L CE，模型/optimizer为 final50状态、eval模式，因此是一步假设试验，不冒充历史 U 联合训练轨迹或新的实际训练。

预计6,300真实IQ读取、9,540真实L网络前向＋2零IQ checkpoint smoke、40临时AdamW步，0真实更新/EMA/另训头/U/V/target读取，CPU2线程。固定 margin/coefficient/temperature=.1，默认norm eps1e−4、数值容差1e−6；不扫描。若来源或参数组无法核实，保留技术失败并停止该run。源诊断之后最多推进一个正式训练消融矩阵，保留BASE、普通SupCon、ungatedORDER与增量贡献版本；具体日程与扩种子条件须在任何新target访问前固定。


SIG_TIME首seedownfinal50真实源诊断已ANALYZED/VERIFIED：pre-gain160固定126000关系未满足6.5397%，joint160未满足11.1143%；辅助anchorORDER真实梯度非零且W/gainNone，原native188参数/72moment/LR1e−6严格重建。四L90×五RX40临时步的sameW跨RX贡献全部低于固定1e−6容差，0/20接受；不降容差、不称早中期无用。6300IQ读取/9540真实L前向＋2zero smoke，0新训练/另训头/U/V/target；252000关系和20决定独立复算。下一最多一个正式SIG_TIME训练矩阵，比较BASE/SupCon/停止ORDER/实际增量CREDIT，同预算default头joint与原U配方保持；源/性能结论分开，84%active。详见[中层增量源报告](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-crossRX-credit-sourceL-manysig-s2026092701-r01/report.md)。


SIG_TIME 匹配训练矩阵已发布并 VERIFIED：BASE、SUPCON、无门控 ORDER、实际增量 CREDIT 首种子均从零训练，原默认头与 33L+17U 配方保持；固定 50 轮，每 16 步执行一次中层辅助评价，前 5 轮预热。首种子源域健康、机制调用及实际梯度条件通过才扩展其余 8 行，每行冻结后自动 clean 预测和独立 truth-last 评分。四组的进程、CWD、命令、GPU 与日志增长已经独立核实；当前没有新的性能结果，84% 仍 active。详见[匹配训练报告](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01/report.md)。


SIG_TIME 训练机制首 seed 四组已完整 ANALYZED / VERIFIED：BASE 83.3625%、SupCon 83.3655%、ORDER 83.2613%、CREDIT 83.2726%；相对 BASE 净判对分别 +5、−170、−151，只有一个 seed，不确认稳定收益。CREDIT 60 次非零梯度调用全部属于 L 阶段，U 的 236 次调用全部停止；固定梯度点仅 1/4 非零，源条件失败，剩余 8 行未启动，不降低阈值或按目标重跑。200 轮/44,400 主更新/672,000 决定/72 混淆矩阵、2,496 调用与16实际点已核实，进程自然退出。浅层及中层参数实际增量存在；Sinc 频率在16点有梯度而更新为0，原因未确认，下一只读核查单位/精度，历史 SincCENTER 等正收益保留。84%仍 active。详见[完整首种子报告](../automation_reports/CV-SincNet/20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01/report.md)。

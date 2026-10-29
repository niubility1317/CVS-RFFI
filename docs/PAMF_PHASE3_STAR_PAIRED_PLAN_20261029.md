# 星地接收与恢复联合拒识证据

run `20261029-phase3-star-paired-evidence-s2026092701-r01`。完整Phase3目标继续ACTIVE，使用用户最新v3宏A_K/R_U/H主报、星地C_K/R_U≥90%、原F0同输入同path闭集到开放损失≤2pp。上一源consensus拒识器原F0/P1的A_K90.1407%、R_U44.9206%、H59.9606%，known保护过但未知失败；训练200步的全量loss仍下降。新研究检验同一星地记录的received与恢复证据是否互补，并用匹配预算控制区分输入证据与训练增量。

比较过继续改骨干、只增加旧单路读出预算、联合P0/P1证据。选择最后一项并纳入P1-only匹配控制：原F0及六类默认头保持冻结，身份始终由P1默认头决定；P0只增加拒识证据，避免改变原未知身份表征。原F0 E50/11100、232993参数，无增量F/旧readout加载；DURA v1.0只作为复用缓存祖先，本run无G前向。全部新的模型IQ都是既有G10 MIX01 P0/P1缓存，不读clean/raw、不重叠加真实信道，不增加独立物理记录。

source角色仍L6300/OE6134/V27000/novel3150，原U56700仅Phase1祖先、本轮新读0。L/OE只原五源RX1/3/4/6/8、三源日；OE21成员监督额外提供“六类之外”，不提供TX/K。novel19-15/19-16/7-1各1050、同源RX/day且已历史开发暴露，非未见RX或新TX盲测。正式与开发target U/Q全部不读。源L标签只拟合各P0/P1八branch geometry，L/OE的100D head-aligned consensus拼为200D，前100为P1、后100为同ID的P0。

两arm均MLP200→16ReLU→1，3233名义参数、每arm独立fresh AdamW CPU全批训练1000步（lr.001/wd.0001/betas.9,.999/eps1e-8），seed2026092701每arm重置、初始权重矩阵逐值相同。P1-only控制在归一化/拟合/预测前把后100D置零，后100均值0/尺度1，不允许P0经normalizer影响控制。其实际仅100D有效输入，不宣称相同有效信息容量；paired为完整200D。每role内15域等权、known/OE loss先验.5/.5，normalizer固定.6/.4 weighted population，均沿既有source定义。域metadata只训练权重，不作部署模型输入。

每条1000步轨迹事前保存step200和step1000快照，200不是另一次训练；总2000实际优化步、4个只读状态。依据是源训练曲线，不根据novel成绩早停。固定5方法顺序：P1 energy、P1-only step200、paired step200、P1-only step1000、paired step1000。唯一主候选paired step1000，其余不能替换；报告同预算paired−P1-only和同输入1000−200配对增益，不能仅以主高分归因于两路证据。

所有geometry/readout/normalizer冻结后才开knownV，42组（overall、5RX、6scene、30RX×scene）用Fraction精确FRR1/10和原F0 closed−open1/50确定各方法最大合法全局scalarτ，完整ties。身份及baseline都为原F0同记录P1，已知cache baseline[:,1]需逐条一致。阈值冻结后才开novel，全部5方法预测固定后独立truth-last scorer连接未知TX/scenario。缺必需类返回N/A且科学不通过；模型/状态不更新，不用目标真值、K、配额或query分数定阈值。

主报宏A_K/R_U/H及C_K、原F0closed、总损pp、宏AUROC/OSCR、逐TX/RX/scene、覆盖与配对净数。主候选须42组known保护及unknownoverall/5源RX/6scene均≥90%；unknownRX×scene仅诊断，无H硬门槛。源通过仅允许下一独立冻结小规模跨RX验证，不等于目标/DA/注册成功。A_N/H_OW均N/A；科学失败完整归档、不改阈值、不挑控制救分、不扩seed或目标读取。

执行顺序：实际原F0合成G10无query smoke（真实cache前、PASS即继续）→ L/OE配对特征/固定两轨迹训练 → knownV源阈值 → novel全预测 → 独立CPU scorer → 同row报告与Git交付。单新不可覆盖run/output，默认N607，实际free≥4096MB一张GPU、一次仅一预测child，本任务每卡新增≤2，不停止/修改其他owner。CPU读出训练在该预测进程内，scorer CUDA不可见。技术输入/非有限/路径碰撞/不能合法预测评分失败时保留partial，不盲重发；科学低分不触发技术停止。

精确commit/命令/环境/CWD/输入输出/owner/PID/GPU/log和预期产物保存在run登记。source配置仍6字段和原F0嵌套12字段，缓存、划分与数据许可直接复用；不重验原raw。一次独立下一实跑P0/P1审查、一份archive一次local/remote SHA比对、一次remote compile及独立启动/终态核实。REJECTED_EXTRA_GATE：不新增设计审批、file/member hash、receipt、固定额外审查或报告完整性发布门槛。原历史结果与阈值不追改。

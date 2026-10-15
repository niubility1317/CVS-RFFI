# 固定 SIG_TIME：双向身份排序 E50 预登记

## 代码审计与研究问题

最新用户附件31cc8a22将主线改为“增强为 clean 提供可用的身份区分”，分别检验 clean 相对无增强、low urban 相对普通增强，两个差值不能相抵。网络固定 SIG_TIME、原内部门控／参考响应和默认 cosine6×160（scale30）；全部原参数按原联合 CE／U 流程更新，不另训或替换分类头。实际身份特征195041参数，完整模型863730参数。

旧八位置族 source 条件失败的结论保留，不扩旧族、不挑目标最高的 S/J。本研究 D2 复用 A-J 是预定的“旧正配对方向”控制。历史浅中层 LOCAL_ORDER 是 clean/full-L6300 与额外 t1/t2 cohort；软关系诊断是 EMA 软图、源梯度条件失败。它们都不是本轮真实 L 标签、clean/practical 交叉视图、joint160、内部5580/720折的直接对照，保留其证据与负结果，不宣称关系学习首次提出。

## 固定矩阵与复用

|逻辑条件|训练／监督|实际身份与新增成本|
|---|---|---|
|D0|无 practical，旧 U 门控／EMA／cos 保持|新 scratch E50；补齐相同内部折 clean-only|
|D1|普通三场景 practical 拼接 CE|复用原八位置 A0 的 final50、预测、独立评分；0新增训练／测试|
|D2|D1＋旧 clean EMA→urban 正配对方向|复用原八位置 A-J 的 final50、预测、独立评分；0新增训练／测试|
|D3|D1＋单向 u→c 相对身份排序|新 scratch E50|
|D4|D1＋双向 u→c/c→u 相对身份排序|新 scratch E50|

首模型seed2026092701，三新模型各 E50=33L+17原 U 阶段、222步/轮、11100次更新；共150新增epoch／33300更新。两复用控制的100历史epoch／22200更新单列，不能算五次新训练。全体固定 split/data392005、L_train5580/L_dev720（fold2026101503）、U56700隐藏TX、V27000health-only。新模型不加载任意父／teacher／optimizer 权重；唯一保留的 EMA 来自各自 scratch 轨迹。baseline路径、code、command、模型／数据／增强／评价seed及逐row配置唯一记录在 [experiment.json](experiment.json)。

## 真实标签关系与梯度边界

复用同一次 native clean／satellite 前向的实际 joint160，辅助路径重组时只 detach 原融合 gain 的值，数值与部署 joint 一致。正样本是同 TX，负样本是不同 TX；两参照必须同 RX/day，且该 RX 与 anchor 不同。首版保持原 shuffled L 采样、独立信道实现和三个训练环境，不另改采样单元来共享传播。

损失为 relu(.1 + cosine(anchor,negative_key) - cosine(anchor,positive_key))。keys 是同次前向的停止梯度副本；按 anchor 先平均合法 triplet，再对完整128个 L 样本平均，满足间隔和缺参考的 anchor 为0。D3只更新 urban anchor；D4为两个方向的半和，clean端也有直接梯度。无新教师网络、投影、分类头、原型或记忆库。

附加目标使用既有 full added-noise SNR 代理 q=sigmoid((S/N-5)/5)，仅 urban 的 q非0：u→c用anchor q；c→u用 sqrt(q_positive*q_negative)。该权重只衰减额外排序，所有合法 satellite 标签仍有 CE，不使用学生置信度／同意门控，不解封 U TX。正负条件同源RX/day不代表信道实现相同，相关传播采样作为后续独立问题。

原头 W 和三个 gain 的辅助梯度必须0；原 CE/U 仍联合更新它们。保留原 native Dropout，观察实际训练的 clean／satellite／aux梯度、层间夹角、clip和AdamW更新；不预设梯度冲突，不增加 Fishr/MixStyle/PCGrad、课程、bridge或分支组合。

## 强度与预算

启动器先做真实默认 checkpoint 无query smoke，随后从固定、合法、同折的原 A0 epoch5（1110更新）严格读取 student 状态，只用于 source 强度统计，不作为三新模型初始化。源 L_dev720，clean与mid/lowurban共2160前向、0更新，eval下分别以全局特征梯度范数中位比 target .1 冻结 D3/D4 系数；默认 W 排除。不是双向梯度翻倍、在线投影或目标调权。系数边界沿用1e-4至10；零／非有限梯度或越界保存错误并停止未启动行，不扫描参数。

D0／D3／D4 都保留与原 A0/A-J 相同的128个额外 clean L EMA 前向；语义 keys 不消费该输出。三者 native U 更新和前向预算一致。D0主前向4769400，D3/D4各6190200；差异来自必要的satellite视图。source dev14×5040、native V健康、真实／预测smoke、校准、正式168000×7推理与墙钟分别计量，前向样本数不冒充FLOPs。

实际信道沿用25MHz、IQ256、native RMS后simulator input RMS、无output RMS；residual/post_sync、EQ=false/G=1、reference_snr_db30、质量适配／相位跟踪／IQ补偿开启、reflect、原24点数值延迟。实际导入file与参数由 channel_execution记录。正式clean及六practical继续复用原VALIDATED_ONCE数组，不调SNR、信道或困难样本。

## 评价、推进和资源

新row final50源冻结后立即预测全部七视图；全部prediction固定后单独exec scorer，以空CUDA_VISIBLE_DEVICES、CPU2线程独立truth-last重算accuracy/Macro-F1。原D1/D2保留其旧实际run/row身份，不重复测试。训练与晋级不消费目标指标。source-family决定在汇总读取target scores之前写出。

首轮源推进固定为：任一D3/D4在final50 L_dev上，clean−D0≥0.5个百分点、low−D1≥1个百分点、urban−D1≥0、最差urban RX−D1≥−1个百分点。失败保留全部负结果，不放宽规则；通过也不自动启动多种子或学生／教师新矩阵。完整性能结论仍需预定义多seed配对，当前单seed不声明稳定双增益。原 low urban≥60%与urban/low增益目标继续逐项报告，不是结果保证。

只有3条新行科学上已独立。发布用已有 receiver_residual_capacity4.lock，在同一临界区读8卡实际／启动中所有owner占用、选卡、Popen并登记，先空卡分散、一卡一行优先，每卡上限3且free≥12000MB。无固定抢卡或保留分区；与另外两chat协调但不等待聊天审批、不扩候选填卡。新行CPU2线程、num_workers=0，source校准及独立评分CPU执行。发布前保存对方近况与实时容量；发布后核验PID/CWD/argv/GPU/log增长。

唯一run/output根、CWD、精确命令与owner见 experiment.json。单Git代码/config身份、一次archive传输SHA比较与remote compile；首次smoke PASS立即继续，无token／额外许可。系统技术错误仅限本run输入／来源／图接线／非有限值／预算／输出冲突／launch；保留产物、停止未启动行，其他健康行完成，不自动重复或操作无关进程。

本地6项检查PASS（6.793s）：真实标签参考条件、单向stopgrad／双向anchor、有界间隔与U负例、真实joint特征与W/gain边界、五条件/复用禁止重训重测、D0/D3/D4实际native graph/CE/EMA管线。新包compile PASS。一次独立P0/P1审查发现 D0 native 后置通用增强替换 capture 输入对象，已显式关闭 D0 的 labeled generic augmentor，以匹配原 concat 控制中未改变的 clean 分量；原 U 强视图保持。加入实际 post-concat conditional 回归，6 项通过，原 P1 定向复核 PASS。真实远端运行尚未开始。

## 未完成与独立支线

当前没有新训练／正式测试结果。源冻结后将补充完整梯度／训练曲线、默认头、分支幅度与身份关系的归因，不能仅凭loss减小宣布有效。已完成HQ源诊断是固定权重降低新增噪声的受控效应，不是训练提升；Bridge-CE/Bridge-S留到语义源证据后独立比较。

跨RX两教师支线已有匹配4464条L子集与L-only CE接线、4项本地测试；当前只有代码草稿，没有登记／发布／模型／教师互补结果，不混入此矩阵。它与A的E20条件不同，后续复用科学证据不能冒充相同预算或直接继承权重。

## 实际发布及首步技术错误

实际发布 commit `e8f51edabdf6b834ad688fc76f3f22370f2ec03c`，独立远端 OID 一致。一次 archive 传输比较 VERIFIED，remote compile PASS，真实 A0/A-J checkpoint 默认头 smoke PASS（4 前向，无 query）。源 epoch5 校准完成：L_dev720、2160 前向、0 更新，状态及 .grad 未变。两系数为 D3=0.6863719838、D4=1.1577435581；全局源比目标 .1，未调目标成绩。

独立读回确认 dispatcher PID3059034、CWD/argv 与发布一致。D0 PID3059123 在 GPU0 继续健康训练，日志 6686→42449 字节，resolved commit 匹配。D3/D4（GPU1/2）均在首步因 labels CUDA、RX/day CPU 的布尔 mask 操作失败并退出，尚无完整 epoch／性能结果；本矩阵当前为 PARTIAL，不能标为完整实验。所有失败输出保留，原 D0 不停止、不重启、不修改。

本地已复现设备错误，mask 在比较前仅把 RX/day 移至 labels.device 并统一 long；修复后7项检查PASS（6.435s），原设备问题定向P1复核PASS。本地修复尚未发布；后续用新不可覆盖run/release仅恢复失败D3/D4，原D0及已完成控制均复用，避免重复预算。原r01发布代码身份不改写为修复commit。源HQ的64.86%→70.97%仍只是冻结源域降噪受控效应，与此训练失败无关。

证据：local artifact `independent_launch_evidence.json`、`independent_initial/readback.json`、`independent_growth/readback.json`、`source_initial.json`、`mixed_device_before.log`、`mixed_device_after.log`；没有正式新性能或跨seed结论。

## D0 匹配性更正

后续按真实 native 前向核查发现，原普通 concat A0/A-J 在拼接前仍执行 generic IQ augmentor（train_ssdg.py8986），因此此前“关闭 D0 通用增强可匹配 concat clean 分量”的表述不准确。r01 D0 的 use_aug=False 使它成为无通用增强诊断，不能作为仅去掉 practical 的严格对照。该健康行继续原自动测试，不停止、重训或改写产物；在 r02 单独补齐 use_aug=True 的匹配 D0，保留同一实际前向后的 capture。r01 D0 的完整 E50／11100 及首步失败两行成本作为技术开销报告，不纳入新五条件源推进或正式配对。

## 原诊断D0自然结束

VERIFIED：PID3059123已自然退出，原D0 row为TRAINED_AND_SCORED。E50／11100更新，训练耗时2327.915211s；head_fit_updates0、source训练target_access=False。原七视图prediction完成且truth_read=False，独立scorer为SCORED_COMPLETE。实际主前向4,769,400、L_dev前向70,560，U TX未读取；其他native V与正式推理成本保留在原产物。本次只读取完成元数据及成本，未读取目标分数作研究反馈。

原r01矩阵最终FAILED的原因仍是D3/D4首步设备错误，没有停止/重启D0或选择性重测。D0 use_aug=False的配方不匹配已在其目标评分前确定；该E50和2327.92s作为独立诊断/技术开销，排除r02五条件配对与源推进。r02正确D0/D3/D4保持原配置运行。证据：local artifact original_D0_terminal.json。

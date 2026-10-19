# 完整 practical 较清晰观测的纯标签监督 E30

状态：REGISTERED，尚未发布或训练。Run 20261019-phase1-FcrossRX-signalBridge-e30-s2026092701-m3-r01。起点 HEAD a0c8ea4a074e4e7dbc42980381f98c0f0553913e。实际代码/config 以本次 Git commit 与 release_commit.txt 为准。

最新附件建议的三条路线已经有后续完成证据：P70 nominal质量分层、P71有界质量风险、UPractical/UQualified和P76共享传播四格均训练并正式评分，固定候选关闭。P70对同池控制low +.101190pp，但源midlow -.027778pp；P71对M2 low +.322024pp，源midlow -.027778pp；UQualified路由增量low+.499405pp而clean-.475595pp；P76共享H单因素对同父控制low+.007143pp、clean-.139286pp。以上均单模型seed，不用于选择新参数或父权重。其他对话当前研究prefix61与随机非参考区段屏蔽，本轮不重复。

唯一假设：额外较清晰的完整practical观测，单独接受真实标签CE，能比等前向raw监督更有利于原始practical身份识别。旧BRIDGE_CE含共同joint anchors，不能回答此问题；本次没有新增teacher、KL、特征/分支对齐、门控或部署模块。完整原As仍监督全部合法L，clean绝对监督份额不稀释。

## 固定矩阵与成本

逐row矩阵唯一维护在experiment.json。N01从原P43 B00 ownNoAs20；C01/S01从同一B01 ownAs20 student/ownEMA。均独立追加E10/2220更新，累计E30/6660，RESET AdamW lr2e-4/wd1e-4/min1e-6/cos10，原Sinc统一参数与ORD128 CROSS_RX及oldU不变。不是所有row共享As父。旧P43 NoAs30/As30完整冻结读出只复用。

原6次L128前向与原损失保持：.5RDordinary+.5RDmixed+.25RDraw+.1SupCon；追加.25×.5×[CE01(extra64pass0)+CE01(extra64pass1)]。两次CE用同一IQ，Dropout各自随机；新增部分无KL。N/C额外raw，S额外HQ。每步896L样本前向/8调用，总L名义系数1.5。N/C/S额外NN与损失尺度匹配，C/S均生成HQ使物理模拟成本匹配；N无物理模拟，照实分列。S-C是HQ增量，S-N是阶段clean比较，C-旧As和N-旧NoAs说明额外raw/成本变化，不能把额外预算都归给HQ。

HQ来自同一raw64，固定reference SNR36dB（旧先验+6dB），原fullAs保持30。外生传播、session硬件值和标准噪声流相同；每包native quality、lock、补偿、AGC按36自然重算，不强设相同接收状态。只改变补充训练观测质量，不修改正式IQ、SNR或困难样本。C生成相同HQ shadow后仍喂raw。

新成本3×E10=30modelEpoch/6660更新；每row L1989120NN，原U学生/teacher各567000，预计native3447122NN、源12600NN、formal1176002NN，共4635724NN。共享首步真实父/hook/末轮边界smoke3592NN/0更新，预计合计13910764NN；以实际计数为准。As full142080/row，C/S额外HQ各142080次，N无模拟；CPU物理时间和墙钟分别计，不冒称无新增成本。

## 输入与评分

F+CROSS_RX、SIG_TIME_X3、统一Sinc、joint160、默认cosine6×160/scale30；骨干和默认头正常联合训练。ManySig fullL6300/hiddenU56700/V27000，RX1/3/4/6/8、day1/2/3、split392005，物理角色匹配，无heldRX或Ldev。现有256点Fs25MHz；内部NoExplicitCFO/逐包补偿沿原实现。沿真实args核对ordinary，RDrop两次不重抽IQ；raw直接生成practical，不叠加ordinary损伤。

使用原P43 commit19aedc3604fed9c193237dd92366164178af0983的scratchFINAL20各自学生/ownEMA；核对实际role_ids、scratch祖先、固定final选模，无目标选父。复用VALIDATED_ONCE source与fixedreceivedviews，不增数据重验。原panel两draw是1800个物理样本，不是两个modelseed；panel与正式fc差异保留，训练2.462GHz，固定formal2.45GHz。实际导入文件在运行产物保存，不依默认值判断。

固定FINAL30，源资格只决定未来E50：S-C pooled两draw等场景midlow≥+.5pp、S-N fullVclean≥-.1pp、S-旧As midlow≥0。任一失败关闭固定+6/.25，不扫描gap/weight/seed或追加预算。失败仍自动clean+6practical全7评分，不以正式结果推翻源决定。源判断在root首次读取新formal指标前保存。最终报告clean配对pp/all6/urbanmean/F1/worstRX/RXdayTX/修复与新增错误/suburban绝对gap/全部成本和缺项。仅1模型seed，微增不宣称可重复双提升。

## 执行与停止

唯一owner codex/root/practical-signalBridge；N607普通非特权既有CVS-RFFI/fullFP32，CPU2。CWD PROJECT/releases/cvs_FcrossRX_signalBridge_E30_20261019_r01；命令 python -u experiments/cvs_signal_bridge30/runner.py。独占输出 PROJECT/runs/20261019-phase1-FcrossRX-signalBridge-e30-s2026092701-m3-r01。dispatcher CPU，队列按每GPU所有owner总进程≤3且free≥12000MB分配，不修改健康任务。

聚焦协议负例和CE数学检查本地通过（87+3，0NN）；17文件/2段发布脚本AST通过。一次独立P0/P1后实际代码commit/push，单归档local/remote SHA比较、ordinary preflight和compile。launcher首先加载真实own20默认头，在三行完整8调用路径及末E30边界运行3592NN/0更新无query smoke，PASS直接训练，无额外授权token。

停止仅限本run系统性协议/父角色/非有限/输出覆盖/技术失败，保留产物并先核实poststate，不自动重启；不因科学负结果停跑或变参。完成后自然退出。预期artifact包括real_checkpoint_smoke.json、synthetic_smoke.json、逐rowsource/result和完整日志、source_matrix_decision.json、7viewprediction、独立score及automatic_summary/complete和同rowfinal报告。


发布状态：RUNNING。实际代码/config commit `80db12fb33a256ef7174f6dcda6f462645fd53ba`，upstream OID 独立匹配。ordinary preflight、单归档传输比较、远端 compile、真实父与首末 hook smoke3592NN 均 PASS/VERIFIED，已直接进入资源队列；尚无本轮性能结果。

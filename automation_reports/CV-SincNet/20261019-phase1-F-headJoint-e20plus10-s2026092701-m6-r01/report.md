# 原默认头联合训练：浅中层特征适配的独立后续对照

状态：RUNNING / VERIFIED，真实父模型8NN无query smoke通过，六行PID/CWD/argv/GPU及日志增长已核实；尚未完成正式评分。

上轮完成六行固定W的E20+10。合法源L实际训练low urban面板上，FULL为65.8784%，LoRA54.2230%，TEMP52.7027%；两种浅中层适配器没有达到全量特征微调水平。不能据此声称物理信息已经丢失，或直接启动raw旁路。目标仍为low urban>60%、clean相对匹配NoAs可重复正收益、逐suburban接近自身clean。

本轮唯一假设：原类别方向W完全固定，是否限制受损特征与既有类别方向的共同适应。依据新附件第七节第3项（原文：“首轮……可以连同原类别权重W一并固定……再独立比较是否放开原默认头的联合训练”），完成其后续独立比较。保留原6×160 cosine头、scale30和同一原W初始化；W与FULL特征或LoRA/TEMP特征模块在同一优化器、同一次反向中联合更新，绝不冻结所有特征后单训头，不重置、不替换、不另训身份/RX诊断头。

六条新行FULL/LORA/TEMP×NoAs/As均从原合法B0 scratch FINAL20出发，新增10×222=2220更新，累计E30；不从已评分的E30适配器权重继续。固定W的六行作为同预算控制全部复用，不重训/重测。直接复用上轮真实284160条As bank、同一L索引流、actual-first L1800及同L新信道面板，不重抽、不改变信道难度、不访问U真值。

除W可训练集合以外，保持上轮r8/alpha8/k5、A/K正常B零、t1/t2 preGN/prePool路径、raw普通槽位、共同eval但学生autograd、L-only CE0(clean)+CE0(second)+mean_all128[parentCorrect×KL(parent_clean||student_clean)]、tau1/lambda1、NoAs第二槽raw、reset AdamW2e-4/min1e-6/cos10/wd1e-4/clip5及FULL CENTER_UNITS。原teacher F和其默认W保持只读。没有U分支、EMA、星地教师或部署门控。新增可训练原W恰为960参数，新增部署参数为0；FULL232193、LoRA3264、TEMP3344个可训练参数，以实际构建核对。

源规则在新正式分数前固定。各joint_As必须相对相同空间frozen_As的V mid/low均值增量≥+.5pp，fullVclean差≥−.2pp，并且joint_As相对自身joint_NoAs fullVclean≥−.2pp。LoRA/TEMP若要晋级，还须源urban分别超过本轮FULL_joint_As和上轮FULL_frozen_As至少+.5pp。FULL本身是共同头规则下的普通As基准，仅当它相对同配方固定W控制满足上述源增量和clean要求时，才有后续匹配延长资格；不再只依据As−NoAs已知大收益推进。

报告固定W→联合W的NoAs差、As差及二者差分，不能把一般联合训练收益全归给星地机制。若失败，关闭此固定联合W/r8/k5/lambda1/E10对照，不扫W学习率、rank、kernel、lambda或seed救回。无自动E50；源支持后累计最多再20轮达E50，必须保留对应固定W、FULL和NoAs控制并预先固定optimizer/sampler/信道计数衔接。不得依据新正式成绩重排候选、选模或延长。

最终均固定FINAL30，默认头直接逐行预测原clean+六practical，每view168000，所有该行prediction完成且跨行source决定固定后独立CPU truth-last评分。复用旧固定W和原B0父预测只作匹配比较，不更改旧结果。报告accuracy/macro-F1/最差RX、RX/day/TX、修复/新增错、每suburban与自身clean差、成本及n1边界。单seed不证明可重复收益，既有benchmark暴露背景保留。

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| W01 | 目标二/三、附件九 | 合法共同B0 E20父、角色/继承不变 | design/model | verified | strict parent/data handles | E30旧适配器不是父 |
| W02 | 附件七.3 | 同一个原默认W联合特征更新，无替换/重训头 | model/optimizer/train | verified | W身份/形状及同一步特征和W变化 | W960非新增部署参数 |
| W03 | 附件七 | LoRA/TEMP原公式/初始化/层位不变 | model/旧layers | verified | 同状态/零函数/参数集合差仅W | 不扫rank/kernel |
| W04 | 目标六、附件九 | 精确复用同一bank和L序列、六匹配控制 | design/bank/train | implemented | 全索引及控制state | 不新模拟 |
| W05 | 用户L-only回复 | 三项loss/有限保持，U隐藏不用 | train/旧loss | implemented | 执行/分母/无U/EMA | teacher原W不更新 |
| W06 | 目标二 | 同cos10/物理Sinc，单优化器全局clip | optimizer/checks | verified | 解析更新/同一步 | 不给W额外lr |
| W07 | 最小流程 | 聚焦负测、真实CP首步立即继续 | checks/smoke | verified | actualF及no-query | 不加门/receipt |
| W08 | 目标七 | 实际joint160与默认W、冻结原状态核对 | model/train/analysis | verified | feature/W同时变，其他固定态不变 | 不能仅W更新 |
| W09 | 目标二.5/六 | 以相对普通As/固定W的源增量推进 | report/dispatch | implemented | source-only预定判定 | 不以As−NoAs替代增量 |
| W10 | 目标三/七 | 自动七环境/终轮/独立truth-last | evaluate/dispatch | implemented | all6×7×168000/独立CM | 不用target调参 |
| W11 | 目标七 | W效应与As交互/修错/分层/seed边界 | report/analysis | implemented | 配对与完整重计 | n1 std N/A |
| W12 | 最小流程 | 一次P0/P1/commit/archive/preflight/owner | publish/dispatch | verified | 实际release及PID/GPU/log | 健康他人任务保持 |
| W13 | 附件九 | 实际As、同L新信道、V分开 | 旧readout/analysis | implemented | 三层源读出/真实IQ保存复用 | 不将面板当全部L |
| W14 | 目标二/八 | 总E30/源支持最多E50/全负结果保留 | design/report | implemented | update/cost/固定source决定 | 不自动延长 |
| W15 | 附件七.4 | 条件式raw IQ旁路 | 无本轮raw模块 | deferred | 仍无必要信息丢失证据 | 不擅自扩架构 |

本地actual-F检查PASS：44合成NN、9负测，0真实CP/IQ/target，3个直接执行train.py最终checkpoint字典并由own_model重建的合成schema；默认预测及joint160逐位回放。source比较/clean限制/FULL控制/失败分支的0NN检查通过。见local_checks.json，真实CP首步smoke与远端执行待返回。

唯一一次独立P0/P1审查PASS，14文件语法/6行配置通过；无额外模型/CP/IQ/target或远端操作。真实CP首步smoke和发布执行待返回。

代码/config commit f4fe173d83245e5ac8a7769d14cacbdf8260dc5a 已push且远端OID独立核实。命令 CVS-RFFI/python -u experiments/cvs_feature_head_joint30/runner.py，CWD=PROJECT/releases/cvs_F_featureHeadJoint_E30_20261019_r01，普通N607既有Torch2.1/FP32/TF32off/CPU2；输出=PROJECT/runs/20261019-phase1-F-headJoint-e20plus10-s2026092701-m6-r01，所有owner每GPU最多2、free≥12000MB，默认首步真实CP8NN通过即继续复用bank和六行训练。逐row配置/路径/stop rule/预期产物见experiment.json。

启动证据见launch_evidence.json：单归档传输与远端compile VERIFIED；六行复用既有实际As bank，新增物理模拟0。没有据运行状态声称性能目标已完成。

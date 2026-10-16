# F＋跨 RX 同 TX：保留基线监督的 additive As 源域反证

状态：RUNNING／VERIFIED。三组已从零启动，尚未完成训练或源评分。唯一假设：完整保留原有 raw/ordinary 损失，再追加一次完整 practical(raw) 监督，能在不牺牲 clean 的条件下改善未见源 RX 的 urban，并优于单纯追加 raw 监督。此处研究 As 训练组织，不是新的物理 A_g；RDrop、SupCon 均不冒称 IQ 泛化增强。

## 依据与既有负结果

代码审计发现，先前固定总权重的 As 配方把第三对 raw.25 替换为 As.25，实际 raw 总系数由 .75 降为 .50。这个事实不证明它造成 clean 下降。旧 normalized 75/25 As 两种子的 clean 配对均值为 −5 个百分点，该固定路线仍关闭；不以当前单种子重新解释或“修复”旧结果。微频移、常增益、PA 旁路、循环移位旁路等固定失败配方也不重跑。最近循环移位旁路 E20：clean +0.396825 个百分点，low urban −1.269841 个百分点，六个 practical Macro-F1 均下降，已关闭。

本轮保留旧完整基线项，显式增加监督项，并加入同成本、同总损失尺度的 extra raw 对照。它能反证 additive As 是否具有超出重复 clean 训练的增益，不能证明之前 clean 失败的唯一原因。另一用户对话的局部平滑衰减 IQ 路线独立进行，不合并不同配方的增益预测联合成绩。

## 数据、模型与权限

固定 source RX1-held fold、model seed 2026092701，三组独立 scratch E20，无父权重、resume 或外部教师。网络为 F＋跨 RX 同 TX 对比学习：三路 SIG_TIME、统一 Sinc、真实 joint160、NoExplicitCFO、必要的内部参考补偿、默认 cosine 头随骨干正常联合训练。无新增部署模块、分类头替换或头拟合。沿用 ordinary L 两次前向的 time_down/t2.pool 源响应 SupCon：temperature .2，weight .1×clip((epoch−5)/10)，同 TX 异 RX 正对，同 TX 同 RX 排除分母，异 TX 负对；无 U 对比或额外教师增强。

复用已验证原 source carrier 的固定物理 ID 角色。训练 RX3/4/6/8、day1/2/3、6TX，实际 L5040/U45360/V21600；held RX1 原 L1260 仅 final20 冻结后作源选择读出。U TX 真值始终隐藏，V 不选 epoch。该训练契约不同于完整 L6300，也不同于 L_train5580/L_dev720。无 target 读取、调参、候选晋级或选择性重跑。匹配旧 B00 checkpoint 只用于 dispatcher 的真实 checkpoint 无 query 合成 smoke，不载入训练；每组 EMA 来自自己的 scratch 学生。

## 三组最小矩阵与损失尺度

每个视图内部两次前向复用同一已生成 IQ。定义 R(x)=两次默认头 smoothed CE 的均值＋同 IQ 双向 KL，label_smoothing=.01、RDrop KL 系数1。CE 按物理包取均值；KL 先求类别和，再按物理包取均值。

共同基线 L_base=.50 R(raw)+.25 R(ordinary(raw))+.25 R(raw第三对)+w(epoch) SupCon。后两组追加项不重新除以视图数；原有绝对监督系数不变。

| row | 第四对输入 | 额外系数 | 总 pair 系数 | raw/ordinary/As 归一化相对份额 | L 前向/物理包 |
|---|---|---:|---:|---|---:|
| D00-RX1-s2026092701 | raw | 0 | 1 | .75/.25/0 | 8 |
| D_raw-RX1-s2026092701 | raw | .25 | 1.25 | .80/.20/0 | 8 |
| D_As-RX1-s2026092701 | 完整 practical(raw) | .25 | 1.25 | .60/.20/.20 | 8 |

D00 的零权重第四对是显式前向成本控制；其损失和默认头梯度与原六前向基线逐位相同的本地验证已通过。D_raw 控制重复 raw 监督、总损失尺度与相同前向数。D_As 与 D_raw 的唯一研究差异是第四对的 IQ 内容。三组绝对基线 raw 系数都是 .75，但新增项改变相对份额；AdamW、梯度裁剪和模型响应使保护 clean 不能由系数推导，必须实测。当前 F 没有带运行统计的 BatchNorm（实际模块数0），仍记录额外 BN 更新数；额外 dropout/RNG/前向成本由三组匹配。新增项只用原有 RDrop 类型，不加 clean-sat 对齐、分支约束或新门控。

每组 20 轮×222=4440 次更新；原 AdamW lr.0002、weight_decay.0001、完整物理 Sinc CENTER_UNITS 修正与 cosine 日程、原13L+7U/旧门控/EMA保持。后7轮继续有标签训练并启用原 U，不追加 U/teacher practical 或置信门槛。所有合法 L satellite 包都监督，urban 不需先高置信。SupCon3330有效步。

原 ordinary 流保持 DC/taper、循环移位、gain/phase/CFO/Wiener/multipath/AWGN/DAC/PA 的实际顺序与 native min(epoch,13) 日程；PA normal 开启但不宣称已证实标签保持，DAC p=0、class_signature=false。ordinary 各自只生成一次。D_As 的第四对直接由 raw 生成一次完整 practical residual，使用既有传播、噪声、接收质量、补偿依赖和六场景均匀物理包抽样；不把 ordinary 的 AWGN/multipath 再送入 practical，不重复施加完整星地信道，不修改测试 IQ/SNR/信道或删除困难包。全局 CPU/Python RNG 由命名物理信道流保持；三组原 L/U 物理序列将完整比较。

## 实际预算与指标口径

三模型合计 60 model-epochs/13320 更新。每组 L NN=4440×128×8=4546560，其中新增1136640；每组568320次 native ordinary 包生成，只有 D_As 追加568320次完整 practical 包生成。另计原 U 学生/自身 EMA、V、scratch smoke、held 源8820（1260×7）及时间。dispatcher 实际 checkpoint smoke 另2次合成前向；teacher复制学生初始eval计数2的继承偏移不算教师真实前向。无续训、父权重成本、辅助头拟合或新增部署参数。

本地测试：20项负例，3858合成NN、3次合成优化更新、128合成ordinary与128合成practical生成、0真实IQ、0target；这与正式成本分开。三组初始化一致，D00损失/默认头梯度等于原六前向，同IQ成对、8前向、只从raw生成一次As、全L监督、无L上下文不做As、默认头联合更新、export160及无新state键验证通过。最后预算元数据/全文件编译检查增加0NN。一次独立P0/P1审查PASS；后续仅更正非行为性预算/命名说明，未重复审查。

源指标固定final20，逐row完成即自动 clean 及六种 practical 读出，不等其他row训练。每场景1260个同物理包。预测全部固定后，独立分析重新计数共26460个硬预测，accuracy、Macro-F1、confusion、相对两个对照的修复/新错、TX/day分层、每个suburban绝对准确率与自身clean差距。测试信道与随机流固定，不按标签选场景。runtime frozen默认头mean CE保留；硬预测没有logits，不能宣称独立重算CE。

实际 native final_L_CE 是第一 ordinary 前向的平滑CE日志，不是完整目标。分别报告 final_native_logged_L_CE、final_base_L_objective（完整共同加权CE＋RDrop＋SupCon）、final_extra_pair_objective与total_L，不能因新总系数1.25直接判健康/失败。完整20轮日志、每row4440步native/view/unit记录、全部physical序列、As覆盖按TX/RX/day/scene/state/lock完整复算。源健康不能替代识别增益。

## 预登记判断与停止规则

D_As 必须同时满足：

- 相对 D00：clean accuracy与Macro-F1均不下降；low urban≥+5个百分点，urban三场景均值≥+3个百分点。
- 相对 D_raw：clean accuracy与Macro-F1下降均不超过.5个百分点；low urban≥+5个百分点，urban均值≥+3个百分点。
- 相对两个对照：六个 practical 每场景准确率下降均不超过.5个百分点；seen源V下降不超过.5个百分点。
- 各组4440步、8L前向、原SupCon3330步；共同基线目标最后222步均值≤.5；D_As全部568320 LAs受监督、其他组0；无As U/teacher、无非有限跳步、无target或头拟合。

任一失败即关闭本固定additive组织，不以调整系数、seed、RX、场景概率、阈值或额外预算救跑。PASS仅支持新的独立源确认，不自动target、多seed或宣称Ag、泛化、联合目标完成。所有负结果保留。完整最终任务仍需要有证据的Ag、统一训练契约下B00/B10/B01/B11及交互项、跨seed、正式target clean正收益/lowurban>60%/suburban接近自身clean；各suburban差≤2个百分点仍为探索目标，不能通过降低clean满足。

唯一owner为 codex/root/F-CROSS_RX-additiveAs-E20-20261016。普通N607用户，CPU dispatcher2线程/loader0，既有共享capacity lock，所有owner每GPU最多3进程、启动worker时至少12000MiBfree。实际GPU选择及PID/CWD/argv/log增长在启动后独立核实。不得停止、重启或热改健康旧任务。

CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01。命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_additive_as20/runner.py；worker追加 --worker <row_id>。精确输出路径、角色、矩阵见experiment.json；代码身份以实际Git commit/release_commit/resolved为准，发布后回填。

一轮N607 preflight、一个release归档本远端SHA比较、remote compile，dispatcher第一步真实CP无query smoke通过即继续。协议/初始化/角色/预算/输出碰撞/系统性非有限/无法产生合法prediction为技术失败：保存partial，取消pending，健康worker自然完成，不自动重试。科学负分不触发停止、重启、改配方。

产物：每row实际args/resolved/roles/init/frozen、私有scratch/final checkpoint、完整stdout和20轮日志、4440步native/view/Sincunit和L/U序列、As全覆盖、固定七视图prediction、result/completion或failure、独立同row分析与汇总图。IQ、权重、逐包轨迹、日志及机器私有信息不入Git；小型汇总、配置和报告提交推送。


## 实际发布与启动读回

执行commit 7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b 已推送，远端OID与本地HEAD独立比对一致。归档单次本远端SHA比对、remote compile VERIFIED；匹配真实CP noquery smoke PASS。dispatcher和三worker的PID/CWD/argv/CUDA环境、实际GPU进程以及日志增长独立核实VERIFIED。三行真实seed2026092701，20/13/7，sourceRX1 fold，无baseline/teacher父权重；resolved为8L前向、保留base raw.75/ordinary.25及extra raw0/raw.25/As.25，updates4440。

- D00-RX1-s2026092701：GPU0，观察完整step437/4440，E2/20；stdout11672字节。仅为运行进度，无最终识别结论。

- D_raw-RX1-s2026092701：GPU1，观察完整step390/4440，E2/20；stdout11693字节。仅为运行进度，无最终识别结论。

- D_As-RX1-s2026092701：GPU2，观察完整step207/4440，E1/20；stdout6516字节。仅为运行进度，无最终识别结论。

仅修改本轮本地报告和状态，不改已发布训练release，不重复启动同run。完整联合目标仍未完成。

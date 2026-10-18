**当前状态：FAILED（固定源缓存的反事实条件轴适配错误），非科学负结果。** ID连接已通过；obs缓存真实形状为2×3×11×1800×2×256，适配代码误写为仅1个条件。原PID1213449已退出，smoke4PASS／四fullV完成；108004NN、6CP、0训练更新／0target，无完整科学结论。0NN读取header已确认实际结构，保留r02，修复r03不改科学条件。

**r02：同一科学矩阵的技术映射修复，尚未远端启动。** r01保留FAILED；实际source panel是既有opaque物理ID，适配代码误与native辅助五元组比较。r02按同一opaque ID连接fullV与panel并逐ID核TX／RX／day，未放宽角色边界。定向两表示正例／乱序正例及6负例PASS，0CP／IQ／NN；唯一原问题定点复核PASS。原科学判定／来源／预算保持。

# 完整 practical E50：既有 own EMA 的源域默认头匹配比较

状态：LOCAL_VERIFIED，尚未远端启动。唯一 owner 为 codex/root/practical-ownEMA-source。本轮只有源域冻结读出，0 新训练轮数／更新／教师更新；不访问正式 IQ、预测、分数或 truth。

唯一假设：当前 F＋CROSS_RX、完整 As 的固定 FINAL50 已有 own EMA，时间平均可能在保留 urban 身份收益的同时改善 clean。当前 source 与正式读出均使用 student，本次在同一默认 6×160 cosine 头和同一输入上比较两个已有完整状态；不另训或替换头，不增加部署分支，不扫描 EMA 衰减、epoch 或 seed。已有 SIGTIME／BASE_SSL clean-only E20 EMA 源比较曾由 501 错增至 504 错，CE 下降不等于身份识别改善；该负结果保留。新问题改变的是当前 F／CROSS_RX／As50 与匹配 NoAs50 的训练来源，而非对旧系数微调救跑。

## 最小矩阵与判定

固定原定第一个 seed2026092701。原 NoAs50 和原 As50 的 student／own EMA 共四条件，完整 V27000 raw 逐物理 ID 配对；原 V1800 clean 与两次固定 low_suburban／mid_urban／low_urban 观测复用，学生 panel 预测直接复用既有产物，EMA 使用同一 IQ。两 draw 不是两个模型种子，也不增加独立物理样本数。四条件各自 readouts/{row_id} 独占汇总路径，整体一次冻结分析。

- full V clean 的 AsEMA−NoAsEMA ≥−0.1 个绝对百分点；另列 AsEMA−AsStudent 的修复／新错／净修复。若后者净值<0，不能称 EMA 保护 clean。
- 源 mid／low 等权合并的 AsEMA−AsStudent 净修复须>0；否则关闭“EMA 带来 urban 增量”的候选。每场景、draw 全部报告，不另加每格涨分门槛。
- `(AsEMA−AsStudent)−(NoAsEMA−NoAsStudent)` 仅作归因，不是新增门槛；两路线相同收益只称通用 EMA 读出收益。

三项真实身份指标的解释预先固定。所有四条件在科学负结果下仍完成源读出；不微调系数或换 seed／预算救跑。单种子源通过只能支持原定其他 seed 的源确认，不证明正式收益或稳定双提升。本轮 automatic_formal=false；没有新增训练模型必须补测，后续正式读出需由合法源结果固定候选后继续，既有默认自动测试规则不被取消。

## 权限、来源与成本

原 L6300／U56700 TX 隐藏／V27000、全部源 RX1/3/4/6/8、day1/2/3、split392005、IQ256／equalized1 不变，不留 RX 或拆 Ldev。只迭代原生 V loader；U loader 不迭代、U TX 不读取。V 与训练／目标的既有物理角色不变，复用 VALIDATED_ONCE 数据，不重验数据、建签名或增加审批。

父 run `20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01`，固定两个 B00/B01 source/final_ssdg.pth。parent release `cvs_FcrossRX_As_budget_E50_20261018_r01`，提交 `0f399c910324e21a8b1a60346d879be5c2964a9c`；原 own_model 的祖先、source metadata、role_ids 和冻结累计 E50 检查全部复用。新增只核实当前 FINAL50 student／EMA 的全部 keys、shape、dtype、finite 与默认头，不能用父 EMA 的合规代替当前状态。模型处于 eval、无梯度／参数／buffer／EMA 更新，前后 state 相等。原 defaultcos、NoExplicitCFO、内部补偿和所有物理处理保持。

预期新增 fullV 4×27000=108000 NN、EMA panel2×12600=25200 NN，真实两CP／双状态 noquery合成 smoke4 NN，共133204次样本前向；以实际 counter 为准。原训练／预测成本引用父报告，不重新计费；学生 panel 只读旧 logits不执行新前向。CP、V IQ、旧 logits、NN 和墙钟成本分开记录。

最小发布流程：一次聚焦协议负例、一次真实 CP noquery smoke（PASS立即运行）、一次 P0/P1 审查、一个实际代码提交、不可覆盖 run root、普通账户 N607 preflight、单 archive SHA 比较／compile、一次 PID／CWD／argv／GPU／日志增长核实。仅本 run 越权输入、错父状态／默认头、非有限值、输出冲突、无法运行等系统技术错误触发保留现场退出；科学负值不当技术失败，不停其他健康任务，无自动重跑。准确命令／环境／路径／GPU规则／逐条件矩阵以 experiment.json 为唯一维护位置。

实现已完成：8个Python文件、2段实际remote脚本AST通过，41个协议负例和合成指标重算PASS（0CP／IQ／NN／更新）；唯一P0/P1审查PASS。资源阈值已按登记对齐12000MB并在本次审查中定点读回。真实父smoke与源读出仍待launcher执行，不将本地检查写成远端结果。输出为源域 logits（远端保留）、summary／CM／F1／最差RX／RXdayTX／配对错误／成本与报告，不含正式预测或分数。

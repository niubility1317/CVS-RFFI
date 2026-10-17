# F＋跨 RX 同 TX：六步 L 重排匹配 E20

状态：ANALYZED／VERIFIED，FAIL_FIXED_SIX_STEP_L_ORDER_CLOSED。两条新 scratch E20 均完成固定 final20／4440 更新及源 RX1 七视图，全部 17,640 个硬判决独立复算。正式 target 未访问；这项实验研究训练组织，不是 IQ A_g，也没有 A_s 训练。完整联合目标 ACTIVE。

本轮未满足预登记源判据，关闭这个固定六步重排方案。不改块长、分配、L 私有 seed、SupCon/CE、RX、模型 seed 或预算救跑；不把统计改善当识别提升。

## 固定数据、模型与唯一差异

fit RX3/4/6/8、day1/2/3，L5040／U45360（TX 真值隐藏）／V21600；held source RX1 原 L1260 在 final20 冻结后用默认头评估。不是完整 L6300，也不是 L_train5580/L_dev720。该源折已有历史暴露，不声称独立盲测。复用既有 VALIDATED_ONCE 物理角色，训练无父权重。

两条新行共用六步 L 预取、私有 L sampler/loader、配额计算、concat/gather；每块 768 个原始观察和全程频次完全相同。O0 交付原顺序，O1 平衡块内 TX/RX 组成。保持 F／SIG_TIME／真实 joint160／统一 Sinc／默认 cosine 头正常联合训练；NoExplicitCFO 和必要内部逐包补偿保留。原四对 RDrop、浅中层跨 RX 同 TX SupCon 与 13L＋7U／oldgate／ownEMA 不变。U 加载器不重排、不进入 L 缓冲、不解析 TX。没有新损失、部署分支、诊断头或 IQ 变换。

共同原 base raw.75＋ordinary.25，额外 raw 对 .25，总 pair 权重 1.25；每物理 L 四对／8 次训练前向，两次 RDrop 复用同一已生成 IQ。旧 ordinary 实际 PA、CFO、时移、增益、AWGN/multipath 等开关仍作为共有历史配方，如实记录，不声称已验证标签保持。practical 只作本轮源评估，传播/噪声/接收质量/补偿/SNR 未弱化，没有修改正式 received IQ。

输入读取口径：原 full-source cache 会反序列化 L6300/V27000，既有 fold.prepare 在模型创建前物化 RX1 L1260 的只读评估缓存及 144 个未训练 fit anchor。RX1 不进入 student/teacher 训练、源 V、校准或额外机制；仅冻结后调用默认头。冻结后评分不等于此前未读 RX1 IQ。

## 七视图身份结果

|视图|O0 accuracy %|O1 accuracy %|差值 pp|O0 Macro-F1 %|O1 Macro-F1 %|修复/新错|
|---|---:|---:|---:|---:|---:|---:|
|clean|80.238095|78.412698|-1.825397|80.034893|78.022230|14/37|
|high suburban|39.206349|38.809524|-0.396825|27.954998|28.034404|22/27|
|mid suburban|39.920635|39.365079|-0.555556|29.231095|28.937840|21/28|
|low suburban|35.634921|36.111111|+0.476190|27.263883|27.590250|39/33|
|high urban|38.015873|38.015873|+0.000000|27.728251|28.159750|26/26|
|mid urban|32.301587|32.222222|-0.079365|26.213825|26.118801|30/31|
|low urban|27.936508|27.380952|-0.555556|24.223685|23.560196|30/37|

每视图相同 1260 个物理包、六 TX × 三 day × 每格 70；完整 TX、day、18 个 TX/day 格及配对身份变化见本地 source_strata.csv／analysis_summary.json，没有筛除困难包。

固定判据：clean accuracy ≥＋1 pp，clean Macro-F1 不下降，每个 practical accuracy ≥−0.5 pp，seen V ≥−0.5 pp；每行原 final base L objective ≤.5 且完整执行要求通过。结果：
- clean：FAIL
- clean_F1：FAIL
- each_practical：FAIL
- seen_V：PASS
- O0/execution：PASS
- O1/execution：PASS

seen V 差值 -0.009259 pp；clean Macro-F1 差值 -2.012662 pp。

## Suburban 绝对表现与自身 clean 差距

|场景|O0 accuracy %|O0 clean 减 suburban pp|O1 accuracy %|O1 clean 减 suburban pp|
|---|---:|---:|---:|---:|
|high suburban|39.206349|41.031746|38.809524|39.603175|
|mid suburban|39.920635|40.317460|39.365079|39.047619|
|low suburban|35.634921|44.603175|36.111111|42.301587|

每场景与 clean 相差不超过 2 pp 是联合研究的探索目标，不是本轮额外晋级阈值。没有通过降低 clean 宣称 suburban 追平。

Urban 的共同 clean 正确子集只作描述，不作新筛选或因果证明：

|视图|共同 clean 正确数|修复|新错|净修复|
|---|---:|---:|---:|---:|
|high urban|974|20|18|+2|
|mid urban|974|22|19|+3|
|low urban|974|24|26|-2|

## 完整训练、输入和成本复算

每行全部 4440 步、740 块物理多重集及原预取/实际交付顺序、568320 次 L 观察／5040 个唯一 L、396368 次 U 观察／45360 个唯一 U 元数据均已复算。原 13 轮无 U／7 轮 U、EMA 4440 次更新、全部 13320 个 SupCon mask term 按实际交付标签核对。U TX 解析 0。共有原 L 顺序一致：True；未修改规则的 U 元数据序列一致：True。

|项目|O0|O1|
|---|---:|---:|
|实际 L 前向包数|4546560|4546560|
|其中额外 raw 前向|1136640|1136640|
|实际总 NN 包数|5823318|5823318|
|训练秒数|1529.1971209049225|1457.1814765930176|
|final native 首次 ordinary CE|0.28347173467412723|0.2816681462163861|
|final 完整 base L objective|0.3214085390707394|0.31453304643835034|
|final 额外 raw CE＋KL|0.09779607672412116|0.0969627822103264|
|完整 L 公式最大复算误差|4.76837158203125e-07|3.2782554626464844e-07|
|重排位置数|0|567571|
|缓冲/gather CPU 秒|16.261185598094016|16.327588016400114|
|最大逻辑 L 缓冲包数|768|768|

每行四类原计数（student train/eval、teacher train/eval）及 held 评估完整保留；教师 deepcopy 继承的初始化计数 2 已扣除，真实 student 初始化 2 次保留。共 40 新 model-epoch／8880 更新，父训练及复用训练均 0；本次额外 dispatcher real-CP no-query smoke 2 NN，合计实际 11646638 NN。源 held 默认头每行 8820 次，其中 practical 7560 次；As 训练 0、附加 teacher 0、channel 训练生成 0。

全部 20 轮原 CSV／JSONL／compact 曲线相符；全部 step 标量有限或显式 null，native 不适用的非有限占位单独列出。完整 stdout 检查无 Traceback/OOM/非有限跳步，原警告保留。完整 base＝.5(raw CE＋KL)＋.25(ordinary CE＋KL)＋.25(s-slot raw CE＋KL)＋原 SupCon；总 L＝base＋.25(extra raw CE＋KL)，不按视图数稀释。native train_loss_tx_labeled 是首次 ordinary CE，不能冒充完整 L；held 硬判决没有 logits，源 CE 仅运行时报告，不称独立复算。U 门控选择数不作识别正确率。

重排会改变原 ordinary 增强及随机子网络抽样与样本的对应关系。即使识别提升，也只能先归因于整套批次组织；不能仅凭正对统计把因果归给 SupCon，也不把训练组织当 IQ A_g。批次统计是机制描述：

|统计全步均值|O0 actual|O1 actual|
|---|---:|---:|
|valid_anchors|128.000000|128.000000|
|total_anchors|128.000000|128.000000|
|positive_min|9.420946|13.202252|
|positive_max|23.104730|18.877252|
|positive_mean|15.873842|15.975123|
|positive_variance|13.178644|2.376925|
|TX_count_range|11.452477|4.750901|
|RX_count_range|11.446622|4.969144|
|missing_TX_RX_cells|0.088288|0.000000|
|missing_TX_RX_day_cells|11.757207|8.740090|
|batch_TX_RX_mutual_information_nats|0.060341|0.010226|

## 产物、执行与边界

实际训练代码/config commit：9059b58502f207331bf9643d9d1399d308a2c2da。离线读者 commit 另见 Git；未复制进运行中的发布目录。发布前 26 项配置/协议反例、206 状态张量与默认 6×160 头一致；合成 NN6／0 更新。实际 N607 CPU 原迭代器合成检查 PASS／0 NN，独立一次 P0/P1 PASS。离线读者 24 项反例 PASS／0 NN/真实 IQ/CP/更新。

ordinary N607 已有 CVS-RFFI 环境，CWD=PROJECT/releases/cvs_F_LbatchOrder_sourceRX1_e20_20261017_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_batch_order17/runner.py，worker 只允许登记 O0/O1 row。owner codex/root/F-LbatchOrder-E20-20261017；GPU4/5，遵守每 GPU 所有 owner 总进程≤3。唯一不可变输出 PROJECT/runs/20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01。单 archive 传输比较、remote compile、真实匹配 CP no-query smoke、原 PID/CWD/argv/GPU/日志增长 VERIFIED，原 worker 自然完成，不停止或重启健康任务。

系统技术失败停止规则：角色/来源/预算/非有限/输出冲突/启动/非法预测失败保存产物、取消未启动行，健康 worker 继续，无自动 retry。科学判定等待两行固定预算和全部源预测；失败按上述固定方案关闭，不改参数救跑。

本地产物：local_artifacts/F_L_batch_order_E20_20261017/final；analysis_summary.json、analysis_O0/O1.json、training_curves.csv、batch_order_curves.csv、source_strata.csv、全部原 metadata/stdout/七视图 predictions，以及 figures/ 的四张 PNG/PDF。Matplotlib 使用已有 ordinary N607 CPU 环境，仅传元数据，新增 NN／更新／target 0；四张 PNG 已实际查看，数值和布局核对一致。

未证明：有效 IQ A_g、稳定跨 seed 的 A_s 收益、同契约 B00/B10/B01/B11 交互、正式跨 RX clean＋六 practical 与 low urban >60%、每个 suburban 接近 clean。不能把单 seed 局部收益、loss 下降或批次统计改善计为联合目标完成。完整目标保持 ACTIVE。

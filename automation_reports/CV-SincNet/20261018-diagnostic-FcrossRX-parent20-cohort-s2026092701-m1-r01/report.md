# P66 实际 As20 学生父模型的源域正确集合读出

目的：补齐用户要求的实际父模型 clean 正确集合上的 satellite 保持率。当前分析的 NoAs30 集合是参照集合；旧 P38 E20 student 缓存虽然源面板 IDs 同序，却来自旧 clock-factorial 父，不能替代当前 P43 As20。该诊断不作 P66 或 E50 晋级门槛。

唯一 row AS20_MODEL-s2026092701，固定 P43 run 20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01/B01-s2026092701/stage20/final_ssdg.pth 的 model 键；epoch20/4440更新、scratch/final_only、完整 L6300/U56700hidden/V27000、RX1/3/4/6/8、day1/2/3、split392005。使用既有来源核验及 VALIDATED_ONCE 数据，不重验数据、不新增hash链。源父来源不变，不加载 EMA 作为父学生。

冻结同一默认 cosine30 分类头和900722参数/206状态键模型，先2个零输入无query真实CP smoke，再读取原合法源V1800 clean和固定2draw×3scene full condition0 received IQ。样本每个TX×RX×day20个。无U标签、无正式目标IQ或truth，无新信道生成，无训练/头拟合/optimizer更新。源IQ仍为25MHz、2×256、原归一化；full缓存中的物理配置不变。

预测主循环复用已验证的 P66 qualification 接口，输出 ids/TX/RX/day/clean/satellite。NN预算为2零输入＋1800clean＋10800satellite＝12602，单列诊断新增成本；历史As20训练共享，不另计E20。所有合法有标签源V保持可见，不能把这项source保持率写成formal保持率，也不能把两draw当两个模型seed。

固定父clean.argmax==TX集合后，对当前行同序source预测计算各draw/scene保持率与固定cos30间隔；可按RX/day及已有物理元数据分层。该集合固定后不根据当前行重定义。不额外拟合诊断头。

环境：普通非特权 N607、既有CVS-RFFI Python、fullFP32/CPU2，pinP43 runtime提交19aedc3604fed9c193237dd92366164178af0983。CWD PROJECT/releases/cvs_FcrossRX_parent20_cohort_20261018_r01，命令 CVS-RFFI/python -u experiments/cvs_parent_cohort66/runner.py。不可变根 PROJECT/runs/20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01，日志 PROJECT/runs/20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01.log。选择空闲GPU，最多2个总进程含其他owner、free≥12000MB；不停止或修改健康P66/其他任务。

停止规则：输入权限/来源不符、非有限logits、冻结状态变化、输出冲突或技术异常，仅停止该读出并保留产物，不自动重跑。正常完成后保留真实CP smoke、predictions.npz、complete及source正确集合附录。这里只读source，按用户明确源诊断范围不启动正式测试。代码提交、必要协议负例、一次真实CP smoke、一次限定P0/P1评审、单release archive比较/编译、一次preflight及独立PID/CWD/cmdline/GPU/日志/产物核验按最小流程执行；当前健康E30不等待此诊断。

# PAMF 第三阶段：固定主度量的原型余弦读出

## 预登记（2026-10-30）

Run `20261030-phase3-star-metric-cosine-s2026092701-r01`，实际执行code `abb2490048d7fcceffbffe814d15f880b79522f4`。只固定父metric-embedding r01/codeefb82a4177e9b59533b7e71408ded6b63fc8733a的compact_oe_structure主g；不按前轮控制较高换g、不新增训练/源IQ或拟合协方差。方法与精确接口见`docs/PAMF_PHASE3_STAR_METRIC_COSINE_PLAN_20261030.md`、`configs/pamf_phase3_star_metric_cosine_20261030.json`。

父训练OE排斥用relu(max原型余弦)^2，本轮检验相同变量的读出。L-only class_means[6,160]仅FP64逐行单位化为prototypes；query父FP32 metric_unit转FP64再单位化，固定einsum+clip[-1,1]计算六余弦。两法predicted_cosine/maximum_cosine，唯一主maximum_cosine。身份仍原F0 defaultlogits argmax，不替换为cosargmax，不按query真TX/RX/场景/K或query统计路由。原MD为独立scorer复用父保存margin列1/六τ的外部匹配参考，不再校准或伪称新四文件可计算的第三分数。

实际gcheckpoint2条synthetic-unit noquery smoke先于known读取；prototypes/同4tensor g先V固定，known27000两法各六τ仅由合法V标签按42组trueTX宏CK90/同输入原F0closed−AK2pp complete ties规则校准；全部12τ/部署包/cal/known写全后才打开novel3150公有特征。producer不读取父score文件、不调用父full_prediction，不把评分派生结果引入模型选择。完整pred后独立CPU scorer先验证来源/模型复制/原defaulthead/父非score数组/protos来自Lmeans/τ/CPU64 g参考原1e-5/2e-6与newraw/margin1e-12，再单次连接source novel私有真值。原MD与新cos在同次truth读取评估和配对，不重算任何未知阈值。

### 数据、祖先与监督

父 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-s2026092701-r01/rows/compact_oe_structure/predict`，同G10MIX01 P1（星地received经冻结DURA）既有原F0/metric单位特征与logits。42584祖先物理记录L6300/OE6134/V27000/novel3150仍同一固定观测，本轮只V/NV公有features和L聚合；不读L/OE逐样本feature、不消耗新的真实IQ/F/G生产前向。source RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）及2021-03-08/15/23已见；旧六TX14-10/14-7/20-15/20-19/6-15/8-20，V各4500；historicalnovel19-15/19-16/7-1各1050，非新RX/TX独立盲测。

原F0为Phase1三视图SSL E50/11100/232993/default6head、sourcecontract392005/L6300/Us56700/V27000/10%标签，DURA v1.0 R69/266951；g为20704参数/1000固定步，祖先用L旧TX/RX与OE成员监督（不是OE真TX/K），L-only均值在V前拟合。本轮无g/头/F/G或optimizer更新、无额外L标签读取；V标签只新τ，不能称整个系统零源监督。targetUQ/正式RX24-18/隐藏Us/未知真K读取0。新source结果不替代跨未见RX、无标签DA或未来匿名注册；AN/HOW=N/A、fullgoalACTIVE。

### 执行与输出

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-cosine-s2026092701-r01-release`，唯一owner `codex-phase3-star-metric-cosine-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_metric_cosine start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-cosine-s2026092701-r01-release/runtime.json`；CPU-only/CUDA_VISIBLE_DEVICES空，新增GPU0、foreign不干预。
- 独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-cosine-s2026092701-r01`、rootconfig.json仅4producer字段（不含truth）、logs、predict和score；source和score为不同subprocess，source成功且全部prediction固定才启动score。private truth `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz` 仅score argv，producer无此字段。新不可覆盖root，启动UNKNOWN先读回同PID/root，不重复提交；技术stop仅权限/来源/非有限/输出冲突/无法合法prediction，保留partial，性能差不停止/换主/重跑。
- 预期launch/pipeline/config/logs；predict smoke/known/NV/raw2/margin2/cal12τ/complete；部署exact4files model.pt（复制原g四FP32tensor）/prototypes.npz FP64[6,160]/thresholds.npz FP64[2,6]/complete.json（sourcefree/无逐样本行）；独立score results/complete及observations。weights/features/pred/log只留private，不提交Git。

NN部署F/G499944+g20704=520648不变；g权重payload82816B/prototypes7680B/12τ96B，容器开销另计。producer新增g2仅actualcheckpoint synthetic-unit smoke，真实g生产forward0/F/G/IQ0、两cosCPU读出30150行；独立CPU64 g30150参考和读出复核另计，均非B1部署时延。父主g训练1000步/284450生产gforward与原三臂总2000步/605350实验成本分别保留，原F6synthetic与父F30152/历史G恢复成本不抹去。

主报trueTX宏AK/RU/H及CK/原F0损、known42/unknown整体5RX6scene12组/逐TXRXscene/AUROC/OSCR/CI、主−predcos及主−原MD的同ID配对。只有固定maximum_cosine known42及unknown12均过90%才可另记小target证伪；control较好不能替主，无H独立90门槛，不扩seed或把source当完整目标。

验证：11 passed, 612 warnings, 35 subtests passed in 5.19s，2AST/实际runtime PASS；唯一独立直接P0P1 PASS（其定向与主suite重叠不加计）。覆盖pure数学/身份不换/父score隔离/θ与原Φ复制/源原型/零训练/实际checkpoint smoke和冻结顺序/独立truth边界/外部MD不校准及主失败不换control。一次N607 preflight核固定主g完整与合法句柄、no parent score/newIQ/target/GPU读取。单code/config commit/一个archiveSHA比较/remotecompile、actualgcheckpoint首步smoke、PID/CWD/cmd/log增长和完整预测后独立评分，不加重复父trace/fullsuite/数据验证或报告审批。

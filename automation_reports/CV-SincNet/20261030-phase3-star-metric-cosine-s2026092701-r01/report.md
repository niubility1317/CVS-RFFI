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

## 固定主度量的余弦读出实际完成（2026-10-30）

VERIFIED / ANALYZED_SOURCE_ONLY。唯一主 maximum_cosine 的 known42 保护通过、unknown 总体/五源 RX/六场景12组90%未通过，NO_PROMOTION。主、控制与旧MD的身份均为同一原F0默认头，旧MD只复用父公开margin及原阈值，不是新部署包支持的第三分数。

| 固定分数 | 宏 A_K | 宏 C_K | 宏 R_U | H-score | 原 F0 closed−A_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 MD 外部参考 | 90.433333% | 98.870370% | 12.476190% | 21.927290% | 0.548148 pp |
| predicted_cosine | 90.385185% | 98.259259% | 31.746032% | 46.988330% | 0.596296 pp |
| maximum_cosine | 90.425926% | 98.348148% | 27.365079% | 42.015307% | 0.555556 pp |

主较旧MD正确接收99得/101失（净−2），未知拒对709得/240失（净+469），RU+14.888889pp、H+20.088017pp；改用训练变量的读出有所改善，但远低于90%。不能用较高的 predicted_cosine 控制替换预先固定的主方法。原F0同输入P1闭集90.981481%，本轮主损失0.555556pp，AUROC80.671049%、OSCR74.911718%。

固定 structured main g 的四FP32 tensor与L-only原型祖先不变，无新模型/optimizer更新、不读L/OE逐样本特征、不新增真实IQ/F/G前向。祖先仍L6300/OE6134/V27000/novel3150共42584原物理记录，五已见sourceRX/三日/G10MIX01P1，历史novel三TX各1050。V只校准两法12τ，全部模型/原型/τ/cal/known冻结后才novel，公有CPU64重建后独立单次truthlast。无targetUQ/正式RX24-18访问，无新目标适应/发现/未来注册；AN/HOW=N/A、fullgoalACTIVE。

实际gcheckpoint两syntheticunit smokePASS。生产新g2仅smoke、真实g/F/G0；CPU余弦读出30150行0.112731s，独立CPU64 g参考30150及读出重建1.525449s，均非部署B1端到端时延。已验证父g/Lmeans复制，unit最大差known6.92904e-7/novel5.32717e-7，原rtol1e-5/atol2e-6不变，FP64cosine/margin容差1e-12。部署exact4files，同g20704参数82816B、原型7680B、12τ96B，source逐样本行0，F/G/g神经参数合520648，祖先训练/缓存成本保留。

执行code abb2490048d7fcceffbffe814d15f880b79522f4、预登记b67a2ed4a6e13e383df2e38dbe6b1a012c75b647，11tests35subPASS5.19s/2AST/实际runtime与一次直接P0P1 PASS。一次preflight/archiveSHA22c06cd3cf83129cdefbc3a9720b92974b807e8714b8eafc930ab94ba2fdb476（363files/data0）比较/compile/runtime VERIFIED。首次独立观察pipeline3222125 live且CWD/argv/CUDA空正确；短时predict3222222和score3222229只由pipeline jobs记录，未抓到独立实时占用，不重跑补遥测。终态live空，predict991B/score1247B/pipeline528B完整日志，全部prediction/cal/deploy/score complete。pipeline9.822261s，predict5.178522s、score4.641403s含进程加载/验证，非readout逐包时延。CPU-only新增GPU0、foreign干预0。

完整逐RX/场景/TX、配对、成本与数据权限见 docs/PAMF_PHASE3_STAR_METRIC_COSINE_RESULTS_20261030.md。本轮结束固定g换读出的证伪，不沿historical novel真值扫阈值/选择控制，下一独立方法先检查开放表征训练和无标签适应的实际作用。当前星地跨RX CK/RU≥90%、AK距原同输入F0最多2pp、未来匿名注册目标保持。

# R54：同一压缩拓扑的源响应与输出敏感度通道筛选

状态：LOCAL_VERIFIED，尚未发布。性能优先；没有新正式结果，不把通道评分、源IQ误差、容差或减参当作识别优化完成。

固定两行 activation_energy、projection_sensitivity，均完整273765参数、context185615；沿用R48最初完整两宽度矩阵的273765拓扑，现有350521宽度保留全通道时筛选恒等，原R49同宽控制保留，不为恒等初始化重复训练。仅用源响应替换原权重能量通道排序，不更换网络族、宽度分配、3prior/6CG或推理分支。唯一直接初始化/冻结IQ教师为用户指定R46 prune_refine350521 E20/4940，方法bbb8daf817de050b958d217b70d4c33f4aefe341；不加载R48/R49/R53学生，不根据其正式分数重排或另选父。固定同宽权重剪枝控制复用R53 prune_decay（方法df36c3d3da4e026b747776b65b58d4921dc1a71e），不重复训练。方案参数在读取R53正式分数之前固定；R53健康训练/预测保持原计划。

源校准在CPU2共同执行一次：seed2026102154从原6300L_s/56700U_s各抽64条，使用原R07 cachedreceived01/20，共128物理样本、256次冻结G输入；0TX标签/F/V/target/optimizer更新。clean仅用于每例输出能量归一化，均方下限1e-4。activation_energy按encoder输出均方评分；projection_sensitivity使用独立Rademacher输出投影（seed2026102155、draw2、除sqrt512和sqrt(clean能量)），求每通道整体gain的方向导数，再平均其平方。通道gain导数为sum_t activation乘output-projection对activation的梯度，不把平方的逐时间均值误写成整体通道gain敏感度。该局部近似不保证识别或GroupNorm裁剪后的全局误差单调。全部模型参数冻结、无参数梯度或状态更新，父逐张量核对不变。

每个GN4组按固定配额、stable排序选通道，保留组内原索引顺序。prior0/1/2先汇总平均评分，再使用完全相同的通道map，避免共享长卷积索引错位；各stage独立bias/GroupNorm保留。coarse spectral、temporal与action分别评分，global head同步切输入列，decoder skip使用原上游宽度偏移。校准分数/map和行位置留私有run，推理只加载独立最终state，不再读取源样本/校准文件或F。

两行继续原pairedclean IQNMSE＋0.05三尺度相对功率STFT＋线性1→0.1冻结父三stage IQNMSE，所有coarse/action/prior联合。原model2026092701、split392005、data2026102001、augmentation2026101912、formal2027，原L/U完整source、R07 epoch01..20、global256micro256/directfloat32buffer、ResetAdamW、LR2e-4暖247/cos至2e-5、WD1e-4/clip1、FP32/TF32关闭。每行新增E20/4940、L126000/U1134000、1260000学生和IQ教师输入；继承每组件80轮后累计100、组件和300、串行140pass，F50成本另计。另单列共用校准256输入（L128/U128次视图曝光），不伪装为0数据初始化、不计成梯度训练epoch或重复计入两行训练。模型/校准均0TX/F/V/target训练。

原source READY/EXACT_MATCH、original7/63/30物理ID及R46/R44/R38/R17/R07/R04来源复用，parent/source契约在实际初始化核对。属于有clean模拟配对的TX无标签源域自监督，不能称无clean盲学习。原F50联合训练默认头仅在训练后评测；无另训头。历史工程基准已暴露，不声称新的独立盲测或逐细节TX真实性。

资源仅物理GPU2/3，R53＋R54本任务合计最多两条GPU worker、每卡一条。activation_energy在GPU3等待R53 retain_anchor完整正式完成且实际所属PID退出；projection_sensitivity在GPU2等待排队中的R53 prune_anchor完整完成且该卡所有R53所属PID退出，不能只等第一条prune_decay。等待进程使用CPU、不预占GPU；不读取前驱性能分数、不停止或热改任何健康进程。普通账户、CPU2、同一共享score.lock最多一个scorer；新不可覆盖release/run/output，唯一owner codex/root/R46-source-prune。

37项聚焦检查与CLIhelp通过；一次限P0/P1审查以实际review.json为准。首步真实checkpoint源校准及R46/F50/2Ls noquery smoke，PASS直接按卡排队训练；一次N607 preflight、单归档两端SHA、compile和实际PID/CWD/argv/GPU/日志核实。技术失败保留受影响输出，无自动重试/热改/回退，低准确率不触发停止。额外设计审批/签名/成员哈希等不属于最小流程，REJECTED_EXTRA_GATE，不等待。

每行固定最终E20/4940，自动source1800×7、同卡G与G+F/B1/B32三轮成本、正式168000×7预测；每行七景预测固定后独立truth-last评分并逐行报告，源域无提升也不取消测试。主要指标为逐场景accuracy/Macro-F1、相对同宽weight-only控制和R17/R46、逐TX/RX/day及repair/damage；时延/吞吐/显存/参数次级。成本为64源V、同步预加载FP32、不含传输/前处理/loader，3G＋1F共享驻留不是单模型显存；ConvLinearMAC不含FFT/CG/norm/激活/F。原R17每景1pp只是退化容差；lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp与更强输入TX细节真实性目标继续ACTIVE。精确commit、命令、环境/CWD、路径和矩阵见experiment.json。

方法commit `554c52baf75ab8da2618a5480851b74271018103`。精确命令、环境/CWD、路径与矩阵见[登记](experiment.json)。

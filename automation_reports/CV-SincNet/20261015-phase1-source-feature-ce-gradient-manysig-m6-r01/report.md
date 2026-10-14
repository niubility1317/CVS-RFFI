# 固定SIG_TIME同L强图身份/特征梯度：最小预登记

状态LOCAL_VERIFIED / NOT_LAUNCHED。代码 da1aafdde1931fcef67ecd509e079cfa371ec166。本次只诊断，不新增训练机制、分类头或目标访问；真实结果尚不存在。

问题依据：此前实际训练的feature effective梯度来自U256，而main梯度来自另一批L128，接近零的余弦不能识别同包身份梯度关系。既有源V错误组feature cosine未稳定更低，也不能据此设可靠度门槛。现在固定现有全部预定义模型，用合法L同强图直接测量；不依据已暴露目标成绩挑父模型、扫系数或改健康矩阵。

矩阵run 20261015-phase1-source-feature-ce-gradient-manysig-m6-r01：3固定seed2026092701/02/03×SIG_TIME ZERO、FEATURE_MT，共6个已有scratch50 final50学生及各自真实ownEMA。父run 20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01 和 20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01，全部保留，不选最高seed/row。加载前复用source_proof核对真实数据角色、物理ID、33+17/11100、终轮选择和继承来源；实际EMA须epoch50/11100。源split392005、L6300/U56700隐藏TX/V27000、RX1/3/4/6/8、day1/2/3及TX集合/NoExplicitCFO不变，VALIDATED_ONCE数据不重验。

测量固定：每模型完整L6300、batch256（末批156），shuffle2026101501、strong/noise/dropout2026101502，各模型同样视图随机数；原strong noise .015与学生train dropout，teacher eval/stopgrad。拒绝BatchNorm，前后逐值比较学生/teacher全部参数buffer，无optimizer/EMA更新，不累积.grad。train是只读测量中的模式，不是新增训练；default cosine头保持原checkpoint数值，用于预测和CE对fθ的导数，头拟合/更新0。仅t1PW3456、t2PW6912、SIG local projection576作autograd.grad对象，实际joint160不变。

同物理L强图trueCE smoothing.01，与原unit joint160 half squared L2 consistency及固定诊断系数.1的导数比较。两臂都评估.1×feature的诊断方向，ZERO的原训练系数仍为0；这不是保存训练更新的回放。教师/学生判断对错的4组同时保存trueCE和feature梯度，全部使用全batch分母；分组贡献必须相加恢复总梯度。空组保留0向量、undefined cosine，不补采、重排或强行归因为错误教师。保存批次和按batch包数加权的完整L全局向量统计、默认头源accuracy、RX/day描述；未拟合任何诊断分类器。

数据访问只有L前向/梯度。共用load_source会物化V27000缓存后丢弃，V前向/梯度0，U返回None且不读IQ/标签/ID；query IQ/truth/统计访问0。所有模型前后参数buffer不变检查。不能把seen L同强图代理解释为真实U准确率、训练累计AdamW、目标DG、unseen TX或Phase2能力。即使有负余弦，也不直接调整SSL权重；若错误组为空或跨seed方向不一致，明确问题仍不可识别。无科学分数停止/扩展规则，无后续自动系数/阈值扫描。

执行：普通N607既有CVS-RFFI/fullFP32，一进程串行6模型、Torch/OMP/MKL/OpenBLAS各2线程，CUDA_VISIBLE_DEVICES为空，不分配GPU、不触碰健康任务。CWD PROJECT/releases/cvs_source_feature_ce_gradient_20261015_r01，不可变run PROJECT/runs/20261015-phase1-source-feature-ce-gradient-manysig-m6-r01；入口python -u -m experiments.cvs_source_feature_ce_gradient.audit；本地experiments.cvs_source_feature_ce_gradient.publish --output local_artifacts/source_feature_ce_gradient_delivery_r01。精确环境/命令/路径/row配置见experiment.json。

预算：每模型0新增epochs、0optimizer/head更新，6300 L观察/12600 forward包次/50 forward batches/25 gradient batches/75层记录/825向量；6模型37800源样本观察（同6300物理样本重复测6模型，不是6套独立数据）、75600 forward包次/300 forward batches/150 gradient batches/450层记录/4950向量。原训练成本每骨干仍50轮，不冒充新训练或新目标预测。

局部13协议负例、samegraph统一分母/空组、实际SIG_TIME合成joint160/三个梯度对象、六上下文、无参数buffer变化、生成CPU入口compile均PASS；唯一独立P0/P1 PASS（真实数据/远端访问0）。发布只做一次普通账户preflight、一次archive本地/远端SHA比较、remotecompile、首步真实checkpoint无query smoke PASS即继续，无签名、receipt、重复审查或审批。

停止规则：任何本run source/provenance/role/query/defaulthead/非有限/状态变化/输出冲突，保存partial并写failure，不自动重试；核对实际owner进程，不停止/热改其他任务。后续核对PID/CWD/argv/CPU环境/loggrowth，源诊断不做目标测试或truth评分。预期artifact为各row resolved_config、L_source_gradient_vectors.pt、analysis、source_gradient_recount、completion，以及全run analysis/completion；原始PT留远端，Git交聚合source_feature_ce_gradient_analysis.md、数据/图与登记。完毕另进程重开保存向量复算，报告完整实际计数和不可识别项。

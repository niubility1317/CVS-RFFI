# 特征一致性：固定02/03条件扩展

状态 LOCAL_VERIFIED / NOT_LAUNCHED，代码 68b3143dbdfcde27c80997f970121a23e34e5f2e。原首seed 20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01 / execution6b23a5b033ee063dea97d6783d5634e220a1d93d 正健康训练。尚未宣称source资格通过、新目标结果或完整三seed性能。仅首seed按原冻结源规则PASS后才发布，不改变配方或按目标筛选。

新run 20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01，固定modelseed2026092702/03，各OLD_GATE_COS50历史复用、SIGZERO、FEATURE_MT三行，共6row、4新scratch50/11100、2历史复用；保留首01不重训。固定SIG112→实际joint160/fθ195041、默认cosine6×160 scale30原联合训练，无新头/模块或另训分类器。完整机制、依据、风险与原矩阵见首runreport，当前不重复设计。

原33label+17pseudo、batch_neighbor/ownEMA.999/cos/λUCE.16/λentropy.01/L smoothing.01、L128/U256/RANDOM顺序RNG不变；L6300/U56700隐藏TX/V27000、sourceRX/day/TX/physicalIDs/split392005与NoExplicitCFO/内部补偿不变。V无梯度、EMA或拟合，无新卫星训练。

单位joint160同U包teacherweak stopgrad/studentstrong的half squared L2 mean保持，eps1e-8，全部实际U、尾批124；权重.1×clip((epoch−33)/5,0,1)，ZERO同图权重0。原UCE对象和头主梯度保持，aux无classW梯度，无新前向或随机数，既有nativeinterface不变。未修改任何健康run archive。仅增加薄的seed封装，并让原审计接受design/build参数，默认调用含义保持。

## 局部证据与源资格

四上下文实际构建和无query合成前向PASS，每seed两臂初态逐值相同；原生args只差seed/run_id/candidate_id/output_dir，完整feature方法字典与首seed一致。剩余seed observer绑定实际验证原UCE对象保持、t1/t2/SIGprojection非零feature梯度、Waux梯度None；11项source资格拒绝条件PASS。没有真实数据/query/checkpoint访问，优化器/头更新0。复用原核心唯一独立P0/P1审查，不重复审未变算法。

发布仅取原source资格，不读目标scores：原两臂终LCE≤.5、候选−ZERO合法L720zL2≥1e-6、34/50三个对象各有非零effective梯度、精确11100updates/963900U/666诊断批和feature边界；最后5轮包加权U特征discrepancy候选/ZERO≤.95。source决定回收/提交先于新numeric target摘要。FAIL保留负结果，不扫权重、门槛、轮次或重跑，不添加额外封印/审批。

## 成本、执行与收尾

本run计划4新200model-epochs/44400更新＋2历史100epochs/22200更新。每新模型原生和feature两套观测合计3996层记录/11988向量；4新15984层记录/47952向量；原17×56700 U每模型、额外模型前向0，诊断反传/存储成本如实报告。完整三seed计划6新300epochs66600更新＋3历史150epochs33300更新，9模型1512000默认decisions/162组指标，不把计划计数冒充已完成。原始产物留远端，Git交聚合结果。

普通N607既有CVS-RFFI/fullFP32/每worker2CPUthreads，唯一owner codex/root/feature-mean-teacher-seeds-20261015；CWD PROJECT/releases/cvs_feature_mean_teacher_seeds_20261015_r01，不可变run PROJECT/runs/20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01；各row独占source/predictions/scores/logs。入口 python -u -m experiments.cvs_feature_mean_teacher_seeds.dispatch，本地发布 experiments.cvs_feature_mean_teacher_seeds.publish --output local_artifacts/feature_mean_teacher_seeds_delivery_r01；全路径/命令/矩阵见experiment.json。

sourcePASS才做标准普通账户preflight、一次archive本地/远端SHA比较、remotecompile和真实源checkpoint无query smoke，PASS自动运行。所有owner共享锁每GPU≤3、分配前空闲≥12000MiB，不改健康他人任务。每row final50自动clean168000默认头预测→prediction固定→独立truth-last评分→报告登记/Git；保留首seed产物合并固定三seed配对、SD、正差数、MacroF1、最差RX/day、纠错/引入错误、完整源曲线和fθ梯度/导出边界。

停止规则为原所属run系统技术失败：数据/query/来源/头权限、非有限或跳过更新、预算/路径冲突、无法执行/合法预测/scorer错误；核对所属PID/CWD/argv后保留partial、无自动重试，不按分数停止、选父、变更或选择性重跑，不触碰其他owner。不声明Phase2/newTX/卫星部署、未验证收敛或84%达成。

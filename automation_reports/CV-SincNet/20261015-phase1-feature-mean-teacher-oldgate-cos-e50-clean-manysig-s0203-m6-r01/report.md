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


## 首seed源决定已回收

原source资格PASS：LCE ZERO=0.247665886、FEATURE_MT=0.248785446；最后5轮U特征discrepancy比=0.821487695≤.95；实际L720 z差=2.058244944，实际t1/t2/SIG梯度与精确预算满足。first三row已TRAINED_PREDICTED_SCORED。当前只读source及计量状态，未回收新目标准确率。决定使用当前已准备的固定02/03原配方，无参数/候选重排；本决定随本次提交冻结后发布，source资格不代表DG收益。详情first_source_decision.json。


## 技术恢复：发布依赖，未启动训练

R01 dispatcher import FAILED：score引用发布包没有包含的旧SPSseeds汇总包。独立post-state确认PID已退出、workers0、run根目录未创建，因此没有新模型训练/更新或健康任务重启。保留R01 release/archive/log。修复提交 729c01e878a0df790c1056d00d1b547f709a57be 改用已发布的相同AST helper，并在实际release preflight导入dispatcher消费依赖链；针对该P1窄复审PASS。

同一run/rows/科学配置/源决定保持，恢复使用不可变 cvs_feature_mean_teacher_seeds_20261015_r02 和本地local_artifacts/feature_mean_teacher_seeds_delivery_r02。本机等价函数AST、dispatcher import与archivepaths检查PASS；真实release仍按原compile/no-query smoke后发布。技术修复发生在目标读回后，但只修启动依赖，不调整方法、系数、种子或预算；原source决定 ec8ea03f7810cb3eb1bcb8108cdb60b4053a59c9 保持。


## R02 恢复发布已验证

状态 RUNNING / VERIFIED，execution ab2c3127e14b48dea5bc2e064ce42c0be7cfdeeb。一次 archive SHA 对比、remote compile、真实checkpoint无query smoke PASS；4新worker的PID/CWD/argv/GPU及后续loggrowth均独立核实，已各完成4至5轮，2历史row复用；所有owner每GPU进程数最大2≤3，失败0。R01失败包保留，未改变科学配置或重启健康任务。原始运行细节留本地/远端，Git聚合证据见 release_R02_launch_evidence.json。


## 完整三seed闭环结论

ANALYZED/VERIFIED：FEATURE_MT82.4359%±1.5874pp，匹配SIG82.6806%±1.7906pp，配对−.2446±.2101个百分点，0/3为正，净−1233/504000；源特征损失三seed都降低但没有DG收益。完整450源epoch/1512000默认决定/162指标、23976实际层批梯度独立复核；头fit0。源决定先提交后读新目标，未择优/扫权重；R01 import失败与R02恢复历史完整保留。详细结果见final_analysis.md/results.json及complete_three_seed_curves_and_results.png。

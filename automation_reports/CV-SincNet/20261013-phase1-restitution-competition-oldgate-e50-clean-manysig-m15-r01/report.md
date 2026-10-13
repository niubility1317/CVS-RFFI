# 归一化信息恢复与分支竞争：最小预登记

目标仍为 Phase1 clean 三种子均值 84%。主实验 RESTORE_COMP；本轮改善特征提取，历史旧伪标签门控＋EMA＋cos 是固定对照。当前首阶段从用户六路线方案执行归一化信息恢复和分支竞争，增加同容量普通残差对照。其他四路线和 Phase2 迁移验证尚未执行。

五组 BASE_REF、ORDINARY、RESTORE、COMP、RESTORE_COMP，种子 2026092701/02/03。BASE_REF 复用此前对应种子 OLD_GATE_COS50 固定第50轮，源角色和来源已核实，原生预测复用；其余四组从零训练，不继承对照或上一轮得分较高变体。每骨干固定50轮，33有标签＋17伪标签，222步/轮，共11100步；历史EMA.999、batch_neighbor/window2/minconfidence.8、strongagreement、rx_day quantile.86、lambda_u.16、lambda_ent.01、AdamW.0002、weight_decay.0001、clip5、cos最小1e-6、L128/U256、label_smoothing.01，关闭卫星训练，NoExplicitCFO保留内部补偿。

RESTORE 在 t1 GN8×72 前统计并保留组内相对均值、标准差；恢复值经1224参数投影和一个有界alpha参数加入原GN输出，alpha初值.05、界限±.2。ORDINARY 使用归一化后的张量，经同参数数目投影残差。COMP 在第1111步起，以.1概率均匀遮蔽真实时序、频域、PA或参考响应中的一组；只作用真实L_s学生前向。老师、U_s、源V和部署使用完整路径。

所有骨干固定后，普通6×160/scale30 cosine头只使用L_s特征训练10轮，AdamW lr.001、wd.0001、batch256，末轮固定，不用V选模。另做五个留接收机身份头和一个5类接收机信息头，每头10轮，所有拟合仅L_s，V只诊断；每row共70个head epoch，骨干零更新。源特征缓存只在Phase1输出目录，禁止作为Phase2输入。普通cosine是诊断，不改主分类头结构或训练目标。

首种子五组源诊断全部固定后才扩展另两种子：各最终L CE≤.5；相对BASE_REF，源V普通TX准确率下降≤1个百分点，leaveRX平均下降≤1个百分点、最差下降≤2个百分点。不排序候选，不把非劣检验称为已有效。每row训练/复用→源诊断→clean双读出预测→fresh CPU truth-last评分。两个预测都固定后才连接truth，全部负结果保留。已有target基准暴露，不称新独立盲测；target结果不改变训练、阈值、候选或重跑。

主native三种子平均≥84%且冻结普通cosine对BASE_REF平均配对差>0，才达到本轮Phase1性能及特征证据要求。Phase2旧新身份联合识别没有完成前，不宣称支持登记改善或目标整体完成。

代码/config身份为CURRENT_COMMIT，完整配置见experiment.json。环境 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_feature_restitution_20261013_r01；命令 python -u -m experiments.cvs_feature_restitution.dispatch。唯一owner codex/root/restitution-competition-20261013；每GPU最多3个总实验进程，计入其他owner，至少12000MB空闲，健康旧任务不变。

新增骨干600轮＋既有参考150轮，分别计费。输出ROOT /home/szu2070436088/2510044040/CV-SincNet/runs/20261013-phase1-restitution-competition-oldgate-e50-clean-manysig-m15-r01，不覆盖；本地传输证据位于local_artifacts/cvs_feature_restitution_20261013_r01。预期每row源冻结、probe摘要/固定W、clean两个prediction、36条指标记录、completion；全矩阵540条记录、汇总配对差和源诊断。

协议/来源不匹配、输出碰撞、隐标签暴露、target拟合、非有限数值、不能合法预测/评分时停止新增派发，保留产物并让健康已启动row完成。源科学界限不满足时不扩种子，首种子照常完成双读出评分。无自动重跑、无目标分数停止规则。发布第一步：最小负向检查和同契约真实旧门控checkpoint无query smoke通过后直接启动。

冻结源V还进行四种确定性单组路径撤除，保持同一full-feature普通cosine W，比较部分特征和完整z。该诊断不用于扩展门槛、target屏蔽或额外头拟合；不把单组撤除冒充单一路径隔离。

唯一独立P0/P1审查 PASS（/root/numpy_boundary_review），未发现阻断合法运行的问题。代码/config实际commit在archive的release_commit.txt、各source resolved_config和dispatcher execution_commit中记录，无自引用commit或报告hash门槛。

发布与启动 VERIFIED。执行commit d1c274d1276a080dcdd66e1d7588a63dbe547e59；单archive SHA比较、remote compile和同契约真实checkpoint noquery smoke PASS。实际进程CWD/参数/GPU、日志增长与总容量核实。基线首seed已完成源冻结诊断、双读出prediction和fresh scorer；新增首seed四骨干已完成5/3/4/4轮（快照），无失败。首seed完整源门槛仍未判定，尚无完整三seed结果或84%结论；主Agent未读取新target指标参与训练。源诊断聚合保存source_diagnostics_available.json，完整结果仍待。

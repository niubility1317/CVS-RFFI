# F＋跨 RX：单侧边缘可见性源身份检验

状态：PLANNED，已登记，尚未启动。当前只检验一个有界IQ候选的源fit身份保持，不训练模型、不拟合额外头、不访问target。唯一逐row矩阵和完整模型/数据配置见[experiment.json](experiment.json)。

候选及原始文献、负面证据与适用限制见[研究说明](../../../analysis/cvs_edge_visibility_source_20261017.md)。本次采用一个端部16点的实数窗：外8点归零、内8点半余弦上升到1，右端镜像。没有中段遮挡、时移、重采样、额外噪声/传播、PA/量化/剪裁或全局功率恢复；归零不是幅值阈值剪裁。窗口不读TX/RX/day或置信度，参考[80:160]及支持外输入精确不变，不要求所有内部特征相同。它是部分观测候选，不声称替换RX硬件或本来就标签保持。

原基线F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、NoExplicitCFO/defaultcos30保持。两份既有own scratchE20 final20 B00只作冻结源诊断，原训练完整来源、final-only/4440更新、实际L5040/U45360/V21600与RX1held角色核对后，在各自原不可变后端独立CPU进程加载。两个模型原训练都遵守隐藏U/EMA规则；本次无U前向不表示父训练未用U。未来增强训练仍从零E20，不把诊断CP当初始化。

统计和默认头前向只用fitL5040，RX3/4/6/8、day1/2/3、6TX、72格各70包。既有载体反序列化L6300、未用源V和历史特征；RX1原L1260先排除，U仅角色ID、无IQ或TX标签。全部包参与，不删低质量或难包。不是完整L6300或L_train5580/L_dev720训练契约，不触发新数据验证。

每模型raw/left/right各5040，预计30240源NN＋一次匹配真实CP的2包合成no-query smoke=30242NN，20160边缘视图生成。0新epoch/更新/教师/头拟合。两个参照历史成本单独40 modelE/8880更新/9373356NN，不重复训练。

原条件在源IQ访问前固定：两模型、每个端部均要求accuracy与macro-F1相对raw≥−.5pp，每RX/TX/day accuracy≥−1pp，raw正确集合保留率≥.99；相对IQ RMS变化median≥.01、max≤.35，reference/支持外精确不变且总功率无增加（浮点相对容差1e-6）。输入变化/接线项通过不能替代身份判据。科学FAIL也完成预定双模型全部三视图，再关闭固定16/8zero8taper；不缩长度、换边、平滑形状、阈值、RX或seed救跑。

PASS仅支持另行登记两组匹配E20 source-heldRX反证：共同raw.75＋ordinary.25/SupCon完整保留，额外.25 raw或edge同IQ RDrop对，8L前向，原U/默认头联合训练；clean至少+1pp、macroF1不降、各practical/seenV下降≤.5pp。该训练尚未实现/登记/启动。默认头fitguard本身没有DG或As证据，不因通过自动访问target。

唯一owner codex/root/F-edge-visibility-source-20261017。普通N607现有CVS-RFFI Python，CPU/OMP/MKL/OpenBLAS2，CUDA_VISIBLE_DEVICES为空，GPU none，两worker顺序；不干预健康旧任务。CWD PROJECT/releases/cvs_F_edge_visibility_fitL_20261017_r01；命令 CVS-RFFI/bin/python -u -m experiments.cvs_edge_visibility17.dispatch。各row精确runner命令在experiment.json。私有runtime保留实际Python/账户信息，不入Git。

发布入口 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_edge_visibility17.publish，CWD当前managed codex/cvs-iq-timing-e20-20261016工作树。代码身份使用本次Git commit，在release_commit.txt/resolved_config记录；不新增配置/报告hash。独占输出 PROJECT/runs/20261017-diagnostic-F-crossRX-edgevisibility-fitL-manysig-m2-r01；日志为该root的各seed.stdout.log和发布根source_edge_visibility.stdout.log。一个N607预检、一份archive及一次传输SHA比较、一个remote compile、launch后PID/CWD/argv/CUDA/log或自然终态独立读回。smoke为首个worker的第一步，PASS立即继续。

技术异常、错输入/来源/角色、输出碰撞、非有限或越界保存failure/partial后停止，不自动重试或修改旧作业。本地19个非法输入/配置与7科学判断反例通过、120合成IQ窗/0真实IQ/NN/更新；参考/支持外、相位/正增益等变、原输入/RNG保持已核实。旧本地Torch有NumPy桥警告，本检查未调用该桥、没有装包。一次独立P0/P1审查PASS，未发现直接阻断缺陷。

产物包括root request/active/launch/result/completion、每seed实际配置/来源/原默认头完整logits与hard判决/全部72格、私有物理ID和逐包窗统计、完整日志、独立全30240判决与分层/保留率/CE/成本复算、图和同row报告/登记。IQ、权重、原始判决、样本ID、机器状态不提交Git，只保存聚合元数据。无target预测或truth-last target评分。

完整联合目标仍ACTIVE：尚缺有效Ag、稳定As、同完整契约四格/交互项、跨seed与正式clean＋六practical；lowurban>60%及suburban距自身clean≤2pp探索目标没有完成。另一对话的practical正负噪声路线不属于本候选。

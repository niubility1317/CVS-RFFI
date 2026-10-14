# SIG_TIME 描述子在末端 Dropout 前后取点：固定首种子

依据用户报告三·路线三和固定源诊断 20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01，L720同投影的feature area噪声/clean RMS31.9940%、time area21.7255%、实际added residual160为25.2901%；全eval的t2.pool与t2相同。隔离回放尚非训练消融或性能因果证据。本轮只检验取点，不依据目标分数选权重、参数、seed或停止规则；旧moment/area六个健康模型不变。

seed2026092701三行：OLD_GATE_COS50历史对照只读复用；SIG_TIME post与SIG_TIME_PRE pre各独立scratch50/11100。所有初始参数和初始eval joint160相同；主干t2 Dropout仍.1，只将descriptor hook从t2移到t2.pool。SIG_TIME净增量＋feature/time area公式、单6D Conv/tanh、4×8窗、112→160/LN/gain初始.05上限.2不变，pool18976参数、fθ195041、W960。未叠加COV/AREA/层级/更多卷积/SiLU/Sinc/PAIR/U权重。

旧batch_neighbor＋自身EMA.999＋cos，33label＋17pseudo；AdamW2e−4→1e−6/wd1e−4、lambdaU.16/entropy.01/smoothing.01，final50固定。L6300/U56700/V27000、10%标注、split392005/原RX/day/TX/物理ID保持，U真值隐藏，V健康不选模/梯度；NoExplicitCFO及内部补偿保持，无新增星地训练。默认cosine6×160scale30与骨干联合训练，不冻结/另训/重置头；native fitted RX/day/channel probes关闭。实际joint160导出保留新pool及norm/gain/reference，W不影响z；teacher/U路径一致。

新增100model-epochs/22200更新，历史baseline50单列复用；无父checkpoint/teacher继承。post控制重新从零用于同路径单因素比较，不使用已暴露目标高分权重，不把不同历史随机训练轨迹合并。单seed不能达84%；只在固定源条件PASS后原样扩预定义02/03。

源资格先于target聚合读入：两模型finalLCE≤.5、所有新增参数真实更新、实际export精确/W独立、初始全参数及初始L720 eval功能相同、PRE−POST最终L720 L2≥1e−6；SOURCE PASS才扩，FAIL保留。该规则确认源健康及机制执行，不能预言目标收益。V不排名，target不得回流调参/父权重/种子或选择性重跑。

实现 087b3f61abb4444f05e7c22f4af6f631865188df。本地2真实native合成IQ路径PASS：训练时实际pre/post取点、主干Dropout.1、所有新增grad/数值更新、默认W联合梯度、精确导出/W独立/teacher、hooks清除、全初态参数/eval函数同、旧默认取点不变、4协议负例。新两条件源汇总/真实初始eval检查/build callback195041及remote3/2/504000统计compile PASS。一次独立新增直接P0/P1审查PASS无发现，只读审查不访问真实数据/远端、不重复旧核心。

入口python -u -m experiments.cvs_read_position50.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_read_position50_20261014_r01；不可变run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s01-m3-r01。共享锁计全部owner每GPU≤3/free≥12000MiB，实时分配并保护健康任务。Git代码/config已提交推送，一个archive一次local/remoteSHA比较/remotecompile/旧seed01真实CP零IQ无查询smoke，PASS立即继续；复用VALIDATED_ONCE和原数据handle，不新增数据重验/hash/seal/审批。

Stop：本run角色/来源/预算/非有限/defaulthead/export/物理ID/prediction/truthlast系统技术失败停止新增派发，保留健康进程和产物，无自动重试；不按target早停/改配方/换seed/重启。每row final冻结即自动clean168000默认头逐包prediction，固定后独立truth-last；3行504000 decisions/54分组指标。保留配对差/正差数、MacroF1、最差RX/day、修正/引入错误、完整50epoch源曲线/U接受/时间、真实feature更新/零fit收尾。只有完整3预定义seed均值≥84才达标，单seed/其他读出/source不能替代。

预期release preflight/submit；run launch/queue/每行initial-final-frozen/resolved/args/roles/50epoch/11100step/source_feature_witness/default prediction/complete/scores/metrics/completion/source_features/source_qualification/analysis。数据、weights、rawlogs及大型prediction留远端，聚合/实现/config/报告Git上传并核remoteOID。只测clean旧6TX，不声称新TX/Phase2/卫星部署。当前已登记实现，尚未发布。

实际发布 VERIFIED：执行 b4d55dba50c8601017604a2a0eb0d037f8d835e8，dispatcher PID 2132655。两新worker PID/CWD/argv/GPU核实，实际resolved读t2/t2.pool且native fits关闭，epoch2日志分别增长10341/10356bytes。真实旧CP no-query smoke、一个archive比较及remotecompile PASS，failure=null；目标结果尚无，详见 launch_evidence.json。

登记文字纠正：复制模板中的4row/3new成本与多描述子representation改为实际3row/2new、100model-epochs/22200新增更新、504000决策/54指标；原文字在 historical_registration_text 保留。实际row configs、3行代码与健康远端release一直为正确3/2预算，未热改/重跑/改变实验。

# 签名内部有界身份排序：首定义种子最小消融

本轮依据全 L6300 四视图源诊断 20261014-phase1-source-signature-order-manysig-s01-m1-r01：原始 SIG112 的4个固定时间步坐标严格为1，平方范数占比均值17.6511%；RAW/DYNAMIC未满足率29.0008%/26.0841%、错误次序19097/18859、活跃物理覆盖99.8889%/99.6508%。两视图梯度都进入真实SIG local_projection和t1/t2。代码与源证据支持检验监督位置/常量归一化，几何值不证明DG收益，不用于调参或候选目标排行。

固定4row：旧门控+cos历史；SIG_TIME ZERO、SIG_RAW_ORDER、SIG_DYNAMIC_ORDER三新模型均scratch50、11100更新、33label+17pseudo、原 batch_neighbor gate+ownEMA+cos；默认分类头普通cosine6×160按原流程联合训练，不单训/重训/替换头。实际部署单路SIG112和joint160、参数195041保持。RAW在实际SIG112辅助距离上计算排序；DYNAMIC只在辅助距离去4固定增量成108，所有时间面积和部署112保留。固定m=.1、coeff=.1、5warmup+10ramp，不叠加SupCon/PAIR/层级/难度权重，不扫描权重/日程。

辅助只用L_s6300真实身份，同日跨RX正样本，异TX负样本与正样本同RX/day。70个固定L90cohort、shuffle2026210712，按native每4步在eval且保留梯度时额外前向一次，恢复mode/RNG；每模型2750extra/247500额外包，三臂含ZERO曝光完全相同，无新增参数/优化步。U56700真TX保持隐藏，旧门控处理；V27000健康用途，无辅助梯度。原物理ID/split392005/RX1/3/4/6/8/day1/2/3/旧6TX/NoExplicitCFO内部补偿不变，无卫星或Phase2。

实际epoch1/17/33/50全部55extra保存mainL128CE、RAW、DYNAMIC、effective真实参数梯度，目标为t1PW3456/t2PW6912/实际SIGlocalprojection576。前向曝光、未满足关系与包出现次数、clock范数、梯度保存复算，不把包出现次数视为唯一物理覆盖、不同L批CE比率视为同包或累计AdamW贡献。fθ导出边界和实际数值更新沿用既有source-only witness，W置零不影响z且删除旧W不丢特征归一化/融合。

首定义seed2026092701（4row/3新150model-epochs33300updates＋1历史50model-epochs11100updates）是最小证伪。固定源资格：三新finalLCE<=.5、实际更新/export/W独立和初态一致、两候选−ZERO L720zL2>=1e-6、两候选t1/t2/SIG投影17/33/50有效梯度非零、2750extra/220梯度批。两候选均PASS才允许原样扩预定义02/03，两者一起扩，不依据任何目标数字挑选。源资格先写后读新target数字；不按源关系比例选优。首seed结果不能替代3seed84%目标。

逐row50终轮冻结后自动clean defaulthead168000预测，prediction固定后独立truthlast评分，首矩阵672000decisions/72组整体RX/day/TX指标。负结果全保留，无目标反馈或选择性重跑。完整budget/data/config/源继承与历史baseline provenance、环境CWD、每row命令/输出/日志和stoprule均在experiment.json。

复现入口：python -u -m experiments.cvs_signature_order50.dispatch；普通N607 CVS-RFFI既有环境、release cvs_signature_order50_20261014_r01。每GPU包含所有owner最多3进程且free>=12000MiB；不修改其他健康任务。新run与输出root不可覆盖；技术失败保留产物、不盲重启。发布为单归档一次local/remoteSHA、remotecompile、真实CP无querysmoke后直接启动。

预期：50epoch/11100step、initial/final/source_frozen/resolved/actualargs、2750extra stream、4期实际gradPT与JSON、sourcewitness/sourcequalification、每rowfixeddefaultprediction/independentmetrics、analysis/completion。原始IQ/CP/预测/日志/gradPT留既定远端，只上传实现/登记/必要聚合结果。当前尚未发布，暂无本轮目标准确率。

代码/config提交 f6dfdb587c505b5720469d4a4a101b8df8b321f4；8negative/3实际native检查及唯一P0/P1审查PASS。所有模块本地编译PASS；真实checkpoint无querysmoke为远端发布第一步，PASS后直接继续。

远端单归档SHA比较/compile/真实SIG checkpoint无query smoke PASS，执行 2658815a2fe5ef931f24558a6c6c61146433fd9c；三实际worker同PID/CWD/argv/GPU及loggrowth VERIFIED，目前3/2/2epoch、GPU1/2/0、无failure。历史旧gatecos默认预测复用已完成。新模型继续原固定50，无本轮目标准确率，不宣称84。

用户最新重心：固定现有SIG_TIME，只研发泛化与半监督训练机制，分类头默认联合训练。当前两个候选均仅trainingloss变化。真实第17轮55extra调用，RAW/DYNAMIC在t1/t2/SIG投影均55/55非零，ZERO均0；辅助/mainCE范数比RAW1.8586%/1.7759%/2.8269%，DYNAMIC2.0883%/1.9840%/3.1007%。这是读取保存训练artifact报告的复算状态，独立全PT重算随完整矩阵audit完成；不同L批梯度不作累计贡献解释。当前进程21/20/22epoch仍健康，未有目标性能。后续半监督先固定SIG_RANDOM/PAIR统一训练参照，不追加结构。见training_mechanism_e17.json。

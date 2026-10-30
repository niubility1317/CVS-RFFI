# DURA单阶段源损失梯度冲突诊断

- run_id：`20261030-diagnostic-DURA-joint-gradient-s2026092701-r01`
- group_id：`phase1-dura-source-gradient`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

FrozenfastLRsourcefinal;source128L+384U xepochs01/20,B32;weightedIQ/prior/spectral/action/noisegradientgeometry;0optimizer/F/TX/V/target

目的：在新单阶段完整source目标中区分拟合不足与目标相互竞争。固定fast LR源finalE20/4940，source commit a0acfbe0c237e38d0460a9f1ea29c46b22cf84a6；不从目标成绩选权重，不读取F/TX/V/target。原R07同128Ls＋384Us位置与epoch01/20 pairedclean/received/source模拟器参数，B32共32批，每批五个weighted autograd.grad，0optimizer或持久p.grad。

分量权重与当前训练严格相同：IQ1、prior0.1、谱0.05、source action0.05、logvariance0.001。报告all/coarse/estimator/prior四组中，每个辅助分量及其和相对IQ梯度的范数、点积、cos和范数比；零范数时cos明确为null。它是batch均值梯度几何，不是逐样本或目标身份因果分析。

一项聚焦几何检查通过，明确相反梯度cos−1/权重范数比2及零范数语义；一次独立P0/P1审查PASS。launcher首步实际两Ls checkpoint noquery smoke后直接诊断。权重/缓冲区逐位不变、p.grad未积累才完成；无新训练恢复器版本或分类prediction。本次CPU2/0GPU，不干预健康恒等初始化训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 完整源域结果

VERIFIED/ANALYZED_SOURCE_ONLY。实际checkpoint两Ls noquery smoke通过，CPU2/0GPU，PID3626846/CWD/argv/日志增长已核实，正常退出。512固定源位置（128Ls＋384Us）×epoch01/20、B32完整32批，耗时34.590秒。模型全部权重/缓冲区不变，p.grad未积累；0优化器/TX/F/V/target。以下从完整32批结果独立聚合；零范数cos为null。

|参数组|辅助项|有效cos批次|负cos批次|平均cos|范数比中位数|
|---|---|---:|---:|---:|---:|
|all|prior_nmse|32|1|0.627635|0.106315|
|all|spectral_error|32|0|0.512010|0.011965|
|all|action_probe_nmse|32|3|0.175827|0.076239|
|all|log_variance_mse|32|25|-0.077822|0.144868|
|all|combined_auxiliary|32|2|0.340449|0.233170|
|frozen.coarse.|prior_nmse|32|1|0.694672|0.122216|
|frozen.coarse.|spectral_error|32|1|0.502809|0.010770|
|frozen.coarse.|action_probe_nmse|32|6|0.298081|0.037225|
|frozen.coarse.|log_variance_mse|32|14|0.056898|0.045298|
|frozen.coarse.|combined_auxiliary|32|1|0.655791|0.167715|
|frozen.estimator.|prior_nmse|32|1|0.626608|0.087240|
|frozen.estimator.|spectral_error|32|0|0.531062|0.011608|
|frozen.estimator.|action_probe_nmse|32|4|0.250412|0.165704|
|frozen.estimator.|log_variance_mse|32|27|-0.193325|0.316012|
|frozen.estimator.|combined_auxiliary|32|11|0.086938|0.404764|
|prior.|prior_nmse|32|0|0.609394|0.084179|
|prior.|spectral_error|32|0|0.511607|0.013453|
|prior.|action_probe_nmse|0|0|null|0.000000|
|prior.|log_variance_mse|0|0|null|0.000000|
|prior.|combined_auxiliary|32|0|0.645446|0.090992|

方差辅助项相对IQ在全参数25/32批、estimator27/32批为负方向，其他分量整体更一致。该证据支持独立匹配地移除方差辅助校准损失，不能证明其必然改善正式识别，也不支持同时删除全部辅助监督。梯度为batch均值，非逐样本或目标因果分析。原始32批与source位置留在忽略的local_artifacts/remote原root，Git仅公开汇总。

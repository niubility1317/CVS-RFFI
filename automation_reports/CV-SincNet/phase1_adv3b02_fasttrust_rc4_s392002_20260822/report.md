# FastTrust-RC4风险校准四态伪标签实验报告

> `SUPERSEDED_DO_NOT_LAUNCH`：用户已将训练预算压缩为50/100epoch。本报告保留为200epoch设计记录，正式发布转入`phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822`。

## 当前状态

`LOCAL_VERIFIED`

run_id=`phase1_adv3b02_fasttrust_rc4_s392002_20260822`

本实验把优化主线从“继续增加星地增强”纠正为“提高无标签伪标签的正确信息密度与有效覆盖”。星地增强只保留为P7中H子集的附加强视图，不作为主体机制。

## 协议与合法性

- Phase1角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，source-only。
- `U_s`只读取IQ与receiver/day域信息，不读取TX真值，不构造proxy unknown，不更新开放集radius/tail。
- `V_cal`只拟合阶段冻结的temperature、correctness logistic与APS阈值；更新epoch固定为`E1/E41/E91/E161`。
- `V_select`仅用于后续source-side盲评分/选模；target clean与三个`leo_*_weak`只在最终checkpoint后独立评估，不反馈训练或选模。
- 固定identity fraction、soft/low quota fill、U prototype update、local/prototype伪标签证据、temporal gate、cross-receiver attraction与nuisance regression均不进入RC4路径。

## 方法与输入输出

输入为冻结`ADV3B02_CORE90_SOFT_E200`anchor、student EMA两种安全弱视图、`V_cal`标签与source receiver/day。correctness calibrator使用7个真值无关特征、class bias、domain bias、L2正则和5-fold cross-fitting，输出估计正确风险。APS风格候选集合采用class→domain→global分层回退。

每个`U_s`样本互斥路由为：H唯一类别hard CE；P候选集内teacher结构KL＋集合外负监督；N只做排除类监督；R无身份方向但保留domain/GRL与安全self-supervision。H/P/N全部按完整`U256`归一化，无选中样本均值放大；H采用class×receiver上限且无下限。

输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_rc4_s392002_20260822`

最终每行必须保存`final_ssdg.pth`、clean指标、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`逐场景指标与日志。

## 最小因果矩阵

|GPU|候选|新增伪标签能力|目的|
|---:|---|---|---|
|0|P0_NO_U_ID|无U identity；保留all-U domain/self|控制组|
|1|P1_NO_FILL_RAW_H|无校准、无anchor的raw H；不回填|验证删除50%fill|
|2|P2_CALIBRATED_H|temperature＋correctness calibration|验证置信度失准|
|3|P3_DUAL_TEACHER_H|anchor＋EMA双时序来源|验证teacher漂移|
|4|P4_PARTIAL|增加P候选内KL＋集合外负监督|利用中置信U|
|5|P5_NEGATIVE|增加N排除类监督|利用低置信U|
|6|P6_CLASS_RX_CAP|增加class×receiver cap|改善均衡与floor|
|7|P7_H_SAT_VIEW|只给H增加星地strong CE|隔离星地辅助增量|

所有行固定seed392002、E200、U batch256、相同初始化、相同source split、相同optimizer-step预算和labeled replay次数。

## 训练加速

- 两个EMA弱视图拼接为一次teacher前向；anchor单独一次冻结前向。
- H/P/N共用一次全U student strong前向；仅P7把H星地行拼接到同一次student调用。
- P/N路由和损失只做logit张量运算，不增加student遍历；P0不会构造星地U视图。
- correctness logistic使用小型Newton求解；只在4个预注册epoch刷新，不逐epoch拟合。
- 最后20epoch冻结阈值并把U identity总权重降至0.4，恢复labeled clean margin与开放世界几何。

## 本地验证与审查

- `test_fasttrust_rc4.py`及SAT-Anchor相邻测试：26项通过。
- `py_compile`：`muse_ssdg.py`与`train_ssdg.py`通过。
- 两个launcher的`bash -n`、矩阵JSON解析、`git diff --check`通过。
- 本地8行dry-run完整生成8条训练命令和clean＋三LEO评估命令。
- 独立P0/P1定点审查修复：N路由风险目标与候选覆盖不一致；低风险N权重可能恒为零；Windows dry-run末列CR残留。修复后未发现会导致协议越权、输出覆盖、无法启动或不能形成合法prediction的剩余P0/P1。

## 停止规则与声明边界

只因协议/query泄漏、错误split/seed/矩阵、输出覆盖、错误checkout、无prediction闭合、确定性重复异常或进程归属不清停止；不得因中期性能差停止。当前只有实现与本地验证证据，没有N607运行结果，不能声称RC4提高了伪标签精度或最终性能。

## N607发布前资源读回

2026-08-22 18:41 CST直连preflight为`VERIFIED`：项目根可见，8张RTX3090可见，home剩余约7.3TiB。GPU0、2–7各有1个现有训练compute app；GPU1有2个。dispatcher设置`RESOURCE_SLOT_LIMIT=2`：有槽位的7行立即启动，GPU1的P1只读轮询并在compute app少于2时启动；不干预SIDFFT96或已运行SAT-Anchor任务。

# 原骨干临时分类器重建：同RX与跨RX独立对照

- run_id：`20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01`
- group_id：`source-episodic-transfer-20261013`；类别：`cvs`；阶段：`phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原旧伪标签门控/EMA/cos保持；临时source ridge任务梯度回到真实joint160，分类器每episode丢弃。6新模型scratch50，3同seed baseline冻结复用，3预登记普通读出automaticclean评分

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

已完成 9 行、三读出 clean 评分及完整复核，结果见 [final_analysis.md](final_analysis.md) 和 [results.json](results.json)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

最小预登记

研究：用户路线六，临时分类器重建监督fθ。可微求解思想已有[Oxford原始论文](https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/)，本轮检验CVS源RX任务是否改善actualjoint160，不将视觉结果移植为RFFI收益。[冻结嵌入后重训分类器研究](https://arxiv.org/abs/2003.11539)提供诊断动机；实际收益以同row实验为准。

矩阵：LEGACY50三seed原E50冻结复用，原执行43bd20494aa470568a68553ce664dc76811b9707、仅反序列化修复9ea84c4d22076532dd613a28d1b2da3988591feb；SAME_RX/CROSS_RX各三seed从零E50×222=11100，32label+18pseudo。新骨干300modelepochs、历史基线150modelepochs分别披露，baseline权重不初始化新学生/教师。

机制：每4step从step4起一个L-only任务，2775episodes/model，6类各support2/query4，12/24包，每row额外99900L包。queryRX轮换1/3/4/6/8，各555episodes；跨RX support为环上后两个源RX每类各1包，同RX控制为同RX不交物理样本。dual solve的两侧梯度均回actualjoint160；ridge.1、scale10、eps1e-4、辅助权重.2固定，临时系数不持久化。source私有generator seed=modelseed+82201，episodeforward seed=modelseed+step+97701，CPU/CUDA主RNG保留。

旧配方实际保留：rx_day_quantile .86、tau .92至.97、batch_neighbor window2/minconf.8、strongagreement、EMA .999 versioned/no startupaverage、lambdaU .16、entropy .01、labelsmoot .01、ordinaryaug true、无satellite；AdamW LR.0002→1e-6 cosine50、WD.0001。分类器不改结构或margin。V仅health，无源性能门槛。native默认callbackNone，现有健康release不热改。

固定终轮读出：ORIGINAL_HEAD、FRESH_COSINE_ALL_L（无旧W、W6x160/scale30/无bias/Adam.01/WD0/20epoch×25batch=500Wupdates、seedmodelseed+60013）、COSINE_PROTO_ALL_L。featurefreeze后源L拟合所有读出，再读targetIQ；之后无queryfit。新头增加0个featureupdates，θ总预算仍50。baseline原头复用已完成recovery NPZ，不重复预测；baseline其余读出为固定新评估。每row三读出各168000clean，全部fixed后独立truth评分，54records/row、总486。84%主指标固定CROSS_RX ORIGINAL_HEAD，其他读出高分不替代；原头84%也不能单独完成特征迁移目标。

数据：原ManySig raw/post_sync/noeq、sourceRX/day/TX、split392005未变，L6300/U56700/V27000。newtraining保留HiddenU，sourceep仅同契约L，V不训练。全部6TX均seen，不能称newTX。target clean沿用source物理ID不交的VALIDATED_ONCE168000观测，不新验未变数据。已暴露基准，不声明独立盲测/Phase2/LEO。

代码身份：本报告同实际代码/config一次Gitcommit，publisher记录release_commit.txt；无额外design/sourceplan SHA。命令python -u -m experiments.cvs_episodic_transfer.dispatch，CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_episodic_feature_20261013_r01，N607普通账号/CVS-RFFI Python。owner codex/root/episodic-feature-20261013；共享capacity lock，每GPU总实验进程≤3含所有owner，free≥12000MB。输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01不可覆盖，row路径/config见experiment.json。

停止：数据/查询越权、source/scratch/clock/cadence不符、非有限loss/gradient、预算/episodecount或预测评分不能闭环等技术失败停止新调度，健康active行完成，保留产物，不影响其他owner；不以性能停止/重排/选择性重跑。step日志每步验证真实episodecadence与12/24packets，避免空机制完成50轮。

focused checks：数学primal/dual与数值梯度、置换/逐query独立、秩亏/zero/autocast及19negatives PASS；actualbackbone辅助W梯度None、backbone梯度finite/nonzero、主RNG保留、2775任务RX555each、实际native总loss插入前backward与默认None不变、11negatives PASS；真实nativeparser/resolved 50/32/18/旧EMA/gates PASS；predict/score36metadata/CM/goal＋10NPZ/preflight/nohook negatives PASS。release首步真实checkpoint/合法sourceL no-query smoke，PASS后直接启动。一次独立下一 run P0/P1 review PASS，证据见后续预登记记录。

独立P0/P1 review：本新候选一次PASS，无直接阻断缺陷。补充本地cold import曾超时；有界2threads配置独立进程读回exit0，已到既定remote preparation文件读取边界（本机不存在，未读dataset），无import错误。不能据此声称remote launch完成，实际release预检仍须PASS。原生默认None路径、预测nohook、source frozen readout三列评分已审查。

完成读回：code d001edf7c03c6c0831679ff37cfbaf1f9ed80eb7，新骨干 6×50，复用基线 3×50；完整 486 聚合记录独立重计 VERIFIED，目标 84% 未达到。实际发布与资源证据保存本地 delivery，原始产物保留 N607；全部进程自然结束。

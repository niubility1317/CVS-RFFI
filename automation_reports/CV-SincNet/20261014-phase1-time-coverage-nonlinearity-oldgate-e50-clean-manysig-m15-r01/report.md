# 时间覆盖 × 分支内非线性：最小预登记

状态：ANALYZED（15row完整预测、独立评分、全源日志、机制及特征导出核验闭合）。唯一维护矩阵见 experiment.json；实际配置见 configs，不复制多份矩阵。

## 科学问题与范围

当前 TIME_MS50 的三个线性深度分支经1×1融合和投影，可在 GN 之前折叠为受约束的长度9卷积。它同时扩大覆盖、增加时间滤波基、增加投影旁路。已有三seed默认头clean均值81.6905%，相对 OLD_GATE_COS50 的80.5268%增加1.1637个百分点，仅2/3 seed正向；不能归因于多尺度本身。当前用户附件要求优先拆解覆盖与分支非线性，分类头使用默认头联合训练，禁止单独训练任何诊断头。本报告覆盖此前目标文件中可另训source diagnostic head的许可；本次所有拟合仅原骨干＋默认分类头joint训练。

只做Phase1固定clean ManySig研究，不进入Phase2、不训练卫星增强、不称卫星部署或独立盲测。84%是LONG_SILU完整三seed默认头clean均值目标，不保证达到，不以目标结果选模/调参/父权重/停止/重跑。

## 候选与控制

四条件为 COMPACT_LINEAR [(3,1),(5,1),(5,1)]、LONG_LINEAR [(3,1),(5,1),(5,2)]、COMPACT_SILU 前者各分支后SiLU、LONG_SILU 后者各分支后SiLU。只替换t1/t2/t3，保留相同fuse/skip/GN/ReLU/pool/dropout。每新条件时序块参数81672，比原时序路径新增60672；identity含W参数237697。PA/频域/参考响应/Sinc/NoExplicitCFO和 ResidualPhysicalHead6×160 scale30 完全保留。没有attention、门控新增、分支GN或辅助loss。

新四条件同seed对应参数张量完全一致；新块初始化使用CPU fork_rng和固定seed+48213，恢复native随机流。SiLU/膨胀率无额外随机抽样；Dropout调用次数及形状一致。初始函数有意不同，不声明同初始函数/相同GPU轨迹。compact t3长度5仍比原长度3长，故compact只代表相对long较短；不声称完全隔离原基线覆盖。原 TIME_MS50 安装消耗native RNG，因此不作为新四条件匹配row复用；独立scratch训练 LONG_LINEAR，历史结果单列。

BASE_REF是原3seed固定final50合法来源仅对照，不用于新模型初始化，所有新模型从零。15row（4新条件×3＋3复用基线），先seed2026092701的5row；全部新条件final训练L CE≤0.5、默认source V总体不低于固定BASE_REF超过1个百分点、最差源RX不低于超过2个百分点，再启动seed02/03。筛选不排序、不访问target指标。失败保留第一seed所有结果，不扩seed、不选择性重跑。固定四条件配对差与交互效应全部报告；LONG_SILU为预定义primary。

## 数据权限与checkpoint来源

数据契约沿用已核验 source_contract：L6300/U56700(hiddenTX)/V27000，源RX索引1/3/4/6/8×day0/1/2，全部6TX；固定6300/56700/27000物理ID逐role EXACT_MATCH。训练池标注率10%。target fixed clean 168000物理样本与所有源角色不交。U loader −1 labels，过滤所有TX metadata；U接受分组只读RX/day和预测类别。V只读eval，无optimizer/EMA/统计状态更新。

旧BASE_REF provenance复用原完整审计和实际source_proof：无checkpoint祖先，scratch50、33+17、11100 optimizer/EMA更新、final_only、同物理角色、无target训练或选模。不得加载未知/不匹配/污染来源。真实checkpoint noquery smoke用 OLD_GATE_COS50 seed01，仅验证strict加载，不继承。历史源码没有native独立probe禁用，新运行明确禁用native闭合的RX/day/channel拟合。

## 执行与预算

代码版本：release_commit.txt内一个Git执行commit，无自引用SHA、无member hashes或额外seal。发布entrypoint：experiments.cvs_time_factorial50.dispatch。远端Python：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_time_factorial50_20261014_r01。精确命令及每row路径在experiment.json。

每新骨干固定50epoch＝33标签＋17旧batch_neighbor门控半监督，222steps/epoch，11100更新。AdamW lr0.0002、原weight decay/cos日程/EMA/强一致门控保持实际历史args；不另拟合头。12新模型600epoch/133200更新；3基线历史150epoch/33300更新单独披露。输出root：/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-time-coverage-nonlinearity-oldgate-e50-clean-manysig-m15-r01，不可覆盖。launch owner codex/root/time-factorial-20261014。

GPU资源按当前用户规则每GPU最多3个总实验进程（含其他owner），启动需≥12000MB free，共用现有capacity锁。旧健康任务不停止、重启、热修改。Sinc量纲修复、本轮外layer allocation/hierarchical分支保持独立后续假设。

## 真实路径和只读诊断

本地检查PASS：4条件同权重同native RNG、81672时序参数、实际L调用计数、optimizer包含全部新参数、各层参数实际数值更新且joint160改变、W正常joint梯度、修改/删除W后pure特征不变、teacher deepcopy和strictcheckpoint一致。纯特征导出只保留 fθ，包含cls_head的base_norm/pa_norm/gain，不保留类别W；实际z等于encoder.features。

历史TIME_MS50现有三seed补等效preGN核、偏移能量、每输入通道核矩阵SVD/有效秩与真实权重fold检查；不读IQ/target/truth。新row冻结后在L做无拟合几何，在source V逐t1/t2/t3撤去一个分支，记录Δz、默认头正确类margin、完整组合纠正/引入错误。核的有效秩不是joint160特征秩，核能量不是TX因果贡献；冻结撤除可能破坏共同适应，不能代替scratch消融。

epoch32/33/34/40/50记录source V/RX指标、Sinc/t1/t2/t3/PA/频域/default head实际参数变化和固定L32的z变化。全伪标签阶段逐epoch记录RX/day U数量、接受数量、预测与接受类别分布、EMA弱视图和student强视图一致度；不读取U TX、不改变原mask、不做额外head拟合，统计实际U扫描56700/epoch。

## 自动收尾与停止规则

每row训练固定final50→来源核验/freeze→source只读诊断→clean默认头prediction全部168000固定→fresh CPU独立truth-last scorer→18同row指标和原始confusion重算。所有15row完成才称完整三seed结果，总2520000 decisions/270metricrecords，其中504000历史默认decisions复用。target结果不回流其他健康训练、source screen或任何选择。

输入/role/checkpoint越界、U真值暴露、碰撞、nonfinite、无法产生合法prediction/score等技术失败停止新增dispatch，保留所有产物与健康active row，不自动重试。科学source gate失败记SOURCE_SCREEN_NOT_PASSED，不伪装技术成功或完整矩阵。成功并不要求84%；目标未达到如实保留负结果。

## 最小验证与发布状态

本地focused检查保存local_artifacts/cvs_time_factorial50_20261014_r01/local_checks.json；原sequence默认接口回归PASS。原运行immutable release未改。下一步只需本候选唯一P0/P1审查、Git提交/push/OID核验、一份release archive SHA传输比较、一次ordinary N607 preflight/remote compile/真实checkpoint noquery smoke，随后立即发布与PID/CWD/argv/GPU/log增长独立读回。禁止旧contract中的额外审批/设计SHA/报告审查/hash链门禁（REJECTED_EXTRA_GATE，不实施）。

历史权重直接检查 VERIFIED：三个 TIME_MS50 seed、三个t层，数值核秩全部4，平均核有效秩3.0682至3.1574，远偏移(|τ|>2)能量占3.4190%至4.7954%，真实训练权重float64折叠最大误差3.33e−15；汇总见historical_time_kernel_summary.json。它证明保留多滤波基表达，不证明TX因果效用。每个BASE_REF worker另做同seed旧TIME的冻结source V尺度撤除，输出 historical_time_diagnostics，独立于新四条件匹配统计。

独立审查发现并修复两个P1：U extra真实tuple格式需n._meta_from_extra；CPU-only fork_rng中torch.manual_seed会改CUDA RNG。现用CPU默认generator独立seed，实际CPU/CUDA状态均逐arm核验一致。真实native.move_batch/threshold/temporal和隐藏TX拒绝检查PASS。无改健康旧release。

唯一独立P0/P1审查及一次问题针对性复查PASS：两项直接缺陷已修复；旧TIME冻结诊断和source-stage闭合无新P0/P1。审查结束，不增加重复审查。

## 已发布的独立证据

实际执行commit 7b1e2fcbe1bf5d05f24bc91f86aa7d4b46d3ee95，本地HEAD与upstream OID独立一致。一次ordinary N607 direct preflight、一次archive local/remote SHA比较、remote compile及真实同契约checkpoint noquery smoke均PASS。不可变release cvs_time_factorial50_20261014_r01 已发布。独立readback VERIFIED（原始证据保留local_artifacts/cvs_time_factorial50_20261014_r01/readback.json和startup_readback.json），dispatcher PID/CWD/argv匹配，四新worker正常、第1至2epoch、日志增长、默认head_only=true、native_fitted_probes_disabled=true、新增参数60672，GPU总进程max3。BASE_REF seed01已source诊断、旧TIME同seed冻结source撤除、复用固定native预测与truth-last评分闭合。其他10row按source gate等待，未称完整15row或84%达标。

2026-10-14最新独立readback VERIFIED：首seed四条件source固定终轮及预测/评分闭合，通过LCE≤.5和默认V总体/最差RX非劣；全部15row已派发、7row闭合，8个seed02/03模型第33至34epoch，pending0、failure0、dispatcher和8worker实际live，GPU总进程max2。source首seed四arm finalCE分别0.250144/0.246923/0.253900/0.253468，默认V98.4037/98.4148/98.2593/98.2667%，0另训head。未读取partial target指标作候选判断，不称完整三seed结果。

## 完整收尾

15row/2520000固定原头决定/270原始CM独立recount VERIFIED；12scratch50＝600新增epoch/133200更新，3历史50成本分开，0另训head。固定LONG_SILU三seed均值81.8115%、sampleSD0.5020pp，目标84%未达。完整因素对照、group指标、paired纠正/引入错误、源阶段/反馈/真实参数变化、pureexport和剩余容量/旁路归因边界统一在主报告 analysis/phase1_feature_evidence_review_20261014.md，机器聚合在本run analysis。所有原产物保留，不按目标选择性重跑。

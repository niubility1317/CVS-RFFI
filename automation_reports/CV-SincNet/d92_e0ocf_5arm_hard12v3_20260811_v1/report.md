# D92-E0OCF五臂Hard12-v3实验报告

## 1.基本信息

|字段|内容|
|---|---|
|实验ID|`D92-E0OCF-5arm-Hard12-v3`|
|run ID|`d92_e0ocf_5arm_hard12v3_20260811_v1`|
|日期|2026-08-11|
|operator|Codex primary；N607唯一runner待指派|
|当前状态|`LOCAL_VERIFIED / TASK_REVIEWS_P0_P1_CLEAR / WHOLE_BRANCH_REVIEW_PENDING`|
|协议|`p2_min_v1`|
|证据范围|`DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN`|
|唯一晋级候选|`E0_OCF25`|

## 2.目标与假设

`E0_FULL_ONLY`已经把注册计算从K折full/block/Fisher图压到单次full主拟合，并在Hard12-v2上取得平均H`+0.336pp`和wall约`−97.7%`，但old floor下降`0.833pp`。错误分解显示这不是单纯old→new侵入，而是少数尾部旧类在旧类内部边界重新分配。

本轮假设：同一个注册后joint state的block几何包含对旧类内部contrast有益的信息；只把该contrast按固定比例融合到FULL_ONLY的旧类行，并保持旧类组均值和全部新类行不变，可以恢复floor而避免`E0_FIXED50`全类融合造成的H损失。

## 3.三轮强制回顾

|轮次|完成证据|保留信号|淘汰项|
|---|---|---|---|
|D92完整部件消融|完整消融分析|E有小幅旧类保护；B效应近零|160维Lite、以B为效率主轴|
|D92-BE Hard12-v1|48/48 job、8/8 shard PASS|E0为低成本底座|B0、B0E0|
|D92-E0D Hard12-v2 v3|60/60 job、8/8 shard PASS|FULL_ONLY平均H与计算显著改善；FIXED50有floor信号|BLOCK_ONLY、全类FIXED50晋级、跨state head拼接|

协议重核：继续复用`VALIDATED_ONCE/p2_min_v1`同一capsule/split；query逐样本对全部注册类竞争，禁止clean/source、query truth/role/quota/fit/update/selection/global reassignment。第四轮仍同时报告注册前旧类、注册后旧类、seen-new、H、逐类floor与forgetting。

## 4.冻结五臂与公式

|arm|机制|角色|K5/K10 two-state fit|
|---|---|---|---:|
|`D92_FULL`|B+E+full/block LOO-soft fusion|原方法对照|48/88|
|`E0_FULL_ONLY`|E关闭，仅after full主几何|低成本对照|2/2|
|`E0_FIXED50`|E关闭，全类full/block固定0.5/0.5|全类融合因果对照|4/4|
|`E0_OCF25`|E关闭，只融合25% block旧类contrast|唯一primary|4/4|
|`E0_OCF50`|E关闭，只融合50% block旧类contrast|diagnostic-only|4/4|

精确公式、禁用项和实现任务见`docs/superpowers/plans/2026-08-11-d92-e0ocf-hard12v3.md`。OCF必须保持FULL_ONLY的新类系数/偏置byte-exact，保持旧类组均值，且不二次全类centering。query MAC和永久state bytes必须与FULL_ONLY完全一致；新增成本仅为after block component fit和旧support RMS/融合代数。

## 5.fresh Hard12-v3

Hard12-v3从125个历史outer的冻结难度中，在排除Hard12-v1/v2共24行后按既有coverage约束选取；未读取本轮候选结果，也未按历史受损类`20-19`选择。

|outer|role|Hard|
|---|---|---:|
|`rx_20_1__seed_713104__k_5__new_20`|performance|0.629334677419|
|`rx_20_1__seed_713106__k_10__new_20`|performance|0.520866935484|
|`rx_20_1__seed_713106__k_1__new_20`|liveness|0.910584677419|
|`rx_3_19__seed_713102__k_10__new_5`|performance|0.429435483871|
|`rx_3_19__seed_713103__k_10__new_20`|performance|0.720463709677|
|`rx_3_19__seed_713105__k_10__new_5`|performance|0.454032258065|
|`rx_7_14__seed_713102__k_10__new_10`|performance|0.412600806452|
|`rx_7_14__seed_713105__k_1__new_20`|liveness|0.875403225806|
|`rx_7_7__seed_713104__k_10__new_10`|performance|0.297479838710|
|`rx_7_7__seed_713106__k_5__new_20`|performance|0.521471774194|
|`rx_8_8__seed_713103__k_10__new_20`|performance|0.456451612903|
|`rx_8_8__seed_713104__k_5__new_20`|performance|0.590826612903|

覆盖：receiver=`20-1:3,3-19:3,7-14:2,7-7:2,8-8:2`；seed=`713102:2,713103:2,713104:3,713105:2,713106:3`；slice=`K1/new20:2,K5/new20:3,K10/new5:2,K10/new10:2,K10/new20:3`。历史Hard总和=`6.818951612903226`。12outer×5arm=60job；三场景合计180scene-arm。

## 6.晋级门

`E0_OCF25`必须同时：相对FULL_ONLY严格提高mean old floor且至少8/10行不降，H、old BA、seen-new不降、forgetting不增；相对D92_FULL的mean ΔH≥0.5pp、至少8/10行非负，old BA/floor/seen-new不降、forgetting不增；median wall相对D92_FULL至少下降60%，peak不高于D92_FULL；fit/query/state和协议门全部精确。还必须报告old→old、old→new、new→old。任一项失败即不晋级。

## 7.最低发布门与审计裁剪

仅保留：Git方法入口、聚焦协议负测、真实checkpoint truth-free smoke、独立P0=0/P1=0、不可覆盖run/output/report、一次N607普通账号preflight和必要交付文件hash。明确不做重复数据验证、整树hash、额外签名/authority层、重复selection receipt或报告润色门。

## 8.本地实施与验证

### 8.1 Git提交

|commit|内容|独立复审|
|---|---|---|
|`b495811e`|新增OCF25/50科学核心、RMS对齐融合与不变量测试|初审后修订|
|`8851ed95`|修正瞬态内存上界、补组内标签置换、闭合lambda收据|`APPROVE，P0=0、P1=0、P2=0`|
|`dd13490d`|新增Hard12-v3矩阵、runner、analysis、CLI与配置|初审后修订|
|`ec6195f0`|将OCF active/lambda/support成本透传到正式fit audit|纳入Task2集成复审|
|`da6422e8`|闭合首次smoke、shard前置收据、非OCF成本与精确分片集合|复审后继续修订|
|`94d84dc4`|冻结双K1矩阵中的唯一技术smoke outer|`APPROVE，P0=0、P1=0、P2=0`|

核心实现位于`probe_d92_registration_balanced_covariance.py`与`stage2_d92_e0d_slim.py`；正式fit audit桥接位于`stage2_d92_e0d_query_evaluation.py`；矩阵、runner、analysis、CLI和method lock分别位于新增的`stage2_d92_e0ocf_*`、`run_d92_e0ocf_hard12v3.py`、`analyze_d92_e0ocf_hard12v3.py`和`stage2_d92_e0ocf_5arm_hard12v3_v1.json`。

### 8.2 本地验证

|验证|结果|
|---|---|
|OCF核心新增与既有probe/slim/query聚焦集|`23 passed`|
|fit-audit桥接evaluator回归|`9 passed`；同时slim回归`9 passed`|
|最终Task2/E0D聚焦回归|`30 passed`|
|Python语法编译、配置JSON解析、runner/analyzer CLI help、`git diff --check`|全部通过|
|Task1独立复审|`APPROVE，P0=0、P1=0、P2=0`|
|Task2最终独立复审|`APPROVE，P0=0、P1=0、P2=0`|

测试已覆盖：新类行byte-exact、旧类W/b组均值、组内标签置换等变、K1/K2别名、OCF fit与support成本收据、非OCF成本篡改拒绝、60-job双K1 smoke、smoke先于shard、跨distinct-outer停派、精确8-shard job集合、OCF50不得改变verdict及混淆率公式。

技术smoke固定为`rx_20_1__seed_713106__k_1__new_20`的`D92_FULL`；另一个K1 liveness outer仍保留用于五臂别名闭包。selection SHA保持`20edc97b914c1031d9f917c63dee8e8ddb94223b31750151a05aabf0375d65f1`。

## 9.发布与结果待办

当前尚未产生Hard12-v3性能结果。下一步只剩：全分支独立P0/P1复审、最小Git运行闭包与launch生成、一次N607普通账号preflight、真实checkpoint truth-free smoke、8-shard完整60-job执行、artifact取回和冻结analysis。精确交付物映射、N607命令、环境/CWD、GPU、日志/output、PID、smoke、artifact计数、同排结果与最终裁决将在同一报告续写。

# D13类条件new-logit侵入保护红队审查

日期：2026-07-17  
范围：只读审查`项目.md`、D6–D12主报告/追踪及D13实现；不修改D13模块、runner、测试或原追踪，不stage/commit。  
当前判定：`SUPPORT_ONLY_D13_NOT_SELECTED_NO_QUERY_OPEN;FINAL_NO_GO`

## 0.核心判据

D13必须把“旧类决策面冻结”和“新类注册仍有效”同时作为硬门：

1. Before与After对任意同一feature的全部旧类logit必须逐位不变；只锁prototype、state SHA或旧类内部argmax都不足。
2. After仅允许对新增类logit施加support-only类条件`delta_c`；不得改feature、旧prototype、旧logit、旧类温度、旧类bias、归一化、全局scale或共同分母。
3. 每fold的`delta_c`只能由该fold的old/new train support得到；old/new held2、full K10统计、query、truth、角色、配额和scorer结果均不可进入。
4. 零侵入不能通过把全部new logit压到不可预测来伪造。每场景held-new overall、floor、`H_old_new`和joint accuracy必须同时不低于预登记基线。
5. 三场景只能共用一套超参数；场景内可由合法fold-train support计算不同类的`delta_c`，但不得按场景选择不同quantile、safety、cap或floor margin。
6. 全部正delta失败时必须保存真实`delta=0`状态；预测、资源和状态内容均与base回退语义一致。

## 1.协议与机制攻击矩阵

|ID|攻击或失败模式|D13硬门|最小反例|
|---|---|---|---|
|RT13-01|After通过共享temperature、normalization或log-sum-exp分母间接改变旧logit|对同一feature直接比较Before旧logit向量与After旧logit前缀，要求逐位相等；比较原始logit，不只比较预测|两旧一新，给新类极大/极小logit；旧类前缀必须完全不变|
|RT13-02|只锁旧prototype，但After重算feature或换operator/view|state绑定Before/After旧support的feature SHA、runtime、checkpoint、operator、view；query也必须用同一策略|同一旧IQ换operator或feature extractor但沿用prototype，必须在打分前失败|
|RT13-03|用full K10统计计算fold的`delta_c`，held2泄漏|每fold只以train old K8+new K8计算prototype、margin分布、quantile、cap和`delta_c`；保存fold-train selection SHA|只改变held2数值，fold state和`delta_c`必须不变，held预测允许变化|
|RT13-04|held样本不进loss但进入normalization、排序、quantile或teacher|所有`delta_c`依赖统计的输入ID必须等于fold-train ID集合；禁止隐式全artifact聚合|held样本设为极端margin；若`delta_c`变化即泄漏|
|RT13-05|query或scorer结果参与delta选择、回退或排序|support-only入口无query/truth/role/quota/scorer参数；COMMIT先于任何query包|传入query sidecar、truth路径或post-score指标必须不可达或fail closed|
|RT13-06|新类生命周期合法但query按真实old/new角色选择是否减delta|delta按registry中的class handle绑定，所有query统一计算全部注册类；无query角色参数或side channel|同一物理query单独预测、混入不同角色/顺序批次，结果必须相同|
|RT13-07|通过类别数量、batch quota或排序推断query角色|物理batch恒1，禁止真实batch类数、quota、global assignment和q-q状态|重复、重排、增删其它query不能改变目标query输出|
|RT13-08|过强delta使old retention=100%但new=0|old门与new overall/floor、H、joint门同时执行；任何new门退化均淘汰正delta|构造`delta=+∞`或大cap候选，必须失败并回退delta0|
|RT13-09|仅平均new不退化，某个新类被完全压死|每场景每个新类held准确率或预登记floor硬门；至少要求new floor不低于alpha0与同口径基线|两新类，一类改善、一类降到0，平均持平；候选必须失败|
|RT13-10|旧类平均不退化但floor类下降|每场景每个旧类After准确率均不低于Before和alpha0；`old_forgetting<=0`只是附加门|一个旧类+20pp、另一个-10pp，平均提高；候选必须失败|
|RT13-11|delta方向或符号错误，实际抬高new logit|强制`delta_c>=0`且After new logit=`base_new_logit-delta_c`；逐类单调性测试|正delta下任一new logit高于delta0即失败|
|RT13-12|delta cap/floor margin依赖显示TX名或old/new特殊列表|规则仅依赖registry生命周期和fold-train support统计；不得硬编码TX、floor类名或场景名|重命名opaque class handle后数值结果应保持等价|
|RT13-13|三场景分别选不同超参后伪称统一|最终selection记录单一超参SHA；所有场景state引用相同hyperparameter lock|让每场景最优arm不同，selector必须选一个统一arm或delta0|
|RT13-14|delta0名义回退但仍保存正delta、执行减法或多报状态/MAC|delta0 state逐类delta严格为0，0额外训练参数/epoch；预测与同prototype base逐位一致|随机feature比较delta0与base logits/prediction；任一差异即失败|
|RT13-15|state字段可篡改，或content SHA未覆盖delta、class order、绑定和资源|readonly bytes-backed state；predict前重算content SHA；覆盖完整delta向量、class order、K、runtime/code/checkpoint/operator/view/support selection/resource|交换两个new类delta、改cap或resource后沿用旧SHA，必须失败|
|RT13-16|普通Mapping、自报SHA或任意callback可伪造feature|只接受runtime-authorized artifact；SHA由实际bytes重算；runner不暴露callback/token工厂|合法形状的普通dict、全+1/-1feature、自报64字符SHA均必须失败|
|RT13-17|K5/K1可达K10余量，或内部切片冒充exact-K|pre-open、allowlist和loader每类恰好K；D13无“取前K”路径|manifest K5但payload含10条/类时，在feature extraction前失败|
|RT13-18|同一物理sample多view被计为多shot，fold只held其中一部分|按`physical_sample_id`成组holdout，全部view共同进退；view不增加K|复制同一physical ID或仅held base view必须失败|
|RT13-19|旧类logit不变只在support成立，formal predict路径另有公式|fit审计与formal predict共用同一受测score原语；formal API测试旧logit前缀逐位锁|构造support审计通过、query路径附加全局scale的替代实现，应被接口测试捕获|
|RT13-20|资源只报delta几十B，遗漏prototype、metadata、序列化容器和安全校验|以实际state package字节执行256KiB门，分别报告增量与总量；MAC/延迟/显存相对同口径single-qKNN|添加未计数字段或使序列化包超限，必须fail closed|

## 2.联合fold数据流与性能门

每个K10 joint fold的唯一合法数据流：

```text
old K8 -> Before旧prototype
old K8 + new K8 -> After追加new prototype
old K8 + new K8 -> support-only计算每个new class的delta_c
old held2 -> Before仅旧类计分；After全部旧+新类计分
new held2 -> After全部旧+新类计分
```

old/new held2及其全部固定接收IQ派生view不得参与prototype、margin分布、quantile、delta、cap、normalization、候选选择、早停、回退或任何state字段。full K10 state只能在统一候选锁定后最终拟合，不能产生promotion指标。

### 正delta候选硬门

每个场景均须同时满足：

1. Before与After旧logit前缀逐位相等。
2. 每个旧类`after_old_acc>=before_old_acc`且`after_old_acc>=alpha0_old_acc`。
3. `old_forgetting=before_old_overall-after_old_overall<=0`。
4. `after_new_overall>=alpha0_new_overall`。
5. `after_new_floor>=alpha0_new_floor`，并不低于预登记同口径参考；D11-v6可作为joint-held参考，D10旧support总体/floor不可直接冒充new-only参考。
6. `H_old_new>=alpha0_H`且`joint_accuracy>=alpha0_joint`。
7. 每个新类不能因delta从非零准确率降为0；若采用更严格逐新类非退化门，应在打开真实support前锁定。
8. 参数0、epoch0、总state≤256KiB、无dense query图；正式资源需报告总state、head MAC、singleton平均/P95时延、峰值显存及相对single-qKNN Pareto。

三场景统一候选排序必须先执行上述硬门，再按最差场景new floor、平均`H_old_new`、平均joint accuracy、平均delta和状态量排序。完全并列优先`delta=0`、更小cap和更少状态。

## 3.delta0真实回退门

如果没有一个正delta候选同时通过三个场景：

- 最终state中所有new-class delta必须为0。
- 状态必须标记`SUPPORT_ONLY_D13_NOT_SELECTED_NO_QUERY_OPEN`。
- `query_package_opened/query_truth_opened/query_prediction_opened/query_score_opened/scorer_opened`全部为false。
- 资源报告必须按实际delta0路径计数；若实现仍执行无效减法，可报告额外安全校验开销，但不得声称零增量MAC。
- 回退结果只能证明注册保护未选中，不能声明D13带来性能改善。

## 4.追踪状态

|ID|要求|状态|验证|
|---|---|---|---|
|RT13-A|旧logit真实逐位不变|verified-module|Before/After旧prototype逐位锁；candidate与delta0对任意held-old/held-new feature的旧logit前缀逐位相等|
|RT13-B|fold-train support-only delta，无held/query泄漏|verified-module|fold0 held ranks0/1极端变异后prototype依赖的train selection SHA、penalty、threshold、strength和calibration diagnostics不变；parent artifact/state provenance变化属于预期绑定|
|RT13-C|无role/quota/global assignment且新类生命周期合法|verified-module|predict签名仅`state,query_artifact`，要求恰好1行并对全部注册类argmax；无truth/role/quota/batch count/global assignment参数|
|RT13-D|过强压制不能伪造old retention|verified-design|已定义new overall/floor/H/joint并列硬门与大delta反例|
|RT13-E|三场景统一超参|verified-real|v3固定6个constant arm含delta0，lock SHA`74feab0c...`，三场景统一选择真实delta0|
|RT13-F|delta0真实回退|verified-real|随机feature与真实v3 state均证明penalty/threshold/strength全0；最终state为delta0且promotion false|
|RT13-G|state/runtime/operator/exact-K绑定|verified-support;blocked-query|v3绑定current module/runner/runtime/checkpoint/code与strict-K包；formal query state/package loader仍不存在，但v3未获query授权|
|RT13-H|实际总state序列化口径|verified-real|v3逐state写NPZ+metadata并复算文件SHA/总字节；After为18,452–18,481B，低于256KiB|

## 5.实现级实测

### 5.1已通过

使用`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`：

```text
python -m py_compile code/cvsrffi/stage2_new_logit_intrusion_guard.py tests/test_stage2_new_logit_intrusion_guard.py
python -m pytest -q tests/test_stage2_new_logit_intrusion_guard.py
.......... [100%]
10 passed

python -m pytest -q tests/test_stage2_new_logit_intrusion_guard.py tests/test_run_d13_support_only_intrusion_guard.py
............. [100%]
13 passed
```

独立只读反例结果：

```text
held2_fold0_penalty_same True
held2_fold0_selection_same True
held2_fold0_diag_same True
delta0_score_array_equal True
delta0_all_penalties_zero True
```

held2变异门的正确判据是拟合数值与train selection SHA不变。完整parent artifact SHA和最终state content SHA因held payload所属父package发生变化而变化，是正确provenance绑定，不是统计泄漏。

联合L2O输出已经包含：

- `old_score_columns_bitwise_unchanged`
- `old_per_class_non_degraded_vs_before`
- `old_per_class_non_degraded_vs_base_cosine`
- `new_per_class_non_degraded_vs_base_cosine`
- `all_new_class_calibration_feasible`
- old/new overall、逐类、floor、joint、`H_old_new`和forgetting

这些字段只有被runner作为三场景统一候选的先验硬门使用时才有效；仅记录字段不构成晋升。

### 5.2公式结论

constant模式的数值关系为：

```text
requested_c=max(0,old_risk_c+safety)
room_bound_c=max(0,new_room_c-new_floor_margin)
delta_c=min(requested_c,cap,room_bound_c)
```

它不必然让held-new准确率下降，因为准确率只在margin越过0时变化；但对类`c`相对旧类的margin会单调减少`delta_c`，因此不能从公式推出new非退化。逐新类、new floor、overall、H和joint门必须保留。

`protection_feasible=(requested_c<=cap且requested_c<=room_bound_c)`对constant模式有明确含义：可在不越过预登记new-room下界的前提下满足old-risk分位保护。若`protection_shortfall>0`，当前delta未达到requested，不得声称该类侵入保护可行。模块已输出shortfall和全新类feasibility字段；runner必须把任一新类不可行作为正delta候选失败，而不是只写诊断。

当前`old_risk`已修正为仅使用Before-correct old support，并比较`new_c-max_old`，避免把Before已存在的旧类内部误分类直接当成new侵入。但是它把全部Before-correct旧类样本池化；floor旧类正确样本较少时权重会降低。该做法不违反协议，但对用户要求的floor优化偏弱。真实runner至少要报告逐旧类侵入计数与逐类held非退化；若D13继续优化，优先改为每个旧类分别估计风险后取最坏合法分位，而不是放宽held门。

### 5.3P0数学阻断：当前hinge模式不可作为正式候选

当前hinge模式使用：

```text
threshold=bounded
correction(q)=strength*relu(threshold-new_margin(q))
```

虽然实现已把`correction`裁剪到`cap`，但`protection_feasible`仍沿用constant模式的`requested<=cap,room_bound`。当`strength<1`时，在`old_risk`对应点实际扣减通常小于`requested`；`new_room`对逐sample hinge correction的约束也不再等价。因此当前feasibility字段不能证明hinge保护成立，且`delta=0`字段会掩盖非零threshold/strength。

正式D13首轮必须：

- runner只预登记`mode=constant`；或
- 对hinge重新推导old-risk/new-room可行区间、逐sample最大修正和独立shortfall，并增加相应反例。

当前runner已选择第一条：正式网格只有delta0和5个constant arm，没有hinge arm。因此hinge仅是未授权模块扩展；在完成第二条前，任何未来hinge arm即使support指标通过也必须标为`FORMULA_GUARD_INVALID_NOT_SELECTED`。

## 6.真实v3独立复核

最终artifact：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d13_new_logit_intrusion_guard_v3_final`

- `COMMIT.json`文件SHA-256：`2d2d7874a66660233189b0c3e5e66545d2f26c4f3b5c9dbf878ef7e923b29860`
- module SHA：`217d90572d263f47af9d442ba5347ec8839958097f514c5bc7e2add2b713ce3b`
- runner SHA：`f6aa801793e39e7ce4c7a7d0665093fd7f669a353cc7937a727e3d529ee28a14`
- 当前工作树module/runner SHA与COMMIT绑定一致。
- hyperparameter lock SHA：`74feab0c409ffb2be2485307cf9cf17d851f8847d1b53179b3d68ab6770a9e5f`
- `support_audit.json`、`training_log.jsonl`、`report.md`和全部6个state NPZ/metadata哈希复算一致。
- 所有artifact文件只读。
- query、truth、prediction、score和scorer五个打开标志全部为false。
- 完整日志276行：三种joint L2O phase各90行，最终Before/After闭式fit各3行；6个candidate、3个scenario归属完整。
- D11-v6参考COMMIT文件SHA、support audit SHA和report SHA均与v3绑定值一致，且D11 reference本身`query_opened=false`。

### 6.1统一选择结果

五个正constant arm无一通过三场景硬门，最终统一回退`d13_delta0_base`：

|场景|After old/floor|After new/floor|H|old forgetting|old逐类>=Before|new逐类>=delta0|feasible|门|
|---|---:|---:|---:|---:|---|---|---|---|
|`leo_clear_weak`|0.6333/0.10|0.5000/0.20|0.5588|8.33pp|False|True|True|False|
|`leo_low_elev_weak`|0.6833/0.20|0.4600/0.10|0.5499|5.00pp|False|True|True|False|
|`leo_rain_weak`|0.6167/0.30|0.6200/0.40|0.6183|11.67pp|False|True|True|False|

所有场景`old_score_columns_bitwise_unchanged=true`，但注册后old仍比Before低5.00–11.67pp。这不是矛盾：旧logit完全不变，新增new logit仍参加全类argmax并抢占旧样本。D13正guard未提供足够有效扣减，因而不能消除D12已经观察到的注册竞争遗忘。

### 6.2正guard实际作用极弱

从完整training log复算：

- `q80/q90`三个arm的75个fold×new-class penalty全部为0，与delta0数值完全相同。
- `q100,r10,safety=.01`仅3个fold-class出现非零delta，最大`0.003656`。
- `q100,r25,safety=0`仅4个fold-class出现非零delta，最大`0.002615`。

这说明当前support train上的`old_risk`大多不为正，或受`new_room`约束后只允许极小delta；但held-old仍发生显著new-logit侵入。问题不是硬门过严，而是当前train-margin分位对held侵入的预测力不足。放宽old门会掩盖机制失败，不应作为下一步。

### 6.3资源

v3真实序列化state：

- Before：每场景9,772B。
- After：clear18,481B、low18,452B、rain18,475B。
- guard tensor增量60B；After数组tensor state12,804B。
- 0训练参数、0epoch、单query head额外5次new-logit减法；无dense query图。

这些数字证明D13很轻，但资源优势不能替代性能门。

## 7.剩余协议边界

support-only闭环已通过。formal `predict_all_registered`仍只消费内存state与artifact，尚无独立state package loader去复算NPZ、metadata、`state_content_sha256`、COMMIT和query package SHA。该缺口当前不构成v3额外失败原因，因为v3的support promotion为false且没有打开query；如果未来D13修订版通过support门，必须先补该loader，再生成candidate-bound query。

## 8.终审结论

D13的constant算法原语达到模块级GO：

- 旧logit前缀真实逐位不变；
- held2极端变异不影响fold-train拟合数值；
- shortfall与逐新类非退化字段已存在；
- delta0与base cosine逐位等价；
- formal API形状为单物理query、全注册类、无role/quota。

真实v3完成了三场景统一锁、strict-K10 support-only、current code/runtime/checkpoint绑定、逐old/逐new硬门、真实delta0回退和实际序列化state审计。结果仍为NO-GO：五个正constant arm全部失败，统一回退delta0；old forgetting仍为5.00–11.67pp，未满足任何场景的逐旧类非退化门。

因此D13不得开放query、不得进入125矩阵、不得声明注册保护有效。最合理的下一机制不是放宽遗忘门，而是提高support对held侵入的可预测性，例如按每个旧类分别估计最坏合法侵入风险，或重新推导有cap的逐样本局部guard；若使用hinge，必须先修复其feasibility数学。任何下一版仍须保持old logit逐位不变、逐new非退化和三场景统一超参。

最高风险：把“旧logit逐位不变”误写成“旧类无遗忘”。v3已经实证二者不同。  

反向追踪计数：`verified-module=5`、`verified-design=1`、`verified-real=4`、`blocked-future-query=1`、`rejected-formal-hinge=1`。

# D75交叉拟合margin安全nuisance投影实验报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d75_crossfitted_margin_safe_nuisance_projection_probe_20260720`|
|候选|`crossfitted_margin_safe_nuisance_projection`|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|以全注册类nested support-held margin安全门过滤D74非可逆方向，同时保护旧类适应、新类注册和通用floor|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.协议与数据复用

- `p2_min_v1`、D18匹配`VALIDATED_ONCE` capsule、receiver`20-1`、seed`713101`、K10/new5、3场景×5fold，outer-fit实际K8。
- 单LEO_weak观测、support-only、query逐样本全注册类argmax；无clean/source/query truth/role/quota/global assignment。
- 数据字节、物理ID、receiver/TX、场景、K、support/query split和schema均未变化，不触发重复数据验证。
- 地面int8组件输入0；D22未获正式资格，不能用于D75候选选择或状态更新。

## 3.机制锁

每个类内物理rank用其余K−1样本同时拟合equal-prior shrinkage LDA和D74方向，并在每类一个held样本上比较固定头投影前后的true-vs-best-other margin。只有全部类平均margin、全体平均margin和held正确数均不退化才接受full-support rank-1投影，否则精确回退D62。容差仅为机器舍入界；无可调阈值、rank、强度、角色或场景分支。

## 4.开发门与结果占位

要求相对D62的`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格提高；失败即负向关闭，不开第二seed或125矩阵。

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|安全门|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D75|D62固定头＋nested margin安全rank-1投影|20-1/new5|K10/713101|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|0接受/15拒绝|见第15节|负向，不晋级|

## 5.版本、验证与运行占位

`E:\type10-7`不是Git仓库；设计、代码、测试和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`。实现后补录commit、clean worktree、source SHA、完整命令、PID/GPU/log/output、105行闭包、机制门、训练、量化、资源、artifact、完整性能和最终判定。

## 6.实现与主工作树验证

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d75_crossfitted_margin_safe_projection.py`|物理rank留一、LDA margin、全类floor安全门与identity回退|`8b4a59ca9b7ded3f144f592dfe710570e595d1e3864814dfc403733b6e60fc46`|
|`code/scripts/probe_d75_crossfitted_margin_safe_nuisance_projection.py`|D62/D74包装、资源核算、Runner闭包和metadata|`0e9b08b410305879153d5d5e936cdc5e6d5ead1da5f4d644b1df4ea0c610d0cc`|
|`tests/test_stage2_d75_crossfitted_margin_safe_projection.py`|held margin拒绝/接受、rank交错、对称support fail-closed|`ae4e8fdddef37d7b4b47c2ad5b18064f63ad28457010f470f722501b52b3d7f5`|
|`tests/test_probe_d75_crossfitted_margin_safe_nuisance_projection.py`|公式、继承结构和query/state/ground闭包|`4aa7a9bb347fbd79cb675ef2706e6faadf21059d4326edc4f915561d93e07751`|

- `ssr-gpu`专项7/7通过，core/probe `py_compile`通过。
- D42–D75相邻完整链43文件、392项全部通过，用时82.2秒；显式仓内basetemp，无数据重验。
- 实现不扫描margin阈值、rank或强度；门限仅为机器舍入界。每个target row预期8次LOO LDA、8次LOO方向、88个held support margin，optimizer/epoch仍为20/20，query额外MAC/state0。

## 7.clean验证与运行锁

- 实现commit=`e2fd8cf8580f3072529460295fb187b7b7a3d0dc`；clean worktree=`E:\type10-7\code\snapshots\d75wt`，detached HEAD且clean。
- clean D42–D75相邻完整链43文件、392项全部通过，用时82.5秒；core/probe `py_compile`通过。
- clean执行SHA：D75 core=`a41456c85437125203a54d069d90dcbebc6462df4519e77b5f4cbbed6fdbc99a`、D75 probe=`6e14688f1049b67c3da57b80d5a9636ca8e27263bc9cc508b43a69dc3147af51`、D74 helper=`427be77328700c524173689567423b861bd18dd57fb8d96d7a4fcd5c6d4e363d`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 启动前输出目录不存在，无D73/D74/D75 Python任务；GPU0 RTX5070Ti显存`1097/16303MiB`、利用率0%。本轮本地执行，不访问N607。
- 预期闭包：105行、30个target row、30次top fit、1080次D62 component execution、每个target row 8次LOO LDA和8次LOO方向；ground/query-fit/clean/source/role/quota访问0。
- 每个target row预期新增：LOO LDA MAC`249,495,552`、LOO方向MAC上界`111,817,728`、full方向＋编译`18,190,656`；相对D62总新增`379,503,936`，总适配MAC`25,270,727,906`，query/state增量0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d75wt\code\scripts\probe_d75_crossfitted_margin_safe_nuisance_projection.py' `
  --d75-arm crossfitted_margin_safe_nuisance_projection `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d75wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d75_crossfitted_margin_safe_nuisance_projection_probe_20260720\crossfitted_margin_safe_nuisance_projection' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 8.启动、完成状态与artifact闭包

- 2026-07-20 02:00:24启动唯一执行，PID`8764`；只读命令行核对与第7节锁定参数完全一致。
- PID于02:02:45前正常退出，Runner实测`elapsed_seconds=134.6343`，launcher stderr为0B。
- RECEIPT=`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，`selected_candidate_id=Z0_SUPPORT_ONLY`，`selected_positive_route=false`，`query_opened=false`；无formal/performance claim。
- 闭包：105/105行、7候选×15fold、30个target量化/FP32行、30次top fit、1080次D62 component execution、240次LOO LDA/方向门审计；target行20/20训练步，ground/query-fit/clean/source/role/quota访问0。
- 安全门：INT8与FP32合计0/30接受、30/30拒绝；15个INT8 fold有15种proposal，但实际删除rank全部为0，精确回退D62。
- 完整摘要：`E:\type10-7\automation_reports\CV-SincNet\d75_crossfitted_margin_safe_nuisance_projection_probe_20260720\d75_full_performance_summary.json`，106,864B，SHA256=`87c4b05347ab278b251f47e1910018507dbab3f6ba751b8ef2494681358a32e0`。

|artifact|字节|SHA256|
|---|---:|---|
|`training_log.jsonl`|14,884,998|`95b1ae26d39670a94bd7b83a1f921b2732d9808b1d2be1115892aba178243d71`|
|`support_audit.json`|313,785|`d3799a00a0b5c010b0535492f960ea8cd0c1adf3cadb1805b56e8ba097187347`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`selection.json`|2,990|`df6cbca03c0e5a38fc0299dcba846a2b18854549d3521e9c75e58eba61f8fd5a`|
|`RECEIPT.json`|5,125|`cd05e6ac6a5adec5be8d1d43b113d6ff3dce6e4da18c8ae04a6a67120dcc5c6b`|
|`D75_PROBE_METADATA.json`|2,601|`c8294c0c88d18a2dabbc4f628c94d97d170e007945aed524d641dab02b038913`|

## 9.同row总体结果与开发门

|candidate|机制|receiver/TX|K/seed|B old|A old|seen-new|unknown|H|forgetting|joint|min-B/A/N|row-floor B/A/N|混淆O→N/N→O/N→N|量化|判定|
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|
|D75 INT8|D74 proposal＋全类nested margin门＋D62回退|20-1/new5|K10实际fit K8/713101|92.78|82.22|84.67|N/A，本开发单元无unknown query|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|INT8=FP32 argmax|负向，不晋级|
|D62 INT8|冻结D42 metric＋crossfitted Fisher row splice|20-1/new5|同上|92.78|82.22|84.67|N/A|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|INT8=FP32 argmax|当前最强|

D75与D62的15/15 outer prediction SHA完全相同，总体、场景、类、fold、floor和混淆逐项相同；但适配计算更高，因此开发正向门失败。相对K10目标：`A`差9.78pp、`min-A`差34.67pp、`new5`差7.33pp。不得运行第二seed、K1/K5/K20或125矩阵。

## 10.三场景表现

|场景|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆O→N/N→O/N→N|主要表现|
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|LEO clear weak|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|新类接近饱和，但仍有fold旧类floor为0|
|LEO low elev weak|91.67|78.33|76.00|75.98|13.33|20.00|80.00/60.00/50.00|70.00/60.00/20.00|8/5/7|旧/新双向混淆，new floor不足|
|LEO rain weak|88.33|76.67|80.00|77.45|11.67|10.00|60.00/30.00/70.00|60.00/30.00/30.00|13/2/8|旧→新侵入最重，min-A仅30%|

## 11.逐类总体准确率

|类|before-old|after-old|遗忘/变化|
|---|---:|---:|---:|
|O1|96.67|90.00|−6.67pp|
|O2|96.67|90.00|−6.67pp|
|O3|96.67|93.33|−3.33pp|
|O4|80.00|53.33|−26.67pp|
|O5|93.33|73.33|−20.00pp|
|O6|93.33|93.33|0.00pp|

|类|seen-new准确率|
|---|---:|
|N1|73.33|
|N2|93.33|
|N3|76.67|
|N4|90.00|
|N5|90.00|

最弱旧类O4=53.33%，最弱新类N1=73.33%。D75只防止了D74进一步伤害，没有改善D62的下尾瓶颈。

## 12.15个outer fold完整同row表

|场景|fold|B|A|N|H|F|J|floor B/A/N|混淆O→N/N→O/N→N|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|clear|0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|
|clear|1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|0/0/0|
|clear|2|91.67|83.33|100.00|90.91|8.33|50.00|50/50/100|1/0/0|
|clear|3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|
|clear|4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|
|low-elev|0|100.00|66.67|80.00|72.73|33.33|50.00|100/50/50|4/1/1|
|low-elev|1|83.33|58.33|70.00|63.64|25.00|0.00|50/50/0|1/0/3|
|low-elev|2|83.33|91.67|70.00|79.38|−8.33|0.00|50/50/0|0/2/1|
|low-elev|3|100.00|100.00|70.00|82.35|0.00|0.00|100/100/0|0/1/2|
|low-elev|4|91.67|75.00|90.00|81.82|16.67|50.00|50/50/50|3/1/0|
|rain|0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|
|rain|1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|
|rain|2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain|3|83.33|75.00|90.00|81.82|8.33|0.00|50/0/50|3/0/1|
|rain|4|83.33|75.00|80.00|77.42|8.33|0.00|50/50/0|3/1/1|

## 13.安全门、训练与失败机理

- 15/15个INT8 fold各有不同full proposal，每个fold另有8种LOO方向；proposal删除能量与D74一致，但最终0/15接受、rank删除全部为0。
- 每fold最弱类margin delta范围`−22.1800`到`−2.4039`、均值`−8.4403`，没有一个fold满足全类不退化。总体margin delta范围`−5.0777`到`+0.9208`、均值`−1.2335`；即使3个fold总体均值为正，仍存在被牺牲的类，按通用floor约束必须拒绝。
- held正确数delta范围−8到+1、均值−1.7333。雨衰最差：总体margin delta均值−3.1240、最弱类delta均值−15.8948、正确数平均−3.6；这与D74雨衰outer退化方向一致，说明门有诊断价值。
- 门的不足是二元且过严：它能拒绝坏方向，却没有生成新边界或保留任何部分收益。0/15接受后D75只能回到D62，不可能突破目标。
- D42 Stage2-B训练完整：epoch1 loss1.031996、support acc95.14%、gradient norm1.08376；epoch20 loss0.102685、support acc100%、gradient norm0.13535。D75全为闭式支持集计算，Stage2-C optimizer step=0、query rows=0。

## 14.近期版本matched比较

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-B/A/N|prediction hash变化|混淆ΔO→N/N→O/N→N|解释|
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
|D75−D62|0.00|0.00|0.00|0.00|0.00|0.00|0/0/0|0/15|0/0/0|完全回退，资源更高|
|D75−D73|0.00|0.00|0.00|0.00|0.00|0.00|0/0/0|0/15|0/0/0|D73也等价D62|
|D75−D74|0.00|+1.67|+5.33|+3.81|−1.67|0.00|0/0/+10.00|12/15|−1/−5/−3|安全门成功消除D74伤害|
|D75−D72|−0.56|−0.56|+2.00|+1.03|0.00|0.00|0/0/+3.33|5/15|+1/−3/0|恢复D62新类但无新进步|
|D75−D71|+1.67|0.00|+0.67|+0.29|+1.67|0.00|−3.33/0/0|1/15|0/−1/0|A未改善且F更高|
|D75−D61|+2.78|−1.11|+8.67|+3.67|+3.89|0.00|+3.33/−6.67/+30.00|15/15|+5/−8/−5|D61低F以新类为代价；两者均不达标|

## 15.量化与资源表现

- INT8与matched FP32的before outer、final outer、before/final support argmax变化均为0，margin符号翻转0；最大score误差最小/均值/最大=`0.000377/0.000882/0.001915`。
- 最差margin：old-new−2.0895、new-old−4.8747、new-new−1.2053。量化不是瓶颈。

|资源|D75|D62|增量/说明|
|---|---:|---:|---|
|trainable parameters|2,016|2,016|不增加|
|optimizer steps/epochs|20/20|20/20|门为闭式0步|
|closed-form LDA fits|80|72|+8 LOO fit|
|LDA fit MAC|18,249,504,768|18,000,009,216|+249,495,552|
|LOO方向MAC上界|111,817,728|0|8个support-held方向|
|full方向＋编译MAC|18,190,656|0|最终虽回退仍完成候选审计|
|total adaptation MAC|25,270,727,906|24,891,223,970|+379,503,936，增加1.5246%|
|query MAC|6,624|6,624|额外0|
|persistent/registry state|8,583/941B|8,583/941B|额外0|
|peak CUDA memory|22,886,912B|22,886,912B|不增加|
|dense query graph|0B|0B|通过|
|ground int8 component input|0|0|D22未获正式资格|

D75满足正式资源上限，但性能与D62完全相同且多耗1.5246%适配MAC，被D62严格支配。

## 16.缺陷、结论与下一步

1.D75的价值是证伪D74并建立可观测安全信号，不是性能提升；全类硬安全门把全部proposal拒绝，无法学习新的正向边界。
2.不能放宽为“总体margin为正即可”，因为这会允许牺牲某些实际注册类，违反通用floor；也不能按old/new角色、场景或具体类设置容差。
3.D75没有使用地面压缩旧类原型。D22仍`formal_phase2_eligible=false`，D66合法读取84个ground int8单元也为负；继续遵守数据协议。
4.停止D75阈值、rank、强度、margin权重、类/角色/场景门和第二seed，不运行125。
5.D73–D75已满三轮，启动D76前必须执行正式技术复盘。下一路线不能再是“提议后硬门回退”，应直接优化全类support-held下尾margin的连续、类对称、可编译残差，同时避免D73的可逆重参数化吸收和D74的盲删。

最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。当前最强协议合法开发版本仍为D62，而不是D75。

## 17.D73–D75三轮技术复盘

### 17.1复盘输入与协议复核

- 已重新核对活动目标，目标文件SHA256=`92f5f155939505dc45b51a1bdde77e606ce56380e706e09127218a1287bda29e`；`项目.md`SHA256=`45683ac1e4f031a8307ac4fcb7745922ed965483975aa7b1258f78d3f6fd4920`。
- 已刷新项目conversation index，共1005条，并搜索`D62/D73/D74/D75/Fisher/margin/nuisance/ground int8`；历史设计中“地面int8旧类锚＋target旧类域校正＋target新类独立注册”仍只是需要合法sealed组件和实测支持的路线，不能覆盖当前D22不具正式资格的事实。
- 已复核D73、D74、D75完整105行日志、逐类/场景/fold摘要及D61/D62报告。三轮均同时保留before-old、after-old、seen-new、H、forgetting、全部旧/新类和混淆；不存在只优化一侧的晋级。
- 协议复核通过：三轮均复用D18 `VALIDATED_ONCE/p2_min_v1`，单LEO_weak固定观测、support-only适配、query一次性评分、全注册类逐样本argmax；clean/source/query truth/role/quota/global assignment/dense query graph均为0/false，ground输入0。

### 17.2三轮同row结果

|版本|核心机制|B|A|N|H|F|J|min-B/A/N|相对D62预测变化|适配MAC|结论|
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
|D73|旧/新等权单步共享metric＋D62 refit|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|0/15|46,145,052,306|可逆变化被refit吸收，完全等价但成本+85.39%|
|D74|类中心正交最大残差rank-1删除|92.78|80.56|79.33|78.81|12.22|26.67|80.00/53.33/63.33|12/15|24,909,414,626|非可逆机制生效，但误删判别方向|
|D75|全类nested margin硬门过滤D74|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|0/15|25,270,727,906|0/15接受，消除伤害但只能回退D62|

### 17.3成功经验、淘汰路线与重复失败

1.当前唯一可复用的正向结构仍来自D62：它直接改变最终仿射类行，用inner-held positive不降、false-positive不增和原子回退约束行级替换；相对D46只取得小幅同row改善，但证明“类对称的row-local边界更新”比共享metric或全局融合更有信息量。
2.D73淘汰“共享可逆metric＋重新拟合统一LDA”：support代理loss下降不代表outer改善，refit会吸收坐标重参数化。
3.D74淘汰“中心正交即可视为nuisance”：类中心保持不变仍可能破坏弱场景所需的类内判别结构，尤其雨衰和新类下尾。
4.D75淘汰“提议＋全类硬安全门”作为主学习机制：门能诊断并拒绝伤害，但0/15接受说明它没有生成正向边界；放宽到总体margin会牺牲实际类，违反floor。
5.重复失败模式是support拟合或单个代理被当成生成机制：D73代理改善被吸收，D74support准确率几乎不变却outer恶化，D75代理只触发identity。下一轮必须直接优化可部署决策行，并用support-held证据定义更新本身，而不是更新后的二元路由。
6.地面压缩旧类知识没有被D73–D75使用；这是有意遵守协议，不是遗漏。D22仍`formal_phase2_eligible=false`且provenance unverified；D66读取84个ground int8单元的诊断为负，不能以“降低遗忘”为由越界或强融合。

### 17.4当前差距与D76决策

当前最强D62仍距K10开发目标：after-old差9.78pp、min-old差34.67pp、new5差7.33pp；最弱旧类O4=53.33%、O5=73.33%，最弱新类N1=73.33%。问题不是INT8量化，D62/D73/D75与matched FP32 argmax均一致；主瓶颈是弱场景下的类间边界和下尾泛化。

D76锁定为`crossfitted_allclass_common_descent_row_residual`：冻结D62 before/final头，在每个物理rank的K−1 support拟合基头，对held support为每个实际注册类计算multiclass margin/CE对最终仿射行的梯度；用类置换等变的minimum-norm convex common-descent方向与解析Lipschitz步长，直接产生一个共享编译的低幅度row residual。它不做role/scene/class-ID门，不扫描步长/rank/阈值，不在更新后refit，因此避开D73吸收；方向由全类held损失共同定义，因此避开D74盲删；它本身是连续学习更新，不是D75硬回退路由。

D76最小门仍为相对D62的`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格提高。开发cell不变，ground/query访问0；完成D76前不启动125。

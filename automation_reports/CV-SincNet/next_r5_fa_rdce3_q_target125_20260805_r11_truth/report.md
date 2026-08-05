# NEXT-R5 FA-RDCE3→qKNN Target125 truth/score报告（r11）

## 身份与目标

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r11_truth`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：复用r10不可变完整prediction manifest，仅完成truth-open与score，不重跑125 prediction。
- r10 prediction闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique+150 alias；manifest=`33027483`bytes，SHA=`29982326a4c3130abea223a56c19362f9cf7e583da186e3498de2b80758c0c49`。
- r10 truth-open失败根因：truth adapter传给D108 sidecar loader的outer漏`old_classes/new_classes`，在任何truth评分前触发KeyError。
- r11从source_context绑定的D92 before_apply/after_apply sealed package manifest取得预truth registry，验证包seal、receiver/seed/K、连续class_index、旧类前缀和new_count，并与r10四状态`registered_classes`元数据逐scene绑定；不读取predicted labels、query truth、role或IQ。
- 科学commit=`d17b29d0aa1840c6d4dffa32034745b952ff828f`；六入口编译和36项聚焦测试通过；独立Terra复核`P0=0，P1=0`。

## 闭包与冻结输入

- closure=`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r11_truth_closure_d17b29d0.tar`；73175040bytes；SHA=`94f106b7c237e034d3c1e50e176adaec528db392fac18660ae03246639d7b810`。
- truth adapter/CLI/D108 scorer/runtime SHA=`91bd6c0c80bad50ec431db37a7941ce8cf9edd62b10703c5b665c08072bfc797`/`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`/`ee64b32359599acba152487b8673ebae386f7d63e2d095ee8186275e5efad766`/`38a69778a3131a144fcece1d7bd066f50c829852aba1c43d265f2c1a29ccbb38`。
- r10远端prediction=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10/merged/prediction_manifest.json`，必须先验证上述SHA与大小。
- r10 prepared plan/context实产SHA由r10报告与runner证据原样传入；不得重新prepare或修改prediction。

## N607执行

- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r11_truth`，先为`ABSENT`；r1至r10只读。
- 只执行：preflight→closure land/hash/compile→读取并验证r10 prediction+prepared→detached truth-open→score→取回。无GPU prediction、无asset build、无prepare、无smoke、无shard、无merge。
- truth catalog和score output-dir必须为r11新路径；truth-open进程必须绑定r11 source CWD和r10 immutable manifest SHA。
- score必须覆盖125 outer、375 scene和四状态；最终按同一行报告DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1及DA/注册主效应和interaction。REG0的新类/H=`N/A`。
- 失败只按truth/score技术错误处理，不得修改或删除r10 prediction；fresh retry authority=`无`。

## 运行前验证与版本状态

|项目|证据|
|---|---|
|N607预检|2026-08-05 14:42 CST；直连`N607`、项目根可见、8张RTX3090可见且空闲；本地`SSH_CLEAN/TCP_CLEAN`。|
|Git承载面|`E:\fa125wt`分支`codex/next-r5-fa-q-target125-20260805`；科学commit=`d17b29d0aa1840c6d4dffa32034745b952ff828f`；报告基线commit=`8f5bddbf89396a2d8ea07ca870fa4d5b1929f5fb`。`E:\type10-7`根目录不是Git仓库。|
|closure|本地与远端均为73175040bytes，SHA=`94f106b7c237e034d3c1e50e176adaec528db392fac18660ae03246639d7b810`。|
|六入口compile|远端`Python 3.10.19`、`CUDA_VISIBLE_DEVICES=`空、`PYTHONPATH=.`；CLI、target、core、matrix、runtime、D108 truth scorer的`py_compile`均PASS。|
|输入复核|r10 prediction=`33027483`bytes/SHA=`29982326a4c3130abea223a56c19362f9cf7e583da186e3498de2b80758c0c49`；plan=`1041594`bytes/SHA=`0d28fc51548907843618524eb31a4b023971535ff861412d53033d3b20f1c292`；context=`28685`bytes/SHA=`5c062dd13516b8a41b2d3ec9597345f2e792606021d2a9eebd13822ed08769d6`。三者均未修改。|

六入口远端文件SHA：CLI=`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`；target=`91bd6c0c80bad50ec431db37a7941ce8cf9edd62b10703c5b665c08072bfc797`；core=`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`；matrix=`935710db6be8c6329e9fb2638a346478ac63a22a536e3666cd74d9a775d8e2e5`；runtime=`38a69778a3131a144fcece1d7bd066f50c829852aba1c43d265f2c1a29ccbb38`；D108 truth scorer=`ee64b32359599acba152487b8673ebae386f7d63e2d095ee8186275e5efad766`。

## 精确N607命令与过程证据

远端source CWD为`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r11_truth/source/code`，Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。完整命令也保存在`retrieved/logs/truth_open.command.txt`和`retrieved/logs/score.command.txt`。

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_next_r5_fa_target125.py truth-open --prediction-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10/merged/prediction_manifest.json --prediction-manifest-sha256 29982326a4c3130abea223a56c19362f9cf7e583da186e3498de2b80758c0c49 --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10/prepared/target125_plan.json --plan-manifest-sha256 0d28fc51548907843618524eb31a4b023971535ff861412d53033d3b20f1c292 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10/prepared/target125_context.json --context-manifest-sha256 5c062dd13516b8a41b2d3ec9597345f2e792606021d2a9eebd13822ed08769d6 --truth-catalog /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r11_truth/truth/truth_catalog.json
```

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_next_r5_fa_target125.py score --prediction-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10/merged/prediction_manifest.json --prediction-manifest-sha256 29982326a4c3130abea223a56c19362f9cf7e583da186e3498de2b80758c0c49 --truth-catalog /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r11_truth/truth/truth_catalog.json --truth-catalog-sha256 6b02dec204143aa56ef8c76b457b9ce6c9d8229965506e8dd892333dad1d09d1 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r11_truth/score
```

|任务|PID|开始/结束|CWD与健康|日志/退出|
|---|---:|---|---|---|
|truth-open|1753612|约14:48:00/14:51:10 CST|CWD与cmdline绑定r11 source和r10三份SHA；CPU约100%，GPU未使用|`truth_open.log`367bytes；进程自然退出；无异常指纹|
|score|1756506|约14:52:43/14:58:49 CST|CWD与cmdline绑定r11 source、r10 prediction SHA和truth SHA；CPU约100%，GPU未使用|`score.log`343bytes；进程自然退出；无异常指纹|

## 原始闭合产物与hash

|产物|远端路径|大小|SHA256|
|---|---|---:|---|
|truth catalog|`.../r11_truth/truth/truth_catalog.json`|57719945|`6b02dec204143aa56ef8c76b457b9ce6c9d8229965506e8dd892333dad1d09d1`|
|score JSON|`.../r11_truth/score/score.json`|5790262|`fa2344ae037e4ab5dfec6fea9bb0f534c7d5c9cdeb3596797bdc403b3c9fcc23`|
|truth-open log|`.../r11_truth/logs/truth_open.log`|367|`5296dffa2a1fbc358df5c97b1a38de5d0b9ba155a2d026aff7552f3d788f5da9`|
|score log|`.../r11_truth/logs/score.log`|343|`aeea34061b43c95b118a799395cf09b3e1099ab452cd14782cf599eeb4e7220a`|

取回目录（两处字节一致）：

- `E:\fa125wt\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r11_truth\retrieved\`
- `E:\type10-7\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r11_truth\retrieved\`

## score结构摘要（不作性能解释）

|字段|原始值|
|---|---:|
|schema|`cvs.phase2.next_r5.fa_rdce3_qknn.target125.score.v1`|
|candidate_id|`NEXT-R5-FA-RDCE3-Q-TARGET125`|
|outer_job_count|125|
|scene_row_count|375|
|logical_state_surface_count|1500|
|unique_prediction_count/alias_count|1350/150|
|four_state_contrasts|375|

|state|rows|每个LEO场景rows|新类/H字段|
|---|---:|---:|---|
|`DA0_REG0`|375|125|`N/A`/`N/A`|
|`DA1_REG0`|375|125|`N/A`/`N/A`|
|`DA0_REG1`|375|125|`REQUIRED`/`REQUIRED`|
|`DA1_REG1`|375|125|`REQUIRED`/`REQUIRED`|

receiver集合为`{3-19,7-7,7-14,8-8,20-1}`，seed集合为`{713102,713103,713104,713105,713106}`，K集合为`{1,5,10}`，new_count集合为`{5,10,20}`；三种LEO场景各500行，四状态×场景各125行。该表只证明结构闭合，不给出性能优劣或晋级判断。

## 主agent性能分析

以下数值均来自同一个不可变`score.json`；准确率、H与floor使用百分数，差值使用百分点。总体值为375个`outer×scene`联合行的等权均值，不拼接不同run或不同候选的边际极值。

### 四状态总体结果

|状态|old BA|old floor|seen-new acc|H old/new|all BA|all floor|
|---|---:|---:|---:|---:|---:|---:|
|`DA0_REG0`（域适应前/注册前）|72.640|39.773|N/A|N/A|N/A|39.773|
|`DA1_REG0`（域适应后/注册前）|72.649|39.453|N/A|N/A|N/A|39.453|
|`DA0_REG1`（域适应前/注册后）|43.071|5.600|23.411|28.851|29.119|0.707|
|`DA1_REG1`（域适应后/注册后）|42.871|5.293|23.375|28.805|29.044|0.653|

### 因果差值

|效应|old BA|old floor|seen-new acc|H old/new|all BA/all floor|判读|
|---|---:|---:|---:|---:|---:|---|
|DA效应（注册前）：`DA1_REG0-DA0_REG0`|+0.009|-0.320|N/A|N/A|N/A/-0.320|old BA近零，floor下降|
|DA效应（注册后）：`DA1_REG1-DA0_REG1`|-0.200|-0.307|-0.037|-0.045|-0.076/-0.053|联合任务总体为负|
|注册效应（无DA）：`DA0_REG1-DA0_REG0`|-29.569|-34.173|N/A|N/A|N/A/-39.067|新类进入竞争后旧类显著退化|
|注册效应（有DA）：`DA1_REG1-DA1_REG0`|-29.778|-34.160|N/A|N/A|N/A/-38.800|DA没有修复注册干扰|
|交互效应（difference-in-differences）|-0.209|+0.013|N/A|N/A|N/A/+0.267|old BA交互为负|

K=1的75个scene行按冻结设计使用exact same-IQ alias，因此所有DA差值严格为0；这只是身份校验，不是性能增益。排除K=1后，100个独立outer（300个scene行）的注册后DA效应为：old BA`-0.250`（按outer聚合的近似95%CI`[-0.458,-0.042]`）、seen-new acc`-0.046`（`[-0.175,+0.083]`）、H`-0.057`（`[-0.153,+0.040]`）。严格同时提升注册后old BA、seen-new acc和H的只有23/300个scene行（7.7%）；同时提升old BA与H的为57/300（19.0%）。这不支持稳定正收益。

### K与新类数分层

|K/new|DA0_REG1 old/new/H|DA1_REG1 old/new/H|DA效应 old/new/H|结论|
|---|---|---|---|---|
|10/5|52.178/41.853/44.716|52.000/41.587/44.553|-0.178/-0.267/-0.162|三项均退化|
|10/10|46.200/28.127/34.013|46.033/28.200/34.042|-0.167/+0.073/+0.029|新类与H微升，但old下降|
|10/20|42.922/17.297/24.217|42.700/17.287/24.176|-0.222/-0.010/-0.041|总体退化|
|5/20|41.856/15.323/21.850|41.422/15.343/21.798|-0.433/+0.020/-0.052|old与H退化|
|1/20|32.200/14.457/19.457|32.200/14.457/19.457|0/0/0|same-IQ alias，不计增益|

### receiver分层与局部正值边界

|receiver|注册后DA效应 old BA|seen-new acc|H|判读|
|---|---:|---:|---:|---|
|20-1|-0.178|+0.047|-0.030|混合偏负|
|3-19|+0.089|+0.093|+0.126|局部均值正向|
|7-14|-0.678|-0.083|-0.221|明确负向|
|7-7|-0.133|-0.190|-0.184|明确负向|
|8-8|-0.100|-0.050|+0.082|old/new下降|

确有局部正值：`receiver=3-19,K=10,new=10`的15个`seed×scene`行均值为old BA`+0.556`、seen-new acc`+0.300`、H`+0.402`。但其中严格同时提升old BA与H的只有5/15行，而且该单元是完整25个`receiver×K/new`单元中的事后切片；不能把它称为“正收益版本”，也不能据此选择receiver或重跑调参。

### 与D92及历史路线的可比边界

`D92-Lite-PR160/r6`因真实top tie在完整预测闭合前技术停止，状态为`NO_PERFORMANCE_RESULT`，没有合法四状态score；因此本报告不能伪造FA-RDCE3+qKNN与D92的数值排名。当前r11证明的是：在同一FA-RDCE3+qKNN预测路径内，DA0到DA1没有形成整体或注册后的联合提升；同时注册本身使old BA下降约29.6个百分点、old floor下降约34.2个百分点，分类头的旧/新类竞争仍是主要缺陷。

### 晋级决定

`NEXT-R5-FA-RDCE3-Q-TARGET125`不晋级。理由不是单个弱指标，而是完整矩阵上的联合证据：注册前DA近零且floor下降；注册后old BA、seen-new acc、H、all BA与floor总体均下降；非K1有效行仅7.7%同时提升old/new/H；receiver方向不一致。停止继续调FA残差强度、qKNN邻居数或事后receiver切片。下一轮研发应更换联合分类头/注册竞争机制，优先处理旧类logit保持、新类容量分配与old/new平衡，再用小型必要矩阵验证；不得围绕本候选继续做125规模盲调参。

## 终态与边界

|项目|终态|
|---|---|
|run状态|`ARTIFACTS_COMPLETE / RAW_TRUTH_SCORE_CLOSED`；未执行GPU prediction、asset build、prepare、smoke、shard、merge，也未重跑r10。|
|r1-r10|只读；r10 prediction/plan/context字节与SHA保持原值。|
|fresh retry authority|`无`；本轮无技术失败、无停止、无重试。|
|资源清理|truth/score进程均已退出；N607八张GPU回到0%/1MiB；本地SSH进程与N607 TCP22连接均为0。|
|报告与版本化|本报告与`E:\type10-7`镜像报告由同一`apply_patch`内容生成，待逐字节校验后仅在`E:\fa125wt`force-add这两份报告提交；不stage`conversation_index`。|

主agent后续只应读取上述不可变truth/score产物进行性能分析；本runner不作候选比较、方法晋级或新实验决策。

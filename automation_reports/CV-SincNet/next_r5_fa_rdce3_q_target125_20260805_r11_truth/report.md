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

## 终态与边界

|项目|终态|
|---|---|
|run状态|`ARTIFACTS_COMPLETE / RAW_TRUTH_SCORE_CLOSED`；未执行GPU prediction、asset build、prepare、smoke、shard、merge，也未重跑r10。|
|r1-r10|只读；r10 prediction/plan/context字节与SHA保持原值。|
|fresh retry authority|`无`；本轮无技术失败、无停止、无重试。|
|资源清理|truth/score进程均已退出；N607八张GPU回到0%/1MiB；本地SSH进程与N607 TCP22连接均为0。|
|报告与版本化|本报告与`E:\type10-7`镜像报告由同一`apply_patch`内容生成，待逐字节校验后仅在`E:\fa125wt`force-add这两份报告提交；不stage`conversation_index`。|

主agent后续只应读取上述不可变truth/score产物进行性能分析；本runner不作候选比较、方法晋级或新实验决策。

# Phase1 ICMT postfreeze v2报告

状态：`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标与唯一判定

|字段|冻结值|
|---|---|
|run ID|`phase1_icmt_postfreeze_20260810_v2`|
|训练输入|`phase1_icmt12_20260810_v1`的12个`final_ssdg.pth`；训练已技术闭合，postfreeze尚未读取性能|
|目标|执行12个ICMT专用clean导出、12个source-only LEO导出、12个proxy评分和6个同fold C/G pair，共42步，一次性完成分类floor与连续unknown proxy双门|
|C|各fold的`F{fold}C_ICMT12`|
|G|各fold的`F{fold}G_ICMT12`；唯一训练差异为冻结`lambda_icmt=.05`|
|结论规则|六折完整门全部通过才可进入主控复核；任一非补偿门失败即`REJECT_P1_ICMT_PERMANENT`，不调参、不换折、不重试|
|实现commit|`7f6f4cfea1fa1af154997c3ab1ccd3d02616d8f3`|
|独立复核|实际实现`P0=0、P1=0、ALLOW`；本地41项聚焦回归、pycompile、bash语法、dry-run42和diff-check均通过|
|当前声明|仅确认实现与训练工件就绪；没有postfreeze性能结果|

## 2.冻结实现、SHA与验证

|文件|作用|SHA256|
|---|---|---|
|`code/export_phase1_icmt_features.py`|严格重建L/U/V；只forward L、V与冻结proxy，U forward=0|`6a84de402784e27af0488037b4f8c9f4aa51be44396356b9fe314164bd349614`|
|`code/export_phase1_icmt_leo_features.py`|source-only三场景LEO导出及数据/物理键绑定sidecar|`fd39c77209bfc5548c80aa9ef45abe8b48a0d71e6466fb71fe776aa5575ec585`|
|`code/evaluate_phase1_icmt_postfreeze_pair.py`|L-only Gaussian-NLL、分类门、proxy双门与F6原始工件重算|`a76123e66fdac7961b4535724aead6f1d7a48cf5deda0468929b5773382ab858`|
|`code/scripts/launch_phase1_icmt_postfreeze_20260810.sh`|冻结42步launcher|`3607b741655823df1c98d8ff2d086e0a3df87673aada50c3db5a460f17a6829b`|
|`code/tests/test_phase1_icmt_postfreeze.py`|数据角色、公式、绑定、F6重算与篡改负测|`cf4fac55510a4d98d071330f8f5950bc9e45002d775d4d6d5dea481cb457c934`|
|`analysis/phase1_icmt_design_20260810.md`|冻结设计、追踪与历史复盘|`ee731b06d6781645e6c137b66883fa761b8d295bf454ab956bb02d4f78382120`|

本地在`ssr-gpu`环境已验证：

- 相关Python文件`py_compile`通过；
- ICMT postfreeze测试`31 passed`，GD模板回归`10 passed`；
- launcher `bash -n`通过；
- dry-run精确展开`42=12 clean+12 LEO+12 proxy+6 pair`；
- `git diff --check`与新增文件空白检查通过；
- 独立复核确认F6不信任历史pair自报字段，而是重读当前原始工件、核SHA并重算摘要、delta与门；
- 独立复核确认LEO逐scenario绑定TX、RX、day，proxy固定400条且JSON、CSV、NPZ、物理键和当前SHA闭合。

## 3.数据角色、Gaussian与proxy固定输入

ManySig固定路径为`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。source days=`2021_03_01,2021_03_08`，source RX=`1-1,1-19,14-7,18-2,19-2,2-1`，三场景=`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`，LEO seed=`7281718`，export seed=`7281105`。

每臂clean导出严格重建训练的local4与L/U/V索引。Gaussian只用L的`z_id`拟合；known连续分数只用V；unknown连续分数只用冻结proxy；U不加载样本、不forward、不持久化特征。分类clean门只读V，LEO分类门只读source-only三场景NPZ。V、proxy与LEO均不参与fit、校准、选参或更新。

几何采用float64 totalized L2：正范数行为`z/||z||₂`，精确零范数行为映射到零向量；所有行保留，nonfinite仍直接失败。每类使用ddof=1方差，逐维class-equal pooled variance、`.9/.1`shrink和`1e-6`floor；完整Gaussian NLL与stable logsumexp产生连续`u`。

proxy选择冻结为：

|字段|冻结值|
|---|---|
|days|`2021_03_01,2021_03_08`|
|RXs|`1-1,1-19,14-7,18-2,19-2,2-1`|
|selection seed|`7281148`|
|max samples per TX|`400`|
|每臂proxy总数|`400`|

clean exporter拒绝上述值漂移，并封存selection SHA与400条唯一physical receipt。pair与F6均强制`expected_proxy_count=400`，同时闭合NPZ physical、proxy JSON、score CSV、路径和当前SHA；同步缩行或替换原始工件必须失败。

## 4.六折非补偿门

|门|冻结判定|
|---|---|
|clean floor|6/6折的overall、min-class、min-RX、min-day均不得低于同折C超过2pp|
|LEO floor|18/18个fold×scenario格的四项floor均不得低于对应C超过2pp|
|LEO overall|每fold三场景overall等权均值`G−C>=0`，且全18格overall等权均值`G−C>=0`|
|proxy AUROC|每fold`AUROC_G−AUROC_C>0`|
|proxy u-gap|每fold`(mean u_proxy−mean u_V)_G−(mean u_proxy−mean u_V)_C>0`|
|proxy联合门|两项严格正增益必须6/6同时成立；均值或其他fold不得补偿|

F6只在F1–F5不可变pair存在后聚合；它必须重读六折C/G clean NPZ、LEO NPZ及binding、proxy JSON/CSV和final checkpoint，核当前SHA、matrix ID、output root、training root、arm、fold、head/class order、source TX与正式proxy选择，然后调用同一冻结函数重算所有摘要、delta和fold gates。任何字段漂移、旧schema、跨run或原始工件替换均失败。

## 5.冻结训练输入与42步矩阵

训练根只读：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_icmt12_20260810_v1`。必须存在以下12个final checkpoint：`F1C/F1G`至`F6C/F6G`，目录名均为`F{fold}{arm}_ICMT12`。训练报告已确认12/12自然闭合、终态合同通过、无技术异常；Runner启动前仍需只读核路径与字节身份，不下载checkpoint。

42步按以下GPU矩阵并行执行每个候选内部的clean→LEO→proxy，全部12个候选成功后CPU串行执行F1至F6 pair：

|GPU|候选|
|---:|---|
|0|F1C、F5G|
|1|F1G、F5C|
|2|F2C、F6G|
|3|F2G、F6C|
|4|F3C|
|5|F3G|
|6|F4C|
|7|F4G；与其它run合计不得超过每卡2个训练进程|

## 6.N607路径与唯一启动命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt_postfreeze_20260810_v2_7f6f4cfe`|
|CWD|`<release>/code`|
|training root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_icmt12_20260810_v1`|
|postfreeze root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_icmt_postfreeze_20260810_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt_postfreeze_20260810_v2`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt_postfreeze_20260810_v2_launcher.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；调用端超时先只读确认是否已landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt_postfreeze_20260810_v2_7f6f4cfe/code && nohup env POSTFREEZE_RUN_ID=phase1_icmt_postfreeze_20260810_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt_postfreeze_20260810_v2_7f6f4cfe/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_icmt12_20260810_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt_postfreeze_20260810_v2_7f6f4cfe/code/scripts/launch_phase1_icmt_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt_postfreeze_20260810_v2_launcher.out 2>&1 < /dev/null &
```

## 7.发布、健康停止与工件回收

唯一Runner先执行direct N607 preflight；启动前确认新release、run、log、outer和临时archive路径不存在，核对实现commit/full-tree archive/member、ManySig、12个训练final、launcher语法与dry-run42。release必须包含同commit完整依赖树，不能只打包6个目标文件。落地后只做必要的pycompile/help/bash-n/dry-run42，不安装包、不添加发布层。

启动后记录wrapper、launcher、12个candidate PID、CWD、cmdline、run/log绑定、GPU映射及日志增长。停止只允许路径/hash/覆盖风险、数据或checkpoint绑定漂移、Traceback/OOM/CUDA、nonfinite、确定性执行异常、缺必要工件或两个distinct候选出现同一预测前确定性异常；不得按accuracy、floor、AUROC、u-gap或任何性能值早停。若技术停止，只终止已证明属于本run的进程树，保留partial，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

技术成功预期：12个clean NPZ、12个LEO NPZ及binding、12个proxy JSON/CSV、6个pair JSON和完整stdout。只回收JSON、CSV、日志、PID、completion、manifest、逐项SHA等小工件；不下载`.pth`和特征NPZ。Runner只报告技术闭合，不解释性能。主控收到完整6折pair与配套小工件后，才读取同run性能并按§4给出唯一裁决。

## 8.结果占位

|fold|clean四floor|LEO 3格四floor|三场景overall|proxy AUROC增量|proxy u-gap增量|fold结论|
|---:|---|---|---:|---:|---:|---|
|F1|待运行|待运行|待运行|待运行|待运行|`NO_PERFORMANCE_RESULT`|
|F2|待运行|待运行|待运行|待运行|待运行|`NO_PERFORMANCE_RESULT`|
|F3|待运行|待运行|待运行|待运行|待运行|`NO_PERFORMANCE_RESULT`|
|F4|待运行|待运行|待运行|待运行|待运行|`NO_PERFORMANCE_RESULT`|
|F5|待运行|待运行|待运行|待运行|待运行|`NO_PERFORMANCE_RESULT`|
|F6|待运行|待运行|待运行|待运行|待运行|`NO_PERFORMANCE_RESULT`|


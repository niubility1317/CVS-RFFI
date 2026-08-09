# Phase1 ICMT postfreeze v2报告

状态：`ARTIFACTS_COMPLETE / TECHNICAL_ONLY / NO_PERFORMANCE_RESULT`

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

## 9.Runner技术记录（落地前）

记录时间：2026-08-10；角色：Luna/max唯一N607 Runner。当前仍未启动postfreeze任务，以下仅为落地前只读核验与归档证据，不构成性能结果。

|项目|证据|
|---|---|
|直连预检|`tools\\n607_ssh_preflight.ps1`通过；N607=`dell-DSS8440`，项目根可见，8×RTX3090可见；预检后本地无残留`ssh.exe`或N607:22连接|
|目标覆盖核验|release、postfreeze run、log root、outer和远端临时archive均`ABSENT`|
|既有任务|GPU7存在既有SCB v4构建PID=`608786`（父PID=`608774`），GPU0–6空闲；不干预该任务或其它run|
|ManySig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|
|完整归档|实现commit=`7f6f4cfea1fa1af154997c3ab1ccd3d02616d8f3`；本地无prefix完整树archive共4900项，bytes=`34881856`，SHA256=`28ee60ccdd5fccce01448b5c41f6eb92a06a38fea5276ca4533e411e6b2adf51`；临时远端目标=`/tmp/phase1_icmt_postfreeze_20260810_v2_7f6f4cfe_fulltree.tar.gz`|

### 9.1训练checkpoint只读身份

|候选|bytes|SHA256|
|---|---:|---|
|F1C|6968891|`26f10abf883e69115db694c6cf963f21a22425dcd95a1af5a4be4f2ea231a6bd`|
|F1G|6970683|`b11753a4d4735997d0b14731a078ef356ebf3f10227ec29ff0de07d204050921`|
|F2C|6968891|`ef563e8ab269c6c8f9967b8e5f002ef199c3baad1ca1f148d11f0789f29d482a`|
|F2G|6970683|`6ad7edea5154b2fbd8491ba81a5251e5d43435c958254e6f111a66307a9ba00a`|
|F3C|6968891|`dd559e12bb6bc77fe2b9971ac9619ed9230717b8ee913cf02171579f9ee14665`|
|F3G|6970683|`4b50a5694883bf2a71449d039f08eb90500c59b45ef66487f6bb08516ed580b9`|
|F4C|6968891|`478c523213dd115f27cc4a32c591607e5bf2b42c9e307b945a30abd337015e57`|
|F4G|6970683|`24fd30fdb9df3ac12ce74ba9e275b5774497c88f097ccb26210fba330386155f`|
|F5C|6968891|`b569682222e837f1286b5f61d67978489c4f01e9b4cc514644c55b567ea6d78e`|
|F5G|6970683|`b0393bbe294968fa40acb241ef764dd0c563b2ae8035c2db7cacfb4a7a35e1d1`|
|F6C|6968891|`864f1164516127d62e789226e9c53a63fb99a511b8e3b833368f05b7f8a4f18c`|
|F6G|6970683|`5cb7708213d5e858cb746e9266edac08cf5de7a80b2366a80410716ed9ea6009`|

### 9.2同commit归档成员的LF映射

说明：§2中的冻结SHA是Windows工作树直接字节SHA；release由实现commit的无prefixarchive解包，Linux成员按LF字节落地。以下远端SHA与本地archive解包成员一致，属于同一commit的明确EOL映射，不是算法改写或替换。

|文件|§2工作树SHA|release归档LF SHA|
|---|---|---|
|`code/export_phase1_icmt_features.py`|`6a84de402784e27af0488037b4f8c9f4aa51be44396356b9fe314164bd349614`|`b0c4c2b6d8e1e31570f4c003dd1fdd24b3af2878f69aacc4ad5605f44f507d33`|
|`code/export_phase1_icmt_leo_features.py`|`fd39c77209bfc5548c80aa9ef45abe8b48a0d71e6466fb71fe776aa5575ec585`|`ec6aedacc36db7e265c1fc5b90f0980ab03020c33821bc513acc86171e2de7c6`|
|`code/evaluate_phase1_icmt_postfreeze_pair.py`|`a76123e66fdac7961b4535724aead6f1d7a48cf5deda0468929b5773382ab858`|`33265cf2f75fd8b322d959489ecdf30873e215ec4394fc57cdd049cbae106f06`|
|`code/scripts/launch_phase1_icmt_postfreeze_20260810.sh`|`3607b741655823df1c98d8ff2d086e0a3df87673aada50c3db5a460f17a6829b`|`3607b741655823df1c98d8ff2d086e0a3df87673aada50c3db5a460f17a6829b`|
|`code/tests/test_phase1_icmt_postfreeze.py`|`cf4fac55510a4d98d071330f8f5950bc9e45002d775d4d6d5dea481cb457c934`|`88a31a212f5e5c0e6a1c96beeaadbc36a7667108529934913cddc359b3a58923`|
|`analysis/phase1_icmt_design_20260810.md`|`ee731b06d6781645e6c137b66883fa761b8d295bf454ab956bb02d4f78382120`|`84b2680809ce3c51acfa3ceeb63c346d88f5171103fcab6f57082df5d8f1ef4e`|

### 9.3远端静态核验

release落地后仅执行必要静态检查：5个Python入口（含测试与实际proxy脚本）`py_compile=PASS`；4个入口`--help=PASS`；launcher`bash -n=PASS`；冻结环境变量下`--dry-run`精确输出`42`条（`12 clean+12 LEO+12 proxy+6 pair`）并通过计数断言。未安装包、未读取性能、未创建postfreeze输出目录。

### 9.4启动前版本与调用登记

Git镜像报告commit=`b012ddaa`（仅本报告文件）；实现commit=`7f6f4cfea1fa1af154997c3ab1ccd3d02616d8f3`。远端release已按完整树archive落地并完成成员SHA核验；`postfreeze run/log/outer`仍不存在。下一步只调用§6逐字冻结命令一次，调用端异常仅只读确认是否landed，禁止重发。

## 10.Runner技术终态

状态：`ARTIFACTS_COMPLETE / TECHNICAL_ONLY / NO_PERFORMANCE_RESULT`。冻结命令实际调用1次；SSH调用端约34秒超时后仅只读确认已landed，未重发。wrapper PID=`630495`、launcher PID=`630496`；launcher按冻结GPU矩阵登记12个candidate PID，随后wrapper/launcher/所有子进程自然退出。GPU7既有SCB v4构建PID=`608786`持续存在，未被干预。

|fold|arm|candidate PID|GPU|candidate|日志|
|---:|---|---:|---:|---|---|
|1|C|630499|0|F1C_ICMT12|`F1C_ICMT12.out`|
|5|G|630500|0|F5G_ICMT12|`F5G_ICMT12.out`|
|1|G|630501|1|F1G_ICMT12|`F1G_ICMT12.out`|
|5|C|630502|1|F5C_ICMT12|`F5C_ICMT12.out`|
|2|C|630504|2|F2C_ICMT12|`F2C_ICMT12.out`|
|6|G|630505|2|F6G_ICMT12|`F6G_ICMT12.out`|
|2|G|630507|3|F2G_ICMT12|`F2G_ICMT12.out`|
|6|C|630508|3|F6C_ICMT12|`F6C_ICMT12.out`|
|3|C|630509|4|F3C_ICMT12|`F3C_ICMT12.out`|
|3|G|630511|5|F3G_ICMT12|`F3G_ICMT12.out`|
|4|C|630513|6|F4C_ICMT12|`F4C_ICMT12.out`|
|4|G|630514|7|F4G_ICMT12|`F4G_ICMT12.out`|

### 10.1矩阵闭合与技术检查

|项|结果|
|---|---|
|42步|12 clean、12 LEO、12 proxy、6 pair；candidate PID清单13行（含表头）|
|小工件计数|12 clean NPZ、12 LEO NPZ、12 binding、12 proxy JSON、12 proxy CSV、6 pair JSON、18非空日志|
|技术绑定|6/6 pair的`technical_binding=true`，schema=`cvs.phase1.icmt_postfreeze_pair.v2`，matrix/root/training-root绑定通过；12/12 LEO binding的`all_scenarios_complete=true`与`all_source_rows_reconstructed=true`|
|错误指纹|Traceback、RuntimeError、CUDA error、OOM、非零`nonfinite_rows`/`total_nonfinite_rows`均为0；未按任何性能值停止|
|进程/GPU收尾|完成检查时本run进程=0；GPU7的SCB v4仍在；本地SSH客户端及N607:22连接均清理为0|

### 10.2小工件bundle与逐项SHA

|项目|路径或SHA|
|---|---|
|远端bundle|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt_postfreeze_20260810_v2/phase1_icmt_postfreeze_20260810_v2_small_bundle_v2.tar.gz`|
|本地bundle|`automation_reports/CV-SincNet/phase1_icmt_postfreeze_20260810_v2/remote_artifacts/phase1_icmt_postfreeze_20260810_v2_small_bundle_v2.tar.gz`|
|bundle|64成员、0个`.npz/.pth`；bytes=`3860792`；SHA256=`dd27a9edb7e51e0ff1937aeb251b0462b1fee2e555af94bf8109aa7995f6fc5f`|
|逐项manifest|`remote_artifacts/small_bundle_v2_extract/logs/phase1_icmt_postfreeze_20260810_v2/phase1_icmt_postfreeze_20260810_v2_small_manifest_v2.tsv`；63项，SHA256=`fd810411ce84aba4470517bb1add443d2ec52759fadef39977854c6a73c545a4`；本地逐项校验`0 mismatch`|
|完成JSON|`remote_artifacts/small_bundle_v2_extract/logs/phase1_icmt_postfreeze_20260810_v2/phase1_icmt_postfreeze_20260810_v2_completion.json`；SHA256=`6d506b74591e1259d59dfa06f4a287438b8d90aee5b3bfd4d896eaa087bba6db`|

首次小bundle尝试因outer日志路径登记层级错误在打包前中止，仅留下未完成manifest，不影响实验输出；未覆盖或删除任何既有工件。校正v2使用实际outer路径并完成上述验证。未下载checkpoint或特征NPZ；远端release、run、log、fulltree archive及全部原始工件保留。

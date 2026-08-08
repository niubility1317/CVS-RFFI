# Phase1 CCPC-LEO v4 postfreeze配对评估报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`ANALYZED / REJECT_CCPC_LEO_NO_RETRY / NO_PHASE3_PROMOTION`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.实验目标与冻结假设

实验ID：`phase1_ccpc_leo12_20260809_v4_postfreeze_v1`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

对已完整训练的六折C/G final-only checkpoint执行一次冻结postfreeze矩阵，回答CCPC是否在不做RX/domain对齐、不训练拒识head、不访问proxy/held/LEO进行fit、校准或选参的前提下，保护clean known并改善LEO表示。训练run为`phase1_ccpc_leo12_20260809_v4`；12个候选均E040、技术终态完整。G相对C仅增加固定`T=0.12、lambda=0.02`的CCPC。

## 2.冻结矩阵与数据角色

共42步：12个clean导出、12个source-only LEO导出、12个source校准proxy连续评分、6个同fold C/G配对评分。每个候选使用自己的`final_ssdg.pth`；C/G同fold继承相同源checkpoint、相同TX划分与seed。clean导出角色固定为source=1600、target_old=400、proxy_unknown=400；LEO导出仅source=1600，场景固定`leo_clear_weak、leo_low_elev_weak、leo_rain_weak`。所有评分为只读，proxy、held和LEO均零fit、零校准、零选参。

GPU映射沿用训练矩阵，每卡最多2条候选pipeline：GPU0=F1C+F5G，GPU1=F1G+F5C，GPU2=F2C+F6G，GPU3=F2G+F6C，GPU4=F3C，GPU5=F3G，GPU6=F4C，GPU7=F4G。候选内部串行执行clean export→LEO export→proxy score；12条候选完成后串行执行6个pair score。

## 3.版本与本地证据

复用已落地不可变release commit：`ad261d2887d867c1993bca2f993f2d7b969000e6`；训练实现commit：`753161c9127f72498507c8bbf4d7994bc4b7e698`。postfreeze文件：

- `code/scripts/eval_phase1_ccpc_leo_pair.py`
- `code/scripts/launch_phase1_ccpc_leo_postfreeze_20260809.sh`
- `code/tests/test_phase1_ccpc_leo_postfreeze.py`

本地`ssr-gpu`验证：postfreeze focused pytest=12 passed，launcher `bash -n`通过，dry-run精确42条；独立复核`APPROVE / Critical=0 / Important=0`。本报告在启动前写入root控制面并镜像到Git承载面。

## 4.N607路径与唯一启动命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28`
- 训练输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4`
- postfreeze run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1`
- postfreeze log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code && nohup setsid env POSTFREEZE_RUN_ID=phase1_ccpc_leo12_20260809_v4_postfreeze_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code/scripts/launch_phase1_ccpc_leo_postfreeze_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.技术门、性能门与预期artifact

启动前核对release hash、12个final checkpoint、ManySig、目标run/log/outer不存在、GPU/活动任务。启动后核launcher/CWD/cmdline、candidate PID/GPU、日志增长、输出计数与异常指纹。仅路径/P0/覆盖错误、输出覆盖风险、OOM/CUDA、至少2条候选同一确定性异常或无进展触发技术停止；不查看性能决定停止，retry=`NO`。

成功artifact：12个clean NPZ、12个LEO NPZ、12个proxy JSON+CSV、6个pair JSON、candidate/pair日志与PID/完成回执、manifest。只回收小JSON/CSV/log/receipt，不下载NPZ/checkpoint。

五项非补偿门：①技术健康；②clean known六折全部overall/minclass/minRX/minday的G-C≥-2pp；③LEO 18个fold×scenario全部四项G-C≥-2pp且aggregate overall改善；④proxy连续排序相对C同向；⑤checkpoint SHA、strict-load、元数据与artifact闭环。任一失败即`REJECT_CCPC_LEO_NO_RETRY`，不进入Phase3。

## 6.运行终态与小artifact回收（2026-08-09）

- Direct N607 preflight为Connection refused；使用已验证lab bridge完成短连接。复用release commit=`ad261d2887d867c1993bca2f993f2d7b969000e6`、训练实现=`753161c9127f72498507c8bbf4d7994bc4b7e698`，release归档SHA=`93532637c41e491b748e522059dd7076face2090de8c5ce031e80e903aa0e559`，ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。postfreeze脚本以`bash`调用；远端脚本/评估器hash已核验，py_compile、help、bash-n及42行dry-run通过。
- 冻结launcher唯一启动一次，detached PID=`4091723`；candidate PID/GPU记录写入`candidate_pids.tsv`：GPU0=F1C+F5G，GPU1=F1G+F5C，GPU2=F2C+F6G，GPU3=F2G+F6C，GPU4=F3C，GPU5=F3G，GPU6=F4C，GPU7=F4G。12条candidate pipeline均完成clean→LEO→proxy，随后6条同fold pair评分完成；detached shell未持久化退出码，completion中的stage exit=0按闭环artifact与pair阶段继续执行记录。
- 结构审计：12/12 clean NPZ各2400行，角色顺序`source→proxy_unknown→target_old`且1600/400/400；12/12 LEO NPZ各1600行，角色为source-only，三场景计数544/544/512；24/24 strict-load、metadata长度/有序绑定、checkpoint SHA全通过。12/12 proxy CSV各2401行；6/6 pair JSON schema与角色/TX/day/RX绑定通过，fit/calibration/model-selection/threshold均为false，错误指纹计数0。
- 远端completion：`logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1/completion.tsv`，18行记录+header，SHA=`e1ba20ba8cc97c572cbe296f3b45a29318dbbdfd1658c5e9ce1939982f595636`。manifest本地路径：`E:\type10-7\automation_reports\CV-SincNet\phase1_ccpc_leo12_20260809_v4_postfreeze_v1\artifacts\runs\phase1_ccpc_leo12_20260809_v4_postfreeze_v1\manifest.json`，SHA=`967010a5a6ad1f1eb5a712fd48bf27407bf14c18e344ad9c94306746993fa2f3`；NPZ/Pair审计均all_ok，51个小文件清单。
- 小artifact已回收到`E:\type10-7\automation_reports\CV-SincNet\phase1_ccpc_leo12_20260809_v4_postfreeze_v1\artifacts`；本地52文件（含manifest），逐项小文件SHA/大小校验0错误，未下载NPZ/checkpoint。传输tar SHA=`cd8ee0c87f8cfaaaded1cbe2368bd5d544148d25ba8fa72b62141d2cb39119a9`（531382 bytes），远端临时tar已删除。
- 终态复核：run-owned进程0；GPU0–7均0%/1MiB；本地SSH进程0、TCP22=0。无retry、无性能解释；该报告与Git镜像已同步更新，未提交commit。

## 7.完整同配对结果

主控完整读取12个训练`metrics_epoch.jsonl`的480条epoch记录、12个训练CSV、12个训练stdout、12个proxy metrics JSON、12个proxy scores CSV的28800条数据行、6个pair JSON及19个postfreeze stdout。训练和postfreeze日志均无Traceback、RuntimeError、CUDA、OOM或fail-closed指纹；已知N/A占位未作为异常。

### 7.1 Clean source-known保护

下表均为准确率百分数；`Δ=G-C`，单位为百分点。六折四项非补偿门全部通过，均值Δ为overall -0.010、min-class 0.000、min-RX +0.063、min-day +0.078个百分点。

|Fold|C overall|G overall|Δ overall|C min-class|G min-class|Δ min-class|C min-RX|G min-RX|Δ min-RX|C min-day|G min-day|Δ min-day|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|F1|99.188|99.000|-0.187|98.000|98.000|+0.000|96.604|95.849|-0.755|99.022|98.900|-0.122|
|F2|99.250|99.250|+0.000|98.000|98.000|+0.000|96.981|96.981|+0.000|99.233|99.233|+0.000|
|F3|99.125|99.250|+0.125|97.500|98.000|+0.500|96.226|96.981|+0.755|98.900|99.144|+0.244|
|F4|99.312|99.250|-0.062|98.750|98.500|-0.250|97.736|98.113|+0.377|99.233|99.233|+0.000|
|F5|97.812|97.875|+0.062|96.000|96.000|+0.000|91.698|92.830|+1.132|97.698|97.800|+0.101|
|F6|97.375|97.375|+0.000|91.750|91.500|-0.250|91.321|90.189|-1.132|95.844|96.088|+0.244|

### 7.2 LEO弱信道配对结果

每一行使用相同fold、相同物理样本顺序和对应C/G final checkpoint；准确率单位为百分数，Δ单位为百分点。只有6/18个fold×scenario单元同时满足四项Δ≥-2pp，远低于冻结的18/18门。

|Fold|Scenario|C overall|G overall|Δ overall|C min-class|G min-class|Δ min-class|C min-RX|G min-RX|Δ min-RX|C min-day|G min-day|Δ min-day|门|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|F1|leo_clear_weak|95.956|95.772|-0.184|90.972|88.194|-2.778|86.598|83.505|-3.093|95.946|95.342|-0.604|FAIL|
|F1|leo_low_elev_weak|95.404|95.404|+0.000|91.406|89.062|-2.344|87.640|87.640|+0.000|95.192|94.712|-0.481|FAIL|
|F1|leo_rain_weak|95.312|95.898|+0.586|92.188|92.969|+0.781|79.747|86.076|+6.329|95.139|95.536|+0.397|PASS|
|F2|leo_clear_weak|94.669|93.750|-0.919|90.278|88.194|-2.083|86.598|84.536|-2.062|94.144|93.243|-0.901|FAIL|
|F2|leo_low_elev_weak|89.706|90.074|+0.368|82.031|78.125|-3.906|68.539|71.910|+3.371|89.423|89.423|+0.000|FAIL|
|F2|leo_rain_weak|90.820|91.406|+0.586|85.938|86.719|+0.781|78.481|78.481|+0.000|90.179|91.319|+1.141|PASS|
|F3|leo_clear_weak|94.485|95.037|+0.551|90.972|90.972|+0.000|87.629|86.598|-1.031|94.099|95.031|+0.932|PASS|
|F3|leo_low_elev_weak|86.029|87.684|+1.654|68.750|78.906|+10.156|67.416|65.169|-2.247|85.417|87.202|+1.786|FAIL|
|F3|leo_rain_weak|85.742|87.695|+1.953|71.094|80.469|+9.375|74.684|77.215|+2.532|81.597|84.722|+3.125|PASS|
|F4|leo_clear_weak|92.647|93.382|+0.735|85.417|86.111|+0.694|81.443|83.505|+2.062|91.615|91.925|+0.311|PASS|
|F4|leo_low_elev_weak|89.890|90.625|+0.735|77.344|77.344|+0.000|73.034|69.663|-3.371|88.393|89.881|+1.488|FAIL|
|F4|leo_rain_weak|89.062|90.039|+0.977|75.000|78.906|+3.906|82.278|84.810|+2.532|84.375|86.111|+1.736|PASS|
|F5|leo_clear_weak|80.147|74.449|-5.699|67.188|47.222|-19.965|55.670|48.454|-7.216|78.571|68.468|-10.103|FAIL|
|F5|leo_low_elev_weak|71.691|68.382|-3.309|56.250|39.062|-17.188|44.944|53.623|+8.679|70.192|62.500|-7.692|FAIL|
|F5|leo_rain_weak|69.141|63.086|-6.055|47.656|29.688|-17.969|43.421|36.842|-6.579|67.857|62.500|-5.357|FAIL|
|F6|leo_clear_weak|79.779|77.390|-2.390|50.694|36.806|-13.889|65.672|49.254|-16.418|78.378|76.126|-2.252|FAIL|
|F6|leo_low_elev_weak|81.801|82.353|+0.551|69.531|60.938|-8.594|57.303|57.303|+0.000|78.846|79.808|+0.962|FAIL|
|F6|leo_rain_weak|80.664|79.297|-1.367|60.156|48.438|-11.719|67.089|65.823|-1.266|78.125|76.042|-2.083|FAIL|

18个单元的平均Δ为overall -0.624、min-class -4.152、min-RX -0.988、min-day -0.978个百分点。overall仅10/18改善，四项全门仅6/18通过。F5三场景和F6多项floor出现集中退化，不能由F3/F4局部收益补偿。

### 7.3 Proxy连续排序诊断

proxy仅为source-held开发诊断，不是真实unknown结果。AUROC在5/6折提高，平均Δ=+0.0107；F6从0.6065降至0.5700，未满足六折同向。冻结source阈值下FAR在6/6折下降，平均下降11.875pp，但所有FAR仍高于5%，且该阈值结果不得替代LEO floor门或Phase3真实unknown结果。

|Fold|C AUROC|G AUROC|Δ AUROC|C FAR%|G FAR%|Δ FAR pp|C known-full%|G known-full%|Δ known-full pp|连续排序方向|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|F1|0.5646|0.5921|+0.0274|15.500|13.750|-1.750|89.500|89.188|-0.313|PASS|
|F2|0.5194|0.5287|+0.0093|86.500|85.750|-0.750|91.688|92.625|+0.938|PASS|
|F3|0.6220|0.6307|+0.0087|55.000|45.750|-9.250|91.562|91.062|-0.500|PASS|
|F4|0.5262|0.5623|+0.0361|82.500|42.500|-40.000|89.938|90.000|+0.062|PASS|
|F5|0.5982|0.6173|+0.0192|60.250|44.250|-16.000|89.375|90.250|+0.875|PASS|
|F6|0.6065|0.5700|-0.0365|30.250|26.750|-3.500|90.062|89.938|-0.125|FAIL|

## 8.机制诊断与大样本heldout旁证

CCPC没有形成稳定的“LEO靠近clean且远离其他类”几何。18个配对单元中，G相对C的paired clean-LEO cosine distance在14/18上升，平均增加0.00841；nearest-other-class centroid margin在10/18下降，平均减少0.01360。F5最严重：三场景margin Δ为-0.06195、-0.07423、-0.09757，同时cosine distance Δ为+0.01978、+0.03270、+0.02897。这与F5/F6的min-class和min-RX退化方向一致。

训练终态自带的冻结大样本heldout评估未参与选模，但提供独立旁证。下表仍为同fold G-C百分点变化；F5的三场景aggregate约下降4.81–4.89pp，F3 receiver floor下降2.463pp。它与postfreeze小型严格配对结果共同表明问题是跨fold不稳定，而不是单个导出样本的偶然波动。

|Fold|overall TX Δ|receiver floor Δ|sat mean Δ|sat floor Δ|sat strict floor Δ|clear Δ|low-elev Δ|rain Δ|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|F1|+0.378|+0.038|-0.668|-0.760|-0.680|-0.474|-0.769|-0.760|
|F2|-0.163|-1.063|+1.094|+0.999|+1.220|+1.049|+1.235|+0.999|
|F3|-0.512|-2.463|+0.639|+0.661|+0.593|+0.599|+0.657|+0.661|
|F4|-0.052|+0.163|+1.221|+1.104|+2.008|+1.421|+1.138|+1.104|
|F5|-1.374|+5.225|-4.852|-4.856|-3.018|-4.809|-4.892|-4.856|
|F6|+0.421|-1.275|-0.052|+0.111|+0.610|-0.295|+0.029|+0.111|

## 9.五门裁决与最终结论

|门|结果|证据|
|---|---|---|
|1.技术健康|PASS|12×E040、42步postfreeze闭环、错误指纹0、raw CCPC nonfinite=0|
|2.clean known保护|PASS|6/6折四项Δ均≥-2pp|
|3.LEO floor与总体改善|FAIL|仅6/18单元四项过门；overall平均-0.624pp；F5/F6严重退化|
|4.proxy连续排序同向|FAIL|AUROC仅5/6折同向；F6 Δ=-0.0365，虽FAR 6/6下降但不可补偿|
|5.checkpoint/artifact闭环|PASS|strict-load、checkpoint SHA、ordered metadata、52个小artifact全部闭环|

最终裁决：`REJECT_CCPC_LEO_NO_RETRY / NO_PHASE3_PROMOTION`。CCPC是可运行且不破坏clean平均性能的机制，但其LEO增益高度依赖fold，并未产生高泛化的稳定表征；尤其F5/F6的类别与接收机floor退化违反Phase1晋级条件。停止该固定`T=0.12、lambda=0.02`路线，不调权、不扫温度、不挑fold、不以proxy FAR改善补偿LEO失败，也不把任何结果表述为真实unknown或多卫星协同能力。

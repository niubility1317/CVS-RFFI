# ADV3B03 CORE90邻域新增5-seed满卡实验

- Run ID：`phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1`
- 当前状态：`ANALYZED`
- 目的：在不干扰已运行near3任务的前提下，使用当前空闲GPU3–7各新增1个邻近seed实验，使8张GPU各有1个正式训练实验。
- 类型：纯Phase1接收机域泛化训练；与Phase2无关。
- 启动所有者：当前主Agent；同一Run ID只允许一次正式启动。
- 方法：仅`ADV3B03_MU10_ALPHA20_E200`，从头训练。
- CORE90原训练seed：`392002`。
- 新增seed与GPU：`392000→GPU3`、`392004→GPU4`、`391999→GPU5`、`392005→GPU6`、`391998→GPU7`。
- 已运行但不属于本Run ID的seed：`392001/392002/392003`位于GPU0/1/2，禁止触碰。
- Git分支：`codex/phase1-fasttrust-eff-src5-20260828`。
- code/config commit：`8af305ed82af15df2e475b7fd94a4b2924d69c39`；自动push及独立远端OID回读均为`VERIFIED`。

## 冻结配置

- 数据：`Dataset_WigSig/ManySig.pkl`。
- source receivers：`1,3,4,6,8`。
- 训练天：day1、day2、day3；day4不用于训练。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- epoch：200；batch size：128；final checkpoint：`final_ssdg.pth`。
- 星地增强：与ADV3B02同款concat masked/CE-only拼接增强，`concat_sat_ce_weight=1.0`；课程为`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 训练完成要求：每行E200、final checkpoint、严格重建、source V_select clean及3种LEO场景artifact完整。
- 选种：near3与near5共8个seed全部完成后，仅按source V_select冻结最佳seed；目标接收机结果不得反馈选种、调参、重训或重跑。
- 后续测试：冻结最佳seed后测试全部7个目标接收机×day1/2/3/4的clean与3种LEO场景，零适配且无状态更新。

## 路径与正式命令

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`。
- N607账户：普通账户`szu2070436088`，禁止管理员账户。
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1_release1`。
- 运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1`。

```bash
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1_release1/code/scripts/launch_phase1_adv3b03_core90seed_near5_day123_e200_20260830.py --root /home/szu2070436088/2510044040/CV-SincNet --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1_release1 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-id phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1 --runs-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1.dispatcher.log 2>&1 < /dev/null &
```

## 验证、artifact与停止规则

- 聚焦测试：`15 passed`。
- Python语法编译与`git diff --check`：通过。
- 独立P0/P1审查：无P0/P1；固定seed与GPU3–7映射、独立Run ID、E200及源域四场景闭合路径均正确，审查者未修改文件。
- 预期artifact：5个候选各自的`config.json`、`train.log`、`final_ssdg.pth`、`metrics_joint.json`、4份场景指标与日志、`status.txt=ARTIFACTS_COMPLETE`；日志根含`plan.json`和`final_status.json`。
- 只在错误stage/receiver/day/seed/GPU/场景、目标域或Phase2被访问、输出碰撞、错误checkout、命令不能启动、同一确定性异常至少两行复现、无final checkpoint、严格重建失败或必要评估artifact缺失时，才绑定并停止该Run ID的精确进程树，同时保留partial artifact。
- 低性能、收敛慢或中间指标差不得停止、重启或热补丁；不得触碰near3或其他无关进程。

## Release、smoke与正式启动

- 单release归档SHA256：本地与N607均为`96e3c37db3f1f879a5f8ab92611a670c6c507502cd9effe993bd2c2e0e32a0cf`，传输`VERIFIED`；远端关键入口编译`PASS`。
- 启动前资源：near3的3个主训练进程继续绑定GPU0/1/2；GPU3–7为空闲；near5正式run/log/release路径均不存在。
- 真实smoke：`phase1_adv3b03_core90seed_near5_day123_smoke_e1_20260830_r1`，seed392000、GPU3、E1；final checkpoint为15,025,055字节，终态`ARTIFACTS_COMPLETE`。
- smoke严格评估：clean=35.9111%、leo_clear_weak=31.0222%、leo_low_elev_weak=29.2222%、leo_rain_weak=30.0741%；4个artifact均为epoch1、strict load、无fallback、missing/unexpected/shape mismatch全为0。该数值仅验证闭合。
- 正式外层启动shell PID：`1910685`；dispatcher PID：`1910686`。
- 直属主训练PID与绑定：`1910692 seed392000 GPU3`、`1910694 seed392004 GPU4`、`1910693 seed391999 GPU5`、`1910696 seed392005 GPU6`、`1910695 seed391998 GPU7`。
- 启动回读：5个候选均为`RUNNING`，epoch2–3/200，日志17,721–23,266字节并持续增长；GPU3–7各1个训练进程。连同near3，GPU0–7当前严格各1个正式训练实验。
- 未发现Traceback、CUDA OOM、TRAIN_FAILED、RuntimeError或AssertionError。启动当时状态：`RUNNING`。

## 终态：8-seed完整闭合与源域冻结

- near3与near5共8行均达到`ARTIFACTS_COMPLETE`：每行`metrics_epoch.csv`与`metrics_epoch.jsonl`均为完整200行、epoch严格为1–200；完整训练日志均到epoch200，无非有限loss/accuracy/LR，无Traceback、CUDA OOM、RuntimeError、AssertionError或Killed。
- 8个`final_ssdg.pth`均存在，大小约15.04MB；clean与3种LEO评估均为checkpoint epoch200、`strict_requested=true`、`checkpoint_load_strict=true`、`fallback_used=false`，missing/unexpected/shape mismatch全为0。
- 选种规则保持预登记不变：依次最大化`H(clean,LEO floor)`、LEO mean、LEO floor、clean；仍相同时seed升序。仅使用source V_select，不读取目标接收机结果。
- 冻结seed：`392005`，来自near5；原ADV3B02 CORE90 seed`392002`在本次ADV3B03训练中排名第2。

|排名|seed|clean|clear|low|rain|LEO mean|LEO floor|H(clean,floor)|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|392005|98.1704|88.2667|85.0296|85.7333|86.3432|85.0296|91.1287|
|2|392002|97.8741|88.4963|84.7185|85.7926|86.3358|84.7185|90.8224|
|3|392000|97.6148|87.7481|84.7852|85.2889|85.9407|84.7852|90.7488|
|4|392001|98.2000|87.3556|84.9481|84.0074|85.4370|84.0074|90.5510|
|5|391998|98.3259|87.9852|83.7111|85.1556|85.6173|83.7111|90.4318|
|6|392003|97.8296|88.2889|83.9185|84.7259|85.6444|83.9185|90.3417|
|7|392004|98.0667|86.9037|83.0815|84.8148|84.9333|83.0815|89.9543|
|8|391999|97.7704|85.5704|82.4148|82.9111|83.6321|82.4148|89.4383|

## 冻结seed的全接收机、全日期零适配测试

- 正式有效目标Run ID：`phase1_adv3b03_seed392005_targetrx_alldays_zeroadapt_20260830_r4`。
- checkpoint：`S392005_ADV3B03_MU10_ALPHA20_E200/final_ssdg.pth`，epoch200。
- 覆盖：7个目标接收机×4天=28个cell；每cell每场景6,000条，共168,000条clean，3种LEO场景各168,000条。
- 边界：严格零适配、`state_updates=false`；目标结果未参与seed冻结、调参、训练或重跑。
- 重建：`strict_requested=true`、`checkpoint_load_strict=true`、`fallback_used=false`，missing/unexpected/shape mismatch全为0。
- 完整性：JSON与CSV各84行（28cell×3种LEO）；每行同时记录对应clean，全部数值与样本计数已全量解析并交叉汇总。
- 评估文件SHA256：本地与N607均为`3fe1228e5d1152a58d559f24d46167bb335d5762d6fa36419e83e8a6b407ca95`。
- r1、r2、r3均在写出指标前因确定性技术配置错误退出，分别为旧入口不识别显式参数、依赖版本导致域头维度不匹配、把自动计算的clean误列为LEO场景；三者均保留为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT`，不构成性能结果。r4仅修正执行绑定与场景参数，不改变seed、checkpoint、receiver、day或评估规则。

### 总体结果

|clean|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|LEO mean|LEO floor|
|---:|---:|---:|---:|---:|---:|
|78.4363|62.8625|60.9440|60.8655|61.5573|60.8655|

### 逐接收机汇总（4天合并）

|接收机|clean|clear|low|rain|LEO mean|LEO floor|
|---|---:|---:|---:|---:|---:|---:|
|1-1|82.2125|72.8208|69.7167|69.3000|70.6125|69.3000|
|14-7|64.7000|42.7333|43.6417|44.0125|43.4625|42.7333|
|2-1|87.7708|72.5792|70.1250|70.5917|71.0986|70.1250|
|20-1|71.6625|51.7417|49.8667|49.2667|50.2917|49.2667|
|7-14|83.4833|71.3167|69.1125|68.9750|69.8014|68.9750|
|7-7|80.5917|65.5750|62.9625|62.4292|63.6556|62.4292|
|8-8|78.6333|63.2708|61.1833|61.4833|61.9792|61.1833|

### 逐日期汇总（7个接收机合并）

|day|日期|clean|clear|low|rain|LEO mean|LEO floor|
|---:|---|---:|---:|---:|---:|---:|---:|
|0|2021_03_01|76.6786|59.6667|57.9286|57.9405|58.5119|57.9286|
|1|2021_03_08|80.9143|64.9048|63.4833|63.1476|63.8452|63.1476|
|2|2021_03_15|74.4929|61.2952|59.2095|59.3881|59.9643|59.2095|
|3|2021_03_23|81.6595|65.5833|63.1548|62.9857|63.9079|62.9857|

### 全部28个receiver×day cell

|接收机|day|日期|clean|clear|low|rain|样本数/场景|
|---|---:|---|---:|---:|---:|---:|---:|
|1-1|0|2021_03_01|68.6333|64.2167|61.6667|59.7167|6,000|
|1-1|1|2021_03_08|85.4833|72.6833|71.0333|72.3833|6,000|
|1-1|2|2021_03_15|90.0333|80.5333|76.4667|75.2167|6,000|
|1-1|3|2021_03_23|84.7000|73.8500|69.7000|69.8833|6,000|
|14-7|0|2021_03_01|61.7167|34.8500|36.1667|37.3833|6,000|
|14-7|1|2021_03_08|60.2667|41.5667|42.5333|42.2000|6,000|
|14-7|2|2021_03_15|59.0500|47.0500|46.3833|47.5833|6,000|
|14-7|3|2021_03_23|77.7667|47.4667|49.4833|48.8833|6,000|
|2-1|0|2021_03_01|84.5333|62.0833|61.5333|60.6167|6,000|
|2-1|1|2021_03_08|92.0667|82.8833|80.1000|79.5333|6,000|
|2-1|2|2021_03_15|92.2833|79.6667|74.0333|77.5167|6,000|
|2-1|3|2021_03_23|82.2000|65.6833|64.8333|64.7000|6,000|
|20-1|0|2021_03_01|89.7500|62.3167|57.7000|59.3000|6,000|
|20-1|1|2021_03_08|72.1500|51.6333|51.4500|49.3833|6,000|
|20-1|2|2021_03_15|48.5000|36.9667|37.7000|36.4167|6,000|
|20-1|3|2021_03_23|76.2500|56.0500|52.6167|51.9667|6,000|
|7-14|0|2021_03_01|70.5667|60.1833|59.3333|60.2167|6,000|
|7-14|1|2021_03_08|98.2667|78.2667|73.5667|73.1667|6,000|
|7-14|2|2021_03_15|80.3167|69.1833|68.3167|67.8167|6,000|
|7-14|3|2021_03_23|84.7833|77.6333|75.2333|74.7000|6,000|
|7-7|0|2021_03_01|82.0500|70.8333|66.4667|66.5667|6,000|
|7-7|1|2021_03_08|78.2500|63.7667|63.6000|63.2833|6,000|
|7-7|2|2021_03_15|76.6167|55.5333|54.3667|53.6000|6,000|
|7-7|3|2021_03_23|85.4500|72.1667|67.4167|66.2667|6,000|
|8-8|0|2021_03_01|79.5000|63.1833|62.6333|61.7833|6,000|
|8-8|1|2021_03_08|79.9167|63.5333|62.1000|62.0833|6,000|
|8-8|2|2021_03_15|74.6500|60.1333|57.2000|57.5667|6,000|
|8-8|3|2021_03_23|80.4667|66.2333|62.8000|64.5000|6,000|

### 结果分析

- 从source V_select到目标域，clean由98.1704%降至78.4363%，下降19.7341pp；LEO mean由86.3432%降至61.5573%，下降24.7859pp；LEO floor由85.0296%降至60.8655%，下降24.1641pp。主要瓶颈仍是跨接收机迁移，并在星地信道下进一步放大。
- 接收机差异显著：clean最好为`2-1`的87.7708%，最差为`14-7`的64.7000%，跨度23.0708pp；LEO mean最好为`2-1`的71.0986%，最差为`14-7`的43.4625%，跨度27.6361pp。`14-7`是总体LEO短板，`20-1/day2`是clean最差cell（48.5000%），`14-7/day0`是LEO最差cell（均值36.4667%）。
- 日期也存在漂移：day3总体clean/LEO mean最高，为81.6595%/63.9079%；day2 clean最低，为74.4929%；day0 LEO mean最低，为58.5119%。训练使用day1/2/3并不意味着这些日期在未见接收机上都优于未参与训练的day0，说明receiver×day交互强于单独日期覆盖。
- 与此前seed713104的同协议目标测试相比，本seed clean从77.6054%升至78.4363%（+0.8309pp），但LEO mean从62.7101%降至61.5573%（-1.1528pp）。这不改变选种：seed392005已在读取任何目标结果前按source V_select冻结；该对比仅说明源域排序不能保证目标域LEO同步提升。
- 科学结论：本轮找到了8个邻域seed中的源域最优seed392005，但其全目标域结果仍存在明显receiver-specific失败，尤其是`14-7`和`20-1`。因此可将seed392005作为本次预登记规则下的冻结checkpoint，但不能据此宣称ADV3B03已经解决接收机域泛化或星地信道鲁棒性。

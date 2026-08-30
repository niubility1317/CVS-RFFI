# Phase1 HCF-DG A0–A5快速源域LORO矩阵

## 预登记

- Run ID：`phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1`
- 当前状态：`LOCAL_VERIFIED`
- 初始集成提交：`49bd116b7af97abacb287113cb389c64b6edf42c`
- P1定点修复提交：`23e8255bca51e8fff4e3603e0df5eb4514ea6d31`
- N607布尔mask兼容修复后的正式代码提交：`7bcc5f73246c642a6cfe4983a3473f490d349ba2`
- Git分支：`codex/phase1-hcfdg-20260830`
- 方法范围：报告定义的HCF-DG V1快速筛选A0–A5；A6–A12不在本run中启动。
- 协议范围：仅Phase1 source-only域泛化；禁止访问Phase2 capsule、target receiver、support、query、truth、target prototype或目标统计。
- 数据：`Dataset_WigSig/ManySig.pkl`；source receiver集合`1,3,4,6,8`；训练日期`day1/day2/day3`；source角色`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- source LORO：fold1为中心receiver，fold8为最远receiver；每行训练时严格排除heldout receiver，训练receiver为其余4个source receiver；最终只在heldout source receiver的day1/2/3上零适配评估。
- 种子：`392001/392002/392003`。
- 预算：每行精确`4000 optimizer updates`。
- 矩阵：`6 candidates×2 folds×3 seeds=36 rows`。
- 训练视图：HCF-DG专用入口采用报告定义的70%clean+30%`mixed_orbit`单前向；不修改旧ADV3B02/ADV3B03入口。
- batch：A3–A5采用`6 TX×4 domain×4 sample=96`矩形episode；receiver/day/channel leave-out比例为`0.65/0.225/0.125`，support完整排除query因素。
- GPU：N607 GPU0–7；dispatcher对每张GPU使用2个并发槽，绝不超过2个本run训练进程/GPU。
- 选模边界：本run只产出source-only证据，低性能不停止健康训练，也不触发目标测试、调参、重训或选择性重跑。

## 候选

|候选|冻结定义|
|---|---|
|A0|ADV3B02闭集精简双分支控制；关闭FastTrust/open/unknown和旧辅助loss|
|A1|`single_parameter_matched`参数量控制|
|A2|单identity主干+48D receiver/day/channel环境编码器|
|A3|A2+矩形batch|
|A4|A3+普通LODO原型分类|
|A5|A4+rank-4公共—特定低秩头|

## 本地验证

- 聚焦与回归测试：`142 passed`。
- Python编译：launcher及HCF-DG config/sampler/satellite/model/losses/trainer/metrics全部通过。
- 真实checkpoint smoke：A0和A4均完成1 update；P1修复后A4最新smoke位于`E:\type10-7\local_artifacts\phase1_hcfdg_smoke\A4-F8-S392002-smoke3`，checkpoint为4,993,633字节，严格重建并分别产生clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`JSON和log，终态`ARTIFACTS_COMPLETE`。
- 已保留两个早期技术失败partial artifact：A2 smoke1为小型数据下`drop_last`空loader；A2 smoke2为CPU generator与CUDA增强器不匹配。两处均已定点修复，未删除失败证据。
- 独立P0/P1审查结论：`P0=0、P1=2`。P1-1为合法路径名包含`phase2`时被整串token扫描误拒；P1-2为逐样本Bernoulli及A3–A5整域mask不能固定每batch 70/30。提交`23e8255b`已定点修复：只检查显式CLI角色option；每个96样本batch固定29个satellite位置；channel episode按实际增强标签重建domain/query/support。原reviewer定点复审结论为`RESOLVED`，未重新全量审查。

## 发布前只读证据

- 2026-08-30 14:46–14:47 CST使用普通N607账户直连preflight通过；项目根、固定Python和`ManySig.pkl`可见。
- GPU0–7均为1MiB、0%利用率，`nvidia-smi`无compute app；目标训练进程计数为0。
- 目标release根、正式run根和dispatcher日志路径均为`ABSENT`，满足不可覆盖要求。
- 旧归档均保留但不发布。正式release归档大小5,461,945字节，本地/远端SHA256均为`3F379910DB1FAA5486C18E59B73A6A9DBC1A81258A96711857432FC411EDC3E6`；归档内已读回launcher、HCF-DG trainer和真实`SSDG/train_ssdg.py`，远端编译通过。解包时仅出现服务器时钟较归档时间慢约13秒的mtime warning，不影响文件或编译。
- N607 smoke r1在Python前因`noclobber`预创建日志后再次重定向而退出1；run根未创建，0字节日志保留。r2真实进入Python后暴露N607旧PyTorch无法推断`numpy.bool` dtype，退出1且无checkpoint/GPU残留，partial根保留。提交`7bcc5f73`改为显式Python`bool`列表和`torch.bool`构造。
- N607 smoke r3使用A4、fold1、seed392002、1 update、GPU0退出0；`final_hcfdg.pt`为4,993,292字节，clean与三种LEO场景均严格重建，missing/unexpected/shape mismatch均为空，终态`ARTIFACTS_COMPLETE`；完整训练日志为空且GPU已释放。1-update准确率仅为启动闭合证据，不作性能判断。
- 首次扩展资源探针被本地PowerShell提前展开远端语法并仅返回远端解析错误；未发生远端写入。随后使用无本地变量展开的短命令重新核对并取得上述有效证据。

## N607路径与命令

- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release归档：`E:/type10-7/local_artifacts/phase1_hcfdg_release/phase1_hcfdg_7bcc5f73.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_hcfdg_7bcc5f73.tar.gz`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73`
- 正式run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1`
- 正式日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1.dispatcher.log`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`

N607 smoke固定为A4、fold1、seed392002、1 update、GPU0，并要求strict final checkpoint及四场景artifact全部闭合。PASS后正式命令为：

```text
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73/code/scripts/launch_phase1_hcfdg_matrix_20260830.py --formal --run-id phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1 --stage quick --folds 1,8 --gpus 0,1,2,3,4,5,6,7 --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
```

## 直接技术停止规则

只有以下情况允许停止对应run并保留partial artifact：Phase1数据越权；错误candidate/fold/receiver/day/seed/update；输出路径冲突；错误checkout/release；主命令不能启动；无法产生prediction/evaluation闭合；同一确定性pre-prediction异常至少重复2行；进程归属不清并可能影响无关任务。低性能、收敛慢或中间指标差均不允许停止、重启或热补丁。

## 预期artifact

每行目录`<run-root>/<candidate>-F<fold>-S<seed>`必须包含：

- `final_hcfdg.pt`，含candidate、fold、seed、source split、精确update、runtime重建参数和推理头边界；
- `metrics.csv`、`metrics.jsonl`；
- `eval_clean.json/log`；
- `eval_leo_clear_weak.json/log`；
- `eval_leo_low_elev_weak.json/log`；
- `eval_leo_rain_weak.json/log`；
- `ARTIFACTS_COMPLETE.json`。

run根在36行全部进入终态后写入`final_status.json`。训练完成不等于实验完成；缺少任一严格评估场景时不得标记`ARTIFACTS_COMPLETE`。
## 正式启动

- 启动时间：N607约2026-08-30 15:05 CST。
- 状态：`RUNNING`。
- dispatcher PID：`2172360`；PPID1；CWD=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73`；cmdline精确绑定正式release、run ID、36行quick stage、fold1/8、GPU0–7和ManySig source路径。
- `plan.json`为12,887字节；首波创建16个row目录。
- 首波16个直属worker与dispatcher绑定正确；GPU0–7各2个本run Python训练进程，没有超过每卡2行。
- 启动30秒快照：GPU利用率69%–90%，显存约1,427–1,429MiB/卡；无`TECHNICAL_FAILURE.json`，dispatcher仍存活。首个update尚未写出metrics，下一次短连接复核增长。
- 启动74秒复核：dispatcher及16个worker仍存活，GPU利用率61%–89%，无技术失败。trainer按设计在每行4000 updates结束时一次性写`metrics.csv/jsonl`，因此运行中dispatcher日志与metrics可保持空/缺失；以精确进程绑定、GPU计算、elapsed和row目录状态监控，不据此停止健康run。
- 低频heartbeat监控ID：`hcf-dg-a0-a5`，每30分钟短连接只读检查；36行闭合后完整解析全部artifact并发布source-only A0–A5结论。

## 完成状态与全量核查

- 完成时间：N607为2026-08-30 15:22:07 CST；从`plan.json`写入到`final_status.json`闭合共15分43秒。
- 最终状态：`ANALYZED`。dispatcher PID2172360已自然退出；36/36行均为`ARTIFACTS_COMPLETE`，`TECHNICAL_FAILURE=0`，GPU0–7均已释放，没有停止、重启、热补丁或选择性重跑。
- 全量读取而非tail/抽样：完整解析36个`metrics.jsonl`和36个`metrics.csv`，两类文件各含144,000条update记录；完整读取144个评估JSON、144个评估log、36个`ARTIFACTS_COMPLETE.json`、36个final checkpoint runtime、`plan.json`和`final_status.json`。
- 训练闭合：36/36行均为连续update1–4000；`optimizer_update`和`backbone_forward_calls`逐行精确等于update；全部数值有限。
- checkpoint闭合：36/36个`final_hcfdg.pt`均可在CPU重新加载；candidate、fold、seed、update、source receiver、heldout receiver、day1/2/3及`target_access=false`与行身份一致。
- 严格评估闭合：144/144个场景均为`checkpoint_load_strict=true`，且`missing_keys=[]`、`unexpected_keys=[]`、`shape_mismatches=[]`；每个场景评估18,000条记录，总计2,592,000次source LORO判决。
- 日志核查：144个评估log均已完整读取，未发现`Traceback`、`RuntimeError`、`TECHNICAL_FAILURE`或独立非有限数值标记。dispatcher日志为0字节，与该launcher仅通过行artifact及`final_status.json`闭合的设计一致，不构成故障。
- 协议核查：全部行仅使用source receiver`1/3/4/6/8`、day1/2/3及fold1/fold8留一源接收机；没有访问Phase2、target receiver、support、query或truth。

## 候选汇总

以下百分数均为6行（2 folds×3 seeds）的均值；`±`为跨6行样本标准差。`LEO floor`是每行三种LEO场景class floor的最小值后再取均值。

|候选|Clean|Clean floor|LEO clear|LEO low-elev|LEO rain|LEO mean|LEO floor|相对A0 Clean|相对A0 LEO mean|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A0|57.20±5.66|30.84|33.35|33.15|33.49|33.33±2.50|6.74|0.00|0.00|
|A1|57.77±3.94|29.46|34.24|34.54|34.66|34.48±1.95|6.27|+0.57|+1.15|
|A2|60.10±5.79|25.47|35.96|35.69|35.91|35.85±4.47|8.56|+2.90|+2.53|
|A3|60.04±5.19|27.08|36.05|35.89|36.20|36.05±4.15|9.39|+2.85|+2.72|
|A4|**60.63±3.82**|27.42|36.19|36.22|36.44|**36.29±3.19**|8.53|**+3.43**|**+2.96**|
|A5|59.63±3.36|27.68|36.24|36.12|36.43|36.26±3.45|7.70|+2.43|+2.94|

按预登记的source-only均值顺序为`A4>A5>A3>A2>A1>A0`。A4同时取得最高Clean和最高LEO mean，因此冻结为A0–A5快速筛选的结构冠军。A5与A4的LEO mean仅差0.02pp，但Clean下降1.00pp、LEO floor下降0.83pp；rank-4 common-specific头没有形成可晋级的边际收益。

## 分fold结果

|候选-fold|Clean|Clean floor|LEO clear|LEO low-elev|LEO rain|LEO mean|LEO floor|
|---|---:|---:|---:|---:|---:|---:|---:|
|A0-F1|58.16|34.71|32.52|33.30|33.73|33.18|7.19|
|A0-F8|56.24|26.98|34.18|33.00|33.24|33.47|6.30|
|A1-F1|60.56|34.47|35.09|36.71|36.73|36.18|7.48|
|A1-F8|54.97|24.46|33.38|32.37|32.59|32.78|5.07|
|A2-F1|61.26|30.11|37.81|38.57|38.73|38.37|9.03|
|A2-F8|58.93|20.82|34.12|32.80|33.09|33.34|8.08|
|A3-F1|60.72|33.19|37.68|38.66|38.98|38.44|11.00|
|A3-F8|59.36|20.98|34.42|33.12|33.42|33.65|7.78|
|A4-F1|60.42|35.22|37.48|38.57|38.70|38.25|10.70|
|A4-F8|60.84|19.62|34.91|33.87|34.18|34.32|6.37|
|A5-F1|60.80|34.97|37.70|38.51|38.85|38.35|8.71|
|A5-F8|58.45|20.40|34.78|33.74|34.01|34.18|6.69|

A4的平均Clean在fold1与fold8间接近，但class floor从35.22%降至19.62%，LEO mean从38.25%降至34.32%，LEO floor从10.70%降至6.37%。因此，A4改善的是平均源域外推，尚未解决最远接收机上的类别尾部崩塌；后续A8的分层尾部风险不能只看总准确率。

## 36行同row结果

|Row|Clean|Clean floor|Clear|Low-elev|Rain|LEO mean|LEO floor|GPU-h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A0-F1-S392001|60.61|40.90|34.60|34.53|35.91|35.01|9.77|0.0487|
|A0-F1-S392002|63.63|42.93|33.96|35.51|34.88|34.78|7.57|0.0485|
|A0-F1-S392003|50.23|20.30|28.99|29.88|30.40|29.76|4.23|0.0507|
|A0-F8-S392001|54.96|24.47|32.65|31.56|32.21|32.14|5.57|0.0482|
|A0-F8-S392002|51.70|20.87|32.49|31.56|31.42|31.82|6.27|0.0490|
|A0-F8-S392003|62.05|35.60|37.38|35.88|36.11|36.46|7.07|0.0504|
|A1-F1-S392001|62.93|45.97|34.48|35.50|36.38|35.45|7.33|0.0511|
|A1-F1-S392002|59.32|25.60|35.21|38.14|36.64|36.66|9.70|0.0508|
|A1-F1-S392003|59.43|31.83|35.58|36.48|37.18|36.41|5.40|0.0510|
|A1-F8-S392001|51.49|20.73|32.93|31.80|32.13|32.29|7.73|0.0533|
|A1-F8-S392002|58.16|25.50|33.11|32.16|32.16|32.48|3.47|0.0501|
|A1-F8-S392003|55.27|27.13|34.11|33.15|33.49|33.58|4.00|0.0502|
|A2-F1-S392001|63.78|36.87|41.26|41.69|42.46|41.80|11.87|0.0541|
|A2-F1-S392002|59.87|21.13|33.26|34.84|33.42|33.84|4.20|0.0547|
|A2-F1-S392003|60.13|32.33|38.90|39.18|40.32|39.46|11.03|0.0554|
|A2-F8-S392001|60.84|25.40|35.06|34.01|34.39|34.49|8.20|0.0555|
|A2-F8-S392002|49.49|23.40|29.74|28.85|28.77|29.12|8.13|0.0632|
|A2-F8-S392003|66.47|13.67|37.57|35.55|36.11|36.41|7.90|0.0647|
|A3-F1-S392001|61.52|30.83|41.14|41.42|41.87|41.48|12.03|0.1303|
|A3-F1-S392002|61.15|32.00|33.25|35.14|34.40|34.26|8.07|0.1302|
|A3-F1-S392003|59.49|36.73|38.63|39.42|40.68|39.58|12.90|0.1305|
|A3-F8-S392001|57.12|24.00|36.08|34.74|35.17|35.33|8.23|0.1340|
|A3-F8-S392002|52.68|23.93|30.08|29.46|29.48|29.67|8.00|0.1310|
|A3-F8-S392003|68.29|15.00|37.10|35.16|35.62|35.96|7.10|0.1296|
|A4-F1-S392001|61.64|38.10|40.13|40.53|40.98|40.55|9.53|0.1462|
|A4-F1-S392002|60.41|38.27|34.54|36.54|35.66|35.58|10.43|0.1431|
|A4-F1-S392003|59.21|29.30|37.78|38.63|39.47|38.63|12.13|0.1457|
|A4-F8-S392001|64.93|21.33|36.16|35.27|35.67|35.70|6.67|0.1425|
|A4-F8-S392002|54.07|22.43|31.63|30.83|31.04|31.17|6.60|0.1417|
|A4-F8-S392003|63.51|15.10|36.93|35.53|35.83|36.10|5.83|0.1410|
|A5-F1-S392001|63.54|43.70|40.64|41.00|41.82|41.15|7.23|0.1408|
|A5-F1-S392002|59.13|34.17|34.81|36.06|35.14|35.34|9.67|0.1415|
|A5-F1-S392003|59.73|27.03|37.66|38.48|39.58|38.57|9.23|0.1100|
|A5-F8-S392001|56.87|22.30|36.06|34.72|35.38|35.39|7.37|0.1082|
|A5-F8-S392002|55.17|23.57|31.15|30.91|30.69|30.92|6.80|0.0700|
|A5-F8-S392003|63.31|15.33|37.14|35.57|35.96|36.22|5.90|0.0661|

## 资源结果

|候选|平均GPU-h/行|相对A0|有效samples/s|训练峰值显存|checkpoint|
|---|---:|---:|---:|---:|---:|
|A0|0.0492|1.00×|2,167|219.5MiB|6.55MiB|
|A1|0.0511|1.04×|2,089|229.1MiB|11.60MiB|
|A2|0.0580|1.18×|1,850|218.5MiB|4.76MiB|
|A3|0.1309|2.66×|815|218.5MiB|4.76MiB|
|A4|0.1434|2.91×|744|218.5MiB|4.76MiB|
|A5|0.1061|2.15×|1,101|218.1MiB|4.78MiB|

36行合计3.232 GPU-hours。表中有效samples/s由`4000×96/(GPU-hours×3600)`计算；峰值显存来自trainer逐update遥测，不等同于`nvidia-smi`启动时每进程约706MiB的进程占用。A5后续波次明显快于A4，但该结果来自同一并发dispatcher的不同时间波次，不能作为隔离速度因果结论；正式性能比较只使用同row准确率与floor。

## 科学解释与冻结决定

1. A1相对A0同时提高Clean 0.57pp和LEO mean 1.15pp，说明第二套完整domain backbone不是当前DG收益的必要条件。
2. A2把Clean提高2.90pp、LEO mean提高2.53pp，并把checkpoint降至4.76MiB；轻量环境因子化是本轮最大结构性跃迁。
3. A3相对A2的Clean基本不变（-0.06pp），LEO mean提高0.19pp，LEO floor提高0.83pp，但GPU成本从1.18×升至2.66×。矩形batch主要改善尾部稳定性，而非平均Clean。
4. A4相对A3把Clean提高0.58pp、LEO mean提高0.24pp，却使LEO floor下降0.86pp。LODO对平均跨接收机识别有效，但仍需A8处理类别尾部。
5. A5相对A4没有增益。当前rank-4 common-specific头不进入下一阶段主干；A4作为V1结构冠军，A3作为去LODO对照保留。
6. A4内部，seed392001跨两fold平均Clean为63.29%、LEO mean为38.12%，均为三seed最高；seed392003的平均LEO floor更高。快速筛选只冻结A4结构，不把单个seed宣称为最终多种子冠军；若下一次单seed实现smoke需要固定种子，使用source-only领先的392001。

本轮绝对准确率不能与历史`ADV3B02_CORE90`旧split结果直接比较：当前证据是day1/2/3、两个source LORO fold、4000 updates和70%clean+30%`mixed_orbit`单前向的快速结构筛选。可支持的结论仅是A4在本矩阵中优于A0–A3/A5的平均source LORO表现，不能外推为目标接收机或Phase2性能。

## A6–A9实现与发布建议

- 当前V2状态：`PENDING_IMPLEMENTATION`；本轮未启动A6–A9，也未访问任何目标域数据。
- 基座：冻结A4作为V2增量基座，A3保留为结构对照；不把A5的rank-4 specific头带入默认V2。
- A6：在A4上实现单因素receiver counterfactual transport，保持same-TX、有界低秩调制和`lambda_CF=0.15`，先验证替换receiver后身份稳定性。
- A7：在A6通过后加入receiver/day/channel组合transport；每次只改变已登记因素，不混入target统计。
- A8：加入hierarchical DRO，固定`lambda_HDRO=0.10`，重点优化并报告RX×TX×day尾部以及fold8 class floor，不允许只按总准确率晋级。
- A9：加入content-conditioned prototypes，缺少内容相似support时跳过强配对，验证是否减少LODO的内容错配。
- 发布顺序：先在本地完成V2实现、聚焦负测、真实checkpoint无query smoke和一次P0/P1审查，再按同一release归档规则发布。V2仍为pending时不得直接在N607启动。
- 推荐矩阵：先按报告要求对A3/A4运行5 folds×3 seeds×6,000 updates的方法确认；确认后用`A4-control/A6/A7/A8/A9×2 folds×3 seeds×6,000 updates=30 rows`做V2快速筛选。两批均只用source-only证据，禁止根据目标结果反馈选种或重跑。

最终决定：A0–A5快速筛选闭合并发布；A4晋级、A3保留为对照、A5不晋级。下一阶段仅进入本地V2实现与发布准备，不自动启动A6–A9。

# PairBiCAD P0–P4 r2正式矩阵预登记

## 当前状态

- 状态：`RUNNING`。
- Run ID：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2`。
- 失败前序run：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1`，保持只读，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 唯一修复：正式评估恢复器按冻结row和候选协议显式重建`num_receivers/num_days/num_channels`；训练方法、loss、数据、矩阵、seed、update预算和四场景评估不变。
- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 代码冻结commit：`c25c6974e0a80108aafe6cfb9fd35f1429cfaad1`。
- N607账户：普通账户`szu2070436088`；禁止管理员账户。

## 候选与矩阵

- 候选：`P0/P1/P2/P3/P4`。
- folds：1、8。
- seeds：392001、392002、392003。
- 总行数：5×2×3=30。
- 每行预算：4000 optimizer updates。
- 训练天：day1/day2/day3。
- fold1 source receivers：3、4、6、8；fold8 source receivers：1、3、4、6。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 物理batch48：16L+32U；clean/LEO拼接网络batch96；单次主干前向。
- 星地信道：`concat_sat_ce_only`兼容边界与`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- target/Phase2/support/query/truth：禁止访问。
- GPU0–7；每GPU最多2个本run训练进程，首波最多16行，其余排队。

## 本地验证与独立审查

- 新回归测试在旧代码上按预期RED：正式恢复器未传入维度。
- 最小修复后GREEN；`code/tests/phase1_bicad_xr`完整相关测试298/298通过。
- launcher与metrics聚焦测试39/39通过；`py_compile`通过。
- r2精确dry-run输出30/30行，候选/fold/seed/day/update/source receiver与本报告一致。
- 真实checkpoint no-query smoke：读取r1的`P4-F1-S392002`U4000 checkpoint，严格重建missing/unexpected均为0，恢复维度为4/3/2；clean和三种LEO弱场景均输出`[256,6]`有限logits；target/Phase2/support/query/truth访问均为false。
- 独立P0/P1定点审查未发现会使r2跑错、越权、覆盖输出、误杀进程、无法启动或无法产生合法prediction的问题。
- `REJECTED_EXTRA_GATE`：不新增seal、成员hash、重复审查、Phase2检查或其他白名单外门槛。

## N607发布计划

- release名称：`phase1_pairbicad_p0p4_c25c6974`。
- 本地归档：`E:\type10-7\release_archives\phase1_pairbicad_p0p4_c25c6974.tar.gz`。
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_c25c6974.tar.gz`。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_c25c6974`。
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2`。
- 远端dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2.dispatcher.log`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（Python3.10.19，与N607正式入口一致）。
- CWD：上述远端release根。
- 正式入口：`code/scripts/launch_phase1_pairbicad_p0p4_n607_20260831.sh`。

正式命令：

```text
PAIRBICAD_RELEASE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_c25c6974 PAIRBICAD_RUN_ID=phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2 PAIRBICAD_MAX_JOBS_PER_GPU=2 bash code/scripts/launch_phase1_pairbicad_p0p4_n607_20260831.sh
```

## 预期artifact与停止规则

- 每行：`metrics_epoch.csv`、`metrics_epoch.jsonl`、`bicad_xr_final.pth`、`checkpoint_runtime.json`、`diagnostics.json`、clean及三种LEO评估JSON/log、`ARTIFACTS_COMPLETE.json`或`TECHNICAL_FAILURE.json`。
- run级：`plan.json`、`final_status.json`、dispatcher日志。
- 只有错误candidate/fold/receiver/day/seed/update、source-only越权、输出冲突、错误release/CWD、命令无法运行、无artifact闭合、同一确定性pre-prediction异常至少2行或进程归属不清时才能停止精确绑定的本run进程树并保留partial artifact。
- 不得因中间或最终低性能停止、重启、热补丁或选择性重跑；不得影响无关进程。
- 30/30行只有在严格checkpoint恢复及clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`全部闭合后才可标为`ARTIFACTS_COMPLETE`。
- 只按source LORO同row证据比较P0–P4；不得访问Phase2、目标接收机、support、query或truth。

## N607落地回读

- 发布前GPU0–7均为0%利用率、1MiB显存，无compute process；`/home`可用7.3TB。
- 远端archive、release根、r2 run根、dispatcher日志和PID文件发布前均不存在，不会覆盖r1或其他任务。
- release归档本地→远端SHA256一致：`ec5ca68e90fb3753b63ca19ae49fd9528e3b0d825ec9e0578d8b371f112ac09c`。
- 归档已解压到预登记release根；使用正式入口同一Python3.10.19解释器对launcher、trainer和SSDG入口执行远端`py_compile`，结果PASS。
- 首次compile探测使用了错误的预登记解释器路径并以exit127结束，未执行代码、未创建run；报告已更正为shell入口实际使用的`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

## N607正式启动回读

- 启动时间：2026-08-31 15:11 CST后；dispatcher PID`2855983`，PPID=1。
- dispatcher cmdline精确绑定release`phase1_pairbicad_p0p4_c25c6974`、r2 Run ID、P0–P4、fold1/8、3个seed、U4000和`--max-jobs-per-gpu 2`。
- 第一波16个直属主训练进程均以正式解释器运行，CWD全部为新release根，输出分别绑定r2唯一row目录。
- GPU0–7各恰好2个本run compute process；启动回读利用率86%–91%，显存约1.60–1.86GiB/卡。
- 首波覆盖P0全部6行、P1全部6行和P2前4行；其余14行由dispatcher排队，不突破每GPU2个活跃训练进程。
- run根已有30个预留row目录和`plan.json`；16个`train.log`已建立。启动早期日志为0字节，但PID/CWD/cmdline/GPU计算均健康；不得据此停止。
- 启动回读为0个`ARTIFACTS_COMPLETE`、0个`TECHNICAL_FAILURE`；尚无性能结论。

## 三级预算后续执行追踪

用户已明确要求继续执行“U4000筛选→前两名U6000–U9000收敛确认→冻结预算→完整多种子确认”。当前追踪如下：

|ID|报告要求|目标|状态|验证|说明|
|---|---|---|---|---|---|
|T1|完整分析P0–P4快速矩阵|本run全部30行|completed|30/30全量artifact解析exit0|只用source-only证据|
|T2|冻结前两名|candidate聚合排名|completed|P2第一、P1第二|主分数为`H(clean,min(LEO三场景))`|
|T3|6000–9000 updates收敛确认|前两名×fold1/8×3 seeds|pending|每500 updates source-LORO曲线|4000后patience=5|
|T4|冻结最终预算|胜出候选各row最佳update中位数|pending|量化到500 updates|不得使用target结果|
|T5|完整多种子确认|胜出候选×fold1–5×3 seeds|pending|15行四场景完整闭合|最终性能结论只来自该阶段|

Ruling：当前U4000 run是快速机制筛选，不是最终收敛结论；前两名及后续预算只能按source-LORO证据冻结。


## U4000全量source-only分析与前两名冻结

分析状态：`ANALYZED`。正式分析器完整读取30/30行的`metrics_epoch.jsonl`、`metrics_epoch.csv`、`checkpoint_runtime.json`、`diagnostics.json`、`ARTIFACTS_COMPLETE.json`以及clean和三种LEO场景JSON/log；逐行验证最终`optimizer_update=4000`、严格重建空missing/unexpected/shape mismatch、source-only身份及target/Phase2/support/query/truth访问均为false。全量命令exit0，没有缺行、缺文件、格式错误、非有限值、访问越权或runtime不一致。独立复核确认JSONL和CSV各390条记录，即每行13条；四场景JSON和log各120/120可读。30个`train.log`均为空，这是trainer只写结构化metrics的已知记录方式，不影响artifact闭合，但训练文本日志本身不可用于复核。`gpu_hours`、吞吐、峰值显存和参数量字段为N/A，不参与性能排序。

选择主分数固定为：

```text
source_sat_hmean=H(clean,min(leo_clear_weak,leo_low_elev_weak,leo_rain_weak))
```

候选排序依次比较6行主分数均值、LEO accuracy均值、最差LEO场景accuracy、clean均值和candidate ID。结果如下：

|排名|候选|H均值±总体标准差|H最差行|LEO均值|LEO场景最差值|clean均值|clean最差值|结论|
|---:|---|---:|---:|---:|---:|---:|---:|---|
|1|P2|41.36%±3.38pp|36.85%|33.33%|28.81%|56.84%|51.11%|冻结进入收敛确认|
|2|P1|39.21%±5.42pp|29.30%|30.74%|21.93%|56.37%|44.12%|冻结进入收敛确认|
|3|P0|37.19%±4.76pp|28.71%|29.20%|20.62%|54.24%|47.24%|不进入下一阶段|
|4|P4|35.08%±4.74pp|26.53%|26.83%|18.59%|53.54%|46.31%|不进入下一阶段|
|5|P3|34.27%±4.16pp|28.90%|27.20%|20.18%|48.62%|43.13%|不进入下一阶段|

P2相对P1的主分数均值提高2.15pp，LEO均值提高2.59pp，LEO场景最差值提高6.88pp，clean均值提高0.46pp。P2不仅均值最高，最差行也明显高于P1；因此前两名正式冻结为`P2`和`P1`，顺序也冻结为P2第一、P1第二。

### 30行同row结果

|row|clean|clear|low-elev|rain|LEO均值|LEO场景floor|H|clean类floor|LEO类floor|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|P0-F1-S392001|58.46%|23.39%|23.06%|23.96%|23.47%|23.06%|33.07%|28.87%|2.43%|
|P0-F1-S392002|49.58%|33.51%|31.72%|32.42%|32.55%|31.72%|38.69%|3.13%|3.83%|
|P0-F1-S392003|47.24%|21.03%|20.62%|21.62%|21.09%|20.62%|28.71%|0.50%|0.50%|
|P0-F8-S392001|55.88%|31.67%|31.15%|31.88%|31.57%|31.15%|40.00%|1.83%|2.53%|
|P0-F8-S392002|65.08%|32.48%|31.70%|31.44%|31.87%|31.44%|42.40%|37.57%|3.60%|
|P0-F8-S392003|49.19%|35.42%|34.09%|34.40%|34.64%|34.09%|40.27%|12.13%|1.93%|
|P1-F1-S392001|61.41%|31.82%|30.81%|31.45%|31.36%|30.81%|41.03%|34.80%|1.67%|
|P1-F1-S392002|48.49%|28.20%|27.13%|27.73%|27.69%|27.13%|34.79%|1.67%|2.80%|
|P1-F1-S392003|44.12%|22.37%|21.93%|22.58%|22.29%|21.93%|29.30%|1.63%|0.93%|
|P1-F8-S392001|61.88%|36.42%|34.93%|35.74%|35.70%|34.93%|44.65%|32.50%|3.70%|
|P1-F8-S392002|61.72%|33.16%|32.15%|32.53%|32.61%|32.15%|42.28%|29.40%|3.40%|
|P1-F8-S392003|60.63%|36.63%|33.59%|34.11%|34.78%|33.59%|43.23%|23.83%|6.43%|
|P2-F1-S392001|54.39%|31.67%|30.41%|31.07%|31.05%|30.41%|39.01%|30.47%|2.07%|
|P2-F1-S392002|51.11%|30.40%|28.81%|29.32%|29.51%|28.81%|36.85%|7.43%|1.60%|
|P2-F1-S392003|51.92%|36.44%|34.50%|34.80%|35.25%|34.50%|41.46%|34.93%|1.80%|
|P2-F8-S392001|57.17%|31.95%|30.19%|30.72%|30.95%|30.19%|39.52%|27.77%|1.43%|
|P2-F8-S392002|61.72%|38.90%|37.61%|37.97%|38.16%|37.61%|46.73%|36.87%|1.60%|
|P2-F8-S392003|64.70%|36.14%|34.03%|35.06%|35.07%|34.03%|44.60%|23.27%|1.60%|
|P3-F1-S392001|50.91%|20.67%|20.18%|20.77%|20.54%|20.18%|28.90%|0.03%|1.53%|
|P3-F1-S392002|46.63%|31.87%|31.74%|32.18%|31.93%|31.74%|37.77%|0.53%|0.00%|
|P3-F1-S392003|46.88%|24.90%|23.51%|24.74%|24.38%|23.51%|31.32%|1.03%|1.57%|
|P3-F8-S392001|54.48%|32.46%|32.59%|33.00%|32.68%|32.46%|40.68%|23.23%|5.90%|
|P3-F8-S392002|49.70%|29.04%|28.09%|28.46%|28.53%|28.09%|35.89%|13.77%|3.37%|
|P3-F8-S392003|43.13%|25.76%|24.27%|25.32%|25.12%|24.27%|31.06%|13.57%|1.67%|
|P4-F1-S392001|56.98%|24.23%|23.29%|23.80%|23.77%|23.29%|33.06%|1.80%|2.50%|
|P4-F1-S392002|50.22%|26.37%|26.85%|26.74%|26.66%|26.37%|34.58%|19.67%|2.63%|
|P4-F1-S392003|46.31%|19.76%|18.59%|19.44%|19.26%|18.59%|26.53%|4.03%|3.17%|
|P4-F8-S392001|56.23%|34.21%|33.61%|34.03%|33.95%|33.61%|42.07%|24.33%|2.00%|
|P4-F8-S392002|55.48%|28.88%|28.21%|28.37%|28.49%|28.21%|37.40%|0.13%|2.33%|
|P4-F8-S392003|56.01%|30.07%|27.44%|28.97%|28.83%|27.44%|36.84%|24.90%|17.30%|

### 解释边界

U4000只承担机制筛选。虽然P2在30行中排名第一，所有候选的LEO类floor仍很低，说明类别级坍缩风险尚未消失；不能把P2的U4000结果当作最终收敛性能。下一阶段固定运行P2/P1×fold1/8×3seed共12行，最大U9000，从U4000开始每500 updates执行一次source-LORO四场景评估，连续5次主分数无改善才停止。完成后只按source-only曲线冻结胜出候选和最终预算。

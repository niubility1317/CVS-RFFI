# PairBiCAD P1 U6500完整多种子确认

## 状态与目的

- 当前状态：`ANALYZED`。
- Run ID：`phase1_adv3b02_pairbicad_final_p1_seed3_u6500_20260831_r1`。
- 目的：对source-only收敛确认冻结的P1和U6500做fold1–5×3seed完整确认；不再选择候选或调整预算。

## 冻结依据

- 12行收敛确认全部`ARTIFACTS_COMPLETE`且无技术失败；完整解析103个source-LORO评估点和全部训练/评估artifact。
- P1最佳H均值`42.1297%`、最小值`39.8983%`；P2分别为`40.7525%`、`38.3515%`，冻结P1。
- P1六个最佳update为`[4000,4500,5500,7000,9000,9000]`，中位数6250按预登记500网格中点向上取整，冻结U6500。
- 冻结提交：`d373e1ed3a52dec8bebaa3844ced65af0eebf758`，分支`codex/phase1-pairbicad-p4-20260831`。

## 固定矩阵与协议

- 候选：仅`P1`。
- fold：`1,2,3,4,5`。
- seed：`392001,392002,392003`。
- 共15行，每行严格`6500 optimizer updates`，禁用在线source-LORO早停。
- 训练/选择输入：ManySig source receiver索引`[1,3,4,6,8]`中的LORO训练子集，day1/2/3；严格source-only。
- 每行训练完成后用固定final checkpoint评估source V_select的clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 禁止target、Phase2、support、query、truth访问或反馈；不得调参、重训或选择性重跑。
- 资源：preflight发现GPU0已有一个无关Stage2进程PID2958727，占用约7.6GB；不得影响该进程。最终矩阵使用GPU顺序`1,2,3,4,5,6,7,0`，因此GPU1–7各2行、GPU0仅1行；含无关进程在内每GPU总训练进程不超过2。

## 代码、验证与命令

- 实际release代码/配置提交：`7e25cc04f4abf298e958ded2e596a80f80d2b549`；PairBiCAD launcher测试40/40通过，15行精确dry-run全部为P1/U6500/source-only/day1–3且GPU映射符合上述容量约束；分析器测试9/9和编译通过。
- Git Bash执行通道被桌面适配器错误路由到WSL并在payload前失败，记为`FAILED_NONBLOCKING`；本次shell只改不可覆盖run/release字符串，未改Bash结构，使用已通过的launcher语义测试和远端一次编译作为正式验证。
- N607普通账户：`szu2070436088`；禁止管理员账户。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_final_p1_u6500_20260831_r1`。
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_pairbicad_final_p1_seed3_u6500_20260831_r1`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_pairbicad_final_p1_seed3_u6500_20260831_r1.dispatcher.log`。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。

```text
PAIRBICAD_FINAL_CANDIDATE=P1 PAIRBICAD_FINAL_OPTIMIZER_UPDATES=6500 bash code/scripts/launch_phase1_pairbicad_final_n607_20260831.sh
```

## 预期artifact与停止规则

每行必须保留完整JSONL/CSV训练遥测、`bicad_xr_final.pth`、`checkpoint_runtime.json`、`diagnostics.json`、clean和三种LEO评估JSON/log，以及`ARTIFACTS_COMPLETE.json`或`TECHNICAL_FAILURE.json`。Run级保留`plan.json`、`final_status.json`、dispatcher日志和PID文件。

只有错误candidate/fold/receiver/day/seed/update、source-only越权、输出冲突、错误release/CWD、命令无法运行、无artifact闭合、同一确定性pre-prediction异常至少2行或进程归属不清时才允许精确停止并保留partial artifact。不得因中间或最终低性能停止、重启、热补丁或选择性重跑；不得影响无关进程。

## N607发布与启动回读

- Release归档：本地`E:\type10-7\release_archives\phase1_pairbicad_final_c1444033.tar.gz`，远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_final_c1444033.tar.gz`；一次本地/远端SHA256均为`33e488fb22ee2b0de71a150b3625b3c50a6fbbc326c6384a7fee37e276fe4e43`。
- 远端release解包后，trainer、launcher和analyzer均通过一次远端编译检查。
- Dispatcher PID：`2983876`；启动命令中的candidate、fold、seed、update、GPU顺序、run根和release根均与本报告冻结项一致。
- 启动绑定回读：dispatcher直属主训练进程`15`个；`plan.json`共`15`行，其中P1=`15`、U6500=`15`、source-only=`15`。
- GPU映射回读：GPU0=`1`行，GPU1–7各=`2`行。GPU0上的无关Stage2进程PID2958727仍在运行，未被干预；启动检查时GPU0–7利用率为`87%–93%`，符合持续计算状态。
- 初始artifact计数：`ARTIFACTS_COMPLETE=0`、`TECHNICAL_FAILURE=0`。后续仅按预登记技术停止规则监控，不使用中间性能触发停止或重跑。

## 完整闭合与分析范围

- 正式矩阵于2026-08-31 18:49:02 CST启动，2026-08-31 19:04:22 CST写入最终状态，矩阵墙钟时间约15分20秒。
- 15/15行均为`ARTIFACTS_COMPLETE`，0行`TECHNICAL_FAILURE`；dispatcher正常退出。
- 完整读取15份`metrics_epoch.jsonl`、15份`metrics_epoch.csv`、15份`checkpoint_runtime.json`、15份`diagnostics.json`、15份completion artifact、60份场景JSON和60份非空场景log。15份`train.log`为0字节，与该trainer使用结构化metrics记录的预期行为一致。
- 15行均精确达到6500 optimizer updates；15个`bicad_xr_final.pth`均存在。所有checkpoint严格重建的missing、unexpected和shape mismatch均为空。
- 15行runtime均为P1、source-only、day1/2/3，target、Phase2、support、query和truth访问均为false；未发现`Traceback`、CUDA OOM、`RuntimeError`、`ValueError`、`TECHNICAL_FAILURE`或`Killed`日志指纹。
- 逐行完整分析器退出码为0；机器可读结果保存在本地同目录`analysis.json`和`rows.csv`，远端run根保留同名文件。
- 15份`diagnostics.json`中的`gpu_hours`、`throughput_samples_per_second`、`peak_gpu_memory_bytes`和`extra_forward_ratio`均为`N/A`。因此本报告只记录已回读的矩阵墙钟时间与启动GPU占用，不虚构逐row GPU小时、吞吐或峰值显存。

## 固定预算总体结果

所有百分比均来自同一行U6500 final checkpoint；`H=H(clean,min(leo_clear_weak,leo_low_elev_weak,leo_rain_weak))`。

|指标|均值|总体标准差|最小值|
|---|---:|---:|---:|
|clean accuracy|59.9378%|13.0433pp|37.6111%|
|LEO三场景accuracy均值|30.2225%|4.4209pp|19.9426%|
|row内最差LEO场景accuracy|29.7237%|4.3116pp|19.6556%|
|clean/最差LEO调和均值H|39.4805%|6.1940pp|28.1029%|
|clean逐类floor|13.5044%|18.7746pp|0.0000%|
|LEO逐类floor|3.0022%|2.0573pp|0.2333%|

三种LEO场景的15行accuracy均值分别为：`leo_clear_weak=30.8459%`、`leo_low_elev_weak=29.7370%`、`leo_rain_weak=30.0844%`。

## 按LORO接收机聚合

|fold|留出source receiver|clean均值|clear均值|low-elev均值|rain均值|H均值|LEO类floor均值|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|1|53.7148%|26.6222%|25.6185%|25.8944%|34.5697%|2.9667%|
|2|3|43.2444%|28.4111%|27.7741%|27.8056%|33.7497%|2.7444%|
|3|4|79.4278%|36.0352%|34.5833%|35.0519%|48.1519%|3.8000%|
|4|6|65.8556%|30.8833%|29.6759%|30.1537%|40.6483%|0.8667%|
|5|8|57.4463%|32.2778%|31.0333%|31.5167%|40.2829%|4.6333%|

receiver4最容易，H均值48.1519%；receiver3最难，H均值33.7497%。receiver6的总体H不低，但LEO类floor仅0.8667%，说明总体accuracy掩盖了个别类别接近失效的问题。

## 按seed聚合

|seed|clean均值|LEO均值|row内最差LEO均值|H均值|LEO类floor均值|
|---:|---:|---:|---:|---:|---:|
|392001|59.1744%|30.2200%|29.7700%|39.4309%|3.7933%|
|392002|60.3644%|27.9574%|27.6311%|37.4596%|2.4200%|
|392003|60.2744%|32.4900%|31.7700%|41.5510%|2.7933%|

seed392002的clean均值最高，但LEO和H最低；seed392003的LEO与H最高。这说明只按clean选择seed会得到错误的稳健性排序。

## 15行同row结果

|row|clean|clear|low-elev|rain|LEO均值|H|clean floor|LEO类floor|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|P1-F1-S392001|54.4556%|26.4944%|25.7889%|26.1500%|26.1444%|35.0018%|0.2000%|5.3000%|
|P1-F1-S392002|49.2833%|20.1556%|19.6556%|20.0167%|19.9426%|28.1029%|0.0333%|2.0333%|
|P1-F1-S392003|57.4056%|33.2167%|31.4111%|31.5167%|32.0481%|40.6044%|7.2667%|1.5667%|
|P1-F2-S392001|40.2333%|27.2333%|26.2722%|26.5667%|26.6907%|31.7874%|1.3667%|2.1000%|
|P1-F2-S392002|37.6111%|26.3333%|25.8500%|25.7389%|25.9741%|30.5625%|6.3000%|3.3000%|
|P1-F2-S392003|51.8889%|31.6667%|31.2000%|31.1111%|31.3259%|38.8993%|0.6000%|2.8333%|
|P1-F3-S392001|84.7000%|37.7000%|36.1889%|36.5833%|36.8241%|50.7110%|42.2667%|6.5667%|
|P1-F3-S392002|82.9667%|35.6111%|34.1444%|34.1667%|34.6407%|48.3789%|64.4000%|3.1000%|
|P1-F3-S392003|70.6167%|34.7944%|33.4167%|34.4056%|34.2056%|45.3657%|19.1333%|1.7333%|
|P1-F4-S392001|62.9389%|31.9167%|30.8611%|31.2389%|31.3389%|41.4150%|0.0000%|0.2333%|
|P1-F4-S392002|71.1667%|25.7389%|24.8444%|25.0167%|25.2000%|36.8311%|2.4667%|1.9667%|
|P1-F4-S392003|63.4611%|34.9944%|33.3222%|34.2056%|34.1741%|43.6990%|0.0000%|0.4000%|
|P1-F5-S392001|53.5444%|30.5889%|29.7389%|29.9778%|30.1019%|38.2394%|2.0000%|4.7667%|
|P1-F5-S392002|60.7944%|34.4611%|33.7722%|33.8556%|34.0296%|43.4226%|25.1000%|1.7000%|
|P1-F5-S392003|58.0000%|31.7833%|29.5889%|30.7167%|30.6963%|39.1866%|31.4333%|7.4333%|

## 跨行逐类均值

|TX类|clean|clear|low-elev|rain|
|---:|---:|---:|---:|---:|
|0|81.9911%|23.3133%|21.2667%|21.2222%|
|1|60.9244%|32.1356%|30.7578%|31.2133%|
|2|30.6756%|16.3489%|15.1622%|15.4622%|
|3|48.6222%|31.1911%|29.7044%|29.5133%|
|4|88.6689%|18.0333%|19.2333%|19.3956%|
|5|48.7444%|64.0533%|62.2978%|63.7000%|

TX2在clean和三种LEO下都是跨行均值最低类别；TX4的clean为88.6689%，但LEO仅18%–19%，存在明显信道脆弱性；TX5则在LEO下最高。总体均值不能替代逐类floor判断。

## 科学结论与边界

1. 固定P1/U6500的15行完整source-only确认已技术闭合，最终固定预算H为`39.4805%±6.1940pp`，不是此前收敛曲线“每行最佳checkpoint”的`42.1297%`。
2. 两个数字不可直接相减作严格退化结论：`42.1297%`来自fold1/8的6行逐row最佳update，当前`39.4805%`来自fold1–5的15行统一U6500 final checkpoint，fold覆盖和checkpoint选择均不同。描述性差值为-2.6492pp，主要说明固定预算后的跨接收机难度和方差比两fold收敛选择阶段更大。
3. P1在receiver、seed和TX类别间仍有明显不稳定性：H最差行28.1029%，clean类floor最低0%，LEO类floor最低0.2333%。因此该run证明了P1/U6500的完整source-only性能，但不支持“稳定替代ADV3B02”或“目标接收机性能已提升”的声明。
4. 本矩阵没有同划分、同预算的ADV3B02控制行，也没有目标域测试；与ADV3B02的数值比较必须等待同协议控制证据，不能引用历史CORE90或其他数据划分拼接结论。
5. 最终状态：`ANALYZED / SOURCE_ONLY_FIXED_BUDGET_COMPLETE`。不因负面floor结果重训或选择性重跑。

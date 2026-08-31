# PairBiCAD P1 U6500完整多种子确认

## 状态与目的

- 当前状态：`RUNNING`。
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

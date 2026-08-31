# PairBiCAD P2/P1 Source-LORO收敛确认预登记

## 状态

`LOCAL_VERIFIED`

## 科学目的与冻结矩阵

- Run ID：`phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r1`。
- 目的：对U4000筛选冻结的前两名P2、P1执行source-only收敛确认；不得根据target或Phase2结果改变候选、预算、超参数或重跑。
- 候选顺序：P2、P1。
- source-LORO folds：fold1和fold8；对应heldout source receiver分别为1和8。
- seeds：392001、392002、392003。
- 总行数：2×2×3=12。
- 最大训练预算：9000 optimizer updates。
- source-LORO时钟：U4000起每500 updates评估clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；主分数为`H(clean,min(三种LEO accuracy))`。
- 早停：连续5个评估点主分数未严格提高超过`1e-12`时停止；U4000前不得停止。
- 训练天：day1/day2/day3。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 物理batch48=16L+32U；clean/LEO拼接网络batch96；单次主干前向。
- 信道协议：`strict_pair_concat`、`concat_sat_ce_only`和三种`LEO_WEAK`。
- target/Phase2/support/query/truth：禁止访问。

## 版本与本地验证

- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 冻结代码/配置提交：`dfb11a63eca29c4ab5afff612ee0304e0b3c8d96`。
- 分析器8/8测试通过；PairBiCAD完整相关测试342/342通过，仅有3个既有AMP弃用警告。
- 收敛trainer聚焦53项通过；launcher/shell聚焦47项通过；trainer、launcher、分析器`py_compile`通过。
- U4000全量分析30/30通过，正式排序P2>P1>P0>P4>P3；前两名冻结为P2/P1。
- 真实checkpoint no-query smoke：使用`P2-F8-S392002`U4000 checkpoint（11,545,089字节）在CPU上严格重建，missing/unexpected/shape mismatch均为空；完成16L+32U、物理batch48、网络batch96、一次optimizer step和clean/三种LEO有限logits`[48,6]`；target/Phase2/support/query/truth访问均为false，结果`PASS`且不产生性能结论。
- 精确dry-run：12/12行，P2/P1×fold1/8×seed392001/392002/392003，全部U9000、day1/2/3和source-only；GPU0–3各2行、GPU4–7各1行，不超过每GPU2行。
- 独立P0/P1审查：pending；只允许报告会直接使本run跑错、越权、覆盖输出、无法启动或无法闭合的问题。
- `REJECTED_EXTRA_GATE`：不增加seal、成员hash、重复审查、Phase2数据检查或其他白名单外门槛。

## N607发布与命令

- 普通账户：`szu2070436088`；禁止管理员账户。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_convergence_20260831_r1`。
- 本地release归档：`E:\type10-7\release_archives\phase1_pairbicad_convergence_dfb11a63.tar.gz`。
- 远端release归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_convergence_dfb11a63.tar.gz`。
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r1`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r1.dispatcher.log`。
- GPU0–7；每GPU最多2个本run训练进程。

正式命令：

```text
PAIRBICAD_CONVERGENCE_CANDIDATES=P2,P1 bash code/scripts/launch_phase1_pairbicad_convergence_n607_20260831.sh
```

## 预期artifact

每行：

- `metrics_epoch.csv`、`metrics_epoch.jsonl`。
- `source_loro_curve.jsonl`。
- `source_loro/checkpoint_u<update>.pth`。
- `source_loro_selection.json`。
- `bicad_xr_final.pth`、`checkpoint_runtime.json`、`diagnostics.json`。
- clean及三种LEO场景JSON/log。
- `ARTIFACTS_COMPLETE.json`或`TECHNICAL_FAILURE.json`。

Run级：`plan.json`、`final_status.json`、dispatcher日志和PID文件。

## 停止与后续冻结规则

只有错误candidate/fold/receiver/day/seed/update、source-only越权、输出冲突、错误release/CWD、命令无法运行、无artifact闭合、同一确定性pre-prediction异常至少2行或进程归属不清时，才能精确绑定本run进程树、保留partial artifact并停止。不得因中间或最终低性能停止、重启、热补丁或选择性重跑；不得影响无关进程。

12行全部闭合后，只按source-only曲线比较P2/P1。胜出候选的6个row最佳update取中位数并量化到500 updates，冻结为后续fold1–5×3seed共15行最终确认预算；不得使用target、Phase2、support、query或truth反馈选择。

## 运行闭合

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- dispatcher PID：`2938180`；启动后已自然退出，无残留直属worker，GPU0–7均未被占用。
- 12/12行均保留`TECHNICAL_FAILURE.json`与`train.log`，`final_status.json`逐行记录技术停止；没有删除或覆盖partial artifact。
- 确定性失败指纹：`ValueError: invalid literal for int() with base 10: '18-2'`，位置为`_validate_bicad_xr_loro_args`。
- 根因：source receiver在冻结矩阵中使用ManySig整数索引`[1,3,4,6,8]`，真实payload的`rx_list`使用`18-2`等字符串标签；新增LORO边界错误地把标签强制转为整数，并在loader中按标签值而非payload索引解析held-out receiver。
- 该run没有进入训练、没有source-only性能结果，不参与P2/P1或训练预算选择。
- 修复边界：只修正receiver索引/标签解析并加入真实字符串标签回归测试；候选、fold、seed、U9000、每500 updates评估、patience5、协议和冻结规则不变。新实验必须使用不可覆盖的`r2` run/release。

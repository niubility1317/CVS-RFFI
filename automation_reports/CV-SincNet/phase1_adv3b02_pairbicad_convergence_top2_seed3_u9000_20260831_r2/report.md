# PairBiCAD前两名source-LORO收敛确认r2

## 状态

- 当前状态：`LOCAL_VERIFIED`。
- Run ID：`phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r2`。
- 目的：仅用source-only证据比较已冻结前两名P2/P1，并冻结后续完整多种子实验的训练预算。
- r1边界：r1因receiver索引/字符串标签解析错误在训练前12/12行技术停止，状态为`NO_PERFORMANCE_RESULT`，不参与选择；partial artifact保留。

## 固定代码与配置

- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 修复提交：`362c634326674a88cb2aacb3ce9cc76f00db9aa3`，远端OID已独立核对一致。
- 修复仅将ManySig held-out receiver按payload整数索引解析；真实`rx_list`标签可为`18-2`等字符串。候选、fold、seed、预算、协议和选择规则不变。
- 完整PairBiCAD测试通过；只有3个既有AMP弃用警告。trainer、launcher和分析器编译通过。
- Luna定点复审：`PASS，可以发布r2`。

## 冻结矩阵

- 候选：`P2,P1`，顺序固定。
- fold：`1,8`；对应held-out source receiver索引分别为1、8。
- seed：`392001,392002,392003`。
- 共12行，每GPU最多2个本run训练进程；GPU0–3各2行、GPU4–7各1行。
- 每行最大`9000 updates`；从U4000开始每500 updates在同row held-out source receiver上评估clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- source-LORO主分数：`H(clean_accuracy,min(three LEO accuracies))`；严格改善阈值`1e-12`，patience=5。
- 仅使用ManySig source receiver索引`[1,3,4,6,8]`和day1/2/3；禁止target、Phase2、support、query或truth。

## 路径与命令

- N607普通账户：`szu2070436088`，禁止管理员账户。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_convergence_20260831_r2`。
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r2`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_pairbicad_convergence_top2_seed3_u9000_20260831_r2.dispatcher.log`。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。

```text
PAIRBICAD_CONVERGENCE_CANDIDATES=P2,P1 bash code/scripts/launch_phase1_pairbicad_convergence_n607_20260831.sh
```

## 预期artifact与停止规则

每行必须保留`source_loro_curve.jsonl`、`source_loro/checkpoint_u<update>.pth`、`source_loro_selection.json`、final checkpoint、strict runtime、clean和三种LEO评估，以及`ARTIFACTS_COMPLETE.json`或`TECHNICAL_FAILURE.json`。Run级保留`plan.json`、`final_status.json`、dispatcher日志和PID文件。

只有错误candidate/fold/receiver/day/seed/update、source-only越权、输出冲突、错误release/CWD、命令无法运行、无artifact闭合、同一确定性pre-prediction异常至少2行或进程归属不清时才允许精确停止并保留partial artifact。不得因中间或最终低性能停止、重启、热补丁或选择性重跑；不得影响无关进程。

12行全部闭合后，只按source-only曲线冻结胜出候选。胜出候选6个row的最佳update取中位数并量化到500 updates，冻结为fold1–5×seed392001/392002/392003共15行最终确认预算；不得使用任何target或Phase2反馈。


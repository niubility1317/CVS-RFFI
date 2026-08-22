# Phase1 ADV3B02 SID-FFT96受控修复最小验证

## 当前状态

- run ID：`phase1_advb02_sidfft96_guarded_20260822_v1`
- 状态：`LOCAL_TDD_RED`
- 目的：验证首轮坍缩是否由SID适配器承受冻结Core90辅助损失并形成无界残差所致。
- 结论边界：本报告当前只记录诊断和预登记，不声明修复有效或优于CORE90。

## 首轮故障定位

- S1/S2/S3均完成200个epoch和独立clean/三种LEO_WEAK评测，进程、数据协议和评测链路无技术异常。
- 三行非SID参数相对基线checkpoint的最大绝对差均为0，排除成熟基座意外更新。
- `train_sid_delta_norm`从E1约0.04增长到E200的21,261.57、9,982.57和16,122.02；raw/SID预测一致率降到0.40%、4.05%和0.38%。
- 加权域对抗损失与SID残差范数的相关系数分别为0.99994、0.99995和0.99995；验证精度跌破50%的epoch分别为45、90和88。
- clean最终准确率仅16.66%、18.40%和16.65%，接近六分类随机水平。首轮矩阵判定为`SCIENTIFIC_FAILURE_NO_PROMOTION`。

## 修复候选与矩阵

|row|候选|作用|
|---|---|---|
|S0|`S0_FROZEN_CORE90`|同checkpoint、同协议冻结基线回读|
|S3G|`S3G_SIDFFT96_GUARDED`|仅训练受控SID投影，验证根因修复|

S3G只包含同一机制边界内的三项修复：逐样本残差范数不超过原始身份嵌入的10%；SID梯度只来自clean TX CE、Core90既定satellite TX CE和轻量身份锚定；checkpoint只按source-only `V_select`的`source_val_sat_hmean`选择。

## 协议与输入输出

- Phase1 source-only：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- `R_s∩R_t=∅`，target/query输入计数必须为0。
- 基座：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- seed：`392002`。
- 必评场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-ntrs-leo-weak-20260820`。
- N607 CWD、release、run root、log root、GPU和精确命令在本地验证完成后回填。

## 停止与判定规则

- 仅因协议/角色/seed/场景错误、输出碰撞、错误release/CWD、进程归属不清、确定性重复异常、无checkpoint或无法闭合独立prediction而技术停止；不得因中间性能低而停止。
- 技术稳定性要求：全程有限loss/gradient、非SID参数零漂移、有效SID残差比例不超过10%。
- 科学晋级沿用首轮预登记：相对S0 clean下降不超过0.3pp；LEO均值至少提升1pp；Strict UDU至少提升1pp；LEO floor至少提升0.5pp；`rescued>harmed`；RX probe相对下降至少20%且TX probe不下降。

## 预期artifact

- `source_validation_selected_ssdg.pth`
- `final_ssdg.pth`
- `metrics_epoch.csv`与`metrics_epoch.jsonl`
- `phase1_terminal_status.json`
- `independent_final_eval/final_eval.json`与`final_eval.txt`

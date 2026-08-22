# Phase1 ADV3B02 SID-FFT96受控修复最小验证

## 当前状态

- run ID：`phase1_advb02_sidfft96_guarded_20260822_v1`
- 状态：`RUNNING`
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
- Git提交：`633da733b9c849592b9f90eeaf11f031095b949e`；远端分支OID与本地`HEAD`一致。
- N607 release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_633da733`。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_guarded_20260822_v1`。
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_guarded_20260822_v1`。
- GPU：S0使用GPU0，S3G使用GPU1；发布前8张GPU均无计算进程。

## 本地与release验证

- 新增测试先因受控接口不存在而失败，修复后SID相关22项测试全部通过。
- 三个Python模块`py_compile`、launcher`bash -n`、训练参数dry-run和`git diff --check`通过。
- release归档：`E:\type10-7\local_artifacts\releases\phase1_advb02_sidfft96_guarded_20260822_v1\phase1_advb02_sidfft96_guarded_633da733.tar.gz`。
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_633da733.tar.gz`。
- 唯一归档SHA-256：本地与远端均为`0023abf9a98c6344d7204361d0b92b297b12ac74d62e653e9c5c0a19c1f36de0`，状态`VERIFIED`。
- 远端编译、import与launcher dry-run为`VERIFIED`；远端未安装pytest，重复单元测试记为`NONBLOCKING`，未增加安装或发布gate。
- 真实ADV3B02 checkpoint无query smoke：只读取4个`L_s`样本，`query_input_count=0`、`target_input_count=0`；raw/SID logits与嵌入最大差均为0，非SID可训练参数为0，梯度只进入4个SID projector参数，状态`VERIFIED`。

## 精确发布命令

在release CWD下分别以不可覆盖输出启动：

```bash
ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_633da733 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_guarded_20260822_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_guarded_20260822_v1 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl BASELINE_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth SID_MASK_PATH=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_leo_weak_20260821_v1/P0_SPECTRAL_AUDIT/sid_mask.npz GPU_S0=0 bash code/scripts/launch_phase1_advb02_sidfft96_guarded_20260822.sh --only=S0
ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_633da733 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_guarded_20260822_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_guarded_20260822_v1 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl BASELINE_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth SID_MASK_PATH=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_leo_weak_20260821_v1/P0_SPECTRAL_AUDIT/sid_mask.npz GPU_S3G=1 bash code/scripts/launch_phase1_advb02_sidfft96_guarded_20260822.sh --only=S3G
```

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

## 启动与早期健康证据

- 2026-08-22 18:12（Asia/Hong_Kong）启动S0和S3G；dispatch PID分别为`1015276`和`1015277`，主Python PID分别为`1015306`和`1015308`。
- S0绑定GPU0，S3G绑定GPU1；PID、CWD、cmdline、release、run root和GPU映射均与预登记一致，其他GPU未占用。
- S3G已完成E4/200，CSV与JSONL持续增长，`best_source_validation_ssdg.pth`已写出；未出现Traceback、RuntimeError、OOM或Killed。
- E4总损失为0.5907，等于受控closed loss；open loss为0。域对抗和正交损失仍被记录，但不计入总损失。
- E4有效SID残差比例为0.00352，低于0.10上界；raw/SID预测一致率为99.91%，训练/验证TX准确率为95.82%/98.86%，非有限loss和gradient计数均为0。
- 当前单epoch约15秒，训练预计约50分钟，随后进行checkpoint闭合与clean/三种LEO_WEAK独立评测；ETA范围为2026-08-22 19:05–19:25（Asia/Hong_Kong）。

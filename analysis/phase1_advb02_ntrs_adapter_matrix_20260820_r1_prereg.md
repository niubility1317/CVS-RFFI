# ADVB02 NTRS Adapter-Only首轮实验矩阵预登记

## 基本信息

- run ID：`phase1_advb02_ntrs_adapter_matrix_20260820_r1`
- 代码提交：`fb7d8871b1689a1e5bf38b7614704560ad339a28`
- 分支：`codex/advb02-ntrs-v2-recovery-20260820`
- 环境：N607，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- 数据：`Dataset_WigSig/ManySig.pkl`
- 成熟初始化：`runs/phase1_advb02_ntrs_v2_recovery_20260820_d1_bypass/ADVB02_NTRS_V2_D1_BYPASS_E200/final_ssdg.pth`
- seed：`392034`
- 数据角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 训练与测试增强：仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，禁止`mixed_orbit`
- checkpoint：E200最终checkpoint

## 首轮矩阵与GPU

|profile|候选|初始化|可训练部分|主要目标|GPU|
|---|---|---|---|---|---:|
|`a0_control`|`ADVB02_NTRS_A0_CONTROL_r1_E200`|从头|Core90|同release基线|0|
|`a0_bypass`|`ADVB02_NTRS_A0_BYPASS_r1_E200`|从头|Core90，NTRS严格旁路|旁路等价性|2|
|`a1_random_q`|`ADVB02_NTRS_A1_RANDOM_Q_r1_E200`|成熟D1|低秩adapter，q冻结|sat CE＋clean-zero＋relative|3|
|`a1_trainable_q`|`ADVB02_NTRS_A1_TRAINABLE_Q_r1_E200`|成熟D1|q＋低秩adapter|sat CE＋clean-zero＋relative|4|
|`a2_teacher_margin`|`ADVB02_NTRS_A2_TEACHER_MARGIN_r1_E200`|成熟D1|q＋低秩adapter|A1＋raw teacher KL＋margin|6|

`a3_support_gate`和`a4_joint_core`已实现但不在首轮同时启动；只有A2满足预登记晋级门槛后才发布A3，只有A3通过后才发布A4。

## 发布命令

每行由同一owner使用不可覆盖候选目录独立后台启动：

```text
env RUN_ID=phase1_advb02_ntrs_adapter_matrix_20260820_r1 NTRS_PROFILE=<profile> REPEAT=r1 GPU=<gpu> bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh
```

- output root：`runs/phase1_advb02_ntrs_adapter_matrix_20260820_r1/<candidate>/`
- log root：`logs/phase1_advb02_ntrs_adapter_matrix_20260820_r1/`
- 每个launcher在训练成功或失败后都尝试依据最终checkpoint执行独立测试；没有`final_ssdg.pth`则明确记录测试不可运行。

## 本地验证与独立审查

- 聚焦编译与测试：`66 passed`。
- 覆盖：q-only低秩残差、q梯度、raw参数和缓冲零漂移、训练raw主路径、评测always-on robust、旧D1到v3安全重建、独立测试重建、LEO_WEAK协议负测、7个profile dry-run。
- 独立P0/P1审查：首次发现2项训练模式P1；定点修复后复审结论`NO_P0_P1`。
- N607只读preflight：项目根目录与成熟D1 checkpoint可见；8张RTX3090各有1个训练进程，本矩阵每个所选GPU新增1个进程，不超过每GPU2个训练实验。

## 直接技术停止规则

只在以下情况停止对应run-owned进程树并保留全部产物：错误seed/角色/LEO场景、target/query泄漏、错误checkout或checkpoint、输出碰撞、launcher级错误、prediction无法闭合、同一确定性训练前异常在至少两行复现。不得因中间或最终性能低而停止合法运行。

## 预期artifact

每行至少应产生：

- `final_ssdg.pth`
- `phase1_terminal_status.json`
- `phase1_resource_summary.json`
- `ntrs_adapter_mechanism.json`（NTRS行）
- `metrics_epoch.csv`与`metrics_epoch.jsonl`
- `independent_final_eval/final_eval.json`
- clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`逐场景结果
- NTRS行raw/robust/fused、rescued/harmed、relative correction p50/p95、rotation angle p50/p95

训练完成但独立测试缺失时，不得标记实验完成。

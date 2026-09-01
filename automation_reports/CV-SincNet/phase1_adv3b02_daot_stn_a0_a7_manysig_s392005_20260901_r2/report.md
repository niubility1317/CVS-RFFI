# ADV3B02-DAOT-STN-V1 ManySig A0～A7 r2预登记报告

## 状态

- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- run ID：`phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2`
- 代码与配置commit：`b57f20cf711cbd150f82e9c28eecfc4c495ff8f6`
- 分支：`codex/adv3b02-daot-stn-v1-20260901`
- 方法声明：`deployment-proxy matched`
- 前序r1：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，产物保留，不复用其run root。

## 冻结候选与矩阵

| Row | GPU | Orbit Teacher | Tangent |
|---|---:|---|---|
| A0 | 0 | 无 | 无 |
| A1 | 1 | EMA两视图 | 无 |
| A2 | 2 | 三教师视图简单平均 | 无 |
| A3 | 3 | 三教师视图部署加权球面聚合 | 无 |
| A4 | 4 | A3 | 单参数tangent |
| A5 | 5 | A3 | 协方差随机方向tangent |
| A6 | 6 | A3 | 分支选择性tangent |
| A7 | 7 | A3 | tangent＋fingerprint keep |

A8 Temporal Memory不属于本次8行矩阵；A2/A3是性能优先三教师视图上界实验。

## 冻结数据与权限

- 数据集：`Dataset_WigSig/ManySig.pkl`，`equalized=true`
- split：`tx_rx_day_1_7_2`，seed=`392005`
- source receiver：`1,3,4,6,8`；source day：`1,2,3`；pool=90000
- `L_s/U_s/V=6300/56700/27000=0.07/0.63/0.30`
- V为单一source validation，只读且不更新模型、EMA、prototype或normalization
- target receiver：`0,2,5,7,9,10,11`；target day：`0,1,2,3`；TX：`0,1,2,3,4,5`
- 每个场景168000样本；场景为`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- target只用于冻结checkpoint后的评测；训练、校准和checkpoint选择不可读取target truth。

## r1定点修复与本地验证

- 修改`code/SSDG/train_ssdg.py`：checkpoint域标签数变化时，仅允许重建`dom_head.net.3`与`adv_head.net.3`的weight/bias；任何其他形状不一致仍硬失败。
- 修改`tests/test_adv3b02_daot_stn.py`：增加14→15正测与非域参数错配负测。
- 更新实现追踪与r1失败记录。
- TDD：新增回归先失败，修复后2项通过。
- 相关回归：84项全部通过。
- 真实checkpoint无query smoke：`PASS`，`query_inputs=0`、`target_inputs=0`。
- 定点P0/P1复审：无遗留直接阻断问题。
- Git状态：修复提交已push并独立核对远端分支OID。

## 环境、路径与启动命令

- N607普通账户；GPU0～7每卡一行，不超过资源上限。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release代码根：`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2/CVS-RFFI-repo`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2`
- 基座checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- release归档本地：`E:\type10-7\local_artifacts\releases\phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2.zip`
- release归档远端：`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2.zip`

```bash
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2/CVS-RFFI-repo PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2 BASE_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth bash code/scripts/launch_phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901.sh > /home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r2/dispatcher.out 2>&1 &
```

## 系统技术停止规则与预期artifact

仅在数据/query边界违反、错误split/receiver/day、输出碰撞、错误checkout/CWD、确定性异常至少两行重复、无prediction闭合或scorer连接错误时停止精确绑定的r2进程树。低性能不停止。

每行预期生成`config.json`、`train.log`、checkpoint、clean与三个LEO弱场景的独立评测日志和指标、`status.txt`；四场景齐全后才可标记`ARTIFACTS_COMPLETE`。

## r2启动、用户矩阵修订与协议停止

- 用户明确覆盖默认GPU并发限制后，r2 dispatcher PID=`3872274`启动；8个主训练进程分别绑定GPU0～7、正确release CWD和独立候选输出目录。
- r1的14→15域输出故障已修复：8行student与teacher均打印4个域输出张量重建marker，未再出现形状异常。
- 用户随后明确“不需要ADV3B02对比基线”，因此A0进程树被精确绑定并停止，partial日志保留，不纳入科学比较。
- 启动日志暴露新的协议错误：命令行虽传入`legacy_l_u_v`及`0.07/0.63/0.30`，旧`_enforce_muse_source_protocol`仍自动改写为`l_s_u_s_v_cal_v_select`及`0.07/0.63/0.15/0.15`。
- 该改写违反用户批准的单一只读V=27000协议，故按协议错误规则停止r2完整进程树；共133个仅属于r2的dispatcher、worker和DataLoader进程被精确停止并核对无残留，其他运行不受影响。
- r2所有release、日志和partial产物均保留，未原地修补或重启。
- 科学判定：`NO_PERFORMANCE_RESULT`。
- 后续：本地修复MUSE自动改写，新增运行时单一V日志负测，并以A1～A7七行、全新r3 release/run root重新发布。

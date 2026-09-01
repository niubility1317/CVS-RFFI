# ADV3B02-DAOT-STN-V1 ManySig A0～A7 r2预登记报告

## 状态

- 当前状态：`LOCAL_VERIFIED`
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

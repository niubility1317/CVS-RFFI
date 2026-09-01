# ADV3B02-DAOT-STN-V1 ManySig A0～A7预登记报告

## 状态

- 当前状态：`LOCAL_VERIFIED`
- run ID：`phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1`
- 代码与配置commit：`b0c876230c761988999ba8d0e1ffbc3cbb13ab9d`
- 分支：`codex/adv3b02-daot-stn-v1-20260901`
- 方法声明：`deployment-proxy matched`

## 冻结数据

- 数据集：`Dataset_WigSig/ManySig.pkl`，`equalized=true`
- split：`tx_rx_day_1_7_2`，seed=`392005`
- source receiver：`1,3,4,6,8`
- source day：`1,2,3`
- source pool：90000
- `L_s/U_s/V=6300/56700/27000=0.07/0.63/0.30`
- V为单一source validation，只读且不更新模型、EMA、prototype或normalization
- target receiver：`0,2,5,7,9,10,11`
- target day：`0,1,2,3`
- target TX：`0,1,2,3,4,5`
- 每个clean或LEO场景目标样本：168000
- 最终场景：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- target只用于冻结checkpoint后的评测；训练、校准和checkpoint选择均不可读取target truth

## 冻结矩阵

| Row | GPU | Orbit Teacher | Tangent |
|---|---:|---|---|
| A0 | 0 | 无 | 无 |
| A1 | 1 | EMA两视图 | 无 |
| A2 | 2 | 三教师视图简单平均 | 无 |
| A3 | 3 | 三教师视图部署加权球面聚合 | 无 |
| A4 | 4 | A3 | 单参数tangent |
| A5 | 5 | A3 | 协方差随机方向tangent |
| A6 | 6 | A3 | 分支选择性tangent |
| A7 | 7 | A3 | tangent+fingerprint keep |

A8 Temporal Memory不属于本次8行矩阵。

## 环境与路径

- N607普通账户
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据/运行根：`/home/szu2070436088/2510044040/CV-SincNet`
- release代码根：`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1/CVS-RFFI-repo`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1`
- 基座checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- release归档本地：`E:\type10-7\local_artifacts\releases\phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1.zip`
- release归档远端：`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1.zip`

## 启动命令

工作目录：
`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1/CVS-RFFI-repo`

```bash
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1/CVS-RFFI-repo PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1 BASE_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth bash code/scripts/launch_phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901.sh > /home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901_r1/dispatcher.out 2>&1 &
```

## 停止规则

只在以下系统性技术失败时停止本run且只处理与该run ID绑定的进程树：数据/query边界违反，source/target receiver或day/split错误，输出目录碰撞，错误checkout/CWD，启动命令无法运行，确定性相同异常在至少两行重复，无法形成完整checkpoint/prediction，或scorer连接错误。低性能不触发停止。

## 预期artifact

每行独立目录应包含：`config.json`、`train.log`、`final_ssdg.pth`、`eval_joint.log`、`metrics_joint.json`、四个场景的`eval_*.log`和`metrics_*.json`、`status.txt`。只有clean与三个LEO弱场景均完整时，单行才能标记`ARTIFACTS_COMPLETE`。


# ADV3B02-DAOT-STN-V1 ManySig A1～A7 r3预登记报告

## 状态

- 当前状态：`RUNNING`
- run ID：`phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3`
- 代码与配置commit：`7504f6669fbc0a02b9b7446f463f561ecbcef6de`
- 分支：`codex/adv3b02-daot-stn-v1-20260901`
- 方法声明：`deployment-proxy matched`
- r1：14→15域输出维度技术失败，无性能结果。
- r2：旧MUSE函数把单一V改回V_cal/V_select的协议技术失败，无性能结果。

## 冻结矩阵

用户明确不需要ADV3B02对比基线，因此A0从r3删除，不启动、不评分、不参与比较。

| Row | GPU | Orbit Teacher | Tangent |
|---|---:|---|---|
| A1 | 1 | EMA两视图 | 无 |
| A2 | 2 | 三教师视图简单平均 | 无 |
| A3 | 3 | 三教师视图部署加权球面聚合 | 无 |
| A4 | 4 | A3 | 单参数tangent |
| A5 | 5 | A3 | 协方差随机方向tangent |
| A6 | 6 | A3 | 分支选择性tangent |
| A7 | 7 | A3 | tangent＋fingerprint keep |

A2/A3仅作为性能优先三教师视图上界实验；A8 Temporal Memory不发布。

## 冻结数据与权限

- 数据集：`Dataset_WigSig/ManySig.pkl`，`equalized=true`
- split：`tx_rx_day_1_7_2`，seed=`392005`
- source receiver：`1,3,4,6,8`；source day：`1,2,3`；pool=90000
- 单一只读V：`L_s/U_s/V=6300/56700/27000=0.07/0.63/0.30`
- 禁止把V重新拆为V_cal/V_select；V不更新模型、EMA、prototype或normalization。
- target receiver：`0,2,5,7,9,10,11`；day：`0,1,2,3`；TX：`0～5`
- 每个场景168000样本；场景为`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- target仅用于冻结checkpoint后的评测；训练、校准和checkpoint选择不可读取target truth。

## 修复与本地验证

- checkpoint只允许重建`dom_head.net.3`与`adv_head.net.3`的4个域输出张量；其他形状错配硬失败。
- 显式`legacy_l_u_v+0.07/0.63/0.30+cal/select=0`不再被MUSE改写。
- 运行时必须打印`L/U/V=6300/56700/27000`，不得打印两个13500验证集。
- 新launcher只包含A1～A7七行，GPU1～7。
- TDD：协议保持和单一V运行时summary均先失败后通过。
- 相关回归：87项全部通过。
- 真实checkpoint无query smoke：`PASS`，`query_inputs=0`、`target_inputs=0`。
- 定点P0/P1复审：无遗留直接阻断问题。
- Git修复提交已push；启动前独立核对远端分支OID。

## 环境、路径与命令

- 用户明确覆盖本次GPU线程/并发数默认限制；仍不干预任何既有任务。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release代码根：`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3/CVS-RFFI-repo`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3`
- 基座checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- release归档本地：`E:\type10-7\local_artifacts\releases\phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3.zip`
- release归档远端：`/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3.zip`

```bash
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3/CVS-RFFI-repo PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3 BASE_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth bash code/scripts/launch_phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902.sh > /home/szu2070436088/2510044040/releases/phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902_r3/dispatcher.out 2>&1 &
```

## 停止规则与预期artifact

仅在数据/query边界违反、错误split/receiver/day、输出碰撞、错误checkout/CWD、确定性异常至少两行重复、无prediction闭合或scorer连接错误时停止精确绑定的r3进程树。低性能不停止。

每行预期生成`config.json`、`train.log`、checkpoint、clean与三个LEO弱场景的独立评测日志和指标、`status.txt`；四场景齐全后才可标记`ARTIFACTS_COMPLETE`。

## r3发布与启动证据

- release归档SHA256：本地与远端均为`0b3b41f9fce013b586fc1ef530d71926a15aa467e6196495779a4e6e9a091672`，状态`VERIFIED`。
- 远端验证：launcher与worker语法、Python编译、7行dry-run、真实checkpoint无query smoke、15域兼容加载、单一V协议保持均`VERIFIED`。
- dispatcher PID：`3884219`，CWD为r3 release代码根，状态活动。
- 7个主训练PID及GPU：A1=`3884248/GPU1`、A2=`3884249/GPU2`、A3=`3884259/GPU3`、A4=`3884252/GPU4`、A5=`3884256/GPU5`、A6=`3884254/GPU6`、A7=`3884257/GPU7`。
- 7个候选目录、7份`train.log`均已建立；A0不存在。
- checkpoint兼容marker共14条（7行student＋teacher）；单一`L/U/V=6300/56700/27000`marker共14条；旧`13500/13500`回退marker为0。
- 两次短时回读均维持7个主训练进程，GPU1～7有对应活动负载；未发现Traceback、RuntimeError、OOM、Killed或其他确定性异常。
- 当前只完成发布和启动健康闭合，尚无性能结果；不得提前作方法优劣判断。

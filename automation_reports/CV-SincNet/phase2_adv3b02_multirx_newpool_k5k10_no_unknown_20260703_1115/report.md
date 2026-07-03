# ADV3B02 Phase2-C多接收机K5/K10低shot搜索

## 运行身份

| 字段 | 值 |
|---|---|
| Run ID | `phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115` |
| 时间 | 2026-07-03 11:15 Asia/Hong_Kong |
| 操作方 | Codex |
| 目标 | 基于`ADV3B02_CORE90_SOFT_E200`，在Phase2-C中使用叠加LEO星地信道的新类样本和旧类目标域样本，执行少样本新类学习和少样本域适应；未知类拒识不纳入目标；同一行必须达到旧类准确率>=85%、新类均值>=80%、每个新类准确率>=85%、新类数量>=2，并以5/10-shot性能为主证据 |
| 比较基线 | `7-14`目标接收机K=5/K=10全135候选和K=10近边界seed sweep均未达标；K=20有2行满足85/85但仅作为higher-shot诊断 |
| 假设 | 不同合法目标接收机的LEO目标域几何和receiver response可能降低旧类/新类混淆；在`20-1,3-19,7-7,8-8`上复用广候选池，有机会在K=5或K=10找到同一行达标组合 |

## 协议与数据

| 项 | 设置 |
|---|---|
| 基础checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| source old TX | `0,1,2,3,4,5`，评估标签为`14-10,14-7,20-15,20-19,6-15,8-20` |
| source receivers | `0,1,2,3,4,5,6` |
| target receivers | `20-1,3-19,7-7,8-8`，均为`项目.md`确认的Phase2候选，且与source receivers不相交 |
| target-old | 每个目标接收机上的`Y_old`，role=`target_old` |
| target-new候选池 | 复用已验证`7-14`广候选池135个ManyTx非旧类TX；每个接收机导出后由sweep按`K+query`样本数过滤 |
| target-new角色 | `target_new,new` |
| 信道视图 | target-old和target-new均为`satellite`视图，`simplified_leo_residual`，场景`leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| unknown | 不导出、不评估、不参与阈值或成功判据 |
| K | `5`和`10` |
| query | 旧类每TX40，新类每TX40 |
| 方法 | `proto,knn1,knn3,knn5` |
| 成功判据 | `old_acc>=0.85`、`seen_new_acc>=0.80`、`min_seen_new_class_acc>=0.85` |

## 本地变更与验证计划

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh` | 为4个目标接收机分别导出features，并在每个接收机上运行K=5/K=10全候选pair sweep |

待执行验证：

```text
conda activate ssr-gpu
bash -n code/scripts/launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh
bash code/scripts/launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh --dry-run
```

## N607计划

| 项 | 值 |
|---|---|
| 远端脚本 | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh` |
| 远端run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115/` |
| 远端log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115/` |
| GPU分配 | `20-1`->GPU0，`3-19`->GPU1，`7-7`->GPU2，`8-8`->GPU3 |
| 预期输出 | 每个receiver一个`features.npz`、`pair_sweep_k5.json/csv`、`pair_sweep_k10.json/csv`和对应日志 |

## 当前状态

| 项 | 状态 |
|---|---|
| 本地脚本 | 已创建 |
| 本地验证 | PASS：`conda activate ssr-gpu; bash -n code/scripts/launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh; bash code/scripts/launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh --dry-run` |
| N607 preflight | 待重新确认 |
| 远端同步 | 待完成 |
| 远端运行 | 待启动 |
| 结果解析 | 待完成 |
| Git提交 | 待完成 |

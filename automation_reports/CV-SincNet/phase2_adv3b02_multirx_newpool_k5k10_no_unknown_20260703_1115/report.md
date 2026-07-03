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
| N607 preflight | PASS：2026-07-03 11:15:54 CST，direct N607可达，项目根目录可见，8张RTX3090空闲显示10MiB |
| 远端同步 | PASS：`scp E:\type10-7\code\scripts\launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115.sh`；远端`bash -n`通过，SHA256=`07db668695dc2e8290f8849af6eb11a4d2a1961b30ffd20586d5e7e57493423f` |
| 远端运行 | 已完成：launcher输出`[ADV3B02-MULTIRX-NEWPOOL-K5K10-DONE]` |
| 结果解析 | 已完成：`multirx_k5k10_summary.json`已拉回并本地解析 |
| Git提交 | 初始launcher/report已提交，结果summary和报告更新待提交 |

## 运行结果

| receiver | K | eligible new TX | rows | 同一行达标数 | 结论 |
|---|---:|---:|---:|---:|---|
| `20-1` | 5 | 135 | 36180 | 0 | 未达标 |
| `20-1` | 10 | 135 | 36180 | 0 | 未达标 |
| `3-19` | 5 | 127 | 32004 | 0 | 未达标 |
| `3-19` | 10 | 127 | 32004 | 0 | 未达标 |
| `7-7` | 5 | 132 | 34584 | 0 | 未达标 |
| `7-7` | 10 | 132 | 34584 | 0 | 未达标 |
| `8-8` | 5 | 131 | 34060 | 0 | 未达标 |
| `8-8` | 10 | 131 | 34060 | 0 | 未达标 |

| 排名 | receiver | K | 新类组合 | 方法 | 旧类准确率 | 新类均值 | 新类逐类最低 | 逐新类准确率 | 解释 |
|---:|---|---:|---|---|---:|---:|---:|---|---|
| 1 | `8-8` | 10 | `18-20,3-2` | `knn1` | 76.67% | 78.75% | 77.50% | `18-20`:77.50%,`3-2`:80.00% | 新类接近但旧类不足 |
| 2 | `8-8` | 10 | `14-9,2-7` | `knn3` | 75.83% | 78.75% | 77.50% | `14-9`:77.50%,`2-7`:80.00% | 旧类不足 |
| 3 | `8-8` | 10 | `14-9,2-7` | `knn1` | 75.42% | 85.00% | 80.00% | `14-9`:90.00%,`2-7`:80.00% | 新类均值达标但逐类和旧类不足 |
| 4 | `7-7` | 10 | `14-13,7-11` | `proto` | 75.00% | 92.50% | 90.00% | `14-13`:90.00%,`7-11`:95.00% | 新类强，但旧类域适应不足 |
| 5 | `7-7` | 5 | `15-19,20-7` | `knn1` | 75.00% | 81.25% | 75.00% | `15-19`:87.50%,`20-7`:75.00% | K=5新类不稳定且旧类不足 |

## 产物与哈希

| 文件 | SHA256 |
|---|---|
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115\multirx_k5k10_summary.json` | `db256e022f28176123d35ef27cf15a97cc74177533cf36f5fb4388f7ea45c875` |
| `ADV3B02_MULTIRX_NEWPOOL_RX20_1.out` | `de9b218c8762917e9ff35f06da011c1250b3e6e6daf13ae20b9313c6ae27708e` |
| `ADV3B02_MULTIRX_NEWPOOL_RX3_19.out` | `9c2a1a171bd41ef992a3dcd2ecb68592e1d97b0020573b7d5fab6433e19ee8b8` |
| `ADV3B02_MULTIRX_NEWPOOL_RX7_7.out` | `4990b9e8fcfaf2d3d96ea2c30dd94f25138551145fdb8a89b12a3776e4981d44` |
| `ADV3B02_MULTIRX_NEWPOOL_RX8_8.out` | `b0c0f9d5955207f4f1fb274bfb1efe3e01c4333a95b934123b758dfe894c43e2` |

## 解释与下一步

`7-14`在K=10最接近目标但新类逐类最低只有82.50%，旧类84.58%；其它receiver的K=5/K=10没有更优同一行。多接收机扩展显示单纯更换target receiver和复用朴素`proto/knn`不足以满足低shot目标。下一步应从方法侧推进：把KNN改造为不保存原始support的压缩原型/子原型记忆库，并加入旧类目标域校准项，使`7-7`这类“新类强、旧类弱”的行获得旧类恢复，同时控制新类逐类不掉到85%以下。

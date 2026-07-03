# ADV3B02 Phase2-C 7-14多新类85/88严格目标扩展搜索

## 运行身份

| 字段 | 值 |
|---|---|
| Run ID | `phase2_adv3b02_rx7_14_newpool_perclass85_old88_no_unknown_20260703_1020` |
| 时间 | 2026-07-03 10:20 Asia/Hong_Kong |
| 操作方 | Codex |
| 目标 | 基于`ADV3B02_CORE90_SOFT_E200`，在Phase2-C中使用叠加LEO星地信道的新类样本和旧类目标域样本，执行少样本新类学习和少样本域适应；未知类拒识不纳入目标；同一行必须达到旧类准确率>=85%、新类均值>=80%、每个新类准确率>=85%、新类数量>=2；最新目标要求样本数越少越好，并以5/10-shot性能为标准 |
| 基础特征 | 复用`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/ADV3B02_RX7_14_NEWPOOL_PAIRSWEEP/features.npz` |
| 协议边界 | `7-14`目标接收机，target-old和target-new均为`satellite`视图，`simplified_leo_residual`，场景`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；未知类不导出、不评估 |

## 现有证据复核

| 检查 | 结果 |
|---|---|
| 旧目标`65/80`通过行 | `19-3,3-8`+`knn1`：旧类84.58%、新类`19-3`82.50%、新类`3-8`85.00% |
| 新目标`85/88`现有通过行 | 0 |
| 现有最接近边界 | 旧类最高且接近新类的行达旧类88.33%，但新类逐类最低只有70.00%；新类逐类最高行旧类只有84.58% |
| 结论 | 旧结果不能作为新目标完成证据，必须扩展搜索 |

## 扩展搜索计划

| 项 | 设计 |
|---|---|
| 候选池 | 使用现有135个`7-14`非旧类TX候选 |
| 组合规模 | 2个新类 |
| 第一阶段 | 取消原先`max_pair_candidates=50`截断，枚举全部135个候选进入两两组合 |
| 方法 | 先复用`proto,knn1,knn3,knn5`，保持与旧结果可比 |
| 成功判据 | 最新目标按`old_acc>=0.85`、`seen_new_acc>=0.80`且`min_seen_new_class_acc>=0.85`重算；由于逐新类至少85%强于均值80%，主瓶颈仍是`min_seen_new_class_acc>=0.85` |
| 第二阶段 | 若K=10失败，在协议允许的few-shot/low-shot边界内提高到`K_old=20,K_new=20`，query仍保持40/类 |
| 若仍失败 | 再实现KNN变体：压缩anchor/加权KNN/旧类保留偏置/信道状态补偿，仍保持不使用未知类 |

## 本地新增KNN变体

| 文件 | 用途 | 验证 |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_classmax_knn_bias_sweep.py` | 对候选组合执行class-max KNN，并扫描旧类score bias，以保留KNN可扩展性同时修正old/new边界 | PASS：`conda activate ssr-gpu; python -m py_compile code\scripts\phase2_classmax_knn_bias_sweep.py` |
| `E:\type10-7\code\scripts\phase2_margin_rescue_knn_sweep.py` | 只对“新类胜出但旧类相似度足够高且边界margin足够小”的query执行old rescue，避免全局old-bias压垮新类 | PASS：`conda activate ssr-gpu; python -m py_compile code\scripts\phase2_margin_rescue_knn_sweep.py` |

| 文件 | SHA256 |
|---|---|
| `code/scripts/phase2_classmax_knn_bias_sweep.py` | `8C7C81BCC11E495C403C26B900D6F37E3C38E99C4D67F96818F39D4D5CA7ED3E` |
| `code/scripts/phase2_margin_rescue_knn_sweep.py` | `B3D5AFCADBC3BBF04F330B179ECD649B114059B9EC6704DFCC3ADFD2FD7EEA4D` |

## 本地版本状态

| 检查 | 结果 |
|---|---|
| 根目录Git | `E:\type10-7`不是Git仓库 |
| Git-backed发布仓库 | `github_publish\CVS-RFFI-repo`分支`codex/cvs-rffi-release-20260626`，编辑前工作区干净，ahead 146 |

## 待完成

| 项 | 状态 |
|---|---|
| N607 preflight和占用检查 | 待完成 |
| 全候选远程扩展搜索 | K=10已完成：全135候选、2新类组合、`proto/knn1/knn3/knn5`，`joint_pass_count=0`；`rows_old88=10582`但`rows_min85=0`，说明瓶颈是逐新类85% |
| K=20远程扩展搜索 | 已完成：全135候选、2新类组合、`proto/knn1/knn3/knn5`，`joint_pass_count=0`；`rows_min85=2`但对应旧类为86.67%，距离88%差约3.2个旧类query |
| class-max KNN旧类bias扫描 | 已完成：`joint_pass_count=0`；全局old-bias能提高old但快速压低新类，不能满足85/88 |
| margin-gated old rescue KNN扫描 | 已完成：`joint_pass_count=0`；触发rescue时通常损伤新类，保持逐新类85%的行没有获得旧类额外修正 |
| 最新5/10-shot目标调整 | K=20/30/40相关结果仅保留为失败诊断，不作为达成目标路径；后续搜索优先K=5和K=10 |
| 结果拉回和严格审计 | K=5/K=10/K=20全135结果已拉回并按更新后门槛重算；K=10近边界seed sweep summary/log已拉回 |
| 报告和Git提交 | 待完成 |

## 2026-07-03 11:10更新后目标复核

| K | 样本量 | 同一行达标数 | 最接近同一行 | 解释 |
|---:|---:|---:|---|---|
| 5 | support总数40 | 0 | `2-13,3-8`+`knn1`：旧类87.50%，新类均值71.25%，逐新类65.00%/77.50% | 旧类可达标，但新类远低于逐类85% |
| 10 | support总数80 | 0 | `19-3,3-8`+`knn1`：旧类84.58%，新类均值83.75%，逐新类82.50%/85.00% | 距离目标最接近，但旧类和一个新类各差约1个query |
| 20 | support总数160 | 2 | `18-20,19-3`+`knn1`：旧类86.67%，新类均值86.25%，逐新类87.50%/85.00%；`19-13,19-3`+`knn1`：旧类86.67%，新类均值85.00%，逐新类85.00%/85.00% | 满足更新后的85/85目标，但不满足“以5/10-shot为标准”的低shot主证据，只作为诊断上限 |

下一步执行K=10近边界seed sweep：候选限制在`19-3,3-8,1-8,6-6,2-13,9-1`，方法先用`knn1`，种子`422001..422100`。目的不是更改协议，而是检查K=10 support/query划分波动中是否存在同一行满足`old_acc>=85%`与逐新类`>=85%`的低shot证据。

## K=10近边界seed sweep结果

| 项 | 值 |
|---|---|
| 远端输出目录 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_rx7_14_newpool_perclass85_old88_no_unknown_20260703_1020/seed_sweep_k10_near/` |
| 远端日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_rx7_14_newpool_perclass85_old88_no_unknown_20260703_1020/seed_sweep_k10_near.out` |
| 本地summary | `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rx7_14_newpool_perclass85_old88_no_unknown_20260703_1020\seed_sweep_k10_near_summary.json` |
| 本地log | `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rx7_14_newpool_perclass85_old88_no_unknown_20260703_1020\seed_sweep_k10_near.out` |
| seeds | `422001..422100` |
| 行数 | 1500 |
| 更新后同一行达标数 | 0 |
| summary SHA256 | `46b3b339b9fd57f9140b118a06677f6818d72eedba7befd648bae29e1064d979` |
| log SHA256 | `049122f8c39d5a25bce5628ad429e2d0406df01f27caf48551a9b7991b7e5644` |

| 排名 | seed | 新类组合 | 方法 | 旧类准确率 | 新类均值 | 新类逐类最低 | 逐新类准确率 | 判定 |
|---:|---:|---|---|---:|---:|---:|---|---|
| 1 | 422001 | `19-3,3-8` | `knn1` | 84.58% | 83.75% | 82.50% | `19-3`:82.50%,`3-8`:85.00% | 旧类和`19-3`各差约1个query |
| 2 | 422001 | `1-8,19-3` | `knn1` | 84.17% | 82.50% | 82.50% | `1-8`:82.50%,`19-3`:82.50% | 未达旧类85%和逐类85% |
| 3 | 422092 | `19-3,2-13` | `knn1` | 83.75% | 80.00% | 80.00% | `19-3`:80.00%,`2-13`:80.00% | 均值达80%，但逐类不足 |

结论：`7-14`目标接收机在K=5/K=10标准下，目前`proto/knn1/knn3/knn5`和近边界seed sweep均没有达到更新后的同一行目标；K=20已有2个同一行满足85/85的诊断证据，但不能作为5/10-shot主结论。下一步应优先扩展到其它协议合法目标接收机`20-1,3-19,7-7,8-8`，同时保留KNN压缩原型变体作为方法创新路线。

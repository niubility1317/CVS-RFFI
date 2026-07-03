# ADV3B02 Phase2-C 7-14多新类85/88严格目标扩展搜索

## 运行身份

| 字段 | 值 |
|---|---|
| Run ID | `phase2_adv3b02_rx7_14_newpool_perclass85_old88_no_unknown_20260703_1020` |
| 时间 | 2026-07-03 10:20 Asia/Hong_Kong |
| 操作方 | Codex |
| 目标 | 基于`ADV3B02_CORE90_SOFT_E200`，在Phase2-C中使用叠加LEO星地信道的新类样本和旧类目标域样本，执行少样本新类学习和少样本域适应；未知类拒识不纳入目标；同一行必须达到旧类准确率>=88%、每个新类准确率>=85%、新类数量>=2 |
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
| 成功判据 | `old_acc>=0.88`且`min_seen_new_class_acc>=0.85` |
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
| margin-gated old rescue KNN扫描 | 待完成 |
| 结果拉回和严格审计 | 待完成 |
| 报告和Git提交 | 待完成 |

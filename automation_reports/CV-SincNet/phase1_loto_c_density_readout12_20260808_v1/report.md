# Phase1 LOTO-C DensityReadout12预注册报告

状态：`LOCAL_VERIFIED / RELEASE_READY`

目标模式：`GOAL_MODE=ACTIVE`

证据标签：`DEVELOPMENT_CROSS_TX_CV_NON_CONFIRMATORY`

版本承载：`E:\type10-7`根目录不是Git仓库；本报告同步镜像到Git工作树同相对路径。

## 1.实验身份与目的

| 字段 | 冻结值 |
|---|---|
| run ID | `phase1_loto_c_density_readout12_20260808_v1` |
| 日期 | 2026-08-08 |
| 主Agent | `/root` |
| 唯一N607 runner | `/root/n607_geosat_lite_runner` |
| 实现commit | `29d488e16408c26a2db081b02acaea6bee66830b` |
| 独立审查 | `VERDICT=APPROVE; P0=0; P1=0` |
| 目标 | 不训练、不对齐，检验冻结GeoSat-C `z_id`是否已有稳定source-only密度拒识信号 |

## 2.方法与矩阵

只使用上一轮6个C arm的现有`features.npz`，不读取G arm、不重导出feature、不改checkpoint。

| Readout | 固定方法 |
|---|---|
| prototype | source closed-correct样本建立每类cosine prototype；按预测类source score Q0.98接受 |
| knn5 | source closed-correct样本建立cosine kNN-5；exclude-self；按预测类source score Q0.98接受 |

Q0.98由`old_drop<=2pp`保护门直接确定，不扫描阈值。评估器的非空proxy诊断集合只来自source误分类样本；`proxy_unknown_roles=__disabled__`且`threshold_policy=source_accept`，因此primary/secondary held不参与建库、阈值或选择。

六fold沿用上一轮TX角色。每fold执行prototype/knn5×primary/secondary，共24条CPU命令。主汇总只含12条primary结果；secondary只作敏感性。

## 3.本地实现与验证

| 文件 | 用途 | archive SHA256 |
|---|---|---|
| `code/scripts/launch_phase1_loto_c_density_readout12_20260808.sh` | 非覆盖24命令launcher与逐命令exit receipt | `52becd4f67ca86d7b097d056d9d287514c91aaa611435bd13ed99eae770992bc` |
| `code/tests/test_phase1_knn_source_only_threshold.py` | held特征改变时source阈值不变的负测 | `56394c2fab27dca068b4cc03dfb4a8fbb4c94b76bffc2299d7c7d59ce6eaaf96` |
| `analysis/phase1_loto_c_density_readout12_design_20260808.md` | 冻结设计与证据边界 | `ff042f8d7e7986b1483e021af49b80a28191c04631fd1a07e8ac061940d7c738` |
| `code/scripts/eval_phase1_prototype_reject.py` | 既有prototype evaluator | `de6c87b369fd306d437d9ac061d9d61c9f08366197087d85b1dd612076a490aa` |
| `code/scripts/eval_phase1_knn_reject.py` | 既有kNN evaluator | `faa0fa08ef8646c12fc92cc812f32ce7c110a908f7afa2ff7d25dfc0362a1f02` |

验证：

- `ssr-gpu: pytest test_phase1_knn_source_only_threshold.py test_phase1_multiview_reject_eval.py`：3 passed。
- `bash -n`：PASS。
- `DRY_RUN=1`：24条；prototype/knn5各12；primary/secondary各12；Q0.98、held proxy disabled和source-misclassified diagnostic proxy均为24条。
- `git diff --check`：PASS。
- 独立P0/P1审查：held不回流，Q0.98只来自closed-correct source，允许发布。

## 4.N607冻结交接

| 字段 | 冻结值 |
|---|---|
| release | `/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_loto_c_density_readout12_20260808_v1_29d488e1` |
| run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_c_density_readout12_20260808_v1` |
| log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_loto_c_density_readout12_20260808_v1` |
| input root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1/postfreeze_audit_v1` |
| input manifest | SHA256=`e81bf52928d812309a1f9b6b4bfb7aab6f1225a1ff817159345d8f0230a57a30` |
| Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| device | CPU；不占GPU |
| retry | `NO` |

冻结启动形式：

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_loto_c_density_readout12_20260808_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1/postfreeze_audit_v1 bash <release>/code/scripts/launch_phase1_loto_c_density_readout12_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_loto_c_density_readout12_20260808_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.产物与健康门

预期24个metrics JSON、24个score CSV、24个stdout日志和header+24的`completion.tsv`。每条命令只能执行一次；任一非零退出立即停止剩余命令并标`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。不得按FAR、coverage、AUROC或old drop中止。

完成后只回收JSON、CSV、日志、completion和manifest；不下载NPZ/checkpoint。runner只判断技术完成，不解释性能。

## 6.晋级门与声明边界

每个readout需要在六个primary上形成稳定方向，并同时满足`FAR<=5%`、safe rejection`>=95%`、`old_drop<=2pp`。不得用单fold或secondary最优值晋级。

本轮不是K-shot、注册、真实unknown或Phase3正式性能，不更新deployment bundle。若两种readout均失败，则停止在当前source-proxy数据上继续优化拒识，等待符合`项目.md`的真实同步事件与物理事件ID数据。

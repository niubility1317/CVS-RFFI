# Phase1 LOTO-C DensityReadout12预注册报告

状态：`ANALYZED / P1-LOTO-C-DENSITYREADOUT12_REJECTED_NO_TRANSFERABLE_THRESHOLD`

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
## 7.N607运行闭环（runner回填，2026-08-08）

| 字段 | 实际证据 |
|---|---|
| status | `ARTIFACTS_COMPLETE`；仅技术完成，`NO_PERFORMANCE_RESULT` |
| release | `/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_loto_c_density_readout12_20260808_v1_29d488e1` |
| run/log/outer | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_c_density_readout12_20260808_v1`；`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_loto_c_density_readout12_20260808_v1`；同目录`phase1_loto_c_density_readout12_20260808_v1.launch.out` |
| archive | commit `29d488e16408c26a2db081b02acaea6bee66830b`；git-archive tar SHA256=`6fa7b866f8fbdef52b0b9dd594026d2f89ba50b1fc87cd344d9aaeff0d8a88dc` |
| code hash | launcher `52becd4f...0992bc`；test `56394c2f...eaaf96`；design `ff042f8d...d7c738`；prototype `de6c87b3...a490aa`；kNN `faa0fa08...2a1f02` |
| input | `postfreeze_manifest.json` SHA256=`e81bf52928d812309a1f9b6b4bfb7aab6f1225a1ff817159345d8f0230a57a30`；仅6个C候选NPZ只读 |
| command/CWD | `cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_loto_c_density_readout12_20260808_v1_29d488e1/code`；冻结`nohup setsid env ... bash scripts/launch_phase1_loto_c_density_readout12_20260808.sh`命令，`PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，`INPUT_ROOT`为上一轮`postfreeze_audit_v1` |
| completion | header+24行；24/24 exit=0；24 JSON、24 CSV、24逐命令stdout；每CSV 2401行 |
| health | error fingerprint=0；外层launch log 0 bytes；结束时无run进程、GPU compute app=0 |

SSH启动调用曾在约34秒处返回客户端timeout=124；随后单次短连接确认同一detached launch已完成，未重试/未重复启动。SSH/SCP每次均主动断开，最终本机TCP22=0。完整小artifact（仅JSON/CSV/日志/completion/manifest）已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_loto_c_density_readout12_20260808_v1\artifacts`，共75项；manifest SHA256=`e8addc8694fb31799e21e8760354e51f83e6662775e1e8a59c0d906b82e5ce8f`；未下载NPZ或checkpoint。retry=`NO`。

## 8.开发性结果

下表只汇总预注册的六个primary held-TX。FAR和safe rejection只表示source-held开发性读数，不是真实unknown性能；K-shot、注册、seen-new、defer和rollback均未发生，统一记为`N/A`。已知closed/full accuracy来自同一行source查询；`old_drop=closed-full`。

| Fold | primary TX | Readout | K-shot | known closed/full | old drop | primary FAR | safe rejection | AUROC | defer/rollback | 同排判决 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| F1 | 14-10 | prototype Q98 | N/A | 99.438/97.438 | 2.000pp | 48.500% | 51.500% | 0.9491 | N/A/N/A | FAR门FAIL |
| F1 | 14-10 | cosine kNN-5 Q98 | N/A | 99.438/97.438 | 2.000pp | 51.250% | 48.750% | 0.9243 | N/A/N/A | FAR门FAIL |
| F2 | 14-7 | prototype Q98 | N/A | 99.500/97.500 | 2.000pp | 90.250% | 9.750% | 0.7283 | N/A/N/A | FAR门FAIL |
| F2 | 14-7 | cosine kNN-5 Q98 | N/A | 99.500/97.500 | 2.000pp | 79.000% | 21.000% | 0.6092 | N/A/N/A | FAR门FAIL |
| F3 | 20-15 | prototype Q98 | N/A | 99.375/97.375 | 2.000pp | 63.750% | 36.250% | 0.8911 | N/A/N/A | FAR门FAIL |
| F3 | 20-15 | cosine kNN-5 Q98 | N/A | 99.375/97.375 | 2.000pp | 53.000% | 47.000% | 0.9063 | N/A/N/A | FAR门FAIL |
| F4 | 20-19 | prototype Q98 | N/A | 99.250/97.250 | 2.000pp | 87.750% | 12.250% | 0.8031 | N/A/N/A | FAR门FAIL |
| F4 | 20-19 | cosine kNN-5 Q98 | N/A | 99.250/97.250 | 2.000pp | 88.000% | 12.000% | 0.7122 | N/A/N/A | FAR门FAIL |
| F5 | 6-15 | prototype Q98 | N/A | 97.438/95.438 | 2.000pp | 72.750% | 27.250% | 0.9014 | N/A/N/A | FAR门FAIL |
| F5 | 6-15 | cosine kNN-5 Q98 | N/A | 97.438/95.438 | 2.000pp | 83.750% | 16.250% | 0.9286 | N/A/N/A | FAR门FAIL |
| F6 | 8-20 | prototype Q98 | N/A | 98.312/96.312 | 2.000pp | 60.750% | 39.250% | 0.7960 | N/A/N/A | FAR门FAIL |
| F6 | 8-20 | cosine kNN-5 Q98 | N/A | 98.312/96.312 | 2.000pp | 39.750% | 60.250% | 0.8949 | N/A/N/A | FAR门FAIL |

### 8.1六折主汇总

| Readout | FAR均值/中位数/范围 | safe rejection均值 | AUROC均值 | old drop均值 | 三门通过 |
|---|---:|---:|---:|---:|---:|
| prototype Q98 | 70.625%/68.250%/48.500%–90.250% | 29.375% | 0.8449 | 2.000pp | 0/6 |
| cosine kNN-5 Q98 | 65.792%/66.000%/39.750%–88.000% | 34.208% | 0.8293 | 2.000pp | 0/6 |

所有行的old drop在十进制意义上恰为`2.000pp`；JSON中的二进制浮点值为`2.0000000000000018`，因此其内置`passes_old_drop_target`显示false。这一数值边界不影响结论，因为12个primary的FAR均远高于5%，safe rejection也均低于95%。

secondary敏感性同样失败：prototype的平均FAR为`70.500%`，kNN-5为`72.208%`，均为0/6通过。它不参与主判决，也没有出现能推翻primary结论的稳定相反方向。

### 8.2与原C-logits读出的配对解释

上一轮同六个C checkpoint、同primary轮转的logits拒识平均FAR为`41.375%`、平均safe rejection为`58.625%`、平均AUROC为`0.5943`。本轮prototype和kNN-5的AUROC分别提高到`0.8449`与`0.8293`，但平均FAR反而分别恶化`29.250pp`与`24.417pp`。这说明`z_id`密度分数在部分fold中保留了source与held-TX的相对排序信息，却发生了明显的跨TX绝对分数平移；仅靠source Q98无法得到可迁移的拒识阈值。AUROC改善不能替代预注册的FAR、安全拒识与已知保护三门。

## 9.最终判决与停止条件

最终判决：`P1-LOTO-C-DENSITYREADOUT12_REJECTED_NO_TRANSFERABLE_THRESHOLD`。prototype与kNN-5均为`0/6`通过，不能更新deployment bundle，也不能形成真实unknown、K-shot注册或Phase3正式性能声明。

本轮完成了预注册的停止条件：不再在当前source-held proxy数据上继续训练拒识头、扫描阈值、增加对齐损失或发布同类source-proxy实验。保留的Phase1经验只有两点：GeoSat-C的clean->LEO一致性仍是当前表征基线；密度读出可提供相对排序诊断，但source-only绝对阈值不具备跨TX泛化性。下一项有效实验必须等待满足`项目.md`的真实同步接收事件、预标签`emission_event_id`与唯一`satellite_reception_id`数据，之后再做正式Phase3多卫星CARE-PoE及fresh-K闭环。

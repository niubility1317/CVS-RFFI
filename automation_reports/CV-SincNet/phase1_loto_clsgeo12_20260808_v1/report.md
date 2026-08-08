# Phase1 LOTO-CLSGeo12十二任务预注册报告

状态：`ANALYZED / P1-LOTO-CLSGeo12_REJECTED_NO_STABLE_GAIN`

目标模式：`GOAL_MODE=ACTIVE`

证据标签：`DEVELOPMENT_CROSS_TX_CV_NON_CONFIRMATORY`

版本承载：`E:\type10-7`根目录不是Git仓库；本报告同步镜像到Git工作树同相对路径。

## 1.实验身份

| 字段 | 冻结值 |
|---|---|
| run ID | `phase1_loto_clsgeo12_20260808_v1` |
| 日期 | 2026-08-08 |
| 主Agent | `/root` |
| 唯一N607 runner | `/root/n607_geosat_lite_runner`（Luna/max） |
| 实现commit | `62e763830603cf1b78d5038ccace24d6816e64db` |
| 独立审查 | `VERDICT=APPROVE; P0=0; P1=0` |
| 目标 | 用6-fold leave-two-TX-out开发交叉验证检验顶层分类特征known-only角度几何是否提高跨TX泛化，同时保持已知类保护指标 |

## 2.假设、继承与唯一干预

本轮继承Phase1 GeoSat探索中已验证的clean→三种`leo_*_weak`一致性经验。每个fold有两臂：

| arm | 机制 | `lambda_sat_cons` | `lambda_open_world_feat` | `ow_feat_key` |
|---|---|---:|---:|---|
| C | GeoSat-C对照 | 0.10 | 0 | 默认`z_id`，不进入新选择器 |
| G | C加顶层分类几何 | 0.10 | 0.0024 | `id_feat_cls` |

G相对C的唯一方法差异是`0.0024 L_OW(id_feat_cls,y,d)`。半径、类间、样本边界固定为12°、55°、5°。proxy、soft-unknown mixup、source episode、OE、open-world domain alignment、tail和vacuum均关闭；不新增对齐操作，不扫描阈值。

## 3.六fold数据锁

TX固定顺序为`[14-10,14-7,20-15,20-19,6-15,8-20]`。fold`Fi`中primary held为`TX_i`，secondary held为`TX_(i+1 mod 6)`，其余4个TX是唯一训练TX。

| Fold | 训练TX | secondary held-known | primary held-proxy |
|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 |

两个held-TX不得进入训练、损失、内部模型选择或Q95校准。`phase1_source_proxy_unknown_tx_ids`只是分割字段，不启用proxy训练。主开发汇总只使用6个primary held-TX同foldC/G配对；secondary held-TX仅作敏感性读数。

## 4.训练矩阵与资源

所有任务从头训练120epoch，`seed=7281105`、`sat_view_seed=9281105`、`checkpoint_selection=final_only`。启动器允许每张GPU最多两个进程，符合用户给定资源边界。

| 物理GPU | 并行任务 | 进程数 |
|---:|---|---:|
| 0 | F1C、F5G | 2 |
| 1 | F1G、F5C | 2 |
| 2 | F2C、F6G | 2 |
| 3 | F2G、F6C | 2 |
| 4 | F3C | 1 |
| 5 | F3G | 1 |
| 6 | F4C | 1 |
| 7 | F4G | 1 |

## 5.本地变更、版本与验证

| 文件 | 目的 | Git blob |
|---|---|---|
| `code/SSDG/train_ssdg.py` | 新增严格`ow_feat_key`选择器，仅改变可选几何损失输入 | `4e40cc88a4ee6d4d32651b155168d5fc64972f4b` |
| `code/tests/test_phase1_ow_feat_key.py` | parser、恒等默认、fail-closed、梯度和真实模型shape smoke | `6d57627e00e5b186887d9b45f1b904c12aebb891` |
| `code/scripts/launch_phase1_loto_clsgeo12_20260808.sh` | 12任务非覆盖启动、PID/completion和GPU映射 | `6cb9cfa23393515e2762e7d0e17f7f4c1a80ef39` |
| `analysis/phase1_loto_clsgeo12_design_20260808.md` | 冻结设计、选择门与证据边界 | `19a8f1111edcfcb4c463faf96b5c21f1d5bf8638` |

验证结果：

- `ssr-gpu: python -m pytest -q code/tests/test_phase1_ow_feat_key.py code/tests/test_phase1_tx_partition.py`：15 passed。
- `bash -n code/scripts/launch_phase1_loto_clsgeo12_20260808.sh`：PASS。
- `DRY_RUN=1`：12条命令；C/G各6条；只有6条G命令含`--ow_feat_key id_feat_cls`；GPU计数为2/2/2/2/1/1/1/1。
- `git diff --check`：PASS。
- 独立审查：默认C计算路径不进入新选择器；G路径严格检查且梯度可达；6fold角色、held排除、资源映射和禁用项均通过。

Git归档成员SHA256：

| 文件 | archive SHA256 |
|---|---|
| `train_ssdg.py` | `bf6b658e8b055f896cb53a76531b6dce0243504407abfd972a892c60a5b06526` |
| `test_phase1_ow_feat_key.py` | `99618897052df1278f32891b62b297af63fce8ee924f24df626a7561ed550acd` |
| `launch_phase1_loto_clsgeo12_20260808.sh` | `e6bea03ba2810ce1db9c7573205939ceb27d20cc1c33096c5dd08550afa5c5ca` |
| `phase1_loto_clsgeo12_design_20260808.md` | `a4209cc014fe669bd1c9e677f420a4dfdadea80353f6a8fe3d3903887dbbc4a0` |

Windows工作树与Git归档中Python/Markdown的SHA差异来自既有换行属性；N607以固定commit的Git归档字节为准。

## 6.N607冻结交接

| 字段 | 冻结值 |
|---|---|
| Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 项目根 | `/home/szu2070436088/2510044040/CV-SincNet` |
| release根 | `/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_loto_clsgeo12_20260808_v1_62e76383` |
| run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1` |
| log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_loto_clsgeo12_20260808_v1` |
| CWD | `<release>/code` |
| launcher | `<release>/code/scripts/launch_phase1_loto_clsgeo12_20260808.sh` |
| 数据 | `<project>/Dataset_WigSig/ManySig.pkl`，已知SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f` |
| retry | `NO` |

冻结启动形式：

```text
cd <release>/code && nohup setsid env RUN_ID=phase1_loto_clsgeo12_20260808_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash <release>/code/scripts/launch_phase1_loto_clsgeo12_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_loto_clsgeo12_20260808_v1.launch.out 2>&1 < /dev/null & echo $!
```

预期产物：每任务`final_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`、terminal manifest和completion receipt；根级`pids.tsv`、`completion.tsv`及外层launcher日志。`final_only`意味着不得要求或伪造`latest_ssdg.pth`。

## 7.健康、停止与成功门

启动后只核对launcher/child PID、CWD/cmdline/run-root、GPU映射、日志增长、分割receipt和首波错误指纹。不得读取性能做停机决定。

仅在下列条件停机并保留partial artifacts：

1. checkout/hash或非覆盖路径错误；
2. source split receipt显示任一held-TX进入训练；
3. query/held真值或角色参与损失、模型选择或校准；
4. CUDA OOM、launcher-wide故障，或至少两个不同任务在产生训练telemetry前出现同一确定性异常指纹。

训练脚本完成120epoch后显式返回`NON_PROMOTABLE_P0_DISABLED exit_code=8`属于预期终态，不是系统异常。任何技术停止均标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重启或覆盖。

成功门是12任务均完成120epoch、错误指纹为零、分割receipt符合冻结fold、同排产物完整。完成训练不等于方法晋级。

## 8.后续分析锁

完成后先回收小型日志、metrics、receipt和checkpoint哈希，不下载checkpoint。随后使用冻结final checkpoint对每fold primary/secondary held切片做一次非训练评估；主结论只基于6个primary C/G配对。

每fold已知保护至少包括clean、三种`leo_*_weak`、min-class、min-receiver和min-day。若任一预指定保护项`G-C<-2pp`，该fold G直接拒绝，其他指标不得补偿。缺少场景、receiver/day或类floor字段的fold不进入主汇总。


### 8.1冻结postfreeze audit命令

只有12个训练任务全部达到E120并形成完整terminal/metrics后，runner才执行一次`postfreeze_audit_v1`；训练不完整时不得执行。审计复用同一release中的`export_spaceborne_features.py`与`scripts/eval_phase1_logits_open_set_reject.py`，不修改代码。

每个candidate的冻结导出参数：

```text
python export_spaceborne_features.py
  --ckpt <run>/<candidate>/final_ssdg.pth
  --wisig_pkl <project>/Dataset_WigSig/ManySig.pkl
  --out_npz <run>/postfreeze_audit_v1/<candidate>/features.npz
  --feature_name z_id
  --source_tx_ids <fold_train_tx>
  --target_old_tx_ids <fold_secondary_tx>
  --proxy_unknown_tx_ids <fold_primary_tx>
  --source_days 0,1 --source_rxs 0,1,2,3,4,5,6
  --target_old_days 0,1 --target_old_rxs 0,1,2,3,4,5,6
  --proxy_unknown_days 0,1 --proxy_unknown_rxs 0,1,2,3,4,5,6
  --max_samples_per_tx 400 --batch_size 512 --device cuda:0 --seed 7281105
  --source_channel_view clean --target_old_channel_view clean --proxy_unknown_channel_view clean
```

12个导出沿用训练的物理GPU映射；wrapper用`CUDA_VISIBLE_DEVICES=<physical_gpu>`，CLI统一`--device cuda:0`。每张卡仍不超过两个进程。每个NPZ预期`source=1600,target_old=400,proxy_unknown=400`，三类TX必须互斥，checkpoint strict load必须`missing=unexpected=skipped=0`。

每个candidate随后执行两条CPU评分命令，阈值固定为source正确分类样本的confidence Q0.05、margin Q0.05和energy Q0.95：

```text
python scripts/eval_phase1_logits_open_set_reject.py
  --feature_npz <features.npz> --source_tx_ids <fold_train_tx>
  --unknown_tx_ids <fold_primary_tx> --known_query_roles source
  --unknown_query_roles proxy_unknown --calibration_roles source
  --conf_quantile 0.05 --margin_quantile 0.05 --energy_quantile 0.95
  --unknown_far_target 0.05
  --output_json <candidate>/primary_metrics.json
  --score_table_csv <candidate>/primary_scores.csv
```

```text
python scripts/eval_phase1_logits_open_set_reject.py
  --feature_npz <features.npz> --source_tx_ids <fold_train_tx>
  --unknown_tx_ids <fold_secondary_tx> --known_query_roles source
  --unknown_query_roles target_old --calibration_roles source
  --conf_quantile 0.05 --margin_quantile 0.05 --energy_quantile 0.95
  --unknown_far_target 0.05
  --output_json <candidate>/secondary_metrics.json
  --score_table_csv <candidate>/secondary_scores.csv
```

audit根固定为`<run>/postfreeze_audit_v1`，日志根固定为`<log>/postfreeze_audit_v1`。每条导出和评分只执行一次，`retry=NO`。runner只核对退出码、行数、角色/TX互斥、strict load、hash和错误指纹，不解释性能；仅回收JSON、CSV、日志、completion和manifest，不下载NPZ/checkpoint。

本轮结论上限为“source-held cross-TX开发性泛化证据”。它不是独立确认，不是K-shot注册，不是正式unknown-FAR，也不更新Phase3能力声明。

## 9.runner终态与postfreeze evidence handoff（2026-08-08）

- 状态：`ARTIFACTS_COMPLETE`（训练与postfreeze技术闭环完成；`NO_PERFORMANCE_RESULT`）。训练launcher PID `3644376`及12个child均已退出；GPU compute apps为空；短连接结束后本地`ssh.exe`与TCP22均为0。
- 固定release/commit：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_loto_clsgeo12_20260808_v1_62e76383`，commit `62e763830603cf1b78d5038ccace24d6816e64db`。训练run/log分别为`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`与`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_loto_clsgeo12_20260808_v1`。
- 训练闭环：`completion.tsv`为header+12；12/12任务E120、terminal=`NON_PROMOTABLE_P0_DISABLED`、exit_code=8（冻结预期）；12×metrics CSV/JSONL和terminal/completion/resource/heldout/tail receipts齐全；checkpoint仅记录SHA，未下载。训练错误指纹计数为0。
- `pids.tsv`绑定GPU0:F1C/F5G、GPU1:F1G/F5C、GPU2:F2C/F6G、GPU3:F2G/F6C、GPU4:F3C、GPU5:F3G、GPU6:F4C、GPU7:F4G；每行四个train TX并排除known validation与held proxy。部分terminal `source_split_receipt`只写day/rx字段，TX绑定以同run`pids.tsv`为准，未修改产物。
- postfreeze根：`<run>/postfreeze_audit_v1`与`<log>/postfreeze_audit_v1`均由空路径创建。12个GPU导出各执行一次并产出2400行NPZ（source=1600、target_old=400、proxy_unknown=400；row-key唯一2400；strict-load日志`missing_keys=unexpected_keys=skipped_mismatch=0`）；导出后台PID退出后未保留shell退出码记录，NPZ存在、strict-load完整且错误指纹为0。
- CPU评分：`score_completion.tsv`记录24个唯一candidate×kind，全部`exit_code=0`；24个JSON与24个CSV存在，每个CSV为header+2400行；primary角色为`proxy_unknown`、secondary为`target_old`，阈值配置按冻结Q0.05/Q0.05/Q0.95与`unknown_far_target=0.05`写入JSON。评分JSON/CSV只作为原样审计artifact，不在此报告解释指标。
- 审计manifest：远端`postfreeze_manifest.json` SHA256=`e81bf52928d812309a1f9b6b4bfb7aab6f1225a1ff817159345d8f0230a57a30`；ManySig SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。本地回收根：`E:\type10-7\automation_reports\CV-SincNet\phase1_loto_clsgeo12_20260808_v1\artifacts`；训练小包SHA=`ee124c192081c41b60dec752fa0a316b5887bcf5fc87f413cbdd9e18c7753d64`，postfreeze小包SHA=`6ee57cb7a5a37dee9472386c6dfe7900f49b909cd5efb2ebb590521e820601b3`。本地非checkpoint文件逐项SHA与manifest匹配；本地无`.pth`/`.npz`。
- 评分driver首次包装失败发生在任何评分命令开始前（远端无driver/score进程和输出），修正后24条命令各仅执行一次；`retry=NO`。本节仅记录运行/结构/哈希/清理证据，保留source-held开发性证据边界，不作性能或晋级结论。

## 10.主Agent同排分析与晋级判决

状态：`ANALYZED / P1-LOTO-CLSGeo12_REJECTED_NO_STABLE_GAIN`

### 10.1证据边界

下表的“primary FAR”只表示每fold预注册primary held-TX在source-only阈值下的开发性false accept rate；它不是正式unknown-FAR。K-shot、seen-new、注册、defer和rollback均未发生，按`N/A`报告。primary是唯一主汇总；secondary只作敏感性检查，不参与调参或补偿主判决。

### 10.2十二任务完整同排结果

已知clean列依次为`closed accuracy/min-class/min-receiver/min-day`；LEO列依次为`aggregate mean/aggregate floor/strict mean/strict floor`。FAR与safe rejection单位为百分比。

| Candidate | 训练TX->primary held | 机制 | K-shot | seed | 已知clean四项 | LEO四项 | primary FAR/safe reject/AUROC | secondary FAR | known coverage | defer/rollback | 同排判决 |
|---|---|---|---|---:|---|---|---|---:|---:|---|---|
| F1C_LOTO_CLSGeo12 | 20-15,20-19,6-15,8-20->14-10 | C:LEO-KL | N/A | 7281105 | 99.44/99.25/98.16/99.36 | 82.58/81.51/78.72/77.42 | 7.75/92.25/0.5921 | 54.50 | 90.00 | N/A/N/A | fold对照；最接近但仍未达5% |
| F1G_LOTO_CLSGeo12 | 20-15,20-19,6-15,8-20->14-10 | G:LEO-KL+CLSGeo | N/A | 7281105 | 99.50/99.25/98.62/99.49 | 81.53/80.41/76.55/75.20 | 69.50/30.50/0.5776 | 92.75 | 90.44 | N/A/N/A | 已知保护FAIL；primary严重恶化 |
| F2C_LOTO_CLSGeo12 | 14-10,20-19,6-15,8-20->14-7 | C:LEO-KL | N/A | 7281105 | 99.50/99.25/97.70/99.38 | 73.80/72.74/68.01/67.04 | 58.00/42.00/0.5315 | 20.75 | 91.00 | N/A/N/A | fold对照 |
| F2G_LOTO_CLSGeo12 | 14-10,20-19,6-15,8-20->14-7 | G:LEO-KL+CLSGeo | N/A | 7281105 | 99.50/99.25/97.70/99.38 | 73.05/71.99/67.26/66.39 | 36.50/63.50/0.5354 | 16.75 | 91.56 | N/A/N/A | 已知门PASS，但FAR门FAIL |
| F3C_LOTO_CLSGeo12 | 14-10,14-7,6-15,8-20->20-15 | C:LEO-KL | N/A | 7281105 | 99.38/99.00/96.31/99.36 | 75.19/74.13/70.34/69.27 | 44.25/55.75/0.6377 | 81.50 | 90.19 | N/A/N/A | fold对照 |
| F3G_LOTO_CLSGeo12 | 14-10,14-7,6-15,8-20->20-15 | G:LEO-KL+CLSGeo | N/A | 7281105 | 99.38/99.25/96.77/99.26 | 74.63/73.60/69.25/68.28 | 42.75/57.25/0.6683 | 81.25 | 92.38 | N/A/N/A | 已知门PASS，primary增益仅1.50pp且FAR门FAIL |
| F4C_LOTO_CLSGeo12 | 14-10,14-7,20-15,8-20->20-19 | C:LEO-KL | N/A | 7281105 | 99.25/98.75/97.24/98.98 | 74.62/73.98/67.75/67.14 | 65.25/34.75/0.5363 | 50.00 | 89.88 | N/A/N/A | fold对照 |
| F4G_LOTO_CLSGeo12 | 14-10,14-7,20-15,8-20->20-19 | G:LEO-KL+CLSGeo | N/A | 7281105 | 99.12/98.50/97.24/98.86 | 73.32/72.72/66.34/65.62 | 66.75/33.25/0.5281 | 19.75 | 90.62 | N/A/N/A | 已知保护FAIL；primary恶化 |
| F5C_LOTO_CLSGeo12 | 14-10,14-7,20-15,20-19->6-15 | C:LEO-KL | N/A | 7281105 | 97.44/94.25/88.02/97.08 | 55.43/54.42/47.25/46.75 | 38.00/62.00/0.6485 | 39.50 | 89.94 | N/A/N/A | fold对照 |
| F5G_LOTO_CLSGeo12 | 14-10,14-7,20-15,20-19->6-15 | G:LEO-KL+CLSGeo | N/A | 7281105 | 97.88/96.25/92.17/97.20 | 54.86/54.07/46.71/46.40 | 61.25/38.75/0.6148 | 30.00 | 89.69 | N/A/N/A | 已知保护FAIL；primary明显恶化 |
| F6C_LOTO_CLSGeo12 | 14-7,20-15,20-19,6-15->8-20 | C:LEO-KL | N/A | 7281105 | 98.31/96.50/93.55/98.22 | 65.28/63.76/59.27/58.05 | 35.00/65.00/0.6195 | 56.00 | 90.38 | N/A/N/A | fold对照 |
| F6G_LOTO_CLSGeo12 | 14-7,20-15,20-19,6-15->8-20 | G:LEO-KL+CLSGeo | N/A | 7281105 | 98.06/96.00/91.71/97.59 | 66.76/65.42/60.68/59.34 | 34.00/66.00/0.6175 | 8.25 | 90.25 | N/A/N/A | 已知门PASS，primary增益仅1.00pp且FAR门FAIL |

### 10.3逐fold配对门

| Fold | primary TX | C FAR | G FAR | G-C FAR | G-C最差已知保护项 | 已知保护门 | fold判决 |
|---|---|---:|---:|---:|---:|---|---|
| F1 | 14-10 | 7.75 | 69.50 | +61.75pp | -4.04pp | FAIL | 拒绝G |
| F2 | 14-7 | 58.00 | 36.50 | -21.50pp | -1.44pp | PASS | FAR仍远高于5%，不晋级 |
| F3 | 20-15 | 44.25 | 42.75 | -1.50pp | -1.34pp | PASS | 改善过小且FAR门失败 |
| F4 | 20-19 | 65.25 | 66.75 | +1.50pp | -2.51pp | FAIL | 拒绝G |
| F5 | 6-15 | 38.00 | 61.25 | +23.25pp | -2.08pp | FAIL | 拒绝G |
| F6 | 8-20 | 35.00 | 34.00 | -1.00pp | -1.84pp | PASS | 改善过小且FAR门失败 |

G在primary上改善3/6、恶化3/6；其平均FAR由C的`41.375%`恶化至`51.792%`，即`+10.417pp`，中位数由`41.125%`恶化至`52.000%`。平均safe rejection由`58.625%`降至`48.208%`；平均AUROC由`0.5943`降至`0.5903`。没有任何C或G通过`FAR<=5%`；最接近的是F1C的`7.75%`，其safe rejection为`92.25%`，仍低于95%。

G的平均已知clean closed accuracy相对C仅`+0.021pp`，但平均strict UDU为`-0.918pp`，且F1、F4、F5三fold各有至少一个预注册保护项下降超过2pp，触发非补偿拒绝。secondary敏感性均值从C的`50.375%`改善至G的`41.458%`，但它不是主汇总，并且与primary的方向相反，说明收益依赖具体held-TX，不能证明稳定跨TX泛化。

### 10.4方法结论与下一实验

最终判决：`P1-LOTO-CLSGeo12_REJECTED_NO_STABLE_GAIN`。顶层分类特征的known-only角度几何并未继承为稳定的open-world方法经验；它保持了平均clean性能，却在不同held-TX之间产生大幅方向反转，并让primary主汇总整体变差。不得用F2的单fold改善、F6的secondary低FAR或任何边际最优值晋级G，也不得用本轮结果更新Phase3正式unknown声明或deployment bundle。

保留结论：GeoSat-C的clean->LEO一致性仍是当前较稳健的Phase1表征基线；后续不再训练相同的角度对齐/几何变体。下一轮应优先做零训练变更的source-only后处理：在六个C checkpoint和相同primary轮转上，预注册一个固定cosine class-prototype Q98与一个固定cosine kNN-5 Q98读出，只检验冻结表征是否存在可用密度信号；Q98由已知下降不超过2pp直接确定，不扫描阈值。若仍无候选同时满足primary`FAR<=5%`、safe rejection`>=95%`和已知下降`<=2pp`，则停止source-proxy拒识优化，等待满足`项目.md`的真实同步事件与物理事件ID数据。

# Phase1 ManyTx RealOE12 Physical-RX v2实验报告

目标模式：`GOAL_MODE=ACTIVE`

## 1.状态与责任

- 实验ID：`phase1_manytx_realoe12_physrx_v2_20260808_v1`
- 当前状态：`LOCAL_VERIFIED / PREREGISTERED`
- 日期：2026-08-08
- 主控：`/root`
- 唯一N607运行器：`/root/n607_geosat_lite_runner`
- 方法标签：`DEVELOPMENT_SOURCE_ONLY_NON_CONFIRMATORY`
- 实现提交：`fc322b598232b6329b9c6965023bcb7052baf1d6`
- 发布提交：以包含本报告的Git HEAD为准，由运行器交接记录精确值；不得从其他工作树拼接代码。

本报告在Git工作树中版本化；根目录`E:\type10-7`不是Git仓库，因此同步副本保存在`E:\type10-7\automation_reports\CV-SincNet\phase1_manytx_realoe12_physrx_v2_20260808_v1\report.md`。

## 2.目的、假设与比较

目的：尽早验证真实source outlier exposure是否能在不增加receiver alignment、不使用虚拟外点、不使用batch轮换known proxy的前提下，提高已知类与外部TX之间的energy可分性，同时保持已知类跨物理receiver与LEO弱信道泛化。

唯一成对比较为GeoSat-C对RealOE-G。每个fold的C/G共享known TX划分、seed、训练配置和GPU资源口径；G相对C只增加ManyTx OE输入与`lambda_manytx_real_oe=0.02`。本实验不训练unknown类别、不产生Phase3确认结论，也不允许用proxy、reserve、locked target_new、target query、truth或role选择epoch、阈值、候选或方法。

## 3.冻结数据与分区

- `ManySig.pkl`：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- `ManyTx.pkl`：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl`；SHA256=`c0319174d40eb64bc49f201743941ebedc5cc0ced284c655cab798b2bdd44275`。
- source physical RX：`1-1,1-19,14-7,18-2,19-2,2-1`。
- source days：`2021_03_01,2021_03_08`；`equalized=1`。
- held target RX：`20-1,3-19,7-14,7-7,8-8`；与source RX无交。
- ManySig known TX：`14-10,14-7,20-15,20-19,6-15,8-20`。
- 分区计数：OE80、proxy20、reserve16、authority-locked target_new20。
- 分区root：`ca3ed65a533359d2abb022fa513c49101ad93235738a39b362b5cdd15879c3d1`。
- OE/proxy/reserve名单SHA256：`919f83d7c8dac57f4dfa6da9b49e4fe868aca009b71469d8d26b032879a81dde`、`29fbcf75c0275189b1357c9c3672bd998be562ad511e3e6f11b8f5e380fac700`、`e5e3fddbd35e89d6e86c59980180331cf94a63ed8ab7fe9271d668a5a05d97e9`。
- 每个OE TX必须在冻结slice上具有至少400条记录、覆盖两个source day和至少两个共同physical RX。raw RX/day index输入fail-closed。

`ManyTx.pkl`是单体容器；训练数据集只索引OE80对应记录。proxy20、reserve16与locked target_new20不进入训练dataset、batch、loss、checkpoint选择或阈值选择。

## 4.机制与关键配置

`E(l)=-T logsumexp(l/T)`，`T=1`。G臂每个known batch均衡抽取16个OE TX、每TX 8样本，共128个真实OE样本；OE标签在进入模型前强制为`-1`，模型调用`y_tx=None`。损失为：

`mean softplus((margin-(E_oe-stopgrad(mean(E_known))))/tau)`，其中`margin=1`、`tau=1`、`lambda=0.02`，epoch61开始，10epoch线性warmup。已知能量锚点停止梯度，因此该辅助项只经OE forward反传。

共同训练：`lite_d`、from-scratch、120epoch、seed`7281105`、sat seed`9281105`、clean与`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`、`final_only`。VOS、virtual proxy、source-Q98、receiver alignment和旧proxy-loss均关闭。

## 5.冻结12任务矩阵与GPU

|Fold|known-validation|known训练TX|C物理GPU|G物理GPU|
|---|---|---|---:|---:|
|F1|14-10|14-7,20-15,20-19,6-15,8-20|0|1|
|F2|14-7|14-10,20-15,20-19,6-15,8-20|2|3|
|F3|20-15|14-10,14-7,20-19,6-15,8-20|4|5|
|F4|20-19|14-10,14-7,20-15,6-15,8-20|6|7|
|F5|6-15|14-10,14-7,20-15,20-19,8-20|1|0|
|F6|8-20|14-10,14-7,20-15,20-19,6-15|3|2|

每张GPU最多两个训练进程；每个子进程设置`CUDA_VISIBLE_DEVICES=<physical>`并使用`--device cuda:0`。

## 6.本地改动与验证

|文件|用途|
|---|---|
|`code/SSDG/train_ssdg.py`|物理标签解析、冻结分区、RealOE loader/sampler、训练接入与回执|
|`code/cvsrffi/losses.py`|停止梯度known anchor的energy ranking loss|
|`code/tests/test_phase1_manytx_realoe.py`|协议负测、梯度与真实`lite_d`smoke|
|`code/scripts/launch_phase1_manytx_realoe12_20260808.sh`|12任务唯一launcher|
|`analysis/phase1_manytx_realoe_design_20260808.md`|冻结科学设计|

验证结果：focused pytest为30 passed；`py_compile`通过；launcher`bash -n`通过；DRY_RUN生成12条且physical source/target标签与partition root均为12/12，raw RX index为0条；`git diff --check`通过。独立审查结论：`P0=0`、`P1=0`、`ALLOW_RELEASE=yes`，仅允许本报告的source-only非确认性12任务。

## 7.N607发布与精确启动模板

- 发布根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_manytx_realoe12_physrx_v2_20260808_v1_<COMMIT8>`。
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v1`。
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_20260808_v1`。
- outer log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_20260808_v1.launch.out`。
- CWD：`<release>/code`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

运行器以包含本报告的精确Git提交替换`<COMMIT8>`并只启动一次：

```bash
cd <release>/code && nohup setsid env RUN_ID=phase1_manytx_realoe12_physrx_v2_20260808_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash <release>/code/scripts/launch_phase1_manytx_realoe12_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_20260808_v1.launch.out 2>&1 < /dev/null & echo $!
```

预期根级artifact为`pids.tsv`、`completion.tsv`和12份stdout；每臂预期`final_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`及terminal/split/resource回执。不得覆盖同名release/run/log，不得下载checkpoint或大型NPZ。

## 8.健康控制、完成和后评估

启动后核对PID、CWD、cmdline、run-root、GPU映射、12日志增长、CONFIG与分区回执。只在协议/路径/哈希错误、输出覆盖、确定性异常、OOM、无进展或至少两个不同任务出现同一预测前异常指纹时停止本run的精确进程树；绝不因accuracy、loss、AUROC或FAR走势停止或调参。失败不自动重试。

训练完成门：12任务均到E120，terminal与completion闭合，checkpoint和metrics齐全。随后每candidate只执行一次冻结postfreeze导出和两类只读评分：

1. 从对应`final_ssdg.pth`导出同一10400行证据：5个source known TX共2000、1个fold-held TX共400、20个proxy TX共8000；只使用冻结source days/RX、`equalized=1`、clean view和每TX 400 cap。导出固定增加`--new_wisig_pkl <project>/Dataset_WigSig/ManyTx.pkl`，source与fold-held从ManySig读取，proxy从ManyTx读取；三类TX必须互斥，checkpoint strict load必须`missing=unexpected=skipped=0`。
2. known保护：同fold同状态比较G-C的clean、三种LEO、min-class与min-receiver floor；任一指标下降超过2pp时该G fold拒绝，不可由其他fold抵消。
3. open-world开发诊断：energy-only gate，关闭confidence与margin gate；阈值仅由source known校准，proxy20只读评分。必须从逐样本`energy`重新计算energy AUROC与FAR，不把旧脚本基于confidence的`roc_auc`误写为energy AUROC。
4. proxy、held-known、locked target_new、reserve均不得参与训练、epoch/checkpoint/候选/阈值选择；结果不能外推为K-shot、target unknown或Phase3正式性能。

每candidate的冻结导出命令为：

```text
python export_spaceborne_features.py
  --ckpt <run>/<candidate>/final_ssdg.pth
  --wisig_pkl <project>/Dataset_WigSig/ManySig.pkl
  --new_wisig_pkl <project>/Dataset_WigSig/ManyTx.pkl
  --out_npz <run>/postfreeze_audit_v1/<candidate>/features.npz
  --feature_name z_id
  --source_tx_ids <fold_train_tx>
  --target_old_tx_ids <fold_held_tx>
  --proxy_unknown_tx_ids <frozen_proxy20>
  --source_days 2021_03_01,2021_03_08
  --source_rxs 1-1,1-19,14-7,18-2,19-2,2-1
  --target_old_days 2021_03_01,2021_03_08
  --target_old_rxs 1-1,1-19,14-7,18-2,19-2,2-1
  --proxy_unknown_days 2021_03_01,2021_03_08
  --proxy_unknown_rxs 1-1,1-19,14-7,18-2,19-2,2-1
  --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256
  --max_samples_per_combo 0 --max_samples_per_tx 400
  --batch_size 512 --device cuda:0 --seed 7281105
  --source_channel_view clean --target_old_channel_view clean
  --proxy_unknown_channel_view clean
```

12个导出沿用训练的物理GPU映射，wrapper设置`CUDA_VISIBLE_DEVICES=<physical>`，CLI统一`--device cuda:0`。每candidate随后只运行两条CPU评分：

```text
python scripts/eval_phase1_logits_open_set_reject.py
  --feature_npz <features.npz> --source_tx_ids <fold_train_tx>
  --unknown_tx_ids <frozen_proxy20>
  --known_query_roles source --unknown_query_roles proxy_unknown
  --calibration_roles source --conf_quantile 0.05
  --margin_quantile 0.05 --energy_quantile 0.95
  --disable_conf_gate --disable_margin_gate --unknown_far_target 0.05
  --output_json <candidate>/proxy_metrics.json
  --score_table_csv <candidate>/proxy_scores.csv
```

```text
python scripts/eval_phase1_logits_open_set_reject.py
  --feature_npz <features.npz> --source_tx_ids <fold_train_tx>
  --unknown_tx_ids <fold_held_tx>
  --known_query_roles source --unknown_query_roles target_old
  --calibration_roles source --conf_quantile 0.05
  --margin_quantile 0.05 --energy_quantile 0.95
  --disable_conf_gate --disable_margin_gate --unknown_far_target 0.05
  --output_json <candidate>/held_metrics.json
  --score_table_csv <candidate>/held_scores.csv
```

audit根固定为`<run>/postfreeze_audit_v1`，日志根固定为`<log>/postfreeze_audit_v1`。每条导出和评分各执行一次，`retry=NO`。运行器只核退出码、10400行结构、角色/TX互斥、strict load、hash和异常指纹；不读取或解释性能，只回收JSON、CSV、日志、completion与manifest，不下载NPZ/checkpoint。

成功只表示技术artifact完整；方法晋级要求6/6 fold通过known保护门，并在完整冻结proxy开发矩阵上报告同一candidate/fold的energy AUROC、energy-only FAR与已知指标。正式目标`unknown FAR<=5%`不可被known精度补偿，但本source proxy结果仍只作为Phase1筛选证据。

## 9.待回填

运行器完成后回填发布commit/archive SHA、代码文件hash、精确launcher PID/children/GPU、completion、artifact hash、异常、资源清理和本地回收路径。主控仅在完整同fold C/G paired artifacts返回后分析性能与决定是否进入Phase1 bundle。

## 10.N607技术终态（2026-08-08）

- `STATUS=STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；仅执行一次冻结launcher，无retry。
- training release=`phase1_manytx_realoe12_physrx_v2_20260808_v1_748a9f41`，发布报告commit=`748a9f41f04a80e8d2eac1a366f1ad250d917e28`，训练实现父commit=`2d44ace3c502c189b1a43965e9d06e6400fffa8e`；archive tar SHA256=`3f1ff8eb36fdded25f9641dd746d0f2374c2cc251920b8548d73cec26ea659a1`。
- exact launcher PID=`3776491`；`pids.tsv`记录12个固定GPU映射子任务，`completion.tsv`为12/12 `exit_code=1`；GPU收口为8卡`0%/1MiB`，无run-owned live process。
- 首波同一确定性异常（12/12，训练telemetry前）：`ValueError: Cannot resolve day 20210301 from ['2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23']`，源自`_resolve_days`解析launcher传入的day标签；因此不执行postfreeze export/score。
- 小型证据已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_manytx_realoe12_physrx_v2_20260808_v1\artifacts`（remote_logs、outer.launch.out）；SSH/SCP客户端与TCP22均清理。

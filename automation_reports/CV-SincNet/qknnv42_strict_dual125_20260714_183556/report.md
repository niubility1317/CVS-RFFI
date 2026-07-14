# qKNNV42严格ADV3B02双125重跑报告

## 实验信息

|字段|内容|
|---|---|
|实验ID|`qknnv42_strict_dual125_20260714_183556`|
|时间|2026-07-14 18:35:56（Asia/Hong_Kong）|
|操作者|Codex|
|目标|修复ADV3B02特征导出的非严格checkpoint重建，重新运行轻量单视图FFT96与完整legacy oracle两组各125任务|
|比较对象|旧兼容加载诊断：轻量组`old=75.12%,new=64.64%,H=68.56%`；完整组`old=82.93%,new=93.37%,H=87.65%`|
|声明边界|轻量组为逐样本Stage2-C诊断；完整组含角色/类别配额Oracle，仅为`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`|

## 根因与修复

旧导出器直接调用通用`build_dual_model`，没有复用ADV3B02训练时`SSDG.train_ssdg.merge_checkpoint_args`、`_apply_model_cli_args`和`build_baseline_model`完整架构路径，导致7个missing key、31个unexpected key和3个shape mismatch仍被`strict=False`兼容加载后继续运行。

本次新增fail-closed严格加载器`code/cvsrffi/checkpoint_loading.py`：

1. 从checkpoint state恢复domain head维度；
2. 从checkpoint args恢复全部训练时模型选项；
3. 调用训练原生`build_baseline_model`；
4. 完整加载state；任何shape mismatch、missing或unexpected key立即失败；
5. 导出manifest固定记录`checkpoint_load_strict=true`与键计数0/0/0。

`code/export_spaceborne_features.py`和`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`均切换到同一严格加载器，避免轻量分支与60epoch adapter分支再次分叉。

## 协议与矩阵

保持`项目.md`不变：5个target receiver×5个seed×K={1,2,5,10,20}=125行；旧类6个、新类2个；support/query同属目标接收机简化LEO视图且不重叠；unknown不参与Phase2；三场景为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

|分支|输入与方法|决策|部署边界|
|---|---|---|---|
|strict singleview FFT96|严格ADV3B02、单个LEO视图、z_id160+FFT96、无adapter|逐样本argmax|轻量Stage2-C诊断|
|strict full legacy oracle|严格ADV3B02、60epoch id_norm_late_feature adapter、5-view TTA、FFT96|角色Oracle+类别配额Hungarian|仅上限诊断|

## 本地变更与验证

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`。根目录`E:\type10-7`不是Git仓库，本报告同步镜像到Git承载面。启动前工作树存在其他任务对SSDG/loss/runner及报告的未提交修改，本次不覆盖、不暂存、不提交这些不相关变更。

本次文件：

- `code/cvsrffi/checkpoint_loading.py`：严格重建与fail-closed加载；
- `code/export_spaceborne_features.py`：单视图导出使用严格加载并写审计；
- `code/scripts/train_apply_phase1_iq_preadapter_20260703.py`：adapter训练与导出使用严格加载并写审计；
- `code/tests/test_exact_ssdg_checkpoint_loading.py`：严格成功与missing key阻断测试；
- 两个launcher增加独立feature root/config覆盖；
- 两个strict配置指向全新输出根，保留旧诊断artifact。

验证命令：

```text
conda run -n ssr-gpu python -m pytest code/tests/test_exact_ssdg_checkpoint_loading.py code/tests/test_stage2_export_target_old_cli.py -q
conda run -n ssr-gpu python -m py_compile code/cvsrffi/checkpoint_loading.py code/export_spaceborne_features.py code/scripts/train_apply_phase1_iq_preadapter_20260703.py
```

结果：4项测试全部通过，三个Python文件编译通过。

## N607预检与资源

2026-07-14 18:36直连预检PASS：目标、身份、项目根与8张RTX3090可见。盘点显示GPU0–7各有1个RIEI训练进程，显存约470–500MiB。按每GPU最多2个任务规则，本实验计划在GPU0–3各增加1个短时导出/adapter任务，不干预现有进程，不超过上限。

## 远端路径与启动计划

|字段|轻量组|完整组|
|---|---|---|
|GPU|0,1,2并行导出；矩阵CPU|3训练adapter并导出；矩阵CPU|
|feature root|`runs/cvs_publication_adv3b02_fft96_singleview_strict_20260714_183556`|`runs/cvs_qknnv42_full_adapter5_fft96_strict_20260714_183556`|
|run root|`runs/cvs_qknnv42_fft96_singleview_strict125_20260714_183556`|`runs/cvs_qknnv42_full_legacy_oracle_strict125_20260714_183556`|
|launcher log|`paper_reproduction/logs/qknnv42_strict_dual125_20260714_183556/single_launcher.log`|`paper_reproduction/logs/qknnv42_strict_dual125_20260714_183556/full_launcher.log`|
|PID文件|同目录`single.pid`|同目录`full.pid`|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|同左|
|工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|同左|

预期artifact：三份严格单视图NPZ、五个接收机adapter NPZ、250个task目录及各自metrics/resolved_config/split_manifest/score table、launcher与逐任务完整日志。

## 成功与停止标准

- 所有feature manifest必须为`checkpoint_load_strict=true`且missing/unexpected/mismatch均为0；否则停止矩阵。
- 两分支均必须完成125/125，support/query无重叠，三LEO场景覆盖完整。
- 日志不得出现Traceback、OOM、Killed、NaN/Inf或静默兼容加载。
- 结果按同一row联合报告old、new、H、coverage/defer与Oracle边界；不以不同row极值拼接结论。

## 结果

### 1.完成状态与严格加载审计

2026-07-14 18:55:34（N607/CST）确认两组父进程均退出，轻量组125/125完成，完整组125/125完成，失败数均为0。8份特征manifest全部满足`checkpoint_load_strict=true`、`missing_keys=0`、`unexpected_keys=0`、`skipped_mismatch=0`，共同加载195个state tensor，恢复14个domain head，输入长度256。轻量组为1-view+FFT96；完整组为60epoch适配+5-view TTA+FFT96。

|审计项|轻量组|完整组|
|---|---:|---:|
|严格特征manifest|3/3|5/5|
|完成任务|125/125|125/125|
|metrics/split/loss/score文件|125/125/125/125|125/125/125/125|
|support/query重叠违规|0|0|
|非有限loss行|0/375|0/375|
|qKNN梯度更新|0|0|
|最终状态|`COMPLETE`|`COMPLETE_NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`|

严格manifest逐项审计见`analysis/checkpoint_manifest_audit.csv`；完整250行指标见`analysis/per_run_results.csv`。

### 2.核心结果

|分支|运行数|old_acc|new_acc|H|平均遗忘|
|---|---|---|---|---|---|
|完整legacy oracle|125|84.07%|93.24%|88.23%|-0.24%|
|轻量单视图FFT96|125|75.12%|64.64%|68.56%|4.12%|

结论：严格ADV3B02下，轻量组为`old_acc=75.12%,new_acc=64.64%,H=68.56%`；完整legacy oracle组为`old_acc=84.07%,new_acc=93.24%,H=88.23%`。完整组相对轻量组的同任务配对提升为old`+8.94pp`、new`+28.60pp`、H`+19.66pp`，但该增益同时包含60epoch适配、5-view TTA及角色/类别配额Oracle，不能解释成可部署qKNN的纯算法增益。

### 3.与旧兼容加载诊断的配对比较

|分支|配对数|Δold|Δnew|ΔH|
|---|---|---|---|---|
|完整legacy oracle|125|+1.14pp|-0.13pp|+0.58pp|
|轻量单视图FFT96|125|+0.00pp|+0.00pp|+0.00pp|

轻量组125行与旧兼容加载诊断逐行完全相同。这不追认旧artifact为严格结果，而是说明新的严格导出在本配置上独立复现了同一组数值；从现在起应引用本次strict run的75.12%/64.64%/68.56%，旧run仍只保留为兼容加载诊断。完整组因适配器会更新此前未正确恢复的结构，严格重跑后old提高1.14pp、new下降0.13pp、H提高0.58pp；新的84.07%/93.24%/88.23%替代旧82.93%/93.37%/87.65%作为严格完整体Oracle诊断。

### 4.K-shot分解

|分支|K|old_acc|new_acc|H|
|---|---|---|---|---|
|完整legacy oracle|1|76.59%|87.27%|81.10%|
|完整legacy oracle|2|81.54%|91.93%|86.23%|
|完整legacy oracle|5|85.47%|94.40%|89.59%|
|完整legacy oracle|10|87.34%|96.00%|91.40%|
|完整legacy oracle|20|89.39%|96.60%|92.81%|
|轻量单视图FFT96|1|62.80%|48.13%|52.70%|
|轻量单视图FFT96|2|70.00%|57.20%|62.01%|
|轻量单视图FFT96|5|78.03%|66.80%|71.33%|
|轻量单视图FFT96|10|81.36%|73.67%|76.89%|
|轻量单视图FFT96|20|83.43%|77.40%|79.89%|

轻量组从K=1的H=52.70%提升到K=20的79.89%；完整组从K=1的81.10%提升到K=20的92.81%。轻量路径主要瓶颈仍是新类低样本识别，而不是旧类完全失效。

### 5.接收机分解

|分支|target receiver|old_acc|new_acc|H|
|---|---|---|---|---|
|完整legacy oracle|20-1|80.06%|93.20%|85.93%|
|完整legacy oracle|3-19|71.91%|80.60%|75.70%|
|完整legacy oracle|7-14|92.24%|97.53%|94.72%|
|完整legacy oracle|7-7|88.44%|97.67%|92.68%|
|完整legacy oracle|8-8|87.68%|97.20%|92.10%|
|轻量单视图FFT96|20-1|74.58%|70.33%|72.02%|
|轻量单视图FFT96|3-19|57.66%|42.80%|47.97%|
|轻量单视图FFT96|7-14|85.96%|73.50%|78.58%|
|轻量单视图FFT96|7-7|81.41%|67.57%|73.17%|
|轻量单视图FFT96|8-8|76.02%|69.00%|71.08%|

接收机`3-19`是两组共同最难域：轻量组H仅47.97%，完整组H为75.70%。完整组在`7-14`达到old=92.24%、new=97.53%、H=94.72%，说明跨接收机域差异仍远大于单一总体均值所呈现的波动。

### 6.场景分解

|分支|LEO场景|old_acc|new_acc|H|
|---|---|---|---|---|
|完整legacy oracle|leo_clear_weak|85.77%|95.48%|90.22%|
|完整legacy oracle|leo_low_elev_weak|82.59%|92.00%|86.78%|
|完整legacy oracle|leo_rain_weak|83.84%|92.24%|87.68%|
|轻量单视图FFT96|leo_clear_weak|78.57%|68.86%|72.51%|
|轻量单视图FFT96|leo_low_elev_weak|72.33%|66.36%|68.67%|
|轻量单视图FFT96|leo_rain_weak|74.48%|58.70%|64.52%|

轻量组在雨天弱链路的new_acc降至58.70%，H=64.52%；完整组最低场景为低仰角弱链路，H=86.78%。这表明5-view与完整体约束显著压低场景扰动，但不能消除接收机域困难。

### 7.阈值覆盖与联合行

|分支|old≥80%|new≥80%|H≥80%|old和new均≥80%|
|---|---|---|---|---|
|轻量单视图FFT96|59|28|38|26|
|完整legacy oracle|85|112|103|84|

联合最优行与最差行如下，所有指标来自同一运行，不拼接边际极值。

|位置|分支|receiver|seed|K|old|new|H|
|---|---|---|---|---|---|---|---|
|Top|完整legacy oracle|7-14|713104|20|94.72%|100.00%|97.27%|
|Top|完整legacy oracle|7-14|713105|10|94.44%|100.00%|97.14%|
|Top|完整legacy oracle|7-14|713105|20|94.44%|100.00%|97.14%|
|Top|完整legacy oracle|7-7|713101|20|94.17%|100.00%|96.99%|
|Top|完整legacy oracle|7-14|713101|10|94.17%|100.00%|96.99%|
|Bottom|轻量单视图FFT96|3-19|713105|1|55.00%|18.33%|26.31%|
|Bottom|轻量单视图FFT96|8-8|713105|1|67.50%|29.17%|37.34%|
|Bottom|轻量单视图FFT96|3-19|713101|2|53.33%|32.50%|38.40%|
|Bottom|轻量单视图FFT96|3-19|713101|1|42.22%|33.33%|35.97%|
|Bottom|轻量单视图FFT96|3-19|713101|5|63.89%|33.33%|43.71%|

最佳联合行是完整组`receiver=7-14,seed=713104,K=20`：old=94.72%、new=100.00%、H=97.27%。最差联合行是轻量组`receiver=3-19,seed=713105,K=1`：old=55.00%、new=18.33%、H=26.31%。

### 8.60epoch适配器训练日志分析

完整组适配器模式为`id_norm_late_feature`，训练参数289,685，输入修复关闭。manifest记录epoch1总loss=4.27497；launcher完整记录epoch5–60每5epoch一次，epoch5为3.66952，epoch55最低2.75568，epoch60为2.77942。epoch1→60期间CE从1.08824降至0.52343，proxy unknown SupCon从9.58522降至6.79505，proxy proto CE从3.76525降至2.54948，proxy-old margin从0.12739降至0.04684。曲线存在小幅非单调波动，但总体收敛，无NaN/Inf、OOM、Killed或Traceback。

250个qKNN任务的`loss_trace.csv`均为3个场景、每场景1行，`gradient_updates=0`，loss仅为约1e-16的数值残差。这符合非参数support-only拟合，不是训练塌缩。

### 9.完整日志与artifact审计

本地完整读取3个日志/artifact根，共2,260个文件、57,764,872字节、1,271,153行；Traceback、OOM、Killed、NaN/Inf命中均为0。250份split manifest逐场景重新计算support/query集合交集，违规为0；250份metrics、250份score table与750行场景级loss均完整且有限。

### 10.结论与声明边界

1.加载问题已经修复：后续导出器与适配器统一走训练时SSDG架构重建，任何missing/unexpected/shape mismatch都会fail-closed。
2.两组strict 125实验已全部重跑完成。新的可引用均值为轻量`75.12/64.64/68.56`，完整Oracle`84.07/93.24/88.23`。
3.轻量结果与旧诊断数值相同，但证据来源必须切换到本次strict run；完整结果已发生实质变化，旧82.93%不再使用。
4.完整体仍是`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`，因为角色筛选和类别配额约束使用了Oracle边界；88.23%不能作为卫星自主部署性能。
5.轻量路径是真正接近卫星约束的诊断，但H=68.56%、接收机`3-19`和K=1仍明显薄弱，当前不能据此宣称部署成功。

### 11.机器可读artifact

- `analysis/summary.json`：完整汇总、严格/兼容配对、阈值、artifact与日志审计；
- `analysis/per_run_results.csv`：250行逐运行结果；
- `analysis/per_scenario_results.csv`：750行逐场景结果；
- `analysis/paired_strict_vs_compatibility.csv`：250行严格与旧诊断配对；
- `analysis/paired_full_minus_light_strict.csv`：125行完整体与轻量体配对；
- `analysis/checkpoint_manifest_audit.csv`：8份严格特征manifest审计；
- `artifacts/`：两组run tree与完整launcher/worker日志。

### 12.250行逐运行同排结果

下表中的unknown/coverage/rollback/defer均未由本次关闭unknown rejection的Stage2-C任务产生，统一记为`N/A`。轻量组无适配器；完整组使用60epoch适配器，且结论仅限Oracle诊断。

|分支|receiver|seed|K|old|new|H|forgetting|unknown/coverage/rollback/defer|adapter|verdict|
|---|---|---|---|---|---|---|---|---|---|---|
|轻量单视图FFT96|20-1|713101|1|64.44%|48.33%|54.32%|6.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713101|10|85.56%|80.00%|82.62%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713101|2|74.17%|67.50%|70.47%|2.22%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713101|20|88.89%|82.50%|85.46%|-0.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713101|5|79.44%|75.83%|77.40%|-1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713102|1|63.33%|57.50%|59.90%|8.06%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713102|10|81.94%|85.83%|83.84%|1.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713102|2|64.44%|60.00%|62.10%|6.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713102|20|86.39%|85.83%|86.04%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713102|5|75.83%|75.83%|75.74%|4.17%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713103|1|43.33%|62.50%|51.10%|6.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713103|10|83.06%|76.67%|79.66%|0.83%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713103|2|60.00%|60.83%|60.14%|11.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713103|20|88.06%|83.33%|85.61%|0.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713103|5|77.50%|70.00%|73.29%|2.50%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713104|1|53.61%|60.83%|56.90%|6.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713104|10|80.28%|75.83%|77.98%|2.22%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713104|2|68.33%|67.50%|67.90%|4.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713104|20|81.94%|77.50%|79.49%|1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713104|5|73.89%|70.83%|72.31%|3.06%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713105|1|56.39%|52.50%|54.16%|9.44%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713105|10|87.22%|75.83%|81.12%|1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713105|2|74.17%|54.17%|62.56%|5.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713105|20|89.72%|79.17%|83.77%|-1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|20-1|713105|5|82.50%|71.67%|76.52%|3.61%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713101|1|42.22%|33.33%|35.97%|3.89%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713101|10|62.22%|47.50%|53.54%|8.06%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713101|2|53.33%|32.50%|38.40%|5.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713101|20|63.33%|45.00%|51.61%|6.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713101|5|63.89%|33.33%|43.71%|4.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713102|1|46.67%|40.00%|42.73%|8.33%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713102|10|68.61%|48.33%|56.60%|6.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713102|2|55.56%|34.17%|41.86%|9.44%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713102|20|69.44%|57.50%|62.89%|6.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713102|5|64.17%|42.50%|50.84%|7.50%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713103|1|43.06%|35.83%|38.97%|4.17%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713103|10|59.44%|48.33%|52.78%|7.78%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713103|2|50.83%|48.33%|49.47%|5.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713103|20|67.50%|51.67%|57.95%|6.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713103|5|56.67%|50.83%|53.28%|6.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713104|1|35.28%|39.17%|35.57%|5.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713104|10|59.72%|59.17%|59.02%|11.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713104|2|55.83%|35.83%|43.46%|5.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713104|20|64.44%|55.83%|59.53%|9.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713104|5|52.78%|49.17%|49.79%|10.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713105|1|55.00%|18.33%|26.31%|7.50%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713105|10|64.44%|38.33%|47.37%|4.44%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713105|2|54.17%|38.33%|44.22%|4.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713105|20|70.00%|48.33%|56.42%|4.17%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|3-19|713105|5|62.78%|38.33%|47.07%|5.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713101|1|80.56%|53.33%|63.74%|3.89%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713101|10|89.72%|82.50%|85.72%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713101|2|83.89%|65.83%|73.42%|6.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713101|20|91.39%|87.50%|89.40%|0.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713101|5|88.33%|70.00%|78.06%|1.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713102|1|78.33%|50.83%|61.64%|7.78%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713102|10|86.94%|83.33%|85.06%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713102|2|83.61%|55.00%|66.35%|1.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713102|20|87.50%|87.50%|87.32%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713102|5|87.50%|67.50%|76.20%|0.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713103|1|81.11%|46.67%|58.94%|5.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713103|10|91.67%|85.00%|87.99%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713103|2|87.22%|68.33%|76.19%|3.06%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713103|20|92.78%|85.83%|89.14%|-0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713103|5|88.89%|67.50%|76.42%|2.22%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713104|1|68.33%|75.00%|71.42%|9.44%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713104|10|85.28%|82.50%|83.84%|2.22%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713104|2|78.89%|79.17%|78.98%|6.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713104|20|86.39%|83.33%|84.77%|1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713104|5|85.56%|81.67%|83.52%|1.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713105|1|86.67%|59.17%|69.37%|3.06%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713105|10|90.28%|85.00%|87.54%|0.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713105|2|88.06%|67.50%|76.03%|1.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713105|20|90.83%|85.83%|88.26%|0.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-14|713105|5|89.17%|81.67%|85.21%|0.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713101|1|71.11%|50.83%|59.27%|8.61%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713101|10|90.00%|74.17%|80.86%|1.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713101|2|76.67%|55.83%|64.43%|6.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713101|20|89.17%|82.50%|85.49%|-0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713101|5|82.50%|77.50%|79.90%|2.78%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713102|1|70.00%|46.67%|55.68%|10.83%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713102|10|85.28%|75.00%|79.57%|1.39%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713102|2|76.39%|48.33%|58.90%|9.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713102|20|88.89%|76.67%|82.15%|0.83%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713102|5|81.94%|70.83%|75.51%|3.33%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713103|1|41.94%|35.00%|37.90%|7.78%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713103|10|90.28%|79.17%|84.22%|0.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713103|2|61.94%|69.17%|65.35%|7.22%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713103|20|90.28%|81.67%|85.70%|-0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713103|5|84.17%|79.17%|81.34%|5.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713104|1|68.06%|60.83%|63.90%|14.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713104|10|88.89%|84.17%|86.41%|-1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713104|2|83.89%|65.83%|73.64%|5.00%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713104|20|85.56%|89.17%|87.31%|-0.83%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713104|5|90.00%|77.50%|83.12%|4.17%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713105|1|80.00%|40.83%|54.00%|4.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713105|10|92.22%|72.50%|81.15%|1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713105|2|82.50%|55.83%|65.45%|5.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713105|20|94.17%|76.67%|84.34%|0.83%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|7-7|713105|5|89.44%|63.33%|73.75%|1.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713101|1|61.94%|61.67%|61.78%|9.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713101|10|85.83%|75.00%|79.96%|0.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713101|2|65.00%|63.33%|64.09%|12.78%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713101|20|86.94%|85.83%|86.34%|0.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713101|5|78.06%|71.67%|74.56%|6.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713102|1|71.11%|40.00%|50.29%|2.50%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713102|10|79.72%|84.17%|81.81%|1.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713102|2|70.83%|58.33%|63.06%|3.61%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713102|20|81.39%|87.50%|84.23%|0.56%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713102|5|77.50%|65.83%|69.99%|1.94%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713103|1|70.83%|49.17%|55.56%|5.83%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713103|10|80.00%|82.50%|81.06%|2.78%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713103|2|69.72%|55.83%|58.79%|4.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713103|20|82.22%|90.00%|85.91%|1.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713103|5|77.78%|69.17%|72.41%|4.72%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713104|1|69.17%|55.83%|60.68%|8.89%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713104|10|85.00%|86.67%|85.81%|1.67%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713104|2|59.44%|65.00%|61.92%|8.33%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713104|20|87.22%|85.83%|86.51%|1.11%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713104|5|80.00%|79.17%|79.24%|3.33%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713105|1|67.50%|29.17%|37.34%|5.28%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713105|10|80.28%|74.17%|76.62%|3.33%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713105|2|71.11%|57.50%|63.18%|9.17%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713105|20|81.39%|82.50%|81.65%|2.22%|N/A|无|轻量Stage2-C诊断|
|轻量单视图FFT96|8-8|713105|5|80.56%|69.17%|74.22%|1.67%|N/A|无|轻量Stage2-C诊断|
|完整legacy oracle|20-1|713101|1|60.28%|86.67%|70.90%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713101|10|81.11%|96.67%|88.13%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713101|2|70.28%|95.00%|80.76%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713101|20|86.94%|96.67%|91.49%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713101|5|81.67%|96.67%|88.41%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713102|1|73.89%|83.33%|78.23%|-2.22%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713102|10|89.44%|98.33%|93.66%|-1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713102|2|77.78%|90.00%|83.42%|-1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713102|20|89.72%|98.33%|93.83%|0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713102|5|84.44%|96.67%|89.99%|1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713103|1|73.61%|88.33%|80.12%|-3.06%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713103|10|86.11%|93.33%|89.54%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713103|2|73.89%|95.00%|83.06%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713103|20|92.50%|95.00%|93.73%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713103|5|75.00%|91.67%|82.49%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713104|1|70.83%|86.67%|77.77%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713104|10|82.78%|95.00%|88.45%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713104|2|72.50%|90.00%|80.29%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713104|20|91.39%|95.00%|93.15%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713104|5|82.22%|95.00%|88.06%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713105|1|69.17%|85.00%|75.79%|1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713105|10|88.06%|96.67%|92.07%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713105|2|76.39%|91.67%|83.31%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713105|20|88.89%|96.67%|92.61%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|20-1|713105|5|82.50%|96.67%|89.02%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713101|1|60.28%|75.00%|66.76%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713101|10|77.50%|75.00%|76.14%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713101|2|70.56%|76.67%|73.38%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713101|20|79.17%|80.00%|79.51%|-0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713101|5|76.67%|73.33%|74.64%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713102|1|51.39%|76.67%|61.08%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713102|10|78.61%|85.00%|81.65%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713102|2|65.00%|78.33%|70.43%|0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713102|20|79.72%|88.33%|83.68%|-0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713102|5|73.33%|81.67%|77.07%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713103|1|61.39%|71.67%|66.11%|-1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713103|10|75.83%|90.00%|82.23%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713103|2|66.67%|73.33%|69.50%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713103|20|79.72%|91.67%|85.28%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713103|5|74.17%|83.33%|78.45%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713104|1|64.44%|68.33%|65.65%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713104|10|75.00%|85.00%|79.63%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713104|2|66.67%|75.00%|70.04%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713104|20|79.44%|86.67%|82.86%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713104|5|73.33%|81.67%|76.99%|1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713105|1|62.22%|66.67%|64.26%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713105|10|79.17%|91.67%|84.94%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713105|2|68.89%|80.00%|73.64%|1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713105|20|82.22%|91.67%|86.67%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|3-19|713105|5|76.39%|88.33%|81.88%|-1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713101|1|87.22%|96.67%|91.62%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713101|10|94.17%|100.00%|96.99%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713101|2|91.94%|96.67%|94.23%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713101|20|93.61%|98.33%|95.91%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713101|5|92.50%|96.67%|94.53%|-1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713102|1|91.67%|96.67%|94.06%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713102|10|93.06%|100.00%|96.40%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713102|2|92.22%|96.67%|94.38%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713102|20|93.61%|100.00%|96.69%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713102|5|92.50%|96.67%|94.50%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713103|1|88.89%|96.67%|92.52%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713103|10|91.94%|100.00%|95.80%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713103|2|91.39%|98.33%|94.70%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713103|20|92.22%|100.00%|95.95%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713103|5|91.39%|98.33%|94.70%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713104|1|88.33%|78.33%|82.23%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713104|10|92.78%|100.00%|96.22%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713104|2|91.11%|98.33%|94.48%|1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713104|20|94.72%|100.00%|97.27%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713104|5|91.94%|100.00%|95.76%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713105|1|93.33%|93.33%|93.32%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713105|10|94.44%|100.00%|97.14%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713105|2|93.89%|98.33%|96.05%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713105|20|94.44%|100.00%|97.14%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-14|713105|5|92.78%|98.33%|95.46%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713101|1|78.89%|78.33%|77.21%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713101|10|90.28%|100.00%|94.88%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713101|2|88.06%|95.00%|91.29%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713101|20|94.17%|100.00%|96.99%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713101|5|87.78%|100.00%|93.45%|-1.11%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713102|1|78.89%|98.33%|87.25%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713102|10|90.28%|100.00%|94.82%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713102|2|88.89%|96.67%|92.51%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713102|20|91.94%|100.00%|95.77%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713102|5|89.44%|100.00%|94.37%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713103|1|78.89%|91.67%|84.70%|-2.50%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713103|10|90.00%|100.00%|94.73%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713103|2|87.78%|96.67%|92.00%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713103|20|90.56%|100.00%|95.03%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713103|5|91.39%|98.33%|94.70%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713104|1|83.33%|100.00%|90.84%|0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713104|10|93.33%|100.00%|96.52%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713104|2|87.22%|100.00%|93.17%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713104|20|93.61%|100.00%|96.68%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713104|5|91.94%|100.00%|95.79%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713105|1|86.11%|93.33%|89.43%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713105|10|90.83%|100.00%|95.18%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713105|2|86.39%|96.67%|91.18%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713105|20|92.50%|100.00%|96.09%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|7-7|713105|5|88.61%|96.67%|92.37%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713101|1|85.28%|98.33%|91.34%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713101|10|91.11%|100.00%|95.34%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713101|2|87.78%|95.00%|91.13%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713101|20|90.00%|100.00%|94.72%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713101|5|91.39%|100.00%|95.48%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713102|1|84.44%|93.33%|88.29%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713102|10|86.94%|100.00%|93.01%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713102|2|86.39%|100.00%|92.64%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713102|20|89.44%|100.00%|94.42%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713102|5|88.33%|100.00%|93.80%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713103|1|88.33%|95.00%|91.46%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713103|10|90.00%|96.67%|93.21%|0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713103|2|87.22%|95.00%|90.88%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713103|20|93.33%|98.33%|95.75%|-0.83%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713103|5|89.72%|95.00%|92.27%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713104|1|78.61%|91.67%|84.49%|0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713104|10|91.11%|98.33%|94.55%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713104|2|88.33%|95.00%|91.51%|0.00%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713104|20|91.67%|98.33%|94.86%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713104|5|88.33%|95.00%|91.50%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713105|1|75.00%|91.67%|82.15%|-1.39%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713105|10|89.72%|98.33%|93.80%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713105|2|81.39%|95.00%|87.64%|0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713105|20|89.17%|100.00%|94.27%|-0.56%|N/A|60ep id_norm_late_feature|Oracle诊断|
|完整legacy oracle|8-8|713105|5|88.89%|100.00%|94.10%|-0.28%|N/A|60ep id_norm_late_feature|Oracle诊断|

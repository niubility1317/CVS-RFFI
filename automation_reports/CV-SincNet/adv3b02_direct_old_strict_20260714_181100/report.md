# ADV3B02地面模型直接旧类评测报告

## 实验信息

|字段|内容|
|---|---|
|实验ID|`adv3b02_direct_old_strict_20260714_181100`|
|时间|2026-07-14 18:11（Asia/Hong_Kong）|
|操作者|Codex|
|目标|不使用域适应、K-shot支持、qKNN、FFT辅助、TTA或角色/配额约束，直接用地面训练的ADV3B02分类头评测旧类星地信道结果|
|比较对象|历史ADV3B02 Phase1 final：`sat_strict_mean=69.8639%`；此前qKNNV42 125任务结果仅作协议差异对照|
|声明边界|Phase1闭集旧类诊断；不是Stage2-C、不是新类学习、不是部署成功证据|

## 假设与方法

假设：若严格重建ADV3B02训练时架构并直接使用其六类`tx_logits`分类头，则结果应复现Phase1闭集旧类卫星评测口径，并解释“均值约70%但局部低于70%”来自均值、场景、接收机和切分粒度差异，而不是域适应退化。

方法：调用现有严格评测器`paper_reproduction/scripts/evaluate_cvs_phase1_ssdg_detailed.py`。该评测器通过`SSDG.train_ssdg`中的训练参数重建模型，要求checkpoint加载`missing=0`且`unexpected=0`；对三个简化LEO场景分别前向六类地面分类头，保存逐样本、逐接收机、逐TX和逐日结果。输入仅为目标旧类原始IQ经单次星地信道后的视图；输出为六类`tx_logits`及top-1预测。

明确禁用：target support、target label训练、adapter、qKNN、FFT96、TTA、原型、标签传播、Hungarian角色/类别配额约束、新类竞争。

## 协议与配置

|字段|值|
|---|---|
|基座checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|数据集|`Dataset_WigSig/ManySig.pkl`|
|旧类|`14-10,14-7,20-15,20-19,6-15,8-20`|
|目标接收机重点|`20-1,3-19,7-14,7-7,8-8`|
|场景|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|sat seed|`2027`，与Phase1正式评测口径一致|
|设备|N607 GPU0，与已有GPU0训练进程并发；不超过每GPU两个任务上限|

## 本地与远端状态

`E:\type10-7`根目录不是Git仓库。本报告镜像到Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。本次不修改评测代码；使用远端已有、与本地Git承载面一致的严格评测脚本。Git承载面启动前存在用户/其他任务的未提交改动和未跟踪artifact，本实验不覆盖、不清理、不纳入提交。

N607直连预检PASS。2026-07-14 18:09盘点显示8张GPU各有1个RIEI训练进程；GPU0显存占用约470MiB。本实验是只读checkpoint/数据的推理任务，计划作为GPU0第2个任务运行。

## 启动信息

|字段|值|
|---|---|
|工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端输出目录|`paper_reproduction/runs/adv3b02_direct_old_strict_20260714_181100`|
|日志|`paper_reproduction/logs/adv3b02_direct_old_strict_20260714_181100.log`|
|PID文件|`paper_reproduction/logs/adv3b02_direct_old_strict_20260714_181100.pid`|
|预期文件|`metrics.json`、`split_manifest.json`、`resolved_config.json`、`score_table.csv`、`detailed_metrics.csv/json`|

最终执行入口为`paper_reproduction/scripts/evaluate_adv3b02_ground_direct_old.py`。它不重建或修改历史source弱标签划分，只从checkpoint state和checkpoint args严格重建模型，并对与125任务相同的target-old样本池直接运行分类头。target-old子集seed使用`713101+29=713130`，星地信道seed使用`713101+811=713912`，与20260714单视图导出的样本池和信道生成顺序一致。

## 成功标准与风险

- checkpoint必须严格重建并加载；任何missing/unexpected key都应直接失败。
- 三场景各应产生204000条逐样本记录，总计612000条。
- 重点报告五个目标接收机的同row准确率、三场景均值/下界和总体结果，不能拼接不同row极值。
- 该口径与Stage2的125个K-shot query子集不同；若需要一一复用125个query ID，须另做严格样本ID对齐，不能把本结果冒充125次Stage2任务。
- 此前20260714特征导出manifest显示`checkpoint_load_strict=false`、`missing_keys=7`、`unexpected_keys=31`、`skipped_mismatch=3`，因此其qKNN结果只能视为兼容加载诊断，不能替代本次严格地面模型结果。

## 执行记录与异常

1. 首次尝试调用严格Phase1详细评测器，发现远端缺少该脚本；本地`ssr-gpu`环境完成`py_compile`后同步到远端。
2. 该评测器在重建历史source划分时按当前`项目.md`硬门触发阻断：历史checkpoint记录`labeled_ratio=0.1`、`unlabeled_ratio=0.7`，即`rho_label=0.125`，高于当前`rho_label<=0.1`。未绕过或篡改该科学协议。
3. 改用专门的target-old直接评测器：不重建source弱标签划分，因此不改变当前协议；模型state严格加载，`missing=0`、`unexpected=0`。
4. 首轮输出发现聚合CSV异构字段写入错误，修复字段并集后重跑。随后发现target-old子集seed应为原导出器的`seed+29`，最终以`713130`重跑并覆盖中间结果。以下仅报告最终校正结果。

完整远端日志已回收并逐行检查；最终日志只有一条成功JSON，无Traceback、NaN或OOM。PID`509677`已退出，结果文件完整。

## 最终结果

### 1.完整target-old样本池

严格地面分类头在7200个星地视图上的总体`old_acc=73.57%`。每个场景2400个样本；没有使用任何target support或target标签。

|场景|正确/总数|old_acc|
|---|---:|---:|
|leo_clear_weak|1827/2400|76.13%|
|leo_low_elev_weak|1700/2400|70.83%|
|leo_rain_weak|1770/2400|73.75%|
|三场景合计|5297/7200|73.57%|

|目标接收机|三场景正确/总数|old_acc|clear|low elev|rain|
|---|---:|---:|---:|---:|---:|
|20-1|905/1389|65.15%|67.39%|62.63%|65.44%|
|3-19|841/1392|60.42%|63.79%|55.60%|61.85%|
|7-14|1201/1353|88.77%|91.35%|88.47%|86.47%|
|7-7|1278/1548|82.56%|84.88%|79.84%|82.95%|
|8-8|1072/1518|70.62%|72.92%|67.39%|71.54%|

|旧类TX|正确/总数|old_acc|
|---|---:|---:|
|14-10|764/1200|63.67%|
|14-7|716/1200|59.67%|
|20-15|937/1200|78.08%|
|20-19|764/1200|63.67%|
|6-15|1044/1200|87.00%|
|8-20|1072/1200|89.33%|

### 2.复用原125任务的旧类query

使用原125个split manifest中的old-query sample ID逐条对齐严格地面分类头预测。每个任务包含三场景×120个旧类query，共360个预测。结果为：

|指标|结果|
|---|---:|
|125行task macro old_acc|73.87%|
|最小task|59.17%，`rx=3-19,seed=713103`|
|最大task|90.28%，`rx=7-14,seed=713103`|
|低于70%的task|45/125|
|低于80%的task|80/125|

|目标接收机|125行中的任务数|old_acc|
|---|---:|---:|
|20-1|25|65.39%|
|3-19|25|60.11%|
|7-14|25|88.72%|
|7-7|25|82.72%|
|8-8|25|72.39%|

|场景|old_acc|
|---|---:|
|leo_clear_weak|76.10%|
|leo_low_elev_weak|71.37%|
|leo_rain_weak|74.13%|

K=1/2/5/10/20五档均为73.87%。这是预期现象：本分支完全不读取K-shot support，原矩阵的old-query集合也不随K变化，因此同一receiver×seed被五个K标签重复。125行实际只有25个独立receiver×seed query集合；K在这里仅是对齐原矩阵的索引，不是模型输入。

## 解释与结论

“ADV3B02星地strict UDU均值约70%”不表示每个接收机、场景和小query子集都不低于70%。历史final记录是`sat_strict_mean=69.86%`，best epoch约70.56%；历史final的rain strict本身约68.77%。本次严格直接评测进一步显示，主要短板集中在`3-19`和`20-1`，分别只有60.11%和65.39%，而`7-14`达到88.72%。因此低于70%不是域适应造成的，因为本实验根本没有域适应；它来自receiver/domain差异、场景差异和小query子集波动。

本次还确认了一个更重要的问题：此前双125实验使用的特征导出manifest为`checkpoint_load_strict=false`，并有7个missing key、31个unexpected key和3个shape mismatch。那两个qKNN分支虽然计算流程完成，但不能继续当作“严格ADV3B02底座”结果；其数值应降级为兼容加载诊断，必须在修复导出器严格重建后重跑才能做正式比较。

## Artifact与验证

|文件|用途|
|---|---|
|`artifacts/metrics.json`|严格直接评测总指标与机制开关|
|`artifacts/split_manifest.json`|target-old样本池、接收机、场景和严格加载声明|
|`artifacts/score_table.csv`|7200条逐样本预测|
|`artifacts/aggregate_metrics.csv`|场景、接收机、TX联合聚合|
|`artifacts/query125_per_task.csv`|原125任务逐行旧类结果|
|`artifacts/query125_per_scenario.csv`|125任务逐场景结果|
|`artifacts/query125_summary.csv`|receiver、seed、K、场景汇总|
|`artifacts/query125_metrics.json`|125任务总体统计|
|`run.log`|最终完整远端日志|

本地验证：两个新脚本均在`ssr-gpu`环境通过`python -m py_compile`；125个split manifest全部找到，所有old-query sample ID均成功对齐，无缺失预测。N607任务完成后，本地`ssh.exe`和N607/bridge TCP22连接均检查为0。

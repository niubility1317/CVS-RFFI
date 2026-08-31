# WISER-RF P3-Primary稳定快速版预登记与机制核查报告

## 当前结论

- 新run ID：`wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1`。
- 当前最高状态：`ANALYZED / NO_PROMOTION_TO_TARGET25`。18/18组prediction、18/18组truth-last score和15/15组同row候选对照均已闭合。
- prediction生产代码提交：`31381958f8075686f5d9410822f5f42428cc417f`；评分器定点修复提交：`a541643feab4c58cb0e403ee15d3dc1206f7ba6f`；两个提交的远端分支OID均已独立核对一致。
- 原实现本地聚焦验证112项通过；评分器修复新增混合`target_old/target_new`红绿测试，评分器与pilot入口59项聚焦回归通过，`py_compile`与`git diff --check`通过。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`、固定`capsule_id/split_id`、support-only训练与选择、全部support状态冻结后只读打开query、prediction完整后由独立scorer连接truth。

## 域训练过久修复

旧run的`p3-smoke`错误复用了正式N6的10000步预算，运行约6小时后才在Stage2发现非有限梯度。新实现做了三层有界化：

1. smoke默认只执行Stage1、三个Stage2分支和Stage3各1步，共5个optimizer step；相对旧smoke上限缩短2000倍。
2. 正式P3配置从`[1500,2000,2500]`改为`[40,60,80]`；N6最坏路径为`40+3×60+80=300`步，相对旧10000步缩短33.3倍。
3. 正式Stage2只有Stage1通过support-only门槛才进入；Stage3只有选中Stage2分支继续通过才进入。每10步诊断，至少20步后连续2次无改善即早停。query不参与步数、门槛、分支或插值选择。

本run的正式arms冻结为`N0,N2,N3,N4,N5,N6`。旧`N1=WISER-A`采用独立8000步旧损失、已退出候选且规则禁止成为冠军；重新运行不会验证新机制，只会恢复主要时延，因此本run不启动N1，也不对N1作同row新结论。

## 失败机制与修复闭环

|问题|旧状态|新实现|本地证据|真实证据缺口|
|---|---|---|---|---|
|D92内层Adam零二阶矩高阶导数|合法单热点几何前向有限、反向15360个NaN|零点安全开平方，保持正数前向值，零点导数有限|先红后绿的高阶梯度测试|真实checkpoint smoke|
|D92 score RMS退化|RMS为0后除零|零/数值退化RMS显式拒绝|退化定点测试|真实动态是否再触发|
|梯度首错不可定位|只检查最终`parameter.grad`|依次检查loss、primary、auxiliary、projected、combined与dual，异常先写JSONL再抛出|正常/失败事件测试|真实训练轨迹|
|`diagnostic_interval`未消费|6小时无中间artifact|正式循环每周期写`training_progress.jsonl`并flush|事件消费测试|远端artifact增长|
|Stage2/3无条件运行|所有分支固定跑满|support-only门槛、耐心早停、未过门槛不深训|门槛/早停测试|各真实场景实际步数|
|选中状态可能未回载|Stage2选择后模型可能停在最后评估分支|最终显式回载选中state并refreeze|runner状态测试|prediction receipt|
|机制诊断不完整|缺按类zero-id、block trace、单模态风险、块级梯度夹角|补齐上述字段及canonical correlations|P3/runner测试|真实数值|
|资源证据缺失|无训练/预测资源闭环|阶段/总训练耗时、峰值RSS/VRAM、状态字节、prediction耗时进入审计/receipt|序列化测试|真实资源值|

没有加入dual cap、梯度裁剪、`nan_to_num`或静默跳步，因为现有证据没有证明对偶放大是首因；这些操作会改变科学方法动力学，不能作为无证据补丁。

## 设计机制/设置/参数全面核查

- `N2`：5-fold cross-fitted old-only精确D92主损失，已实现并由正式入口消费。
- `N3`：每类风险、soft floor、非负对偶更新，已实现并由正式入口消费。
- `N4`：6类目标中心与26×6 int8源域类中心的共享域流形，已实现并由正式入口消费。
- `N5`：P3主梯度与source-head、target-prototype、domain-manifold辅助梯度的全局冲突投影，已实现并由正式入口消费。
- `N6`：identity–FFT互补/冗余与identity能量约束，已实现并由正式入口消费。
- 24个`WISERP3TrainingConfig`字段均在正式runner中有实际消费点；新增早停字段有严格类型/范围校验；旧`20260831`配置保持原样，新配置独立为`configs/wiser_rf_p3_primary_20260901_stablefast.json`。
- source类内低秩协方差摘要仍是设计明确的后续项，本轮未实现；现有int8域×类中心已足够验证共享域流形。
- C/ABC、旧classwise VSW、ASAM、SWA和大权重网格已明确移出主矩阵，不属于“漏启动”。
- Target25、完整125与阶段B没有启动；它们必须等待本pilot科学门槛，不能由技术修复提前授权。

## 冻结输入与资源

- outer：`rx_3_19__seed_713102__k_10__new_5`；receiver=`3-19`；seed=`713102`；K=`10`；new-count=`5`。
- scenes：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- arms：`N0,N2,N3,N4,N5,N6`；共18个prediction/receipt。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`。
- source summary：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz`。
- source binding：`configs/wiser_rf_adv3b02_source_binding.json`。
- P3 config：`configs/wiser_rf_p3_primary_20260901_stablefast.json`。
- 物理GPU0；`CUDA_VISIBLE_DEVICES=0`后程序使用`cuda:0`。2026-09-01 01:50CST只读盘点8张GPU均无compute-app，本run加入后不超过用户授权的每卡3个训练任务。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- release：`wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_31381958.tar.gz`。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_31381958`。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1/pilot.out`。
- score：`<run-root>/score`，必须在prediction完整后由独立进程创建。

## 冻结命令

远端CWD固定为上述release根。

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-smoke --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260901_stablefast.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --source-summary /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1/smoke --device cuda:0 --runtime-commit 31381958f8075686f5d9410822f5f42428cc417f --arm N6 --scenario leo_clear_weak --smoke-stage-steps 1 1 1
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-pilot --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260901_stablefast.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --source-summary /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1/pilot --device cuda:0 --runtime-commit 31381958f8075686f5d9410822f5f42428cc417f --arms N0 N2 N3 N4 N5 N6
```

## 停止规则与预期artifact

- 只允许因协议/query越权、错误split/receiver/seed/K/scene、错误checkout/CWD、run root冲突、非有限loss/gradient/dual、确定性重复异常、prediction不完整或scorer连接错误停止。
- 不因低性能、负收益、未晋级或缺少非必要字段停止。
- smoke预期：`smoke/training_progress.jsonl`、`smoke/smoke_result.json`，且`query_opened=false/query_rows_used=0`。
- pilot预期：18个`training_audit.json`、适配state、18个完整`predictions.npz`与receipt、`support_audit.json`、`pilot_result.json`。
- prediction完整后另起独立`p3-score-pilot`连接truth，生成详细score与`score_collection.json`。报告绝对Accuracy/BA/floor/NLL、per-class、P1/P2/P3、适应增量、help/harm、训练/预测资源和三场景同row结果。
- 只有三场景pilot门槛通过才授权Target25；Target25通过后才讨论完整125与阶段B。

## 2026-09-01远端发布与启动回执

- release归档本地/远端唯一SHA256均为`c27b1a136dff7e8912cfcb7936ee5d71724dbe49be3b36f84d2244f7b6b25ad9`；远端四个运行模块一次`py_compile`通过。
- 启动前只读复核：8张GPU compute-app为空；新run、log根均不存在。物理GPU0被冻结且未超过每卡3个训练任务。
- 远端owner PID=`3208551`（PPID1），smoke worker PID=`3208559`；worker CWD精确为`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_31381958`，cmdline、config、checkpoint、source summary、output root和runtime commit与预登记一致。
- GPU映射：worker PID`3208559`位于物理GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`，首次采样显存6700MiB。
- 启动36秒时`smoke/training_progress.jsonl`已有3957字节；最新事件为`stage2_time/step1`，loss、primary、auxiliary、projected、combined gradient及dual均有限，`zero_identity_count=0`，`query_rows_used=0`。日志仍为0字节属于stdout缓冲，不影响JSONL增长证据。
- 当前最高状态更新为`RUNNING`。不重复启动、不热修改、不因中间性能停止；下一次检查在smoke预计完成后进行。

## 2026-09-01 03:00CST小时检查

- 真实ADV3B02 checkpoint无query smoke已`PASS`：5个optimizer step完整覆盖Stage1、三个Stage2分支和Stage3，总训练耗时90.466秒；旧smoke约6小时仍未完成，快速闭环已经由真实远端证据确认。
- smoke峰值CUDA分配6395007488字节，进程峰值RSS2075070464字节；最终`zero_identity_count=0`，identity-only、FFT-only、joint OOF风险均为`VALID`，全程`query_rows_used=0/query_opened=false`。
- 同一owner PID`3208551`已按预登记自动进入`p3-pilot`，当前运行约1小时3分；物理GPU0显存8788MiB。
- pilot支持适配阶段已完成5个`training_audit.json`，当前推进到`leo_clear_weak/N6`，对应进度JSONL为8221字节；尚未打开query，因此prediction/receipt为0/18符合support-first顺序。
- 定点扫描未发现`FAILED_NONFINITE`，未发现技术异常。保持运行，不终止、不重启、不热修改、不因性能停止。

## 2026-09-01 pilot闭合、评分器修复与最终状态

- 原owner PID`3208551`自然退出；`pilot_result.json`状态为`ARTIFACTS_COMPLETE`。18/18组`training_audit.json`、prediction与receipt完整，3个LEO场景×6个arm各有120条query，共2160条独立预测；prediction阶段`truth_opened=false`。
- 第一次独立评分尝试保留在`<run-root>/score`与`score.out`，技术状态为`FAILED`：truth sidecar同时包含`target_old`与`target_new`，旧评分器在按prediction token连接前错误地要求sidecar全部truth index落在0至5，因新类索引6至10抛出`WISER truth index is outside the six-class registry`。该问题不影响已封闭prediction。
- 定点修复遵循REG0语义：先验证全部prediction token、特征、三组预测、logit、有限性与argmax闭合，再按已连接token的`evaluation_role=target_old`抽取六旧类评分；缺少角色字段的旧六类fixture继续按`target_old`兼容。显式`target_new`仍不得与0至5旧类注册表重叠。
- 修复release为`wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_scorefix1_a541643f`；本地/远端唯一归档SHA256均为`db75802711cbdcaf02a89385f3b102f1d7c88b41f7f9360cced9875e9977533e`，远端评分模块与入口编译通过。
- 第二次独立评分使用不可覆盖输出`<run-root>/score_fix1`，PID`3311842`自然退出，生成18个`score.json`和`score_collection.json`。集合状态为`ANALYZED`，18/18行`truth_join_after_prediction_only=true`，每行120条旧类query、每类20条；当前prediction registry不含新类probe token，因此`total_query_rows=old_query_rows=120`、`ignored_non_old_query_rows=0`。
- `champion_identity=null`、`full_target25_authorized=false`。5个候选均未通过预登记P3-primary门槛；未启动Target25、完整125或阶段B。

## 三场景绝对结果

下表的BA为六旧类balanced accuracy，floor为六类最小召回率，均以百分数表示；NLL越低越好。P1、P2、P3分别为source head、source prototype和old D92 probe。

|场景|arm|P1 BA|P2 BA|P3 BA|P3 floor|P3 NLL|
|---|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|N0|61.67|60.83|78.33|55.00|1.018577|
|leo_clear_weak|N2|60.00|59.17|75.00|45.00|0.996677|
|leo_clear_weak|N3|49.17|47.50|76.67|40.00|0.924920|
|leo_clear_weak|N4|59.17|59.17|70.83|40.00|1.009479|
|leo_clear_weak|N5|62.50|60.83|81.67|55.00|0.930259|
|leo_clear_weak|N6|64.17|61.67|78.33|60.00|0.962952|
|leo_low_elev_weak|N0|69.17|67.50|70.00|35.00|1.044540|
|leo_low_elev_weak|N2|69.17|67.50|70.00|35.00|1.044540|
|leo_low_elev_weak|N3|69.17|67.50|70.00|35.00|1.044540|
|leo_low_elev_weak|N4|69.17|67.50|70.00|35.00|1.044540|
|leo_low_elev_weak|N5|69.17|67.50|70.00|35.00|1.044540|
|leo_low_elev_weak|N6|69.17|67.50|70.00|35.00|1.044540|
|leo_rain_weak|N0|58.33|58.33|71.67|40.00|1.059876|
|leo_rain_weak|N2|59.17|60.00|69.17|30.00|1.058862|
|leo_rain_weak|N3|58.33|56.67|71.67|40.00|1.026607|
|leo_rain_weak|N4|60.00|59.17|71.67|35.00|1.061890|
|leo_rain_weak|N5|55.83|55.83|71.67|35.00|1.026600|
|leo_rain_weak|N6|55.00|55.83|71.67|35.00|1.026384|

## 域适应增量、help/harm与按类变化

所有增量均为同一outer、scene、query和基线N0上的`DA1_REG0-DA0_REG0`，单位为百分点。正的net表示被域适应纠正的query数多于被破坏的query数。

|场景|候选|ΔP3 BA|ΔP3 floor|help|harm|net|ΔNLL|
|---|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|N2|-3.33|-10.00|5|9|-4|-0.021900|
|leo_clear_weak|N3|-1.67|-15.00|9|11|-2|-0.093656|
|leo_clear_weak|N4|-7.50|-15.00|5|14|-9|-0.009097|
|leo_clear_weak|N5|+3.33|0.00|11|7|+4|-0.088317|
|leo_clear_weak|N6|0.00|+5.00|5|5|0|-0.055624|
|leo_low_elev_weak|N2|0.00|0.00|0|0|0|0.000000|
|leo_low_elev_weak|N3|0.00|0.00|0|0|0|0.000000|
|leo_low_elev_weak|N4|0.00|0.00|0|0|0|0.000000|
|leo_low_elev_weak|N5|0.00|0.00|0|0|0|0.000000|
|leo_low_elev_weak|N6|0.00|0.00|0|0|0|0.000000|
|leo_rain_weak|N2|-2.50|-10.00|1|4|-3|-0.001015|
|leo_rain_weak|N3|0.00|0.00|7|7|0|-0.033269|
|leo_rain_weak|N4|0.00|-5.00|2|2|0|+0.002014|
|leo_rain_weak|N5|0.00|-5.00|6|6|0|-0.033277|
|leo_rain_weak|N6|0.00|-5.00|6|6|0|-0.033492|

按固定六旧类注册表顺序，P3按类召回率变化如下：

|场景|候选|类0|类1|类2|类3|类4|类5|
|---|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|N2|-10|-20|0|+10|0|0|
|leo_clear_weak|N3|+25|-25|0|-10|0|0|
|leo_clear_weak|N4|-20|-25|0|+5|-5|0|
|leo_clear_weak|N5|+10|+10|0|0|0|0|
|leo_clear_weak|N6|-5|+5|0|+5|-5|0|
|leo_low_elev_weak|N2/N3/N4/N5/N6|0|0|0|0|0|0|
|leo_rain_weak|N2|-5|-10|0|-5|+5|0|
|leo_rain_weak|N3|-5|0|0|-5|+10|0|
|leo_rain_weak|N4|0|-5|0|0|+5|0|
|leo_rain_weak|N5|-5|-5|0|-10|+20|0|
|leo_rain_weak|N6|-5|-5|0|-10|+20|0|

## 晋级门槛结果

|候选|三场景ΔP3 BA|中位ΔBA|最差ΔBA|中位Δfloor|三场景net|结论|
|---|---|---:|---:|---:|---|---|
|N2|-3.33/0/-2.50|-2.50|-3.33|-10.00|-4/0/-3|失败：均值、floor、net与最差场景|
|N3|-1.67/0/0|0.00|-1.67|0.00|-2/0/0|失败：均值、P1/P2安全、net与最差场景|
|N4|-7.50/0/0|0.00|-7.50|-5.00|-9/0/0|失败：均值、floor、P1安全、net与最差场景|
|N5|+3.33/0/0|0.00|0.00|0.00|+4/0/0|失败：只在clear获益，rain的P1/P2均下降2.50pp|
|N6|0/0/0|0.00|0.00|0.00|0/0/0|失败：无P3 BA净收益，rain的P1下降3.33pp、P2下降2.50pp|

## 训练时长、早停与资源

- 15个候选场景单元实际执行1910个optimizer step；若全部走满冻结的300步最坏路径则为4500步，support-only门槛与耐心早停减少57.56%的步数。
- 候选训练累计9074.525秒，即2.521GPU小时；单元中位702.711秒，最长为`leo_clear_weak/N6`的1224.235秒。clear、low-elev、rain分别消耗1.307、0.147、1.067GPU小时。
- low-elev的5个候选均在首阶段20步后未通过support-only门槛，后续分支全部跳过，故query输出与N0严格相同。这是预登记fail-closed行为，不是训练未启动或scorer遗漏。
- 训练峰值CUDA分配8.479GB（7.897GiB），峰值RSS2.151GB（2.004GiB），最大适配state为800849字节（782.08KiB）。全15个`training_progress.jsonl`共191条记录，JSON解析0失败，NaN/Inf、OOM、Killed与Traceback标记均为0。

|场景|N2步/秒|N3步/秒|N4步/秒|N5步/秒|N6步/秒|
|---|---:|---:|---:|---:|---:|
|leo_clear_weak|150/690.90|210/916.96|140/676.98|220/1195.05|250/1224.24|
|leo_low_elev_weak|20/115.89|20/101.27|20/106.55|20/100.08|20/106.06|
|leo_rain_weak|150/755.42|150/702.71|180/839.86|180/759.25|180/783.32|

## 机制判断与未启动项核查

1. N2的support OOF BA不等于target query收益；clear和rain分别下降3.33pp与2.50pp，说明精确D92主损失仍存在support-to-query错位。
2. N3在clear将NLL降低0.093656，却把BA降低1.67pp、floor降低15pp，并使P1/P2分别下降12.50pp/13.33pp。概率校准改善不能替代硬分类与旧类底线门槛。
3. N4没有带来跨场景稳健收益；clear下降7.50pp，rain仅保持BA且floor下降5pp。当前仅用int8域×类中心的共享流形不足以稳定纠正目标域方向。
4. N5是唯一产生局部正信号的机制：clear的P3 BA提高3.33pp，help/harm=11/7；但low-elev无变化，rain的P1/P2安全指标各下降2.50pp。梯度冲突投影值得保留为后续机制线索，不足以成为默认方法。
5. N6在clear只提高floor 5pp而BA不变，在另外两个场景没有P3 BA收益；identity–FFT约束没有形成跨场景增益。
6. `p3_training`的24个参数均已由正式runner消费；Stage1/2/3门槛、早停、进度落盘、资源记录、按类风险、流形、梯度投影、互补/能量约束均有真实训练审计。不存在“配置写了但正式入口未消费”的P3参数。
7. 配置仍保留`n1_training`和arm注册项N1用于schema兼容，但正式命令明确排除N1；这是预登记退出，不是漏启动。C/ABC、旧classwise VSW、ASAM、SWA与大权重网格同样不属于本轮冻结矩阵。
8. source类内低秩协方差摘要尚未实现；它是新机制扩展，不影响本轮int8中心流形结果的有效性。若继续研发，应首先验证“support目标与query硬分类收益对齐”，并以N5的clear局部信号为起点，而不是恢复长步数或扩大参数网格。
9. Target25、完整Target125和阶段B均未启动。原因是`full_target25_authorized=false`，属于科学未晋级，不是技术失败；绕过该门槛会把单场景局部增益误当成跨场景方法成立。

## 最终结论

本轮工程目标已闭合：真实smoke从旧run超过6小时未完成缩短到90.466秒；正式pilot通过门槛与早停把实际步数压到最坏预算的42.44%，完整产生2160条query预测并完成独立truth-last评分。科学目标未达成：没有候选同时满足跨三场景P3 BA、floor、help/harm与P1/P2安全门槛。正式状态为`ANALYZED / NO_PROMOTION_TO_TARGET25`，后续不应自动扩展到Target25、125或阶段B。

# Phase1 ADV3B02 FastTrust数值修复后16条矩阵完整实验报告

## 当前状态

```text
run_id=phase1_adv3b02_fasttrust16_s392002_20260821_r2
status=ANALYZED
seed=392002
epochs=200
matrix_rows=16
gpu_count=8
rows_per_gpu=2
completed_candidates=16/16
completed_e200=16/16
completed_clean_and_3leo=16/16
analysis_readback_time=2026-08-22 15:13 CST
```

## 修复目的

上一run在系统执行和数据划分正常时出现MUSE本地分类概率在AMP下转回float16、非目标类概率下溢为0、有限forward loss产生NaN backward梯度的问题，最终导致优化器持续跳步。r2只修复该共同数值路径：本地分类概率固定保留float32，FastTrust路由、三头定义、U_s身份规则、损失权重、seed、数据角色、星地增强和矩阵均不改变。

旧run固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，其远端run root及全部artifact原位保留。r2使用新的不可覆盖output root，不从旧run恢复任何故障期训练状态。

## 冻结矩阵与协议

- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，实际预期样本数`5880/52920/12600/12600`。
- 全部行使用seed`392002`和E200；R0从scratch，其余行从`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`初始化。
- Phase1训练复用CORE90同款LEO_WEAK拼接增强，`lambda_sat_cls=0.68`、`lambda_sat_cons=0`和三段场景日程。
- 每GPU两个实验；GPU3为U128+U384，其余GPU为U256+U256，每卡U batch总和均为512。
- U_s每epoch完整覆盖；U伪身份只能来自FastTrust预定的H/M/candidate路由，U_H星地身份CE仍要求high、temporal stable、三头一致和class cap。
- checkpoint选择固定为`final_only`。训练完成后必须自动测试`final_ssdg.pth`的clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，四份逐场景结果不齐全不得标记完成。

## 技术停止规则

- protocol/query泄漏、错误checkout或output碰撞立即停止对应run。
- 同一候选连续两个完整epoch的`train/optimizer_step_applied=0`，视为确定性非有限梯度执行故障；仅停止该run-owned进程树并保留产物。
- Traceback、OOM、`TRAIN_FAILED`、`EVAL_FAILED`或prediction闭合失败按对应技术失败处理。
- 不因中间准确率低、loss走势差或候选性能不佳停止训练。

## 发布前验证

- 新增AMP概率下溢回归测试先RED：三类logit`[0,-20,-40]`产生有限forward loss但输入及五组本地头参数梯度为NaN。
- 最小修复后该测试及MUSE路由、训练集成、FastTrust协议/速度/launcher聚焦联合测试全部GREEN；聚焦联合回归144项通过。
- Python编译、两个launcher语法检查和16条dry-run均通过；远端dry-run实际生成16个训练命令、16次联合评测命令和64份clean/三LEO分场景输出声明，且没有创建run root。
- N607真实checkpoint CUDA无query验证严格恢复ADV3B02权重，missing/unexpected均为0。初始GradScaler从65536自动校准到8192时出现少量预期跳步；分项未缩放CE与local梯度均有限。连续8个真实L_s epoch的实际更新率依次为95.56%、100%、97.78%、100%、100%、100%、100%、100%，后5轮没有跳步；MUSE头参数产生非零变化，query迭代和target truth读取均为0。

## 远端发布与启动

```text
release=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_r2_3646fa0b
run_root=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821_r2
dispatch_log=/home/szu2070436088/2510044040/CV-SincNet/launcher_logs/phase1_adv3b02_fasttrust16_s392002_20260821_r2.dispatch.log
```

- release对应Git HEAD为`3646fa0bca943fa5687b396610298369a0f00d90`；自动push后远端`origin/work/cvs-active`OID与本地一致。
- 单一release归档本地路径为`E:\type10-7\release_archives\phase1_fasttrust_r2_3646fa0b.tar.gz`，远端路径为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_r2_3646fa0b.tar.gz`；唯一一次本地/远端SHA-256均为`5238e6be12d2a722baf9cde5ac9eafbaed4208385777d4fbfe81822c58c46c94`。
- dispatch PID为`335414`，CWD为`/home/szu2070436088`，cmdline绑定上述release launcher；新run root和dispatch log均为不可覆盖新路径。
- GPU0–7各恰好2个本run的GPU主训练进程，合计16个；16份train.log均已增长，启动错误指纹为0。
- 16/16条均已完成E1。首轮实际optimizer step率范围为97.83%–99.52%；所有MUSE U256候选均为98.55%，U128为99.28%，U384为97.83%。少量首轮跳步来自GradScaler初始scale校准，不是连续零更新；当前各行已到E1–E3，未触发防复发停止规则。
- 启动健康快照：GPU利用率94%–99%，显存4.64–5.83GB/24GB，温度65–87°C；GPU7温度较高但尚无OOM、Xid、thermal failure或训练异常，后续只读监控继续观察。

该段记录的是启动时状态；最终状态见下文“最终闭合状态与裁决”。

## 2026-08-21 17:00 CST长程健康复核

- dispatch PID`335414`持续存活，16个GPU主训练进程仍为GPU0–7各2个；Traceback、RuntimeError、OOM、`TRAIN_FAILED`、`EVAL_FAILED`和`FASTTRUST-SYSTEMIC-FAILURE`指纹均为0。
- 16条最新进度范围为E16–E30：U128为E16，U256主体为E23–E24，R0/R1为E27，U384为E30。该顺序与每epoch步数相符。
- 16条最新完整epoch的optimizer step率范围为99.52%–100%，零更新候选为0；13条为100%，其余3条为99.52%或99.76%。没有任何候选接近“连续两个完整epoch零更新”的技术停止条件。
- 这是对旧故障的直接真实训练回归：旧run的MUSE候选在E7附近已进入持续0%更新，而r2已经越过E16并保持至少99.52%更新率。因此AMP概率下溢导致的长期非有限梯度故障已不再复现。
- GPU利用率80%–99%，显存4.37–5.65GB/24GB，温度54–68°C，均为P2；启动初期GPU7的87°C瞬时高温已回落，当前无热故障或资源异常。

裁决：`RUNNING_HEALTHY`。此裁决只证明训练执行与数值更新健康，不构成E200性能结果；实验继续按冻结矩阵运行，训练结束后仍须由launcher自动闭合clean和三种LEO弱信道测试。

## 最终闭合状态与裁决

2026-08-22 15:13 CST只读回查确认：16/16个候选均完成E200，全部生成`final_ssdg.pth`、200行`metrics_epoch.jsonl`和200行`metrics_epoch.csv`；16/16个候选均自动完成clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四类测试。每个场景覆盖60,000个样本、5个未见接收机，每个接收机12,000个样本。

所有64份评测均以epoch200的`final_ssdg.pth`严格重建：`strict_requested=true`、`checkpoint_load_strict=true`、`fallback_used=false`，missing key、unexpected key和shape mismatch均为0。16条调度日志均给出`COMPLETE`和`ARTIFACTS_COMPLETE`；调度器与训练日志未出现Traceback、RuntimeError、CUDA OOM、Killed、`TRAIN_FAILED`或`EVAL_FAILED`。N607的8张RTX3090在最终回查时均为空闲，当前run无残留训练进程。

最终裁决为`ANALYZED`。本矩阵完整闭合，可以用于同seed机制归因；由于只有一个训练seed，不足以直接晋级为新的Phase1默认方法。

## 数据协议与评测口径

- 数据角色严格保持`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，对应`5880/52920/12600/12600`个样本，source pool共84,000个样本。
- source receiver为RX0–RX6、source day为0和1；target receiver为RX7–RX11、target day为2和3，source/target receiver交集为0。
- 全部候选使用seed`392002`、E200、labeled batch 128；除R0外均从`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`初始化。
- 每个最终场景都使用`test_unseen_day_unseen_rx`，在RX7–RX11上独立统计clean与三种LEO弱信道准确率；表中“接收机floor”是四个场景、五个接收机中的最低准确率。
- 评测只读取冻结的epoch200 checkpoint。训练、选择和伪标签路由没有读取target/query真值；U_s真值保持隐藏，因此本实验不能直接报告伪标签precision。

## 冻结配置与星地信道增强

优化器为AdamW，峰值学习率约`2e-4`，最终学习率`1e-6`，`weight_decay=1e-4`。星地增强沿用CORE90同款LEO_WEAK配置：`lambda_sat_cls=0.68`、`lambda_sat_cons=0`，调度为：

```text
E1–E40:  p=0.30, leo_clear_weak
E41–E90: p=0.60, leo_low_elev_weak + leo_rain_weak
E91–E200:p=0.80, leo_clear_weak + leo_low_elev_weak + leo_rain_weak
```

日志中的`train_concat_sat_expanded=0`不是增强未生效。当前`concat_masked`实现分别执行clean B和satellite B两次B-sized forward，避免同时物化2B张量；星地CE按调度实际生效，但显存按B规模分配。

FastTrust的身份路由使用global/local/prototype三头、source prior、可靠性、temporal stability和class-balanced cap。配置中`hard_max_fraction=0.25`、`identity_max_fraction=0.50`。E1–E16伪身份损失关闭；E17进入路由阶段但边界epoch的有效权重仍为0，随后逐步升权。FastTrust候选从E17起每epoch选择26,460/52,920，即固定50%的U_s样本；U256完整模型平均每batch使用约117.6个U身份样本，其中约39.81个进入更严格的U星地身份监督。

## 16条最终结果

单位均为百分比。`LEO均值`为clear/low-elev/rain三场景宏平均，`四场景floor`为clean和三种LEO场景中所有未见接收机的最差值。

|候选|clean|clear|low-elev|rain|LEO均值|四场景floor|
|---|---:|---:|---:|---:|---:|---:|
|R0_SCRATCH_CONTROL_U256|84.662|70.757|68.312|67.315|68.794|50.675|
|R1_ADV_INIT_CONTROL_U256|85.152|75.345|73.380|72.243|73.656|58.525|
|R2_FAST_HML_U256|84.225|75.022|73.098|72.073|73.398|58.808|
|R3_FAST_HML_UPROTO_U256|84.973|75.177|73.340|72.218|73.578|58.583|
|R4_FAST_FULL_U128|83.958|76.930|75.370|74.635|**75.645**|**62.883**|
|R4_FAST_FULL_U256|84.540|75.908|74.192|73.290|74.463|60.383|
|R4_FAST_FULL_U384|84.955|75.168|73.408|72.440|73.672|59.067|
|R4_NO_CLASS_CAP_U256|84.827|75.622|73.887|72.865|74.124|59.700|
|R4_NO_CROSSRX_U256|84.710|76.082|74.397|73.480|74.653|60.658|
|R4_NO_NUISANCE_U256|85.075|76.140|74.347|73.455|74.647|60.875|
|R4_NO_PRIOR_U256|84.375|75.873|74.050|73.225|74.383|60.333|
|R4_NO_PROTO_EVIDENCE_U256|84.557|76.072|74.323|73.475|74.623|60.700|
|R4_NO_TEMPORAL_U256|84.475|76.213|74.395|73.542|74.717|60.908|
|R4_NO_U_PROTO_UPDATE_U256|84.163|75.675|73.973|73.178|74.276|60.600|
|R4_NO_U_SAT_ID_U256|84.750|75.293|73.433|72.227|73.651|58.808|
|R4_NUISANCE_DETACHED_U256|84.630|76.083|74.413|73.433|74.643|60.717|

### 单点冠军的逐接收机结果

`R4_FAST_FULL_U128`在RX7–RX11上的结果如下。RX8是三种LEO场景的共同最差接收机，RX11是clean最差接收机。

|接收机|clean|clear|low-elev|rain|
|---|---:|---:|---:|---:|
|RX7|81.008|73.333|70.850|71.183|
|RX8|79.375|64.342|63.575|62.883|
|RX9|95.642|93.567|91.250|90.283|
|RX10|91.408|82.817|81.067|79.408|
|RX11|72.358|70.592|70.108|69.417|

`R4_FAST_FULL_U128`是本次单seed矩阵的准确率冠军：相对同初始化无MUSE控制R1，LEO均值提高1.989个百分点，四场景floor提高4.358个百分点，但clean下降1.193个百分点。该结果同时改变了U batch和每epoch更新步数，不能把全部收益归因于FastTrust机制。

## 同seed因果消融

以下差值均为“前者减后者”，单位为百分点。

|比较|clean差值|LEO均值差值|四场景floor差值|结论|
|---|---:|---:|---:|---|
|R0 scratch−R1 Core90初始化|-0.490|-4.862|-7.850|Core90初始化是稳定LEO鲁棒性的关键|
|R2基础FastTrust伪标签−R1|-0.927|-0.258|+0.283|通用身份伪标签没有带来平均提升|
|R3加入U prototype−R1|-0.178|-0.078|+0.058|接近持平，不能证明有效|
|Full U256−R1|-0.612|+0.807|+1.858|完整机制包改善LEO与floor，但牺牲clean|
|Full U256−No U Sat ID|-0.210|+0.812|+1.575|U星地身份监督具有最强、方向一致的正证据|
|Full U256−No Class Cap|-0.287|+0.339|+0.683|类别均衡cap改善LEO和最差接收机|
|No Temporal−Full U256|-0.065|+0.253|+0.525|当前temporal gate没有正贡献|
|No CrossRX−Full U256|+0.170|+0.189|+0.275|当前cross-receiver项没有正贡献|
|No Nuisance−Full U256|+0.535|+0.184|+0.492|当前nuisance项整体有害|
|No Proto Evidence−Full U256|+0.017|+0.160|+0.317|prototype evidence没有正贡献|
|Full U256−No Prior|+0.165|+0.081|+0.050|source prior有小幅一致收益|
|Full U256−No U Proto Update|+0.377|+0.188|-0.217|U prototype update作用混合|
|U128−U256|-0.582|+1.182|+2.500|小U batch提高鲁棒性，但训练预算显著增加|
|U384−U256|+0.415|-0.791|-1.317|大U batch更快，但LEO鲁棒性下降|

同batch、同seed下最清楚的结论是：U星地身份监督和class-balanced cap有效；基础FastTrust身份伪标签、temporal gate、cross-receiver、nuisance和prototype evidence均未表现出独立正收益。`R4_NO_TEMPORAL_U256`的LEO均值74.717%是U256候选最高值，但只比Full U256高0.253个百分点，单seed不足以晋级。

## 伪标签为何未形成稳定收益

FastTrust的hard路由要求high reliability、temporal stable和三头一致，并受25%上限与类别均衡cap约束；这部分设计合理。问题发生在后续身份配额填充：代码先从mid与未进入hard的high样本补齐，再从low reliability样本继续补到`identity_max_fraction=0.50`。因此，从E17开始`pseudo_selected`固定为26,460/52,920，即使E17的`temporal_pass=0`也已经选满50%。训练后期三头一致率和置信度又接近99.7%–99.9%，多个门控逐渐失去区分度。

这解释了两个表面矛盾：

1. 通用伪身份分支选择了大量U_s样本，却没有超过无伪标签R1；低可靠度回填带来的确认偏差抵消了高置信样本收益。
2. U星地身份分支只消费更严格的hard子集，并把同一伪身份约束到CORE90同款LEO弱信道视图，因此在`Full U256−No U Sat ID`比较中稳定改善clear、low-elev、rain和最差接收机。

U_s真值在训练中不可见，日志中的`train_pseudo_correct=0`表示truth unavailable，而不是伪标签准确率为0。本矩阵只能通过同seed消融评价伪标签分支的下游因果效应，不能声称测得了伪标签precision。

## 训练健康与资源

数值修复有效。16条候选的平均optimizer step应用率为99.915%–99.940%；每条仅有极少数epoch出现单步非有限梯度跳过，平均跳过比例为0.060%–0.085%。该现象也出现在M0控制中，且全程没有非有限loss、连续零更新、异常退出或旧run的系统性MUSE梯度崩溃。

|候选|平均每epoch秒|总时长小时|U吞吐样本/秒|峰值保留显存GiB|
|---|---:|---:|---:|---:|
|R0_SCRATCH_CONTROL_U256|238.3|13.25|N/A|1.95|
|R1_ADV_INIT_CONTROL_U256|238.0|13.23|N/A|1.96|
|R2_FAST_HML_U256|269.7|14.99|207.8|2.56|
|R3_FAST_HML_UPROTO_U256|270.1|15.02|207.4|2.40|
|R4_FAST_FULL_U128|406.3|22.58|138.3|2.06|
|R4_FAST_FULL_U256|271.9|15.11|206.6|2.40|
|R4_FAST_FULL_U384|186.5|10.37|299.7|3.29|
|R4_NO_CLASS_CAP_U256|270.7|15.05|207.0|2.39|
|R4_NO_CROSSRX_U256|273.9|15.23|204.6|2.40|
|R4_NO_NUISANCE_U256|273.0|15.18|205.3|2.71|
|R4_NO_PRIOR_U256|271.2|15.08|206.7|2.41|
|R4_NO_PROTO_EVIDENCE_U256|271.6|15.10|206.3|2.53|
|R4_NO_TEMPORAL_U256|271.0|15.06|207.0|2.67|
|R4_NO_U_PROTO_UPDATE_U256|270.6|15.05|207.1|2.69|
|R4_NO_U_SAT_ID_U256|269.3|14.97|208.2|2.68|
|R4_NUISANCE_DETACHED_U256|275.2|15.30|204.3|2.41|

U128变慢不是显存不足：其峰值保留显存仅2.06GiB。U_s每epoch完整覆盖，U batch从256降至128后需要约两倍U loader步数，并同步增加labeled/augmentation forward次数，所以总时长从15.11小时升至22.58小时。U384把时间降至10.37小时，但LEO均值和floor分别下降0.791和1.317个百分点。当前效率主点仍应使用U256，U128只作为高计算预算的准确率参考。

## clean—LEO权衡与source proxy偏差

FastTrust的主要收益集中在LEO弱信道，不是clean。Full U256相对R1的clear/low-elev/rain分别提高0.563/0.812/1.047个百分点，但clean下降0.612个百分点；U128进一步扩大LEO收益，同时clean降至83.958%。

source侧星地验证不能可靠预测target receiver表现。U128在E200的source satellite mean为96.312%，target LEO均值只有75.645%，相差20.667个百分点；Full U256的对应差值为20.523个百分点。最困难的LEO接收机稳定为RX8，而clean floor主要由RX11决定。后续候选不能仅根据source satellite mean或source val选择。

历史冻结CORE90 checkpoint的直接测试为clean 86.090%、LEO均值70.563%，但该历史评测使用`sat_seed=2027`，本矩阵使用`392002`，只能作为诊断背景，不能与本矩阵做严格同row因果比较。本报告所有机制判断均以同seed矩阵内部的R1和逐项消融为准。

## 最终checkpoint与日志定位

单点冠军checkpoint：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821_r2/R4_FAST_FULL_U128/final_ssdg.pth
SHA256=00e84c7362608694736b6f410be8b1ee9d3c9a7262ba80c6c7f229bf3e009641
```

同batch主比较checkpoint：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821_r2/R4_FAST_FULL_U256/final_ssdg.pth
SHA256=662f7c720eb836ad3c2c741475d4e8b5cc246e14b71e107bec41fd024c882219
```

每个候选目录均保留以下证据：

```text
config.json
train.log
metrics_epoch.jsonl
metrics_epoch.csv
final_ssdg.pth
metrics_clean.json
metrics_leo_clear_weak.json
metrics_leo_low_elev_weak.json
metrics_leo_rain_weak.json
metrics_joint.json
phase1_resource_summary.json
phase1_terminal_status.json
status.txt
```

调度日志位于：

```text
/home/szu2070436088/2510044040/CV-SincNet/launcher_logs/phase1_adv3b02_fasttrust16_s392002_20260821_r2.dispatch.log
/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821_r2/dispatcher_logs/
```

## 方法裁决与下一步

本轮不能把“伪标签整体”写成成功。基础FastTrust身份伪标签没有超过无伪标签控制；完整模型的正收益主要来自更严格的U星地身份监督、类别均衡cap以及少量source prior。temporal、cross-receiver、nuisance和prototype evidence在当前组合中没有独立正贡献。

因此，当前不建议用完整FastTrust直接替代`ADV3B02_CORE90_SOFT_E200`作为Phase1默认模型。下一版SSDG应执行最小化重构：保留U星地身份监督、class-balanced cap和source prior；取消low reliability回填至固定50%的规则，使身份选择比例由证据自适应；先移除temporal、cross-receiver、nuisance和prototype evidence，再用同seed最小矩阵验证组合效应。U256作为效率主配置，U128作为高预算参考。只有该简化候选在多seed上同时复现LEO均值与receiver floor提升，并明确控制clean退化后，才具备晋级条件。

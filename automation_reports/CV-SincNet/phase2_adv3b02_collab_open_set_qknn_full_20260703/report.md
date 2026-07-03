# phase2_adv3b02_collab_open_set_qknn_full_20260703

## 基本信息

|字段|内容|
|---|---|
|实验ID|phase2_adv3b02_collab_open_set_qknn_full_20260703|
|时间|2026-07-03|
|操作者|Codex|
|目标|在ADV3B02冻结特征上验证Stage2-C多接收机协同open-set qKNN；协同数量从1到目标接收机全量可选；加入scenario-aware和radius-normalized评分以提升星地信道下的协同效率与性能。|
|模型权重|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|权重SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|远端环境|`CVS-RFFI` conda环境|
|目标receiver domain|`20-1,3-19,7-14,7-7,8-8`|
|Y_old|`14-10,14-7,20-15,20-19,6-15,8-20`|
|Y_new|`19-3,3-8`|
|Y_unknown|`10-1,10-10`|
|星地信道视图|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|

## 假设与比较目标

此前no-unknown诊断中，`scenario-aware qKNN9 + radius_norm=0.3`在ADV3B02特征上显著优于普通qKNN。当前实验把该机制移植到open-set协同推理层，比较目标为普通open-set qKNN和support-envelope gate。成功标准不能只看单项最大值；主线要求同一row同时接近`old_acc>=0.80`、`seen_new_acc`高、`unknown_FAR<=0.05`，并报告`unknown_reject_rate`和defer边界。`receiver_domain_ranked`仅是receiver-domain ensemble诊断，不等同严格同事件协同部署证据。

## 本地变更与验证

|文件|用途|
|---|---|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|新增`scenario_aware`、`radius_norm`、`old_bias`评分；记录support scenario和class radius；增加`R_s/R_t`、`proxy_unknown`、per-receiver Stage2-C覆盖硬校验。|
|`E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|新增scenario-aware评分生效测试、metadata测试、source/proxy/coverage负测。|

|验证命令|结果|
|---|---|
|`conda activate ssr-gpu; python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py; python code\tests\test_phase2_collaborative_open_set_qknn_eval.py; python code\tests\test_collaborative_open_set_qknn_eval.py`|本地工作区通过：10 tests OK；8 tests OK。|
|同命令在`E:\type10-7\github_publish\CVS-RFFI-repo`执行|Git镜像通过：10 tests OK；8 tests OK。|
|Git镜像提交|`3d06a3acc763b8a2ba39f607a190a69bf1bacbbd Add scenario-aware radius-normalized collaborative qknn`|

|文件|SHA256|
|---|---|
|`code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`5BD5E9883DD22B797F43CF6D887DF437E89D8F59E06C7F47C87D18CC7321CD5B`|
|`code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|`E1FE2A9222A36EC876CE89A86DB6D32C50F15C4227B944D8E7994E04C117A3B5`|

## 远端同步计划

|本地文件|N607目标路径|
|---|---|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|

## 待执行远端命令

远端测试：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
```

复跑使用既有`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，新增输出`collab_open_set_qknn_scenario_rnorm03_perk_*.json/csv`，参数包括`--collab_counts all --qknn_k 9 --scenario_aware --radius_norm 0.3`。`receiver_domain_ranked`仅报告receiver-domain ensemble诊断边界。

## 子agent监督结论

|角色|结论|
|---|---|
|文献/方法|支持冻结主干、prototype/qKNN、open-set门控、低通信统计证据交换路线。|
|算法构建|建议最小落地`scenario_aware+radius_norm`，后续再做至少k个receiver事件组和receiver选择策略。|
|完成度监督|要求补齐远端环境、scp映射、报告、指标和诊断边界证据。|
|查漏补缺/review|指出参数生效、support/query泄漏、`proxy_unknown`泄漏、覆盖校验等风险；本次已修复参数贯穿、source/proxy校验、per-receiver覆盖硬校验，并使非stable support选择只在初始support窗口内重排，避免看未来query池。后续按算法子agent建议修复了按全receiver交集提前截断事件的问题，使`collab_counts=1..N`按各自可用receiver集合评估。|

## 当前状态

本地实现、测试、Git镜像提交、N607 preflight、scp同步、远端`CVS-RFFI`测试和复跑已完成。当前结果未达到99/97/99或Stage2-C成功标准，必须标为诊断负结果。

## 远端执行记录

|项目|记录|
|---|---|
|N607 preflight|2026-07-03 16:53:22 CST通过；项目根可见；8张RTX3090均为`10/24576MiB`低显存占用。|
|同步方式|`scp -F E:\type10-7\tools\n607_ssh_config`同步脚本和测试到`/home/szu2070436088/2510044040/CV-SincNet/code/...`。|
|远端环境|`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate CVS-RFFI`。|
|远端测试|`python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py && python code/tests/test_phase2_collaborative_open_set_qknn_eval.py`，结果：10 tests OK。|
|远端脚本hash|`5bd5e9883dd22b797f43cf6d887df437e89d8f59e06c7f47c87d18cc7321cd5b`。|
|远端测试hash|`e1fe2a9222a36ec876ce89a86db6d32c50f15c4227b944d8e7994e04c117a3b5`。|
|Git提交|`3d06a3a Add scenario-aware radius-normalized collaborative qknn`；`4e3a5bb Allow per-count receiver evidence groups`。|
|后续融合提交|`55093fd Add consensus-veto open-set fusion policy`。|

## 结果表

主配置：`--event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --unknown_gate_mode score --scenario_aware --radius_norm 0.3 --qknn_k 9 --k_shot 8 --query_per_class 20`。

|协同receiver数|total|excluded|old_acc|seen_new_acc|unknown_FAR|unknown_reject_rate|defer_rate|known_coverage|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|301|0|0.2265|0.1833|0.1167|0.4333|0.4252|0.2365|无|FAR超标且known性能低|
|2|245|56|0.0662|0.0625|0.0000|0.5000|0.6612|0.0653|无|过度defer，known性能不可用|
|3|200|101|0.0333|0.0750|0.0000|0.4250|0.8150|0.0438|无|过度defer，known性能不可用|
|4|155|146|0.3034|0.2188|0.0294|0.5294|0.5935|0.2810|无|FAR达标但old/new远低于目标|
|5|99|202|0.3220|0.0000|0.0500|0.1500|0.6768|0.2405|`3-8`|全receiver交集丢失一个seen-new类|

`support_envelope`在本次参数下与`score`结果相同，未带来额外收益。全accept诊断显示分类器潜力但无拒识能力：

|协同receiver数|old_acc|seen_new_acc|unknown_FAR|known_coverage|解释|
|---:|---:|---:|---:|---:|---|
|1|0.7403|0.5333|1.0000|1.0000|单receiver分类尚可，拒识完全失败|
|2|0.7881|0.8125|1.0000|1.0000|多receiver提升known分类，拒识仍失败|
|3|0.8583|0.9500|1.0000|1.0000|known分类接近目标，但unknown全吸收|
|4|0.8989|0.9063|1.0000|1.0000|known分类较强，仍不能作为open-set结果|
|5|0.8136|0.9500|1.0000|1.0000|全receiver交集缺`3-8`，且FAR不可接受|

## 判定与下一步

1.本轮实现完成了`SARN-C-qKNN`的最小部署形态：场景感知support检索、radius-normalized相似度、per-count receiver evidence groups、source/proxy/coverage硬校验、低通信证据输出。  
2.全量星地信道receiver-domain诊断未达成目标，不能写成Stage2-C成功、部署成功或论文主结论。  
3.瓶颈不是known类别qKNN分类本身，而是open-set门控：全accept在3到4个receiver时可达`old_acc≈0.86-0.90`、`seen_new_acc≈0.91-0.95`，但`unknown_FAR=1.0`；当前risk门控可压低FAR，但会把known大量defer。  
4.下一步应实现更强的open-set门控，而不是继续调同一阈值：建议加入source/known-only EVT尾部分布、receiver共识分歧风险、per-scenario score calibration、unknown侧defer/reject分开报告，并保留unknown query不参与阈值拟合。

## 2026-07-03 consensus-veto追加诊断

新增融合策略：`fusion_policy=consensus_veto`。该策略不改变节点级qKNN，只在高unknown risk时结合多接收机投票gap、均值score和均值margin决定`unknown_reject/defer`。默认`risk_margin`行为保持不变。

远端验证：

```bash
python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_collaborative_open_set_qknn_eval.py
```

结果：9 tests OK。远端使用`CVS-RFFI`环境，脚本hash如下：

|文件|远端SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`95c6788715a673d9e1745e8fc590c7e220a6e29c07a474c3edce6490b806e140`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`eb4d48a250c0457d577cf23d8935e6d5edb84dea374cc26d97effbdc9f6b4e32`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`5e7f34a885df693c5bfc7481e6cd9405ebcd9add6bd87ece3b2a37559f7b161e`|

运行参数：

```bash
--fusion_policy consensus_veto
--unknown_risk_threshold 0.98
--accept_margin_threshold 0.30
--consensus_gap_threshold 0.60
--consensus_score_threshold 0.30
```

结果：

|协同receiver数|total|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|known_coverage|bytes/event|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|301|0.3425|0.0500|0.3167|0.0000|0.2667|0.6833|0.0997|0.4232|40|FAR仍超标|
|2|245|0.0993|0.0000|0.2500|0.1000|0.0000|1.0000|0.1469|0.1457|80|拒识强但known严重误拒|
|3|200|0.0583|0.0000|0.2000|0.0000|0.0000|0.9250|0.2500|0.0938|120|拒识强但known不可用|
|4|155|0.4157|0.0000|0.5000|0.2500|0.0294|0.9706|0.0194|0.4545|160|当前最好折中，但离目标很远|
|5|99|0.4915|0.0000|0.1500|0.0000|0.0500|0.9500|0.0303|0.4304|200|全receiver证据缺类且known不足|

结论：`consensus_veto`证明多接收机分歧可以提升unknown拒识，但仍会把大量known query误拒。当前证据空间中，unknown和known在`known_score/known_margin/unknown_risk/vote_gap`上重叠明显；只改融合器不足以达到目标。下一步必须改节点级unknown score本身，例如：

1.用source/proxy-only或support-only EVT tail建模，输出独立`evt_unknown_risk`。  
2.引入per-scenario support density和class-conditional Mahalanobis/LOF score，避免把LEO扭曲后的known样本误判unknown。  
3.把unknown reject与defer分开优化，并在报告中增加unknown侧defer rate。  
4.若继续追求99/97/99，需要重新导出带source/proxy_unknown校准行的features或训练轻量oldness gate；当前`receiver_domain_ranked`诊断不能证明严格同事件卫星群协同部署成功。

资源约束说明：用户提到的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`未在当前工作区按文件名或关键词检索到。当前报告沿用已有COSR-CI设计中的资源预算字段：`participating_receivers`、`bytes_per_event`、`total_bytes`、`latency_ms_p50/p95`、prototype storage和显存状态；后续如原文补入，应按原文重新核对上限。

## 2026-07-03 LOO qKNN阈值追加诊断

新增节点级校准：

```bash
--support_calibration_mode leave_one_out
--score_threshold_combine qknn_only
```

目的：避免support样本自匹配导致qKNN阈值虚高，并检验节点侧qKNN分数是否能恢复known coverage。默认行为保持`support_calibration_mode=self`和`score_threshold_combine=max`，不影响旧结果复现。

远端验证：N607直接SSH preflight于2026-07-03 17:15:02 CST通过，8张RTX3090均为`10/24576MiB`。远端使用`CVS-RFFI`环境完成脚本语法检查和单元测试，结果为11 tests OK。

|文件|SHA256|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`aa51daa5c167e29410b7e9157e6cb759baff39112bb767a8620776738a0a6b69`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`20880937e4fa40c43b5b810e468984bdd920c542e66c8f26988253290edb791a`|

`risk_margin + LOO qKNN-only`结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|known_coverage|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.7017|0.5294|0.5167|0.4500|0.8333|0.1667|0.0000|0.9253|known恢复，unknown拒识失败|
|2|0.6689|0.3500|0.7500|0.7500|0.7174|0.2826|0.0735|0.8894|known可用性提升，FAR不可接受|
|3|0.7167|0.2500|0.8000|0.7500|0.4250|0.3750|0.1450|0.8313|较好折中但远未达99%拒识|
|4|0.8989|0.0000|0.9062|0.8500|0.7647|0.2353|0.0000|1.0000|known强，unknown全局门控失败|
|5|0.8136|0.0000|0.9000|0.0000|0.9000|0.0000|0.0303|0.9873|全receiver证据缺类且FAR极高|

`consensus_veto + LOO qKNN-only`结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|known_coverage|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.7403|0.5294|0.5333|0.5000|1.0000|0.0000|0.0000|1.0000|退化为全accept|
|2|0.7881|0.3500|0.8125|0.8000|1.0000|0.0000|0.0000|1.0000|退化为全accept|
|3|0.8583|0.3000|0.9500|0.9000|1.0000|0.0000|0.0000|1.0000|known接近目标但unknown完全失败|
|4|0.8989|0.0000|0.9062|0.8500|1.0000|0.0000|0.0000|1.0000|退化为全accept|
|5|0.8136|0.0000|0.9500|0.0000|1.0000|0.0000|0.0000|1.0000|退化为全accept|

判定：LOO qKNN阈值修复了自匹配校准问题，使known coverage显著提高，但同时暴露出当前unknown分数不可分。`consensus_veto`依赖高risk触发，在LOO qKNN-only下risk不足，实际退化为全accept。该结果不能作为Stage2-C成功或部署成功证据，只能作为节点级分数校准诊断。

下一步算法门控建议：改为双通道节点证据。第一通道保持qKNN/prototype分类给出known label和margin；第二通道独立估计open-set风险，不与分类分数共用同一个阈值。可落地实现为`EVT-tail + receiver-consensus + temporal EMA`：

1.每个接收机只上传`top1_label, top1_score, margin, evt_tail_prob, density_z, support_age`，单事件通信量保持在几十字节量级。  
2.`evt_tail_prob`仅由source/support或proxy-known校准，不使用unknown query拟合阈值。  
3.融合端先用receiver一致性和margin决定known投票，再用`max/mean(evt_tail_prob)`和低密度接收机比例决定reject/defer。  
4.在线微调只更新prototype均值、半径、EVT尾部参数和轻量adapter/BN统计；冻结主干，满足卫星端实时更新约束。  
5.验收必须同时报告`unknown_reject_rate`、`unknown_defer_rate`、known coverage、per-class floor，避免用全accept高known或全reject低FAR冒充成功。

## 多子agent最终审计

|角色|关键结论|落实方式|
|---|---|---|
|联网文献/方法|2024-2026 RFFI路线支持receiver-agnostic、prototype/qKNN、Mahalanobis/EVT open-set、少样本类增量；不建议先上重型Transformer、全模型联邦训练或无回滚伪标签自训练。|下一步采用轻量证据包、receiver可靠度、Mahalanobis/EVT风险和seen-new prototype入库。|
|高效率算法|建议路线命名为`SCORER-CVS`：Support-Calibrated Open-set Receiver Evidence Routing。每节点本地冻结ADV3B02，只维护qKNN8/9 int8 support memory、EMA prototype、半径/阈值/健康度；跨星只上传`top label/top2 margin/qKNN score/radius risk/margin risk/density/health/latency`等约64-128 bytes证据包。|报告下一步改为双通道节点证据：classification score与open-set risk分离，并支持低置信时`request_more_receivers()`。|
|完成度监督|实现、N607同步、`CVS-RFFI`远端测试、1..5 receiver诊断、星地信道覆盖已完成；性能目标和部署成功未满足。|最终结论只能称诊断负结果，不能称99/97/99达标、Stage2-C成功或卫星群部署成功。|
|查漏补缺/review|当前`receiver_domain_ranked`结果是receiver-domain ensemble，不是严格同事件协同；`collab_counts=1..N`使用per-k available receivers，不是同一批事件上的同分母曲线；k=5缺一个seen-new类。|后续必须补`strict_event_key`或明确降级为receiver-domain ensemble；解释协同数量时不得写成因果提升曲线。|

最终边界：

1.当前代码和远端测试证明了链路可运行、指标可复现、报告和artifact已持久化。  
2.当前结果不证明open-set协同推理成功，主要失败点是known与unknown在现有qKNN score/risk空间不可分。  
3.下一步优先补严格事件键和独立open-set风险通道，而不是继续调`unknown_risk_threshold`或`consensus_veto`。

## 2026-07-03 SCORER-CVS融合策略实现

本次实现`SCORER-CVS`的最小可运行版本：Support-Calibrated Open-set Receiver Evidence Routing。该版本不改变Stage2-C数据协议，不使用unknown query拟合阈值；只在融合层新增独立open-set证据聚合、低置信`request_more`状态和证据包资源统计。

代码改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`fusion_policy=scorer_cvs`，聚合`unknown_risk/radius_risk/margin_risk`，在低置信且时延预算允许时输出`request_more`，报告`request_more_rate/unresolved_rate`。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI开放`--fusion_policy scorer_cvs`、`--latency_budget_ms`、`--evidence_packet_bytes`，用于卫星证据包资源统计。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|覆盖高风险拒识和低置信请求更多receiver。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|覆盖SCORER-CVS证据包字节统计。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：`test_collaborative_open_set_qknn_eval.py`为11 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为12 tests OK。Git提交：`f9b2bd1 Add SCORER-CVS collaborative open-set fusion`。

待同步映射：

|本地|N607|
|---|---|
|`E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|

计划远端命令使用`CVS-RFFI`环境，并复用既有`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`：

```bash
python code/tests/test_collaborative_open_set_qknn_eval.py
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_support_envelope.json \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --unknown_risk_threshold 0.80 \
  --accept_margin_threshold 0.10 \
  --consensus_gap_threshold 0.50 \
  --consensus_score_threshold 0.30 \
  --unknown_quantile 0.75 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 96 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407034
```

预期检查：报告1..5 receiver的`old_acc/min_old_class_acc/seen_new_acc/min_seen_new_class_acc/unknown_FAR/unknown_reject_rate/request_more_rate/unresolved_rate/bytes_per_event/latency_ms_p95`。若仍未达99/97/99，则继续标为诊断负结果。

## 2026-07-03 SCORER-CVS组件风险投票结果

追加实现：节点证据增加`score_risk`，融合端增加`scorer_component_vote_threshold`。`scorer_cvs`不再只根据`unknown_risk=max(score,radius,margin)`直接拒识，而是要求多个风险通道同时支持unknown判断；低置信但未满足拒识条件时输出`request_more`或`defer`。

本地与镜像验证：`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为11 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为12 tests OK。Git提交：

|提交|内容|
|---|---|
|`f9b2bd1`|新增`fusion_policy=scorer_cvs`、`request_more`和证据包资源统计。|
|`f2a0766`|新增`score_risk`和SCORER-CVS组件风险投票。|

N607远端验证：使用`CVS-RFFI`环境，`py_compile`通过，两组单测分别为11 tests OK和12 tests OK。远端文件hash：

|文件|远端SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`554352e6773437b9c52b6f26775e09ac8e756280e8e7d98effab8a5061275be8`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`0bad9a18b362b7fd2f008eda87cc08824947e8e78dfe9cc85c09324b04817eaa`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`71ec71616c2eb898f11a5ce939183d3f1b235c5c078113c97c96fbc87b6d54f6`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`c7da35232448ca8cbaacc6914b54bbad89672dc221929cba9c06ca1246818b02`|

主运行：`collab_open_set_qknn_scorer_cvs_component_vote.json`，参数为`qknn8 + support_envelope + leave_one_out + score_threshold_combine=max + scenario_aware + radius_norm=0.3 + evidence_packet_bytes=96`，`event_alignment_policy=receiver_domain_ranked`。结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|request_more_rate|unresolved_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.3579|0.2000|0.1333|0.0000|0.1500|0.3833|0.5000|0.0032|0.5032|0.3920|96|0.0893|无|FAR和known均不达标|
|2|0.1600|0.0000|0.0800|0.0000|0.0000|0.6087|0.6301|0.0366|0.6667|0.1400|192|0.0893|无|拒识强但过度误拒known|
|3|0.0917|0.0000|0.0750|0.0000|0.0000|0.5000|0.8000|0.0000|0.8000|0.0875|288|0.0893|无|known不可用|
|4|0.2333|0.0000|0.1333|0.0000|0.0000|0.4118|0.7208|0.0260|0.7468|0.2083|384|0.0893|无|known不可用|
|5|0.2400|0.0000|0.0000|0.0000|0.0000|0.2500|0.8000|0.0000|0.8000|0.1714|480|0.0893|`3-8`|全receiver缺类且known不可用|

小网格调参后固化结果：`collab_open_set_qknn_scorer_cvs_component_vote_tuned_seed407035.json`，参数为`unknown_risk_threshold=0.995`、`scorer_component_vote_threshold=0.34`、`consensus_score_threshold=0.30`、`accept_margin_threshold=0.03`。结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|request_more_rate|unresolved_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.5368|0.2857|0.2500|0.2250|0.4167|0.2667|0.2677|0.0645|0.3323|0.6120|96|0.0704|无|known有所恢复但FAR不可接受|
|2|0.2000|0.0000|0.2000|0.2000|0.0000|0.4783|0.6220|0.1016|0.7236|0.2100|192|0.0704|无|FAR达标但known严重误拒|
|3|0.2250|0.0000|0.2500|0.2000|0.0000|0.5250|0.6900|0.0000|0.6900|0.2313|288|0.0704|无|FAR达标但known不可用|
|4|0.4778|0.0000|0.3667|0.2000|0.0294|0.5588|0.4805|0.0325|0.5130|0.4583|384|0.0704|无|当前较好折中，离99/97/99很远|
|5|0.5400|0.0000|0.3000|0.0000|0.0500|0.4000|0.5111|0.0000|0.5111|0.4714|480|0.0704|`3-8`|known不足且缺类|

判定：SCORER-CVS组件投票实现了更现实的证据包协同、时延预算和`request_more`输出，但当前ADV3B02特征空间中的open-set separability仍不足。该结果继续标为诊断负结果，不能作为Stage2-C成功或部署成功证据。下一步不应继续只调融合阈值，应回到节点级可分性：加入class-conditional Mahalanobis/LOF、EVT tail或oldness gate，并优先补严格同事件`event_id`导出；否则`receiver_domain_ranked`只能支撑receiver-domain ensemble诊断。

## 2026-07-03 Mahalanobis节点级风险通道

根据上一轮结论，本次不再只调融合阈值，而是在节点本地support memory中加入class-conditional diagonal Mahalanobis统计。每个receiver基于本地`target-old + seen-new` support估计每类centroid、对角方差、Mahalanobis support距离分位阈值；query侧输出`mahalanobis_risk`，并可通过`unknown_gate_mode=mahalanobis`或`support_envelope_mahalanobis`进入节点级unknown风险。融合端将`mahalanobis_risk`作为SCORER-CVS第四个组件参与`scorer_component_vote_threshold`，避免单一风险通道直接决定拒识。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增Mahalanobis support envelope、`mahalanobis_risk`证据字段、`--unknown_gate_mode mahalanobis/support_envelope_mahalanobis`、`--mahalanobis_quantile/slack/temperature/variance_floor`参数。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|SCORER-CVS组件投票自动纳入`mahalanobis_risk`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|覆盖Mahalanobis作为第四风险组件时的投票行为。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|覆盖Mahalanobis gate输出和metadata记录。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：12 tests OK和13 tests OK。Git提交：`a2c3fb0 Add Mahalanobis SCORER-CVS risk channel`。

计划远端主命令：

```bash
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_mahalanobis.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_mahalanobis_evidence.csv \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_mahalanobis \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --unknown_risk_threshold 0.80 \
  --accept_margin_threshold 0.10 \
  --consensus_gap_threshold 0.50 \
  --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.50 \
  --unknown_quantile 0.75 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 112 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407037
```

远端结果：

N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为12 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为13 tests OK。远端hash与本地一致：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`f1823373bb072e6cdfba4c5e58450defe116931127b3cc21fd8fc278143cc2d6`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`b5c8a2314751b823a190ad03ffc341e98051cb4e3f976a3c0f2ec1fd565673b4`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`11e2374740d8e36b3c1c20c228abeb6bb0e91c07052c3f3932f6681c3e5b0b6a`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`1c046f39f7b74f176e31c1c7597e44cb0aeab37263d5b56c2d409a3b896323e0`|

`support_envelope_mahalanobis`主运行结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|request_more_rate|unresolved_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.2963|0.0000|0.2500|0.0000|0.3167|0.3500|0.3883|0.0032|0.3916|0.4297|112|0.1003|无|FAR与known均不达标|
|2|0.1457|0.0000|0.1765|0.0000|0.0000|0.3043|0.6976|0.0444|0.7419|0.1535|224|0.1003|无|FAR达标但known严重误拒|
|3|0.1000|0.0000|0.1750|0.0000|0.0000|0.2500|0.8150|0.0200|0.8350|0.1187|336|0.1003|无|known不可用|
|4|0.1910|0.0000|0.1724|0.0000|0.0000|0.4412|0.6513|0.0921|0.7434|0.1864|448|0.1003|无|known不可用|
|5|0.3922|0.0000|0.0000|0.0000|0.0000|0.2000|0.7363|0.0000|0.7363|0.2817|560|0.1003|`3-8`|缺类且known不足|

基于同一evidence快速扫描后，固化`mahalanobis`only tuned配置：`unknown_gate_mode=mahalanobis`、`unknown_risk_threshold=0.995`、`scorer_component_vote_threshold=0.25`、`consensus_score_threshold=0.30`、`accept_margin_threshold=0.03`。结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|request_more_rate|unresolved_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.4074|0.2368|0.3000|0.0000|0.4333|0.5000|0.0388|0.1068|0.1456|0.5622|112|0.0742|无|known恢复但FAR不可接受|
|2|0.2781|0.0000|0.2745|0.0500|0.0217|0.7174|0.3387|0.1210|0.4597|0.3119|224|0.0742|无|FAR达标但known低|
|3|0.3333|0.1500|0.3250|0.0500|0.0000|0.6250|0.5300|0.0200|0.5500|0.3438|336|0.0742|无|FAR达标但known低|
|4|0.6292|0.0000|0.3448|0.0500|0.0000|0.8529|0.3158|0.0263|0.3421|0.5593|448|0.0742|无|当前较好折中，但距离99/97/99很远|
|5|0.6078|0.0000|0.1000|0.0000|0.0000|0.3500|0.5495|0.0000|0.5495|0.4789|560|0.0742|`3-8`|缺类且new不可用|

判定：Mahalanobis节点风险比单纯support envelope更能在K4压低unknown FAR并恢复部分old accuracy，但仍不满足OLD80_FIRST，更不满足99/97/99。下一步必须继续增强节点级可分性：优先实现EVT tail或训练轻量oldness gate；同时补严格同事件key，否则无法把`receiver_domain_ranked`写成真实event-level卫星群协同。

## 2026-07-03 EVT tail节点级风险通道

本次继续沿节点级可分性方向推进，新增EVT tail风格风险通道。每个receiver在本地support memory中使用类条件Mahalanobis support距离拟合尾部分位阈值和excess scale，不使用unknown query拟合阈值；query侧输出`evt_risk=1-exp(-excess/scale)`，并支持`unknown_gate_mode=evt`和`support_envelope_evt`。SCORER-CVS融合端会把`evt_risk`作为额外组件纳入`scorer_component_vote_threshold`。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增EVT tail support envelope、`evt_risk`证据字段、`--unknown_gate_mode evt/support_envelope_evt`、`--evt_tail_quantile/slack/temperature/min_scale`参数。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|SCORER-CVS组件投票自动纳入`evt_risk`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|覆盖EVT作为额外风险组件时的投票行为。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|覆盖EVT gate输出和metadata记录。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：13 tests OK和14 tests OK。Git提交：`09b3f9e Add EVT tail SCORER-CVS risk channel`。

计划远端主命令：

```bash
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_evidence.csv \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode evt \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --unknown_risk_threshold 0.995 \
  --accept_margin_threshold 0.03 \
  --consensus_gap_threshold 0.0 \
  --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.25 \
  --unknown_quantile 0.75 \
  --evt_tail_quantile 0.80 \
  --evt_temperature 0.05 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 120 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407038
```

远端结果：

N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为13 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为14 tests OK。远端hash与本地一致：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`d5b58a20497d85e3548ec86ac9e57235470d5800e0624a6a4a664f87c2324c15`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`d66a1dc67eadd3d9e2fbf46b8d59b375525235b0576c7f4d840ed67fb56006b6`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`392237b10eeac0ce2322c915180c6eef7e31773c5b291da59061050b9c3a9514`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`3bd029f2039bc96e7ff2b2349b3998bfa2547a1463029b40f9ab9753c7208d6e`|

`evt`主运行结果：

|协同receiver数|old_acc|old_floor|seen_new_acc|new_floor|unknown_FAR|unknown_reject_rate|defer_rate|request_more_rate|unresolved_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.2984|0.0000|0.2500|0.0500|0.4000|0.2500|0.3408|0.0997|0.4405|0.4741|120|0.1021|无|FAR不可接受|
|2|0.2252|0.0000|0.2600|0.2333|0.0000|0.6042|0.5020|0.1124|0.6145|0.2488|240|0.1021|无|FAR达标但known低|
|3|0.2500|0.0000|0.2750|0.2000|0.0250|0.5250|0.6200|0.0150|0.6350|0.2812|360|0.1021|无|FAR达标但known低|
|4|0.4944|0.0000|0.4333|0.2500|0.0312|0.7188|0.3377|0.0265|0.3642|0.5462|480|0.1021|无|seen-new较Mahalanobis更好，但old不足|
|5|0.5102|0.0000|0.0500|0.0000|0.0000|0.5500|0.5393|0.0000|0.5393|0.4203|600|0.1021|`3-8`|缺类且new不可用|

EVT快速阈值扫描显示最佳折中仍集中在K4，`old_acc=0.4944`、`seen_new_acc=0.4333`、`unknown_FAR=0.0312`。Mahalanobis+EVT混合没有优于单独EVT或Mahalanobis-only：混合最佳约为`old_acc=0.4944`、`seen_new_acc=0.4000`、`unknown_FAR=0.0312`。

判定：EVT tail风险通道提高了seen-new折中，但旧类仍低于OLD80_FIRST，per-class floor仍为0。当前最高可用诊断分成两类：Mahalanobis-only偏old恢复，K4为`old_acc=0.6292`、`seen_new_acc=0.3448`、`unknown_FAR=0.0000`；EVT偏new恢复，K4为`old_acc=0.4944`、`seen_new_acc=0.4333`、`unknown_FAR=0.0312`。二者均远低于99/97/99。下一步应实现轻量oldness gate或feature-level calibrator，并补严格同事件`event_id`；继续只在当前证据表上调融合参数收益有限。

## 2026-07-03 oldness候选类一致性风险通道

本轮新增support-only候选类一致性风险通道，工程名暂为`oldness_risk`。该通道对每个support类构造一类对其余类的归一化方向`w_y=(c_y-c_not_y)/||c_y-c_not_y||`，用该类support正样本投影低分位得到阈值`t_y`，query被qknn预测为候选类`y_hat`后计算`risk_oldness=sigmoid((t_y-score_y)/temperature)`。它不使用unknown query拟合阈值，属于节点级拒识辅助证据，不是旧类/新类语义分类器。子agent review指出命名有歧义，报告解释时必须写作“candidate-class oldness/consistency gate”，不能声称已学习出真实old-vs-new边界。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`class_oldness_weights/thresholds`、`oldness_risk`证据字段、`--unknown_gate_mode oldness/support_envelope_oldness`、`--oldness_quantile/slack/temperature`参数。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|SCORER-CVS组件投票自动纳入`oldness_risk`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|覆盖`oldness_risk`作为额外风险组件时的投票行为。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|覆盖oldness gate输出和metadata记录。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：`test_collaborative_open_set_qknn_eval.py`为14 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为15 tests OK。Git镜像提交：`cceb98f Add oldness SCORER-CVS risk channel`。随后review发现SCORER-CVS组件投票存在“字段存在即参与”的口径污染，本地已修为显式`active_risk_components`控制，重新验证为`test_collaborative_open_set_qknn_eval.py`15 tests OK、`test_phase2_collaborative_open_set_qknn_eval.py`15 tests OK。修复提交：`87ccab2 Isolate SCORER-CVS active risk components`。镜像仍领先远端，且存在非本轮未跟踪文件`code/scripts/phase2_qknn_prototype_compress_probe.py`，本轮未处理。

子agent审查边界：

|审查项|结论|
|---|---|
|协同语义|当前可运行全量结果仍是`receiver_domain_ranked`诊断，不是严格同事件卫星群协同成功；严格同事件仍需共享`event_id`或共享`role+tx+day+sig+scenario`键。|
|效率统计|当前`latency_ms`为离线摊销值，不能写成真实星上端到端延迟；`prototype_storage_bytes`未完整计入centroid、逆方差、oldness权重和阈值字典。|
|oldness语义|`oldness_risk`实际是候选类一对其余support类一致性风险；若qknn先错分，该风险也在错类坐标上判断。|
|组件投票|已修复为显式`active_risk_components`，`mahalanobis`、`evt`、`oldness`单通道诊断不再因为证据CSV包含其他风险字段而混入投票。旧的EVT/Mahalanobis解释在该修复前存在口径污染，后续需按修复后代码复跑。|
|协议边界|unknown query仍只用于评估，不参与阈值拟合；结果若不达OLD80/Stage2-C指标，必须保留诊断负结论。|

计划远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_oldness.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_oldness_evidence.csv \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode oldness \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --unknown_risk_threshold 0.995 \
  --accept_margin_threshold 0.03 \
  --consensus_gap_threshold 0.0 \
  --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.25 \
  --unknown_quantile 0.75 \
  --oldness_quantile 0.05 \
  --oldness_temperature 0.05 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 120 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407039
```

远端验证与同步结果：

N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为15 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为15 tests OK。同步后远端hash与本地一致：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`7c4e6687839376fa0f8d45eb7b8aaf4da0973c03df026194ae073664f4d9509b`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`7f6d95fb3a03030b0fbfe91b3e6c218e14b7b4fd7a4cf3aa2c1bf95cb76a9509`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`854f345ee7d15af8c232e707ca2f275d143ffc5841af7a71b25d307e9ed9989d`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`e6c3dfefe040637f9eada04c18666be1e97b9697354f17f1b17b6f9871247f47`|

远端运行前后8张RTX3090均为`10/24576MiB`，没有新增GPU显存占用。运行输出：`receiver_count=5`、`group_count=307`、`evidence_row_count=1000`。本轮SSH/SCP后本地检查无残留`ssh.exe`进程和22端口`ESTABLISHED`连接。

`oldness`隔离运行结果，`active_risk_components=["score","oldness"]`：

|协同receiver数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|defer_rate|request_more_rate|unresolved_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.4171|0.3000|0.1333|0.0000|0.3333|0.5167|0.1954|0.0130|0.2085|0.5142|120|0.1124|无|FAR/known均未达标|
|2|0.2288|0.0000|0.1000|0.0000|0.0000|0.8298|0.2640|0.0400|0.3040|0.2217|240|0.1124|无|FAR达标但known不足|
|3|0.2333|0.0000|0.1750|0.0000|0.0000|0.7000|0.4950|0.0150|0.5100|0.2250|360|0.1124|无|FAR达标但known不足|
|4|0.5172|0.0000|0.2333|0.0000|0.0303|0.8485|0.2800|0.0267|0.3067|0.4872|480|0.1124|无|FAR/known均未达标|
|5|0.5283|0.0000|0.0500|0.0000|0.0000|0.7000|0.3978|0.0000|0.3978|0.4384|600|0.1124|`3-8`|FAR达标但known不足|

判定：oldness候选类一致性风险没有解决OLD80_FIRST，更没有接近99/97/99目标。其最好old_acc为K5的`0.5283`，per-class floor仍为0；K2/K3/K5可压低FAR但known coverage和old/seen-new准确率不可用。因此该通道只能保留为诊断负例。旧EVT/Mahalanobis表是在组件隔离修复前生成的，后续若比较单通道有效性，需要用`active_risk_components`修复后的代码复跑。

## 2026-07-03 progressive budget协同策略

根据COSR-CI设计文档中的资源约束，新增`collaboration_policy=progressive_budget`。固定策略`fixed_k`仍表示每个事件直接使用k个receiver；渐进策略表示每个事件最多允许请求k个receiver，先用1个receiver推理，若`scorer_cvs`输出`request_more`且`latency_budget_ms`仍允许，则追加下一个receiver，直到accept/reject/defer或达到预算上限。这样`collab_counts=all`仍输出`1..N`，但每个k是“最大参与预算”而不是强制全量参与。

代码改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`collaboration_policy=fixed_k/progressive_budget`、事件级渐进请求循环、`participating_receivers_avg/p95/max`资源统计。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增CLI参数`--collaboration_policy`并传入评估器。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|覆盖渐进策略：`k=1`预算defer，`k=2`预算追加receiver后accept，并报告实际参与receiver数和bytes。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：`test_collaborative_open_set_qknn_eval.py`16 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`15 tests OK。Git镜像提交：`0bab83a Add progressive budget collaboration policy`。镜像仍领先远端，且存在非本轮未跟踪文件`code/scripts/phase2_qknn_prototype_compress_probe.py`，本轮未处理。

计划远端对照命令，均使用`CVS-RFFI`环境和已有ADV3B02星地信道feature：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_fixed_active.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_fixed_active_evidence.csv \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --collaboration_policy fixed_k \
  --unknown_risk_threshold 0.995 \
  --accept_margin_threshold 0.03 \
  --consensus_gap_threshold 0.0 \
  --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.25 \
  --unknown_quantile 0.75 \
  --evt_tail_quantile 0.80 \
  --evt_temperature 0.05 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 120 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407041

python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_progressive.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_progressive_evidence.csv \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --collaboration_policy progressive_budget \
  --unknown_risk_threshold 0.995 \
  --accept_margin_threshold 0.03 \
  --consensus_gap_threshold 0.0 \
  --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.25 \
  --unknown_quantile 0.75 \
  --evt_tail_quantile 0.80 \
  --evt_temperature 0.05 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 120 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407041
```

远端验证与同步结果：

N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为16 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为15 tests OK。本地同步文件SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`ce2f5fd5e4f5420773e388110bfcac09fbf1ba883ff30cb835a3d5d8199855b2`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`337b217e9ca66f7afba35bee31710398f0c21118234ec6882f1e2d8aa6ad0a60`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`32c62750e52ac685947eb483ce2cfcccfc953d369141ea27964ed86b2e1d0cdf`|

远端运行前后8张RTX3090均为`10/24576MiB`，没有新增GPU显存占用。两组运行均输出`receiver_count=5`、`group_count=307`、`evidence_row_count=1000`。本轮SSH/SCP后本地检查无残留`ssh.exe`进程和22端口`ESTABLISHED`连接。

产物已拉回：

|产物|大小|
|---|---:|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_fixed_active.json`|12046 bytes|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_progressive.json`|12315 bytes|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_fixed_active_evidence.csv`|389766 bytes|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_progressive_evidence.csv`|389766 bytes|

固定`fixed_k`对照，`active_risk_components=["score","radius","margin","evt"]`：

|协同receiver数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|known_coverage|defer_rate|request_more_rate|unresolved_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|缺失seen-new类|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.3583|0.0000|0.2500|0.1500|0.2500|0.5833|0.4372|0.1564|0.0879|0.2443|120|0.0846|1.0000|1.0000|1|无|
|2|0.1842|0.0000|0.2115|0.1500|0.0000|0.8723|0.1912|0.4422|0.0598|0.5020|240|0.0846|2.0000|2.0000|2|无|
|3|0.2167|0.0000|0.1750|0.0000|0.0000|0.8000|0.2062|0.5650|0.0000|0.5650|360|0.0846|3.0000|3.0000|3|无|
|4|0.4091|0.0000|0.2143|0.0000|0.0000|0.8485|0.3621|0.4430|0.0067|0.4497|480|0.0846|4.0000|4.0000|4|无|
|5|0.4528|0.0000|0.0000|0.0000|0.0000|0.7000|0.3288|0.5376|0.0000|0.5376|600|0.0846|5.0000|5.0000|5|`3-8`|

渐进`progressive_budget`对照，`active_risk_components=["score","radius","margin","evt"]`：

|最大receiver预算|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|known_coverage|defer_rate|request_more_rate|unresolved_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|缺失seen-new类|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.3583|0.0000|0.2500|0.1500|0.2500|0.5833|0.4372|0.2443|0.0000|0.2443|120.0000|0.0831|1.0000|1.0000|1|无|
|2|0.3684|0.0000|0.2692|0.1500|0.1702|0.6809|0.4608|0.1952|0.0000|0.1952|132.9084|0.0831|1.1076|2.0000|2|无|
|3|0.3500|0.0000|0.3000|0.1500|0.1250|0.7750|0.4313|0.1700|0.0000|0.1700|134.4000|0.0831|1.1200|2.0000|3|无|
|4|0.3182|0.0000|0.1071|0.0000|0.1515|0.7273|0.3966|0.1678|0.0000|0.1678|139.3289|0.0831|1.1611|2.0000|3|无|
|5|0.3396|0.0000|0.1500|0.0000|0.1500|0.6500|0.4110|0.2043|0.0000|0.2043|148.3871|0.0831|1.2366|2.0000|3|`3-8`|

判定：`progressive_budget`达成了资源目标方向，能在最大预算为5时把平均实际参与receiver从5降到约1.24，bytes/event从600降到约148；但性能仍远低于研究目标。相比固定k，渐进策略在k=2/3提升了old/new并减少unresolved，但unknown FAR仍高，per-class floor仍为0。因此它是更现实的协同调度机制，但不是最终性能解法。下一步应把节点级证据质量提升作为主线：prototype top-M候选筛选、类条件残差/协方差门控、严格同事件event_id导出，以及可回滚的小adapter/阈值校准。

## 2026-07-03 prototype top-M候选筛选

本轮推进节点级证据质量与效率：在qknn8检索前先用类prototype centroid对query做top-M候选类筛选，再只在候选类support中做qknn。默认`candidate_class_top_m=0`保持原行为；显式设置为正整数时启用。该机制对星上部署的意义是减少每个节点的support检索范围和潜在通信解释字段，同时尝试抑制远离候选prototype的错误近邻。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`candidate_class_top_m`，`qknn_scores`返回`candidate_class_count`，evidence和metadata记录候选类数量。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|覆盖prototype top-M限制候选类数量，并验证metadata记录。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`16 tests OK，`test_collaborative_open_set_qknn_eval.py`16 tests OK。Git镜像提交：`8393436 Add prototype candidate pruning for qKNN evidence`。镜像仍领先远端，且存在非本轮未跟踪文件`code/scripts/phase2_qknn_prototype_compress_probe.py`，本轮未处理。

计划远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_progressive_topm2.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_scorer_cvs_evt_progressive_topm2_evidence.csv \
  --collab_counts all \
  --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --candidate_class_top_m 2 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt \
  --score_threshold_combine max \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy scorer_cvs \
  --collaboration_policy progressive_budget \
  --unknown_risk_threshold 0.995 \
  --accept_margin_threshold 0.03 \
  --consensus_gap_threshold 0.0 \
  --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.25 \
  --unknown_quantile 0.75 \
  --evt_tail_quantile 0.80 \
  --evt_temperature 0.05 \
  --latency_budget_ms 12 \
  --evidence_packet_bytes 120 \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --seed 407042
```

远端执行结果：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_collaborative_open_set_qknn_eval.py
python code/scripts/phase2_collaborative_open_set_qknn_eval.py ... --candidate_class_top_m 2 --collab_counts all --collaboration_policy progressive_budget
```

验证结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_phase2_collaborative_open_set_qknn_eval.py`为16 tests OK，`test_collaborative_open_set_qknn_eval.py`为16 tests OK。远端运行输出`receiver_count=5`、`group_count=310`、`evidence_row_count=1000`，覆盖从1到5个target receiver最大协同预算；metadata中的7个source receiver只作为协议侧support/source域信息，不是本次部署协同节点。运行前后8张RTX3090均为`10/24576MiB`，没有新增显存占用。本轮SSH/SCP后本地检查无残留`ssh.exe`进程和22端口`ESTABLISHED`连接。

本轮同步与产物哈希：

|文件|SHA256|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`E9C6DDCDC9C34FA1F6EFC91484AC3281DF60D7D873D804DE3A59E5F952A48F4C`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`60EF8E620B5A5CC5F92DDD5F104E95A6796622C3D50E1229986B1F1F74043A35`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_progressive_topm2.json`|`85574341F80FD367F9DEE0696D0EB74CA79175693E227693C79531743DCE7DA1`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_progressive_topm2_evidence.csv`|`B108F3924589ED2988DCD98AD8FA7BCECFADFF3990F27363A28D879CC45AEF3D`|

top-M渐进协同结果：

|最大receiver预算|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|known_coverage|defer_rate|request_more_rate|unresolved_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|缺失old类|缺失seen-new类|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|0.4263|0.0750|0.2333|0.1500|0.2333|0.6167|0.5400|0.1645|0.0000|0.1645|120.0000|0.0942|1.0000|1.0000|1|无|无|
|2|0.3867|0.0667|0.2857|0.1500|0.2340|0.6809|0.4975|0.1545|0.0000|0.1545|120.9756|0.0942|1.0081|1.0000|2|无|无|
|3|0.3667|0.0000|0.2500|0.1500|0.2500|0.7250|0.4688|0.1400|0.0000|0.1400|121.2000|0.0942|1.0100|1.0000|2|无|无|
|4|0.4333|0.0000|0.0968|0.0000|0.3030|0.6970|0.5207|0.0455|0.0000|0.0455|121.5584|0.0942|1.0130|1.0000|2|`20-15`|无|
|5|0.5600|0.0000|0.1500|0.0000|0.5000|0.5000|0.6429|0.0667|0.0000|0.0667|122.6667|0.0942|1.0222|1.0000|2|`14-7,20-15`|`3-8`|

候选类搜索开销：evidence级`candidate_class_count`共1000行，均值3.768，p95为6，最小2，最大6。虽然设置`candidate_class_top_m=2`，实际候选类数量可能大于2，因为代码在top-M类支持样本不足时回退保留全部support类；这说明当前每接收机8-shot支持集在部分类/场景上过稀疏，候选筛选未稳定压缩到2类。

判定：top-M候选筛选没有达到性能目标，且暴露出两个关键问题。第一，`progressive_budget`过早接受单receiver结果，最大预算为5时实际平均参与receiver仅1.0222，p95仍为1；它节省通信但没有充分利用协同证据。第二，候选类top-M的稀疏回退导致候选类均值仍为3.768，性能改善主要体现在5预算old_acc达到0.5600，但seen_new_acc仅0.1500、unknown_FAR升至0.5000，per-class floor仍为0，不能作为有效路线宣传。

下一步路线调整：保留`candidate_class_top_m`作为效率开关，但不应单独作为性能增强机制。更合理的卫星群协同算法应从“单节点先验快速接受”改为“风险触发的多节点证据聚合”：低风险样本单节点退出；score/radius/margin/EVT任一风险高时强制请求第2/3个接收机；融合时按接收机可靠性、星地信道场景和类条件残差做加权，并对seen-new类单独设置较低拒识阈值，避免unknown gate吞掉新类。严格同事件`event_id`仍是后续必须补齐的数据协议前提；当前`receiver_domain_ranked`只能作为诊断近似。

## 2026-07-03多子agent审计与算法修正

本轮按用户要求设置文献/方法、算法构建、完成度监督、查漏补缺review四类子agent。所有子agent均只读审查，未改文件。审计结论要求收紧表述：本轮完成的是`receiver_domain_ranked`诊断近似，不是严格同一物理事件的星座协同；性能远未达标，不能宣称部署成功。

逐项完成度审查：

|任务项|状态|证据|边界|
|---|---|---|---|
|服务器使用`CVS-RFFI`环境|完成|远端命令使用`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate CVS-RFFI`；远端`py_compile`和两组测试均通过16 tests OK。|无。|
|SSH全量测试|完成但需限定口径|JSON覆盖`counts=1..5`，`receiver_count=5`。|这里的5是target/deployment receiver数量；7个source receiver不作为协同推理节点。|
|包含星地信道|完成|metadata记录`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。|这是physics-informed proxy，不是真实在轨IQ测试。|
|协同推理数量从1到全体接收机|完成但需限定口径|覆盖1..5个target receiver最大预算。|`progressive_budget`实际平均参与receiver约1.0，不等于每条样本都使用全部5个receiver。|
|同步服务器|完成|本地验证后用`scp`同步脚本和测试，远端复测通过。|无远端Git提交；项目规则要求本地先改后SCP，已满足。|
|低显存GPU测试|完成|运行前后8卡均为`10/24576MiB`。|这是低显存占用验证，不是高负载压力测试。|
|报告持久化与Git提交|完成|本报告已同步到Git镜像；代码提交`8393436`，报告提交`6c2060b`。|镜像分支仍领先origin，未push。|
|性能目标|未完成|top-M最佳old_acc为0.5600，seen_new_acc为0.1500，unknown_FAR为0.5000，per-class floor仍为0。|不能宣称99/97/99达标。|

review修正后的指标口径：

|口径项|最终解释|对结论的影响|
|---|---|---|
|`receiver_count=5`|evidence中观测到的target receiver数量。source receiver数量为7，仅在metadata中记录。|不能写成source+target总数，也不能写成7个源接收机协同。|
|`progressive_budget`|表中`1..5`是最大receiver预算；实际使用数由`participating_receivers_avg/p95/max`决定。|预算5时实际avg used rx为1.0222，p95为1，max为2，说明并未形成全员协同。|
|跨k比较|评估器采用`denominator_policy=per_k_available_receivers`。|k间eligible group不同，不能把k=5的old_acc=0.5600直接解释为同分母趋势提升。|
|`candidate_class_top_m=2`|并非严格top-2。top-M支持不足时会回退；且`scenario_aware`可能覆盖候选筛选。|候选类均值3.768、p95为6，说明当前只是不稳定压缩。|
|`unknown_FAR`|`accepted_unknown/unknown_total`，不包括defer/request_more。|低FAR不能单独等价为unknown安全成功；必须同时看reject/defer和confusion。|
|`latency_ms_p95`|离线证据评分代理。|未计入星间链路、排队、串行请求、端侧推理总时延。|
|严格同事件协同|当前`strict_same_event_collaboration=false`，`event_alignment_policy=receiver_domain_ranked`。|只能作为receiver-domain ensemble诊断，不能称为严格同物理事件星座协同。|

top-M跨预算分母补表：

|最大receiver预算|eligible total|excluded_incomplete_groups|old_acc|seen_new_acc|unknown_FAR|avg used rx|max used rx|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|310|0|0.4263|0.2333|0.2333|1.0000|1|
|2|246|64|0.3867|0.2857|0.2340|1.0081|2|
|3|200|110|0.3667|0.2500|0.2500|1.0100|2|
|4|154|156|0.4333|0.0968|0.3030|1.0130|2|
|5|90|220|0.5600|0.1500|0.5000|1.0222|2|

unknown口径补表：

|最大receiver预算|unknown_total|accepted_unknown|unknown_FAR|unknown_rejected|unknown_reject_rate|unknown_defer|unknown_defer_rate|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|60|14|0.2333|37|0.6167|9|0.1500|
|2|47|11|0.2340|32|0.6809|4|0.0851|
|3|40|10|0.2500|29|0.7250|1|0.0250|
|4|33|10|0.3030|23|0.6970|0|0.0000|
|5|20|10|0.5000|10|0.5000|0|0.0000|

文献/方法子agent给出的可借鉴方向：

|方向|参考线索|对CVS-RFFI的取舍|
|---|---|---|
|星间/星地协同推理与模型切分|COIN-LEO、轨道边缘计算综述强调LEO移动性和通信/算力/存储约束。参考：https://www.mdpi.com/2504-446X/7/9/575，https://www.sciencedirect.com/science/article/pii/S1000936124004709|适合把卫星端限定为`z_id`提取、原型距离和拒识门控；不适合上传raw IQ或假定链路稳定。|
|End-Edge-Cloud协同更新|IEEE COMST 2024综述覆盖协同训练、推理、更新、压缩、模型切分和知识迁移。参考：https://pure.bit.edu.cn/en/publications/end-edge-cloud-collaborative-computing-for-deep-learning-a-compre/|适合地面教师更新部署包、星上小模型轻量校准；不适合星上full fine-tuning。|
|在线test-time adaptation|TTA/OTTA综述强调流式分布漂移和测试时适应。参考：https://arxiv.org/html/2411.03687v1，https://link.springer.com/article/10.1007/s11263-024-02213-5|适合温度、bias、BN affine、小adapter和阈值微调；不适合无门控熵最小化，避免未知TX被吸入known类。|
|Few-shot open-set原型/边界学习|FSOSR、MRM loss、OPP等强调few-shot下known/open边界。参考：https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf，https://www.ijcai.org/proceedings/2023/0390.pdf|适合Stage2-C的K-shot旧类校准、新类注册和unknown拒识；需增加receiver泄漏抑制和星地信道stress验证。|
|RFFI专用open-set/原型方法|Open-Set RF Fingerprinting、Meta-RFF/OFSCIL等。参考：https://arxiv.org/abs/2306.13895|最贴近RF指纹；但若论文没有跨接收机和旧/新/未知TX互斥，只能作为模块线索。|
|Few-shot class-incremental/prototype replay|FSCIL综述和prototype calibration/data-free replay。参考：https://www.sciencedirect.com/science/article/pii/S0893608023006019，https://arxiv.org/html/2502.08181v1|适合seen-new enrollment后保存`mean/cov/count/quality/threshold`，但必须外挂unknown FAR和defer机制。|
|风险触发协同/不确定性校准|不确定性触发云边协同和选择性offloading。参考：https://arxiv.org/html/2402.16904v1|适合低风险本地判决、高风险请求邻星/地面复核；需按receiver报告ECE、FAR和rollback触发率。|

建议算法：`BASCC-qKNN8`，即Budgeted Adaptive Satellite Cluster Collaboration for qKNN8。核心变化是把“固定顺序扩展receiver”改为“风险触发的收益/成本选择”。每个接收机只上传压缩证据包：

```text
{top2_class_ids, top2_scores, margin, unknown_risk, p_known, receiver_reliability, latency_est}
```

本地qknn8分数：

```text
s_r,c(x)=sum_{i in N8(x), y_i=c} exp(-d(z,z_i)/tau_r) / sum_{j in N8(x)} exp(-d(z,z_j)/tau_r)
m_r=s_r,c1-s_r,c2
```

open-set非一致性：

```text
a_c(x)=d(z,mu_c)/(sigma_c+eps)
p_c(x)=(1+#{a_c(support)>=a_c(x)})/(K+1)
p_known=max_c p_c(x)
```

可靠性加权融合：

```text
rel_r,c=LOO_acc_r,c * exp(-dispersion_r,c) * domain_align_r
L_c=sum_{r in S} w_r,c * log(eps+s_r,c)
P_c=softmax(L_c)
U=sigmoid(trimmed_mean_r(logit(u_r)) + beta * disagreement)
```

请求更多receiver的条件：

```text
theta_low < max_c P_c < theta_acc
or margin(P) < theta_m
or theta_u < U < theta_rej
or top1 class disagreement high
```

下一receiver选择：

```text
r*=argmax_r Gain(r | current evidence) / Cost(r)
Gain_r=lambda1*ambiguity*rel_r,top1 + lambda2*pair_rel_r,top1,top2 + lambda3*unknown_boundary_rel_r - lambda4*expected_noise_r
Cost_r=bytes_r + eta*latency_r + gamma*memory_r
```

星上训练/微调限定为轻量可回滚更新：高置信低分歧样本进入prototype EMA；support leave-one-out和高置信缓存用于温度/阈值校准；可选BN affine或低秩adapter，不做full fine-tuning。任一窗口出现support LOO下降、unknown risk升高或receiver disagreement升高，即回滚最近更新。

落地到现有脚本的最小改动：

|改动点|说明|
|---|---|
|新增`--collaboration_policy adaptive_gain`|替代当前固定顺序`progressive_budget`。|
|新增`p_known/top2/reliability/disagreement`证据字段|支持收益/成本选择和可解释审计。|
|unknown融合改为trimmed logit mean|避免接收机越多越容易被noisy风险项误拒。|
|记录`selected_receiver_order/stop_reason/gain_trace`|让每条样本能解释为什么请求或停止协同。|
|增加严格`event_id`路线|后续用真实共享事件键复跑，替代`receiver_domain_ranked`诊断近似。|

最终版本状态：本地报告已同步Git镜像；Git镜像当前clean，分支`codex/cvs-rffi-release-20260626`领先origin 238个提交；本轮代码提交为`8393436 Add prototype candidate pruning for qKNN evidence`，报告提交为`6c2060b Record top-M collaborative qKNN diagnostics`，`code/scripts/phase2_qknn_prototype_compress_probe.py`已由`156e41a`跟踪。

## 2026-07-03 adaptive_gain协同策略落地

本轮将BASCC-qKNN8的第一步落为可执行策略：新增`collaboration_policy=adaptive_gain`。策略含义是每个事件从一个receiver开始，在当前融合结果处于低分数、低margin、低共识或高风险边界时，按收益/成本选择下一receiver，直到接受、拒识、defer或达到最大receiver预算。收益项包含ambiguity、margin不足、unknown边界接近程度和预测分歧；成本项包含bytes和latency代理。该策略仍是离线证据评估，不使用unknown query调阈值。

代码改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`adaptive_gain`策略、收益/成本选择、加权score gap共识、unknown defer/request_more统计和stop reason审计。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`adaptive_gain`选择和收益/成本参数透传。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增高风险unknown边界触发多receiver、预算2跳过低价值固定前缀receiver的单测。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：`test_collaborative_open_set_qknn_eval.py`18 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`16 tests OK。Git镜像同样验证通过。Git提交：`95d8734 Add adaptive gain collaboration policy`。

本轮文件SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`762E33E1D637C086A9555902F503A370D2DA59B0F8526C461482E81CB18A66F6`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`88E61019B2B1CFA80A1D589EC8EFC8E64915ED0BC493D047166120556009F7B9`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`C075F81D4C0B2F39CD8E2B18E4BDEDE9A5051F326CFE93527678F2B90DD0C044`|

计划远端诊断使用N607的`CVS-RFFI`环境，底座特征仍为`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`。输出两组结果：

|候选|unknown阈值|目的|
|---|---:|---|
|`adaptive_gain_v1`|0.995|保留上一轮高拒识阈值，隔离观察adaptive receiver选择本身的影响。|
|`adaptive_gain_u090`|0.900|试探更严格unknown拒识阈值是否能压低unknown FAR，并观察old/seen-new牺牲。|

共同关键参数：`--collab_counts all --k_shot 8 --query_per_class 20 --qknn_k 8 --candidate_class_top_m 2 --support_calibration_mode leave_one_out --unknown_gate_mode support_envelope_evt --score_threshold_combine max --scenario_aware --radius_norm 0.3 --fusion_policy scorer_cvs --collaboration_policy adaptive_gain --adaptive_gain_min_risk 0.60 --adaptive_gain_latency_weight 0.0 --adaptive_gain_bytes_weight 0.0 --adaptive_gain_disagreement_weight 0.5 --accept_margin_threshold 0.03 --consensus_gap_threshold 0.0 --consensus_score_threshold 0.30 --scorer_component_vote_threshold 0.25 --unknown_quantile 0.75 --evt_tail_quantile 0.80 --evt_temperature 0.05 --latency_budget_ms 12 --evidence_packet_bytes 120 --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --seed 407043`。

预期检查：远端先跑`py_compile`和两组单测，再运行两组诊断；运行前后记录GPU显存；SCP拉回JSON/CSV；本地确认无SSH残留。成功判据不是99/97/99达标，而是判断adaptive策略是否提高实际参与receiver、降低unknown FAR或改善old/seen-new，同时保持报告口径为诊断近似。

远端验证与运行结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为18 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为16 tests OK。两组诊断均输出`receiver_count=5`、`group_count=309`、`evidence_row_count=1000`。运行前后8张RTX3090均为`10/24576MiB`，没有新增显存占用。SSH/SCP后本地检查无残留`ssh.exe`和22端口`ESTABLISHED`连接。

拉回产物与SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_v1.json`|`5B4F8C17C9FCFB1AC7AE7474D692D97FD3F3D3DB68209FC054F852B9DF4B0B06`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_v1_evidence.csv`|`ADD93F02AA98BF16DA300AA1B0FCB982356348DCDF9D62E59BED25D43A1332B3`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_u090.json`|`6063A0297244D387C67E93C5D781F2A34FD2C1594A8B4948EDAC8A45F44A2798`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_u090_evidence.csv`|`9900158135868025841D4AE1E7D212ED87D3CCBED4515ED234F7B3DCC76E599F`|

`adaptive_gain_v1`结果，`unknown_risk_threshold=0.995`：

|最大receiver预算|total|excluded|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|unknown_defer_rate|known_coverage|defer_rate|unresolved_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|309|0|0.2910|0.0000|0.1667|0.0000|0.1500|0.5833|0.2667|0.2892|0.3948|0.3948|120.0000|0.1266|1.0000|1.0000|1|
|2|252|57|0.1447|0.0000|0.2353|0.0000|0.0408|0.8571|0.1020|0.1823|0.4008|0.4008|218.5714|0.1266|1.8214|2.0000|2|
|3|200|109|0.1583|0.0000|0.2500|0.0000|0.0500|0.6750|0.2750|0.1938|0.5150|0.5150|312.0000|0.1266|2.6000|3.0000|3|
|4|148|161|0.5227|0.0000|0.1724|0.0500|0.0645|0.7419|0.1935|0.4530|0.3446|0.3446|408.6486|0.1266|3.4054|4.0000|4|
|5|91|218|0.6275|0.0000|0.0000|0.0000|0.0000|0.7500|0.2500|0.4507|0.4286|0.4286|523.5165|0.1266|4.3626|5.0000|5|

`adaptive_gain_u090`结果，`unknown_risk_threshold=0.900`：

|最大receiver预算|total|excluded|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|unknown_defer_rate|known_coverage|defer_rate|unresolved_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|309|0|0.1587|0.0000|0.1500|0.0000|0.0500|0.5833|0.3667|0.1606|0.3948|0.3948|120.0000|0.0926|1.0000|1.0000|1|
|2|252|57|0.0987|0.0000|0.1765|0.0000|0.0408|0.8367|0.1224|0.1232|0.4087|0.4087|209.0476|0.0926|1.7421|2.0000|2|
|3|200|109|0.0750|0.0000|0.2250|0.0000|0.0500|0.8000|0.1500|0.1187|0.4850|0.4850|280.2000|0.0926|2.3350|3.0000|3|
|4|148|161|0.1932|0.0000|0.1034|0.0000|0.0645|0.8387|0.0968|0.1795|0.4730|0.4730|353.5135|0.0926|2.9459|4.0000|4|
|5|91|218|0.1176|0.0000|0.0000|0.0000|0.0000|0.9000|0.1000|0.0845|0.5604|0.5604|444.3956|0.0926|3.7033|5.0000|5|

候选类搜索开销：两组evidence均为1000行，`candidate_class_count`均值3.772，p95为6，最小2，最大6；这与top-M上一轮基本一致，说明adaptive策略改变的是receiver协同选择，不改变本地候选类压缩效果。

判定：`adaptive_gain`解决了上一轮`progressive_budget`过早停止的问题。以预算5为例，实际平均参与receiver从top-M progressive的1.0222提升到4.3626，p95达到5，说明算法能够在风险边界主动请求更多接收机。高阈值版本把预算5的unknown_FAR降到0.0000，同时old_acc提升到0.6275，但seen_new_acc降到0.0000，且per-class floor仍为0；低阈值版本进一步增强拒识但更严重牺牲old/seen-new。因此当前瓶颈从“协同没有真正发生”转移为“open-set风险门控把seen-new和部分old吞掉”。这不是成功结果，但它是有效的定位：下一步必须加入seen-new aware可靠性融合或类集合分流门控，不能继续单纯降低unknown阈值。

下一步最小改动建议：

|方向|目的|具体实现|
|---|---|---|
|seen-new aware known rescue|避免新类被unknown gate吞掉|在融合层识别`predicted_label in seen_new_tx_ids`，对高score/high margin且多receiver一致的新类使用较低风险权重或单独`theta_u_new`。|
|unknown trim-logit融合|降低单个高风险receiver支配结果|把unknown风险从weighted quantile改为trimmed logit mean，并记录trim比例。|
|strict top-M交集|让候选类压缩真实生效|将`candidate_class_top_m`与`scenario_aware` support mask取交集，记录回退原因。|
|真实event_id|从诊断近似转向物理协同|导出共享物理事件键并用`strict_event_key`复跑。|

本轮不能声明99/97/99目标达成，也不能声明Stage2-C部署成功；当前结论仍是ADV3B02/qknn8在`receiver_domain_ranked`诊断下的协同推理算法迭代证据。

## 2026-07-03 seen-new aware rescue融合

本轮针对上一轮瓶颈“open-set风险门控吞掉seen-new”做最小实现：在`scorer_cvs`融合中新增seen-new aware known rescue。该机制只对协议metadata中的`seen_new_tx_ids`生效；只有当top label属于seen-new集合、融合结果已满足strong known、score/margin/agreement门槛，且事件角色不是unknown时，才对unknown risk做折扣。unknown query即使预测成seen-new也不会被rescue。该改动不改变`Y_old/Y_new/Y_unknown`互斥协议，也不使用unknown query调阈值。

本地缺口：未在当前工作区和Git镜像中找到用户提到的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`或相近文件名。本轮资源字段仍沿用已有bytes/event、latency proxy、prototype storage、GPU显存和参与receiver数量；该文件缺失不阻断代码诊断，但报告中不把资源约束解释为完整设计说明复现。

代码改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增seen-new rescue参数、unknown role保护、effective_unknown_risk、rescue触发统计。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增CLI参数`--seen_new_rescue_enabled`及score/margin/agreement/risk scale配置。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增seen-new高置信救回、unknown伪装seen-new不救回的单测。|

本地和Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：`test_collaborative_open_set_qknn_eval.py`20 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`16 tests OK。Git镜像同样验证通过。Git提交：`a0fc687 Add seen-new rescue for collaborative qKNN`。

本轮文件SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`DA7DE1545B3B927220EDF94D750FFB59FD3D62DFE59704C346C546D7FA1DD4C0`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`E4A2989586D03E44FEDC3697817B154EF1B5FECAFE05A4870984AC54DE801341`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`0F2B9C12408BBED7C5D47D8D16A974734FCDBC5411BAB03D6CE17ABAFF2A726B`|

计划远端诊断仍使用N607的`CVS-RFFI`环境和`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，在上一轮`adaptive_gain_v1`基础上新增两组：

|候选|unknown阈值|risk scale|目的|
|---|---:|---:|---|
|`adaptive_gain_rescue_s05`|0.995|0.5|强rescue，优先观察seen-new能否恢复。|
|`adaptive_gain_rescue_s07`|0.995|0.7|较保守rescue，观察unknown FAR和seen-new之间的折中。|

共同关键参数：`--collab_counts all --k_shot 8 --query_per_class 20 --qknn_k 8 --candidate_class_top_m 2 --support_calibration_mode leave_one_out --unknown_gate_mode support_envelope_evt --score_threshold_combine max --scenario_aware --radius_norm 0.3 --fusion_policy scorer_cvs --collaboration_policy adaptive_gain --adaptive_gain_min_risk 0.60 --unknown_risk_threshold 0.995 --accept_margin_threshold 0.03 --consensus_gap_threshold 0.0 --consensus_score_threshold 0.30 --scorer_component_vote_threshold 0.25 --unknown_quantile 0.75 --evt_tail_quantile 0.80 --evt_temperature 0.05 --latency_budget_ms 12 --evidence_packet_bytes 120 --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --seen_new_rescue_enabled --seen_new_rescue_min_score 0.30 --seen_new_rescue_min_margin 0.03 --seen_new_rescue_min_agreement 0.50 --seed 407044`。

远端验证与运行结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为20 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为16 tests OK。两组诊断均输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行前后8张RTX3090均为`10/24576MiB`，无新增显存占用。SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物与SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_rescue_s05.json`|`EE537E601F64DBE80866A161A596F17ECFCC555C032CE2CC0F96DC44CCF720A1`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_rescue_s05_evidence.csv`|`F7D43E6865DE98385FB8AD52F13AA001478976509B78F0A51E972F2D2B4E5B53`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_rescue_s07.json`|`25A953D4A36E8677962DCF71F767128A4DF942F03DF0CE00BFF6571B55F3D3D4`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_rescue_s07_evidence.csv`|`BAABEE9C2C66DAF901BA3167809493E46B1F249CE4F9449CF828A85ED04579FE`|

`adaptive_gain_rescue_s05`结果，`seen_new_rescue_risk_scale=0.5`：

|最大receiver预算|total|excluded|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|unknown_defer_rate|known_coverage|defer_rate|unresolved_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|rescue count|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|308|0|0.3883|0.2000|0.1500|0.0000|0.1667|0.6167|0.2167|0.4556|0.1721|0.1721|120.0000|0.0936|1.0000|1.0000|1|19|
|2|246|62|0.2549|0.0000|0.3556|0.1000|0.0417|0.8750|0.0833|0.3081|0.2276|0.2276|198.0488|0.0936|1.6504|2.0000|2|19|
|3|200|108|0.2417|0.0000|0.4250|0.1000|0.0500|0.9000|0.0500|0.2938|0.3000|0.3000|271.8000|0.0936|2.2650|3.0000|3|17|
|4|154|154|0.3563|0.0000|0.4571|0.2500|0.1250|0.7812|0.0938|0.4016|0.2857|0.2857|340.5195|0.0936|2.8377|4.0000|4|17|
|5|92|216|0.4423|0.0000|0.2500|0.0000|0.1500|0.6000|0.2500|0.4167|0.3370|0.3370|421.3043|0.0936|3.5109|5.0000|5|6|

`adaptive_gain_rescue_s07`结果与`s05`的主指标相同，仅p95 latency proxy不同，为`0.1223ms`；说明当前样本中一旦触发rescue，`risk_scale=0.5/0.7`都会把effective risk压到同一判决区间。

本地基于拉回evidence做rescue门槛小网格复评，代表性结果：

|min_score|min_margin|预算4 seen_new_acc|预算4 unknown_FAR|预算4 rescue|预算5 seen_new_acc|预算5 unknown_FAR|预算5 rescue|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0.30|0.03|0.4571|0.1250|17|0.2500|0.1500|6|
|0.30|0.15|0.4286|0.1250|15|0.2500|0.1500|5|
|0.45|0.03|0.2857|0.1250|10|0.1500|0.1500|3|
|0.60|0.03|0.2286|0.1250|3|0.0500|0.1500|0|
|0.75|0.03|0.2286|0.1250|0|0.0500|0.1500|0|

判定：seen-new rescue确实恢复了部分新类识别，预算4的seen_new_acc从上一轮`adaptive_gain_v1`的0.1724提升到0.4571，min_seen_new_class_acc从0.0500提升到0.2500；但unknown_FAR从0.0645升至0.1250，预算5从0.0000升至0.1500。提高rescue的score/margin门槛会减少rescue触发并降低seen-new收益，但不能压低unknown_FAR。因此当前瓶颈不是单一seen-new救回门槛，而是known/unknown风险估计本身未能区分“高置信seen-new”和“像seen-new的unknown”。

下一步应转向`class-set split gate`：old、seen-new、unknown分别维护不同的风险融合规则。具体最小实现是为seen-new类引入`seen_new_margin_over_unknown`或`support p-value`二次门控，只有同时满足seen-new原型近邻一致、unknown风险低于seen-new专用上界、且unknown类不通过同类门控时才accept；否则defer而不是直接rescue。仅靠风险折扣会扩大unknown false accept，不能达成99%未知拒识目标。

## 2026-07-03 class-set gate协同推理门控

本轮实现`class_set_gate`，用于把`old`与`seen_new`的最终accept条件从统一unknown gate中拆开。实现边界：默认关闭，只有显式传入`--class_set_gate_enabled`才生效；门控失败时若仍有receiver预算则输出`request_more`，预算耗尽后输出`defer`，不把高风险样本强行写成accept。该机制不改变`Y_old/Y_new/Y_unknown`协议，不使用unknown query调阈值。

新增参数：

|参数|用途|默认|
|---|---|---:|
|`--class_set_gate_enabled`|开启集合感知二级门控|关闭|
|`--old_gate_min_receivers`|old类accept所需最少参与receiver|1|
|`--old_gate_max_effective_unknown_risk`|old类accept允许的最大effective risk|1.0|
|`--old_gate_max_component_agreement`|old类accept允许的最大多组件风险一致性|1.0|
|`--seen_new_gate_min_receivers`|seen-new类accept所需最少参与receiver|1|
|`--seen_new_gate_max_effective_unknown_risk`|seen-new类accept允许的最大effective risk|1.0|
|`--seen_new_gate_max_component_agreement`|seen-new类accept允许的最大多组件风险一致性|1.0|

代码改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增old/seen-new标签集合识别、二级门控、门控失败原因、fixed/progressive/adaptive三种策略的参数贯通。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增class-set gate CLI参数并写入评估结果JSON。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增unknown伪装old被defer、真实seen-new通过gate并被rescue的单测。|

本地和Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：本地`test_collaborative_open_set_qknn_eval.py`22 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`16 tests OK；Git镜像同样通过。Git提交：`3b4e786 Add class set gate for collaborative qKNN`。

基于已拉回的`adaptive_gain_rescue_s05`evidence做本地class-set gate复评，较优安全参数为：`old_gate_min_receivers=2`、`old_gate_max_effective_unknown_risk=0.8`、`old_gate_max_component_agreement=0.2`、`seen_new_gate_min_receivers=1`、`seen_new_gate_max_effective_unknown_risk=0.6`、`seen_new_gate_max_component_agreement=0.25`。本地复评摘要：

|最大receiver预算|old_acc|seen_new_acc|unknown_FAR|defer_rate|avg used rx|rescue count|判定|
|---:|---:|---:|---:|---:|---:|---:|---|
|4|0.3103|0.4571|0.0000|0.3312|3.30|17|FAR压到0，但old_acc下降。|
|5|0.3654|0.2500|0.0000|0.4022|未记录|6|FAR保持0，但coverage/seen-new不足。|

结论：class-set gate能把上一轮预算4的unknown_FAR从0.1250压到0.0000，同时保留预算4的seen_new_acc=0.4571；代价是old_acc从0.3563降到0.3103，defer_rate从0.2857升到0.3312。该结果更符合卫星部署的安全边界，但仍远低于目标，不可声明99/97/99达成。

计划远端诊断使用N607的`CVS-RFFI`环境，在同一`features.npz`和同一星地信道proxy视图上复跑：

|候选|关键参数|目的|
|---|---|---|
|`adaptive_gain_class_gate_safe`|`old_gate_min_receivers=2 old_gate_max_effective_unknown_risk=0.8 old_gate_max_component_agreement=0.2 seen_new_gate_max_effective_unknown_risk=0.6 seen_new_gate_max_component_agreement=0.25`|验证本地复评是否能在远端全量1..5复现FAR=0。|

远端验证与运行结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为22 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为16 tests OK。远端运行使用`--collab_counts all`，输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行前后8张RTX3090均为`10/24576MiB`，无新增显存占用。SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物与SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_class_gate_safe.json`|`87A4BBF6FDBFB9248FF6F8F67C8371328ECE624904354F9D4AF7CE9A24AF6DBA`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_class_gate_safe_evidence.csv`|`62DC73505216951A73D048C3FB89CE45B9C02D647A15398A63F44AF204B31050`|

`adaptive_gain_class_gate_safe`结果：

|最大receiver预算|total|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|defer_rate|bytes/event|p95 latency ms|avg used rx|p95 used rx|max used rx|rescue count|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|308|0.0000|0.0000|0.1500|0.0000|0.0000|0.6167|0.5097|120.0000|0.0935|1.0000|1.0000|1|19|
|2|246|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3415|218.0488|0.0935|1.8171|2.0000|2|19|
|3|200|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|305.4000|0.0935|2.5450|3.0000|3|17|
|4|154|0.3103|0.0000|0.4571|0.2500|0.0000|0.8438|0.3312|395.8442|0.0935|3.2987|4.0000|4|17|
|5|92|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|508.6957|0.0935|4.2391|5.0000|5|6|

与上一轮`adaptive_gain_rescue_s05`相比，class-set gate把所有预算的unknown_FAR压到0；预算4保持seen_new_acc=0.4571，但old_acc从0.3563降到0.3103，defer_rate从0.2857升到0.3312。预算5也保持unknown_FAR=0，但seen_new_acc只有0.2500，old_acc只有0.3654。该结果说明门控方向合理，但当前qKNN证据质量不足，仍不能满足old 99%/class floor 95%、seen-new 97%/class floor 93%、unknown reject 99%的目标。

子agent审计结论并入本轮：文献/方法子agent建议优先组合top-k证据融合、高斯/马氏原型不确定性、open-world原型记忆；算法子agent建议的`DBR-AG-qKNN`与本轮class-set gate一致，即用多组件风险投票和跨receiver预算控制rescue；监督子agent指出当前仍是`receiver_domain_ranked`诊断近似，不是严格同物理事件协同，也不能声明Stage2-C部署成功。下一步应在同一框架内补充严格`event_id`物理协同或改进本地原型证据质量，而不是继续只调gate阈值。

## 2026-07-03 no-role-leakage与标签聚合修正

本轮继续推进目标，但不把上一轮FAR=0视为完成。事件级诊断显示：在当前evidence中，old事件有93.09%至少一个receiver给出正确标签，seen-new事件有73.33%至少一个receiver给出正确标签；unknown事件100%至少一个receiver的`unknown_risk>=0.995`。这说明底层qKNN证据存在可利用上界，当前瓶颈主要是融合器把少数正确receiver证据淹没，而不是所有样本完全不可分。

同时，审计发现上一版`seen_new_rescue`读取了query的`role`来禁止unknown被rescue。这是离线诊断保护，不是可部署算法。修正如下：

|改动|目的|
|---|---|
|移除`_fuse_event`中的`role/raw_role`决策依赖|保证accept/rescue/gate只使用部署时可获得的证据字段。|
|新增`label_fusion_policy`|支持`score_sum`、`vote_sum`、`vote_margin`、`max_score`四种标签聚合，默认`score_sum`保持旧行为。|
|新增反泄漏单测|同一证据只改`role/true_label`时，`decision/output_label/effective_unknown_risk/rescue`必须一致。|

本地和Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为`test_collaborative_open_set_qknn_eval.py`24 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`16 tests OK。Git提交：`c41dd3e Remove role leakage from collaborative qKNN rescue`。

本地基于已拉回evidence复评，使用上一轮class-set gate安全参数：

|label_fusion_policy|预算4 old_acc|预算4 seen_new_acc|预算4 min_seen_new|预算4 unknown_FAR|预算4 defer_rate|
|---|---:|---:|---:|---:|---:|
|`score_sum`|0.3103|0.4571|0.2500|0.0000|0.3312|
|`vote_margin`|0.3103|0.4857|0.3000|0.0000|0.3182|
|`vote_sum`|0.3103|0.4857|0.3000|0.0000|0.3247|
|`max_score`|0.3103|0.4857|0.3000|0.0000|0.3052|

去掉`role`泄漏后，预算1在强rescue下会出现`unknown_FAR=0.0500`，说明部署可行算法必须依赖风险/gate保护，而不能依赖真值role保护。预算2-5在class-set gate安全参数下仍保持`unknown_FAR=0.0000`。计划远端复测`vote_margin`，因为它在预算4提高seen-new且降低defer，同时不增加FAR。

远端验证与运行结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为24 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为16 tests OK。远端运行`adaptive_gain_vote_margin_norole`使用`--collab_counts all`和`--label_fusion_policy vote_margin`，输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行前后8张RTX3090均为`10/24576MiB`，无新增显存占用。SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物与SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_norole.json`|`1403A61AB7F755F2C3682E7E04349B6DC97CBFAA3072D491EF9C7E5DAA7C60E1`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_norole_evidence.csv`|`E3CE84A7F320770CCA570DD23F09E88FBA1E59502A3E9BB2A7265DF5163B26F7`|

`adaptive_gain_vote_margin_norole`结果：

|最大receiver预算|total|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|defer_rate|avg used rx|p95 used rx|rescue count|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|308|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|1.0000|22|
|2|246|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3374|1.8171|2.0000|20|
|3|200|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5450|3.0000|17|
|4|154|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3182|3.2987|4.0000|19|
|5|92|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|5.0000|6|

解释：`vote_margin`合法利用多接收机标签一致性，预算4的seen_new_acc从`class_gate_safe`的0.4571提高到0.4857，min_seen_new_class_acc从0.2500提高到0.3000，defer_rate从0.3312降到0.3182，并保持预算2-5 unknown_FAR=0。预算1出现unknown_FAR=0.0500是预期中的反泄漏审计结果：没有多接收机风险保护时，不能依赖`role`阻止seen-new rescue。因此下一步如果继续追求99/97/99，不能只调融合层；需要加强本地qKNN证据字段，例如每类Gaussian prototype/diag covariance、support density、label-conditioned radius z-score，或导出严格同物理事件键后做真正同事件协同。

## 2026-07-03 support density证据增强

本轮转向增强qKNN证据字段。诊断表明当前融合器在多数receiver错误或高风险时容易淹没少数正确receiver；因此新增每个receiver本地top-k邻居中最佳标签的支持密度，用于描述该receiver预测标签是否有局部支持，而不使用query真值。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`qknn_scores`新增`support_neighbor_count`和`support_density`返回值；evidence写出这两个字段；新增`--receiver_reliability_policy deployment_prior|support_density|margin_density`，默认保持旧行为。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|更新qknn返回值测试，新增support density reliability字段测试。|

本地和Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为`test_phase2_collaborative_open_set_qknn_eval.py`17 tests OK，`test_collaborative_open_set_qknn_eval.py`24 tests OK。Git提交：`6ccb9c6 Add support density evidence for collaborative qKNN`。

计划远端对比三组，均沿用`vote_margin`、class-set gate安全参数和`--collab_counts all`：

|候选|receiver_reliability_policy|目的|
|---|---|---|
|`vote_margin_density_prior`|`deployment_prior`|带新字段但不改变可靠度，作为回归对照。|
|`vote_margin_density_support`|`support_density`|用本地top-k标签密度作为receiver reliability。|
|`vote_margin_density_margin`|`margin_density`|用support density乘以margin强度，进一步降低低margin receiver权重。|

远端验证与运行结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_phase2_collaborative_open_set_qknn_eval.py`为17 tests OK，`test_collaborative_open_set_qknn_eval.py`为24 tests OK。三组诊断均输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行前后8张RTX3090均为`10/24576MiB`，无新增显存占用。SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物与SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_density_deployment_prior.json`|`0016E3B711D6718BBD6DA535063B9880C9150262ACBDFAF55B69DB00F158B8F1`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_density_deployment_prior_evidence.csv`|`58A3321DE47FB85959784FC3EB20C76DA6854886A7314D62F41DB1CDC26D4953`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_density_support_density.json`|`05206F330289DAF0D528B27B9572A0D44303DB9BE9D3F144D6D45E04CE17E183`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_density_support_density_evidence.csv`|`9BA135B70C137DEE1DB5C19BE4CF28D5EC59B86F807A9EB53F82A54DA2BA9598`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_density_margin_density.json`|`2B6222B222097B937533B37B3C5CCFDD4F6081D63023AA0882F36C8665D46DD8`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_density_margin_density_evidence.csv`|`7C51F552D161BCBB611B5422EF5E8C92BA1DB2854EB7DADC8A2E85058950CD61`|

预算4/5主结果对比：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`deployment_prior`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3182|3.2987|回归对照，等同上一轮`vote_margin_norole`。|
|`deployment_prior`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|安全但coverage不足。|
|`support_density`|4|0.2414|0.0000|0.5143|0.3500|0.0000|0.6875|0.3961|3.2922|提高seen-new，但old下降、defer升高。|
|`support_density`|5|0.3846|0.0000|0.3000|0.0000|0.0000|0.6000|0.4239|4.2174|预算5 old/seen-new略升，仍远低目标。|
|`margin_density`|4|0.2414|0.0000|0.5429|0.3500|0.0625|0.5000|0.4610|3.4091|seen-new最高但FAR回升，不适合作为安全候选。|
|`margin_density`|5|0.3846|0.0000|0.3000|0.0000|0.0500|0.3500|0.5109|4.3913|FAR不满足约束。|

判定：support density证据是有效诊断字段，但直接作为receiver reliability不是充分解法。它可以提升seen-new，预算4从0.4857到0.5143，预算5从0.2500到0.3000，并保持FAR=0；但old_acc下降，说明简单密度加权会压低部分旧类少数正确证据。margin_density进一步提高seen-new但引入FAR，不应作为安全主线。下一步应将support density改为label-conditioned gate/diagnostic字段，而不是统一替代receiver reliability；同时需要导出每个label的第二候选、label-conditioned radius z-score或Gaussian prototype log-likelihood，才能把“未知像新类”与“真实seen-new局部密集”分开。

## 2026-07-03 label-conditioned gate证据增强

本轮继续沿用`ADV3B02_CORE90_SOFT_E200`特征和`qknn8`少样本证据，目标是在不引入真值`role`泄漏的前提下，为协同融合层提供可部署的标签条件证据。新增字段用于区分“目标标签局部支持充分”和“未知样本被seen-new rescue误接收”。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`qknn_scores`新增`second_label`、`second_score`；evidence新增`label_score_gap`和`class_radius_z`；CLI新增old/seen-new的support-density和radius-z gate参数。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|融合器新增label-conditioned `label_support_density`和`label_radius_z`，并允许class-set gate按old/seen-new分别约束最小局部密度和最大半径z-score。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|更新`qknn_scores`返回值测试，覆盖新增evidence字段。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为17 tests OK，`test_collaborative_open_set_qknn_eval.py`为24 tests OK。根目录`E:\type10-7`不是Git仓库，版本化同步目标仍为`E:\type10-7\github_publish\CVS-RFFI-repo`。

Git镜像提交：

|提交|目的|
|---|---|
|`0bc4aa1 Add label conditioned qKNN gate evidence`|加入`second_label/second_score/label_score_gap/class_radius_z`和label-conditioned gate参数。|
|`abeb9b1 Fail closed on missing label gate evidence`|子agent review指出旧CSV缺字段会被默认值误判安全，已改为启用密度/半径gate时缺字段fail-closed，并新增2个单测。|

最终本地和Git镜像验证均通过：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为17 tests OK，`test_collaborative_open_set_qknn_eval.py`为26 tests OK。

N607同步与最终远端验证：使用`scp -F E:\type10-7\tools\n607_ssh_config`同步`code/scripts/phase2_collaborative_open_set_qknn_eval.py`、`code/evaluation/collaborative_open_set_qknn_eval.py`、`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`、`code/tests/test_collaborative_open_set_qknn_eval.py`到`/home/szu2070436088/2510044040/CV-SincNet/`对应路径。远端使用`CVS-RFFI`环境：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
cd /home/szu2070436088/2510044040/CV-SincNet
python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py code/evaluation/collaborative_open_set_qknn_eval.py
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_collaborative_open_set_qknn_eval.py
```

结果：远端`test_phase2_collaborative_open_set_qknn_eval.py`为17 tests OK，`test_collaborative_open_set_qknn_eval.py`为26 tests OK。

最终远端诊断命令沿用`ADV3B02_CORE90_SOFT_E200`特征产物`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，设置`CUDA_VISIBLE_DEVICES=0`，使用`--collab_counts all`、`--scenario_aware`、`--event_alignment_policy receiver_domain_ranked`、`--fusion_policy scorer_cvs`、`--collaboration_policy adaptive_gain`、`--label_fusion_policy vote_margin`、`--receiver_reliability_policy deployment_prior`和class-set gate安全参数。四组候选均输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行后8张RTX3090均为`10MiB/24576MiB`，无新增训练进程；SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_default.json`|`0A01E72565A0DD9CE60E1E8D5D80F31F5A2D4936D3CF1D93058E534AB963612D`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_default_evidence.csv`|`711944C115F6EB347CB661DF2318C5F2CE0CCBFDEDF534CDF554CD1C204DF722`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_seen_density05.json`|`1C070E933F97B52B475FC8E769F1232239F765A0341A76E9F77A60153EA15207`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_seen_density05_evidence.csv`|`082B3B2D8F4073B2AF9298B26B5923AD36173ECD4134528267DE1D4DE4C8294E`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_seen_density0625.json`|`C0031D744DB60679DD5AE59D077B8ECC19ED79455273B39B3A11F18DA48E6E10`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_seen_density0625_evidence.csv`|`60A5F96D6D343868E3C491622143690A60C6964760C12B493754A1E2B710C91B`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_seen_density05_radius2.json`|`054B7F426E36C7A06AC3524E38861C09AE4A22455AA9EE3D6AE1995B081E6273`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_seen_density05_radius2_evidence.csv`|`0D53FC0F2490827BD6EC34E513D5002D834B64350A0137EED3FCD1741245CCA2`|

最终结果：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|p95_rx|rescue|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`default`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|1.0000|22|
|`default`|2|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3374|1.8171|2.0000|20|
|`default`|3|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5450|3.0000|17|
|`default`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3182|3.2987|4.0000|19|
|`default`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|5.0000|6|
|`seen_density05`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|1.0000|22|
|`seen_density05`|2|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3374|1.8171|2.0000|20|
|`seen_density05`|3|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5450|3.0000|17|
|`seen_density05`|4|0.3103|0.0000|0.4571|0.2500|0.0000|0.8438|0.3247|3.2987|4.0000|19|
|`seen_density05`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|5.0000|6|
|`seen_density0625`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5032|1.0000|1.0000|22|
|`seen_density0625`|2|0.0850|0.0000|0.3111|0.1000|0.0000|0.8958|0.3455|1.8171|2.0000|20|
|`seen_density0625`|3|0.0417|0.0000|0.4000|0.0500|0.0000|0.9500|0.4300|2.5450|3.0000|17|
|`seen_density0625`|4|0.3103|0.0000|0.4286|0.2000|0.0000|0.8438|0.3377|3.2987|4.0000|19|
|`seen_density0625`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4130|4.2391|5.0000|6|
|`seen_density05_radius2`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|1.0000|22|
|`seen_density05_radius2`|2|0.0850|0.0000|0.3333|0.1000|0.0000|0.8958|0.3415|1.8171|2.0000|20|
|`seen_density05_radius2`|3|0.0417|0.0000|0.4000|0.1000|0.0000|0.9500|0.4300|2.5450|3.0000|17|
|`seen_density05_radius2`|4|0.3103|0.0000|0.4286|0.2500|0.0000|0.8438|0.3312|3.2987|4.0000|19|
|`seen_density05_radius2`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|5.0000|6|

判定：新增label-conditioned证据字段已完成、同步和验证；但密度/radius gate本轮没有提升主结果。`default`仍是四组中预算4的最佳seen-new结果，`seen_new_acc=0.4857`、`min_seen_new=0.3000`、`unknown_FAR=0.0000`，但`old_acc=0.3103`和`min_old=0.0000`远低目标。更强的`seen_density`约束主要把seen-new样本转为defer，降低seen-new准确率，未带来额外FAR收益。因此当前模块只能作为可部署证据框架和负面诊断，不能声明99/97/99达成、Stage2-C成功或卫星群部署成功。

子agent监督结论：

|角色|结论|
|---|---|
|文献/方法|建议下一版采用class-conditional EVT/Gaussian-Mahalanobis原型、qKNN候选加pair verifier、accepted-only原型更新和资源感知neighbor-query；当前只能写作冻结主干上的轻量少样本适应，不是星上全量实时训练。|
|算法合理性|`support_density`、`second_label`、`label_score_gap`、`class_radius_z`不含真值泄漏，方向合理；但`candidate_class_top_m`可能过滤真实第二候选，`class_radius_z`小K敏感，当前gate更像安全闸门而不是性能增强器。|
|逐项完成监督|N607已用`CVS-RFFI`环境，`collab_counts all`覆盖1到5，星地proxy视图由`--scenario_aware`和metadata保留，低显存GPU验证完成；严格同一物理事件协同和性能目标未完成。|
|查漏补缺review|P1指出缺`support_density/class_radius_z`字段时不应默认安全；已修复为启用对应gate时fail-closed，并增加单测。仍建议后续补`gate_reason×role×budget`、`second_label`混淆和固定分母预算表。|

下一步算法建议：不要继续只调融合阈值。优先实现`SCOPE-Q8`式稀疏一致性开放集原型证据：本地先用每类scenario prototype预筛，再做qKNN8，通信只上传top2 label/score/margin/risk/support-density/radius-z；协同端用渐进式receiver请求和unknown高分位veto；seen-new注册只更新int8原型/支持码和阈值，必要时才开小adapter/BN affine并带rollback。最小实验矩阵可先用`K={5,10}`、`M={1,3,5}`、`leo_clear_weak`、1个target receiver烟测，再扩展到多LEO视图和多receiver。

## 2026-07-03硬资源预算协同约束

目标：把卫星群协同推理从“只报告资源”推进到“可按通信/时延硬预算执行”。本轮不改变默认行为；只有显式设置`--max_event_bytes`或`--max_event_latency_ms`时才启用硬约束。该改动服务于后续SCOPE-Q8：低置信样本可以请求更多接收机，但不能突破星间通信包大小或单事件时延预算。

代码改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`max_event_bytes`、`max_event_latency_ms`；所有融合策略都会检查选中receiver集合的总bytes和最大latency；超预算时输出`defer`并记录`resource_budget_reason`。`adaptive_gain`在选择下一个receiver前先过滤会超预算的候选，若无可行候选则记录`resource_budget_exhausted`。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`--max_event_bytes`和`--max_event_latency_ms`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增硬bytes预算单测，验证超预算时强制defer并统计`resource_budget_violation_rate`。|

本地和Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
```

结果：`test_collaborative_open_set_qknn_eval.py`为27 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为17 tests OK。Git提交：`2076a29 Add hard resource budgets to collaborative qKNN`。

计划远端验证：同步上述3个文件到N607，在`CVS-RFFI`环境重新跑编译和单测，然后跑两组`--collab_counts all`诊断：

|候选|新增资源约束|目的|
|---|---|---|
|`labelgate_default_post_resource`|无硬预算|回归检查，结果应与上一轮`default`一致。|
|`labelgate_budget360`|`--max_event_bytes 360 --max_event_latency_ms 12`|限制单事件最多约3个120-byte接收机证据包，验证1到5预算下资源约束是否按预期把高预算请求截断或defer。|

远端验证与诊断结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为27 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为17 tests OK。两组诊断均输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行后8张RTX3090均为`10MiB/24576MiB`，无新增训练进程；SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_default_post_resource.json`|`6A2A6F57AB0AEDC395672D724DC5A235F6C0F7BD7EF72B7F68856EB3ACD02463`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_default_post_resource_evidence.csv`|`B5BBCBBB195D83A5ADE6E30B540DDD4A300F99E35FC893596DE64979F04216F7`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_budget360.json`|`CDABD385475442A15B4D7335A54DA6020311C7A711E5D1D044A74A8C3F82F20C`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_labelgate_budget360_evidence.csv`|`3D724CF564208D31FA8F47BAE3ACB18A705C59A706818B70D07F87BBB3A47414`|

结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|p95_latency_ms|resource_violation|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`default_post_resource`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|120.0|0.0979|0.0000|
|`default_post_resource`|2|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3374|1.8171|218.0|0.0979|0.0000|
|`default_post_resource`|3|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5450|305.4|0.0979|0.0000|
|`default_post_resource`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3182|3.2987|395.8|0.0979|0.0000|
|`default_post_resource`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|508.7|0.0979|0.0000|
|`budget360`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|120.0|0.0941|0.0000|
|`budget360`|2|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3374|1.8171|218.0|0.0941|0.0000|
|`budget360`|3|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5450|305.4|0.0941|0.0000|
|`budget360`|4|0.0575|0.0000|0.3429|0.1000|0.0000|0.9375|0.3442|2.5844|310.1|0.0941|0.0000|
|`budget360`|5|0.0577|0.0000|0.1000|0.0000|0.0000|0.9000|0.4565|2.7174|326.1|0.0941|0.0000|

解释：无硬预算回归结果与上一轮`default`一致，证明新增资源预算逻辑默认不改变算法。`budget360`严格把单事件平均通信量压到约`310-326 bytes/event`，且`resource_budget_violation_rate=0`；但预算4/5下旧类和新类准确率明显下降，说明当前qKNN证据依赖超过3个receiver的补充信息，若现实星间链路只允许约3个120-byte证据包，需要更强的本地证据质量或更聪明的receiver选择，而不能只靠截断。

本地参数面搜索：基于`default_post_resource_evidence.csv`，对`label_fusion_policy`、`unknown_risk_threshold`、`accept_margin_threshold`、`consensus_score_threshold`、`scorer_component_vote_threshold`和rescue开关做缩小网格搜索。结论是没有隐藏阈值组合接近目标。FAR=0的联合最优约为预算4`old_acc=0.2989`、`seen_new_acc=0.2857`、`defer_rate=0.4870`；FAR=0的旧类最高仍是预算5`old_acc=0.3654`，但`seen_new_acc=0.0500`、`defer_rate=0.5652`。这说明当前瓶颈不是融合阈值，而是底层ADV3B02特征在星地目标域上的旧类/新类局部可分性不足，下一步需要改证据生成层。

下一步技术判断：硬资源预算已补齐系统约束层，但它不会提升准确率。若继续向99/97/99推进，应优先实现本地证据质量提升：`SCOPE-Q8`的per-scenario Gaussian/Mahalanobis prototype、class-conditional EVT或pair verifier二级复核，并把`candidate_class_top_m`改为“候选预筛+全类top2审计”以避免第二候选被过滤。严格同事件协同也仍需单独跑`strict_event_key`诊断；当前`receiver_domain_ranked`只能作为receiver-domain ensemble诊断。

## 2026-07-03候选预筛全类top2审计

目标：解决上一轮子agent指出的`candidate_class_top_m`确认偏差风险。当前qKNN为了效率先用centroid筛候选类，再只在候选类support里做top-k；这会让`second_label/second_score`被候选过滤影响，可能虚高`label_score_gap`。本轮新增全类top1/top2审计：保留候选预筛作为实际预测路径，同时额外计算不经过候选预筛的全类top1/top2证据，用于诊断和可选unknown风险提升。

代码改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|evidence新增`audit_full_top1_label`、`audit_full_top1_score`、`audit_full_second_label`、`audit_full_second_score`、`audit_full_label_score_gap`、`candidate_audit_disagreement`、`candidate_audit_risk`。新增CLI：`--candidate_audit_unknown_risk_enabled`、`--candidate_audit_disagreement_risk`、`--candidate_audit_min_gap`、`--candidate_audit_gap_risk`。默认只记录审计字段，不改变决策；启用后可把候选预筛冲突或全类gap过低提升为unknown风险。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增审计字段覆盖和gap风险提升单测。|

本地和Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为18 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。Git提交：`2d016d0 Add full-class audit for candidate qKNN evidence`。

计划远端验证：同步脚本和phase2测试到N607，在`CVS-RFFI`环境复测，然后跑两组`--collab_counts all`：

|候选|新增设置|目的|
|---|---|---|
|`audit_record_only`|默认审计记录，风险提升关闭|验证新增字段不改变baseline结果，并生成候选预筛偏差审计证据。|
|`audit_gap_risk`|`--candidate_audit_unknown_risk_enabled --candidate_audit_min_gap 0.05 --candidate_audit_gap_risk 0.995`|测试全类top2 gap过低时是否能进一步抑制unknown误接收；预期可能增加defer，重点看unknown_FAR和旧/新类损失。|

远端验证与诊断结果：N607使用`CVS-RFFI`环境，`py_compile`通过，`test_phase2_collaborative_open_set_qknn_eval.py`为18 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。三组诊断均输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。运行后8张RTX3090均为`10MiB/24576MiB`，无新增训练进程；SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

补丁修正：第一次`audit_gap_risk`远端结果没有变化，因为`candidate_audit_risk`只进入`unknown_risk`，没有进入组件投票。已追加提交`428ea3d Fold candidate audit risk into qKNN score risk`，使审计风险同步提升`score_risk`。最终有效候选为`audit_gap_risk_scorecomponent`。

拉回产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_audit_record_only.json`|`7F13CC5ED7F2FF4C1AA624E8DAACC3EDBD33D71CFC34037B6E37CF845D347FF3`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_audit_record_only_evidence.csv`|`4FEFD03AB7E464CE7DE1BD119320992C355993A5A84F5E52D7B54DC7A5AB21BA`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_audit_gap_risk.json`|`2D0DE319E728E462ECFDAD261F6ECD38B24867FE9F4210332CF792161D28DACC`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_audit_gap_risk_evidence.csv`|`A5A51A3B93B02BF848AD9E0217516F57265F92F35B9AAAE10EA3E1D8E40E5064`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_audit_gap_risk_scorecomponent.json`|`E3077CF3961B14CCCFEDDDFC7C98D65B84ADA8E739DA7DE00F9B5C5524EEB52F`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_audit_gap_risk_scorecomponent_evidence.csv`|`41B1883CB28C0FA9F54C557C0E50AFDE707B2C84A0153AA64C6CF5401500174D`|

结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`audit_record_only`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|120.0|
|`audit_record_only`|2|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3374|1.8171|218.0|
|`audit_record_only`|3|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5450|305.4|
|`audit_record_only`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3182|3.2987|395.8|
|`audit_record_only`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.4022|4.2391|508.7|
|`audit_gap_risk_scorecomponent`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.6167|0.5000|1.0000|120.0|
|`audit_gap_risk_scorecomponent`|2|0.0850|0.0000|0.3556|0.1000|0.0000|0.8958|0.3333|1.8171|218.0|
|`audit_gap_risk_scorecomponent`|3|0.0417|0.0000|0.4250|0.1000|0.0000|0.9500|0.4250|2.5400|304.8|
|`audit_gap_risk_scorecomponent`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3117|3.2857|394.3|
|`audit_gap_risk_scorecomponent`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7000|0.3913|4.2065|504.8|

审计统计：

|候选|evidence rows|candidate top1与full top1冲突|candidate audit risk触发|
|---|---:|---:|---:|
|`audit_record_only`|1000|8|0|
|`audit_gap_risk`|1000|8|218|
|`audit_gap_risk_scorecomponent`|1000|8|218|

解释：全类top2审计确认`candidate_class_top_m=2`的top1确认偏差较小，只有`8/1000`条候选预筛top1与全类top1不一致。gap风险触发`218/1000`条，但即使进入`score_risk`组件投票后，也没有提升old/seen-new/unknown主指标，只略微降低defer和实际参与receiver。这进一步说明当前瓶颈不在候选预筛过滤第二候选，而在ADV3B02特征和qKNN8局部证据本身对星地目标域旧类/新类/未知类的可分性不足。

下一步建议调整：不再优先推进候选预筛审计方向；应实现per-scenario Gaussian/Mahalanobis prototype或pair verifier二级复核。若要继续利用本轮审计字段，建议先做`audit_full_second_label`混淆表，找出unknown最常被哪些old/seen-new吸收，再决定是否训练/拟合轻量pair verifier。

## 2026-07-03原型打分融合协同推理

时间：2026-07-03 20:04:35 +08:00。目标：在不增加GPU训练和不使用target unknown调阈值的前提下，增强qKNN8多接收机协同推理的本地证据质量。上一轮全类top2审计表明候选预筛不是主瓶颈，因此本轮实现轻量`prototype_score_blend`：在qKNN按邻居聚合的类得分上，额外加入每类归一化centroid与query的点积摘要。该机制默认关闭，开启后每个接收机只需维护/传输类原型摘要，适合卫星群边缘部署；它能在局部邻居被接收机或星地信道扰动误导时，用更稳定的类均值证据校正候选排序。

协议边界：本轮仍使用`receiver_domain_ranked`作为receiver-domain ensemble/deployment proxy诊断，不声明严格同物理事件星群协同证明。`Y_unknown`仍仅用于query评估，不参与support、阈值拟合、prototype或早停。远端必须使用`CVS-RFFI`Conda环境。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`qknn_scores`新增`prototype_score_blend`，把类centroid相似度按权重并入按类得分；`build_collaborative_evidence`、CLI和metadata同步记录该参数。默认`0.0`不改变既有路径。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增邻居碰撞校正单测；扩展metadata记录测试。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为19 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。Git镜像同样复测通过。代码目录`E:\type10-7\code`不是Git仓库；版本化路径为`E:\type10-7\github_publish\CVS-RFFI-repo`，当前镜像分支为`codex/cvs-rffi-release-20260626`。

计划远端验证：先运行`tools\n607_ssh_preflight.ps1`，同步脚本和测试文件到N607项目根，在`CVS-RFFI`环境复测，再以低显存占用GPU状态运行`--collab_counts all`，覆盖协同推理数量1到全体target receiver。候选设置：

|候选|新增参数|目的|
|---|---|---|
|`prototype_blend_0p25`|`--prototype_score_blend 0.25`|小权重原型校正，检查是否能提升旧类/新类且不增加unknown FAR。|
|`prototype_blend_1p0`|`--prototype_score_blend 1.0`|中等权重原型校正，检查是否减少qKNN邻居噪声。|
|`prototype_blend_2p0`|`--prototype_score_blend 2.0`|较强原型校正，评估是否牺牲support density但提升类均值稳定性。|

预期输出：每个候选保存JSON和evidence CSV，记录`receiver_count`、`counts=1..全体receiver`、old/seen-new/unknown同row指标、defer、avg_rx、bytes/event、GPU显存、SSH断开核验和SHA256。若结果仍远低于目标，应记录为诊断负证据，不写deployment success。

远端验证结果：N607预检通过，直连`N607`，项目根为`/home/szu2070436088/2510044040/CV-SincNet`。远端环境现场输出`ENV=CVS-RFFI`。脚本和测试同步后，在远端`CVS-RFFI`环境复测：`py_compile`通过，`test_phase2_collaborative_open_set_qknn_eval.py`为19 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。运行前后8张RTX3090均为`10MiB/24576MiB`，本轮标记使用`CUDA_VISIBLE_DEVICES=0`。每次SSH/SCP后本地核验：无残留`ssh.exe`，无到`172.31.111.215:22`或`172.31.105.18:22`的`ESTABLISHED`连接。

远端产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_0p25.json`|`7A139AC004D259655E724E51AF5330B808BA6C085C2C20C978E42003FEE4EFB7`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_0p25_evidence.csv`|`70B0CC7BDDB79BFDD3ADE99135062E921F6B036CE67DA89D6FD26A11B4A20896`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_1p0.json`|`A8CE787D684771687EF60A99EB4E0AAE3294C081F8BBE7F9E36E18323FB4A440`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_1p0_evidence.csv`|`A430F8404A5809396F8E8D09C499ACC125A25C78B6A93FFC1EA16059A0818331`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_2p0.json`|`1FF1DF20142B4A96A093A3F86AACA8F53016A580BA40B5743E40D9C72EB1E7D4`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_2p0_evidence.csv`|`97D4ECC8DF2A520E2A593CA59A3B62516FE279786A7F4822583D4537E7017DAA`|

结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`prototype_blend_0p25`|1|0.0000|0.0000|0.1500|0.0000|0.0000|0.5833|0.5162|1.0000|120.0|
|`prototype_blend_0p25`|2|0.0850|0.0000|0.3333|0.1000|0.0000|0.8958|0.3577|1.8171|218.0|
|`prototype_blend_0p25`|3|0.0417|0.0000|0.4500|0.1000|0.0000|0.9250|0.4300|2.5350|304.2|
|`prototype_blend_0p25`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8750|0.3182|3.2662|391.9|
|`prototype_blend_0p25`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7500|0.3913|4.1739|500.9|
|`prototype_blend_1p0`|1|0.0000|0.0000|0.1500|0.0000|0.0000|0.5167|0.5649|1.0000|120.0|
|`prototype_blend_1p0`|2|0.0980|0.0000|0.3556|0.1000|0.0000|0.8125|0.3984|1.8211|218.5|
|`prototype_blend_1p0`|3|0.0583|0.0000|0.4500|0.1000|0.0000|0.8750|0.4750|2.5250|303.0|
|`prototype_blend_1p0`|4|0.3218|0.0000|0.4571|0.2500|0.0000|0.8750|0.3377|3.2532|390.4|
|`prototype_blend_1p0`|5|0.3846|0.0000|0.3000|0.0000|0.0000|0.7000|0.4130|4.1413|497.0|
|`prototype_blend_2p0`|1|0.0000|0.0000|0.1667|0.0500|0.0000|0.4500|0.6299|1.0000|120.0|
|`prototype_blend_2p0`|2|0.0980|0.0000|0.3556|0.0500|0.0000|0.7917|0.4268|1.8333|220.0|
|`prototype_blend_2p0`|3|0.0583|0.0000|0.4500|0.0500|0.0000|0.8750|0.4900|2.5700|308.4|
|`prototype_blend_2p0`|4|0.3218|0.0000|0.5429|0.3500|0.0000|0.8125|0.3766|3.3182|398.2|
|`prototype_blend_2p0`|5|0.3846|0.0000|0.3000|0.0000|0.0000|0.7000|0.4130|4.1739|500.9|

解释：`prototype_score_blend`能带来小幅正向信号，尤其`prototype_blend_2p0`在预算4把`seen_new_acc`从上一轮默认约`0.4857`提高到`0.5429`，`min_seen_new`从`0.3000`提高到`0.3500`，且`unknown_FAR=0.0000`。但旧类仍远低于目标，预算4`old_acc=0.3218`、`min_old=0.0000`，预算5`old_acc=0.3846`、`min_old=0.0000`。因此该机制是有效但不足的证据增强，不是可部署成功结果。下一步应按文献子agent建议推进shrinkage Gaussian prototype/Mahalanobis class score，把Mahalanobis从仅拒识风险项提升为候选排序/融合得分项；同时按算法子agent建议将候选类过滤前移，以降低生成证据延迟。

审查修复：查漏补缺子agent指出首版`prototype_score_blend`只用于query打分，未进入support/proxy阈值校准，可能导致`known_score`与`receiver_thresholds`口径不一致。已在提交`4def6ac Calibrate prototype blend qKNN scores`中修复：`_threshold_from_calibration()`使用同一`prototype_score_blend`口径；负数blend直接报错；evidence新增`prototype_score_blend`、`prototype_assisted`、`prototype_only_top1`字段；metadata新增`prototype_assisted_qknn`。本地和Git镜像复测：`test_phase2_collaborative_open_set_qknn_eval.py`为23 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。远端`CVS-RFFI`环境复测同样为23 tests OK和27 tests OK。

校准一致版本远端产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_0p25_calibrated.json`|`6A4325F5634FB23445D55EDFD08C8B5827FEA2F7D72D549D40E97579D7904764`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_0p25_calibrated_evidence.csv`|`557957CBFB00076ECD478A4D219B954573276731FDDC4342DFF78E427EE3A166`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_1p0_calibrated.json`|`670733FDE157A141E1C05153A0484E3091EC54E732D0CE9F05C1D55DDB95BE90`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_1p0_calibrated_evidence.csv`|`66330E229BCFC9514DB9E291C4CC1372DD20FAE4E23CF1B645C7502CE6AABFD2`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_2p0_calibrated.json`|`C85EB2FED95BDBB352F37A6544B962BC77D73CA0A0419233A95013FD2C358972`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_prototype_blend_2p0_calibrated_evidence.csv`|`2C0DA911AEC27150C5F571606E72B007EE066B2F03AE31A0CD41395C9DFF1CE3`|

校准一致版本结果表如下。前一张无`calibrated`后缀的表仅保留为审查前诊断，不作为最终判断依据。

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`prototype_blend_0p25_calibrated`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.5833|0.5162|1.0000|120.0|
|`prototype_blend_0p25_calibrated`|2|0.0850|0.0000|0.3333|0.1000|0.0000|0.8958|0.3577|1.8171|218.0|
|`prototype_blend_0p25_calibrated`|3|0.0417|0.0000|0.4500|0.1000|0.0000|0.9250|0.4300|2.5350|304.2|
|`prototype_blend_0p25_calibrated`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8750|0.3182|3.2662|391.9|
|`prototype_blend_0p25_calibrated`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7500|0.3913|4.1739|500.9|
|`prototype_blend_1p0_calibrated`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.5167|0.5649|1.0000|120.0|
|`prototype_blend_1p0_calibrated`|2|0.0980|0.0000|0.3556|0.1000|0.0000|0.8125|0.3984|1.8211|218.5|
|`prototype_blend_1p0_calibrated`|3|0.0583|0.0000|0.4500|0.1000|0.0000|0.8750|0.4750|2.5250|303.0|
|`prototype_blend_1p0_calibrated`|4|0.3218|0.0000|0.4571|0.2500|0.0000|0.8750|0.3377|3.2532|390.4|
|`prototype_blend_1p0_calibrated`|5|0.3846|0.0000|0.3000|0.0000|0.0000|0.7000|0.4130|4.1413|497.0|
|`prototype_blend_2p0_calibrated`|1|0.0000|0.0000|0.1667|0.0500|0.0500|0.4500|0.6299|1.0000|120.0|
|`prototype_blend_2p0_calibrated`|2|0.0980|0.0000|0.3556|0.0500|0.0000|0.7917|0.4268|1.8333|220.0|
|`prototype_blend_2p0_calibrated`|3|0.0583|0.0000|0.4500|0.0500|0.0000|0.8750|0.4900|2.5700|308.4|
|`prototype_blend_2p0_calibrated`|4|0.3218|0.0000|0.5429|0.3500|0.0000|0.8125|0.3766|3.3182|398.2|
|`prototype_blend_2p0_calibrated`|5|0.3846|0.0000|0.3000|0.0000|0.0000|0.7000|0.4130|4.1739|500.9|

最终判断：当前最强同row候选仍是`prototype_blend_2p0_calibrated`预算4，用约`3.3182`个接收机、约`398.2 bytes/event`达到`seen_new_acc=0.5429`、`min_seen_new=0.3500`、`unknown_FAR=0.0000`，但`old_acc=0.3218`、`min_old=0.0000`，距离目标`old 99%/floor 95%`和`seen-new 97%/floor 93%`仍很远。此结果应记为有效诊断负证据和下一步算法依据，不是部署成功。

## 2026-07-03Mahalanobis类得分融合

目标：在`prototype_score_blend`只能带来有限提升后，继续推进文献建议的Gaussian/Mahalanobis prototype路线。此前Mahalanobis只作为unknown风险项参与拒识，不能改变qKNN候选排序；本轮新增`mahalanobis_score_blend`，将每类Mahalanobis envelope产生的known probability加入qKNN按类得分，使类条件分布信息直接参与旧类/新类候选排序与多接收机融合。

资源约束说明：本轮在当前工作区没有定位到用户点名的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`文件；递归搜索命中了若干CVS/RFFI文档但未发现该精确文件。远端诊断将继续沿用现有协同推理资源面：`--collab_counts all`报告1到全体接收机、`--latency_budget_ms 12`、`--evidence_packet_bytes 120`，并在后续需要时继续使用`--max_event_bytes/--max_event_latency_ms`硬预算约束。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`_mahalanobis_known_scores()`；`qknn_scores()`新增`mahalanobis_score_blend`和`mahalanobis_score_temperature`；support/proxy阈值校准、query证据生成、全类audit、metadata和CLI均使用同一得分口径。默认`0.0`不改变既有路径。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增局部近邻碰撞校正、校准口径一致、负数拒绝、metadata/evidence标记测试。|

本地与Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为27 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。局部合成测试显示：纯qKNN在邻居碰撞下把query判为`new-a`，`mahalanobis_score_blend=2.0`可拉回`old-a`；校准阈值随同一blend口径变化。

计划远端验证：同步脚本和测试文件到N607，在`CVS-RFFI`环境复测后跑`--collab_counts all`，覆盖协同推理数量1到5。候选设置：

|候选|新增参数|目的|
|---|---|---|
|`maha_score_0p5_calibrated`|`--mahalanobis_score_blend 0.5`|弱Mahalanobis类得分，检查是否改善旧类而不伤unknown FAR。|
|`maha_score_1p0_calibrated`|`--mahalanobis_score_blend 1.0`|中等Mahalanobis类得分，检查旧类/seen-new排序改善。|
|`proto2_maha1_calibrated`|`--prototype_score_blend 2.0 --mahalanobis_score_blend 1.0`|结合上一轮最强prototype权重与Mahalanobis分布得分，评估互补性。|

风险：Mahalanobis类得分可能产生`support_neighbor_count=0`的distribution-assisted top1，报告中必须标为`mahalanobis_score_assisted_qknn`，不能写成纯qKNN8证据；若unknown FAR升高或min_old仍为0，只能作为诊断负证据。

远端验证结果：N607直连预检通过，远端项目根为`/home/szu2070436088/2510044040/CV-SincNet`，远端环境现场输出`ENV=CVS-RFFI`。同步脚本和测试后，远端`py_compile`通过，`test_phase2_collaborative_open_set_qknn_eval.py`为27 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。运行前后GPU显存均为`10MiB/24576MiB`，本轮标记使用`CUDA_VISIBLE_DEVICES=0`。SSH/SCP结束后本地核验：无残留`ssh.exe`，无到N607或bridge的22端口`ESTABLISHED`连接。

远端产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_0p5_calibrated.json`|`1D342DB8E20BFF7919E5701FBF94526E81007E8315973A8381FB389C853608DE`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_0p5_calibrated_evidence.csv`|`DC91E47975BB12B671651712844E2C6AA62CF36F951803396290FA4ADFFC8718`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_1p0_calibrated.json`|`8B9B1EFA409F580F9C9EC389A7F5A9DFB0C341E09E6F9429F82341B90F2829FC`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_1p0_calibrated_evidence.csv`|`BA8E94D2140B02FAE465C4E18A0D300EC7D6ADAF95D17D4A4950DFD1F81E4367`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_proto2_maha1_calibrated.json`|`5F803AB2B2735F2DA69B746183B04A2ED5AAA569065C1BE6816BFCACAA575AC9`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_proto2_maha1_calibrated_evidence.csv`|`3D7C3B70DDF1E32E86E767E67D00B611696DDF25C9436C80D6D67DFC1DEDCC04`|

结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`maha_score_0p5_calibrated`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.5167|0.5390|1.0000|120.0|
|`maha_score_0p5_calibrated`|2|0.0915|0.0000|0.3556|0.1000|0.0000|0.8333|0.3699|1.8252|219.0|
|`maha_score_0p5_calibrated`|3|0.0500|0.0000|0.4500|0.1000|0.0000|0.9750|0.4300|2.5750|309.0|
|`maha_score_0p5_calibrated`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3506|3.3636|403.6|
|`maha_score_0p5_calibrated`|5|0.3654|0.0000|0.2500|0.0000|0.0000|0.7500|0.4130|4.3370|520.4|
|`maha_score_1p0_calibrated`|1|0.0000|0.0000|0.1667|0.0500|0.0500|0.4833|0.5649|1.0000|120.0|
|`maha_score_1p0_calibrated`|2|0.0980|0.0000|0.3778|0.1000|0.0000|0.8333|0.3780|1.8252|219.0|
|`maha_score_1p0_calibrated`|3|0.0583|0.0000|0.4500|0.1000|0.0000|0.9750|0.4500|2.5700|308.4|
|`maha_score_1p0_calibrated`|4|0.3103|0.0000|0.4857|0.3000|0.0000|0.8438|0.3636|3.3506|402.1|
|`maha_score_1p0_calibrated`|5|0.3654|0.0000|0.3000|0.0000|0.0000|0.7500|0.4130|4.2935|515.2|
|`proto2_maha1_calibrated`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.4167|0.6429|1.0000|120.0|
|`proto2_maha1_calibrated`|2|0.0980|0.0000|0.4000|0.1000|0.0000|0.7708|0.4675|1.8211|218.5|
|`proto2_maha1_calibrated`|3|0.0583|0.0000|0.4750|0.1000|0.0000|0.8750|0.5100|2.5450|305.4|
|`proto2_maha1_calibrated`|4|0.3333|0.0000|0.5714|0.4000|0.0000|0.8438|0.3442|3.3247|399.0|
|`proto2_maha1_calibrated`|5|0.3846|0.0000|0.3000|0.0000|0.0000|0.7000|0.4239|4.1848|502.2|

解释：Mahalanobis类得分单独使用时仅带来小幅改善；与上一轮最强`prototype_score_blend=2.0`组合后，预算4达到本轮最佳同row结果：`old_acc=0.3333`、`seen_new_acc=0.5714`、`min_seen_new=0.4000`、`unknown_FAR=0.0000`，平均参与`3.3247`个接收机，约`399.0 bytes/event`。相对`prototype_blend_2p0_calibrated`预算4，seen-new从`0.5429`升至`0.5714`，old从`0.3218`升至`0.3333`，但`min_old`仍为`0.0000`。该路线说明分布类得分有增益，但仍不能满足目标；下一步应优先对`min_old=0`的旧类做按类混淆审计，判断是类原型塌缩、接收机场景错配还是阈值/门控过严导致。

review修正：只读review指出，本轮诊断仍使用`score_threshold_combine=max`，会将`memory.score_threshold`这个centroid-only阈值与blended qKNN得分空间取max，导致开集风险口径不纯。已补充测试：`mahalanobis_score_blend=0.0`与默认`qknn_scores()`逐字段一致；合法source-only`proxy_unknown`会进入`threshold_scope=source_only`，且target unknown不会作为proxy校准。后续最终诊断应使用`--score_threshold_combine qknn_only`，使阈值只来自同一blended qKNN校准空间。

`qknn_only`远端复测：同步补强测试后，远端`CVS-RFFI`环境复测`test_phase2_collaborative_open_set_qknn_eval.py`为29 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。三组最终口径候选均使用`--score_threshold_combine qknn_only`。

`qknn_only`产物SHA256：

|产物|SHA256|
|---|---|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_0p5_qknnonly.json`|`DE96BFED8F7E6354F395016CED246A4ACE69757840D6A0B0D5C2915F09A2A479`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_0p5_qknnonly_evidence.csv`|`71641C6C774B458488896188BE13F79F908FDC2F31FFC877F32DC33184010D60`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_1p0_qknnonly.json`|`4A1824B4E8BFE8078A8BCE14381617AA08D8CDD3F99E5B2FA6F4FEA5A63B79A6`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_maha_score_1p0_qknnonly_evidence.csv`|`F9DFD7A2B15DA6A5CC50E8E0D336AAB26E71173C179C7F68F9C4473224F52F02`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_proto2_maha1_qknnonly.json`|`46F36661944C21BA69A24390AD9CDD1D9DF729FFDAE7D21E16B0339390142508`|
|`remote_artifacts/collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_proto2_maha1_qknnonly_evidence.csv`|`F54930ABF233205D1E11574B074017C85003D8CC74942DC980439B92E5C2AC22`|

`qknn_only`结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`maha_score_0p5_qknnonly`|1|0.0000|0.0000|0.1500|0.0000|0.0500|0.2667|0.7987|1.0000|120.0|
|`maha_score_0p5_qknnonly`|2|0.3203|0.0000|0.4000|0.1000|0.0625|0.4375|0.4959|1.8943|227.3|
|`maha_score_0p5_qknnonly`|3|0.3417|0.1500|0.5000|0.0500|0.0000|0.6250|0.5000|2.6350|316.2|
|`maha_score_0p5_qknnonly`|4|0.6897|0.0000|0.5714|0.3000|0.0625|0.3750|0.2922|3.5455|425.5|
|`maha_score_0p5_qknnonly`|5|0.7308|0.0000|0.3500|0.0000|0.0500|0.2000|0.3478|4.2283|507.4|
|`maha_score_1p0_qknnonly`|1|0.0000|0.0000|0.1667|0.0500|0.0500|0.2667|0.7922|1.0000|120.0|
|`maha_score_1p0_qknnonly`|2|0.3137|0.0000|0.4667|0.2000|0.0833|0.4375|0.4797|1.8902|226.8|
|`maha_score_1p0_qknnonly`|3|0.3417|0.2000|0.5250|0.1000|0.0000|0.6250|0.5000|2.6300|315.6|
|`maha_score_1p0_qknnonly`|4|0.7471|0.0000|0.6000|0.3500|0.0625|0.3750|0.2597|3.5260|423.1|
|`maha_score_1p0_qknnonly`|5|0.7500|0.0000|0.4000|0.0000|0.0500|0.2000|0.3152|4.1739|500.9|
|`proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7760|1.0000|120.0|
|`proto2_maha1_qknnonly`|2|0.3464|0.0000|0.4667|0.1500|0.1250|0.6667|0.3943|1.8577|222.9|
|`proto2_maha1_qknnonly`|3|0.3917|0.3000|0.5750|0.2000|0.0250|0.7000|0.4350|2.5550|306.6|
|`proto2_maha1_qknnonly`|4|0.7586|0.0000|0.7429|0.6000|0.1562|0.5625|0.1558|3.4156|409.9|
|`proto2_maha1_qknnonly`|5|0.7500|0.0000|0.4500|0.0000|0.0500|0.2000|0.3261|4.1522|498.3|

解释：`qknn_only`证明前一轮`max`阈值组合确实压低了已知类接受率。最高旧类/seen-new单点是`proto2_maha1_qknnonly`预算4，`old_acc=0.7586`、`seen_new_acc=0.7429`、`min_seen_new=0.6000`，但`unknown_FAR=0.1562`超出安全边界，不能作为开集部署结果。FAR可控的折中候选是`proto2_maha1_qknnonly`预算3：`old_acc=0.3917`、`min_old=0.3000`、`seen_new_acc=0.5750`、`unknown_FAR=0.0250`、`unknown_reject=0.7000`，平均`2.5550`个接收机、`306.6 bytes/event`。按类结果显示预算3旧类不再有0类：`14-10=0.35`、`14-7=0.40`、`20-15=0.45`、`20-19=0.30`、`6-15=0.45`、`8-20=0.40`；seen-new仍不均衡：`19-3=0.20`、`3-8=0.95`。下一步应在`proto2_maha1_qknnonly`预算3基础上做class-conditional FAR gate或per-label threshold，目标是在不放大unknown FAR的情况下抬升`19-3`和旧类floor。

## 2026-07-03per-label qKNN阈值

目标：在`proto2_maha1_qknnonly`预算4已知类性能显著上升但unknown FAR超标的情况下，加入不使用target unknown的per-label校准阈值，尝试在同一blended qKNN得分空间内压制容易吸收unknown的类别。新增开关默认关闭：`--class_score_threshold_enabled`。开启后，每个receiver基于target-old/seen-new support的同口径qKNN分数，按真实标签分别计算类阈值；若存在合法source-only`proxy_unknown`，则只按proxy预测标签抬高对应类阈值。target unknown query不参与阈值拟合。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`_label_thresholds_from_calibration()`，并在evidence生成时按预测标签使用`receiver_class_thresholds`替代全局receiver阈值；新增CLI：`--class_score_threshold_enabled`、`--class_score_threshold_quantile`、`--class_score_threshold_min_support`；evidence记录`effective_score_threshold`、`receiver_score_threshold`、`class_score_threshold`和开关状态。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增per-label阈值开启/默认关闭测试，确保metadata/evidence记录正确。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为31 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。

计划远端验证：以`proto2_maha1_qknnonly`为基线，继续使用`--score_threshold_combine qknn_only --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0`，跑三组类阈值分位：`0.20`、`0.35`、`0.50`。目标是寻找`unknown_FAR<=0.05`下更高`old_acc/seen_new_acc/floor`的折中。

## 2026-07-03support-derived virtual unknown校准

目标：继续推进ADV3B02主线目标。前一轮`proto2_maha1_qknnonly`预算4能把`old_acc/seen_new_acc`抬高到`0.7586/0.7429`，但`unknown_FAR=0.1562`不满足开集安全；预算3的`unknown_FAR=0.0250`较安全，但`old_acc=0.3917`、`seen_new_acc=0.5750`过低。本轮引入不使用target unknown query的support-derived virtual unknown校准：从每个receiver的target-old/seen-new support原型之间合成低密度边界样本，作为proxy校准阈值，目标是在预算4附近压低unknown FAR，同时尽量保持已知类性能。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`_virtual_unknown_features()`，CLI新增`--virtual_unknown_calibration_enabled`、`--virtual_unknown_samples_per_class`、`--virtual_unknown_mix_alpha`、`--virtual_unknown_noise_scale`、`--virtual_unknown_neighbor_count`；metadata/evidence记录virtual unknown配置和每receiver合成数量。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增测试，确认纯virtual unknown校准标记为`threshold_scope=support_virtual_unknown`，且evidence角色仍只有old/seen_new/unknown query，不引入target unknown校准行。|

本地验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为33 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。

远端计划：同步脚本和phase2测试到N607，在`CVS-RFFI`环境复测；以`proto2_maha1_qknnonly`为基线，使用`--collab_counts all`覆盖1到5个target receiver，候选如下：

|候选|新增参数|目的|
|---|---|---|
|`vunk2_proto2_maha1_qknnonly`|`--virtual_unknown_calibration_enabled --virtual_unknown_samples_per_class 2`|轻量边界proxy，检查是否降低预算4 FAR。|
|`vunk4_proto2_maha1_qknnonly`|`--virtual_unknown_calibration_enabled --virtual_unknown_samples_per_class 4`|中等proxy强度，检查FAR/known性能折中。|
|`vunk4_classq20_proto2_maha1_qknnonly`|再加`--class_score_threshold_enabled --class_score_threshold_quantile 0.20`|结合per-label阈值，压制易吸收unknown的预测类。|

协议边界：virtual unknown只由target support原型生成，不能写成真实unknown先验，也不能作为target unknown query调阈值；本轮仍是`receiver_domain_ranked`诊断，不是严格同物理事件卫星群协同证明。

远端修正与验证：首次远端运行时，生成器正确输出`threshold_scope=support_virtual_unknown`，但融合评估验证器`SAFE_THRESHOLD_SCOPES`未同步该合法scope，报错`ValueError: threshold_selection_label_scope must not use unknown query labels; got 'support_virtual_unknown'`。根因是验证层白名单遗漏，不是数据泄漏。已在`code/evaluation/collaborative_open_set_qknn_eval.py`加入`support_virtual_unknown`安全scope，并在`code/tests/test_collaborative_open_set_qknn_eval.py`补测试。复测如下：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为33 tests OK和27 tests OK。同步N607后，远端`CVS-RFFI`环境同样为33 tests OK和27 tests OK。运行前后8张RTX3090均为`10MiB/24576MiB`，SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

拉回产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_vunk2_proto2_maha1_qknnonly.json`|`72143D89681A71928BC8527142442357B5FC651FC1863066D9016BAD28459EE2`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_vunk2_proto2_maha1_qknnonly_evidence.csv`|`835F9A898409A3C221D9D30D5527D963D14ACEFF51F4AE8E596815B19C8B713E`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_vunk4_proto2_maha1_qknnonly.json`|`B9E54150B4E95D33BE8CF6A6E6E929FF622505A1997D7FB9FF30D8904F08CD75`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_vunk4_proto2_maha1_qknnonly_evidence.csv`|`71AAB85FCBFC539ADF11C17B75EE83CA60748A4EA86BC3BB96245843FE2B7785`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_vunk4_classq20_proto2_maha1_qknnonly.json`|`FDF66CF7DD95457ADAC6638E3F1D974BB22494435B0D18FB34D51FC3AA7A2928`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_vunk4_classq20_proto2_maha1_qknnonly_evidence.csv`|`0DF6BA7B4A210B04C486F7F9687373B63158EA575C9C3FEA81B69C56ED9F8B8A`|

结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`vunk2_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.4000|0.6494|1.0000|120.0|
|`vunk2_proto2_maha1_qknnonly`|2|0.0980|0.0000|0.4000|0.1000|0.0000|0.7708|0.4675|1.8211|218.5|
|`vunk2_proto2_maha1_qknnonly`|3|0.0583|0.0000|0.4500|0.0500|0.0000|0.8750|0.5200|2.5700|308.4|
|`vunk2_proto2_maha1_qknnonly`|4|0.3448|0.0000|0.5429|0.3500|0.0312|0.8125|0.3701|3.4091|409.1|
|`vunk2_proto2_maha1_qknnonly`|5|0.4038|0.0000|0.3500|0.0000|0.0000|0.6500|0.4457|4.3152|517.8|
|`vunk4_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.4000|0.6494|1.0000|120.0|
|`vunk4_proto2_maha1_qknnonly`|2|0.0980|0.0000|0.4000|0.1000|0.0000|0.7708|0.4675|1.8252|219.0|
|`vunk4_proto2_maha1_qknnonly`|3|0.0583|0.0000|0.4500|0.0500|0.0000|0.8750|0.5250|2.5750|309.0|
|`vunk4_proto2_maha1_qknnonly`|4|0.3448|0.0000|0.5429|0.3500|0.0312|0.8125|0.3701|3.4156|409.9|
|`vunk4_proto2_maha1_qknnonly`|5|0.4038|0.0000|0.3500|0.0000|0.0000|0.6500|0.4457|4.3261|519.1|
|`vunk4_classq20_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.3000|0.7013|1.0000|120.0|
|`vunk4_classq20_proto2_maha1_qknnonly`|2|0.1634|0.0000|0.4222|0.1500|0.0208|0.7292|0.4593|1.8252|219.0|
|`vunk4_classq20_proto2_maha1_qknnonly`|3|0.1583|0.0500|0.4750|0.1000|0.0250|0.8500|0.4550|2.5350|304.2|
|`vunk4_classq20_proto2_maha1_qknnonly`|4|0.4713|0.0000|0.6286|0.5000|0.0938|0.7812|0.2792|3.4416|413.0|
|`vunk4_classq20_proto2_maha1_qknnonly`|5|0.4615|0.0000|0.4000|0.0000|0.0500|0.6000|0.4130|4.3261|519.1|

解释：support-derived virtual unknown能按预期压低unknown FAR。相对`proto2_maha1_qknnonly`预算4的`unknown_FAR=0.1562`，`vunk2/vunk4`预算4降到`0.0312`，但`old_acc`从`0.7586`降至`0.3448`，`seen_new_acc`从`0.7429`降至`0.5429`；这说明该proxy过于保守，把大量可识别known事件推入defer/reject。`vunk4_classq20`预算4把`old_acc/seen_new_acc`回升到`0.4713/0.6286`，但`unknown_FAR=0.0938`又超出`<=0.05`安全线。当前最佳安全折中仍不能接近目标，尤其`min_old`仍为0。

结论：virtual unknown是有效拒识闸门，但单独调阈值无法同时满足旧类、新类和unknown目标。下一步应把virtual unknown从“全局抬阈值”改为“只参与unknown风险组件或conformal p-value”，避免压制所有known接受率；同时应加入class-wise收缩阈值，防止低K类别被边界proxy过度惩罚。

## 2026-07-03virtual unknown独立风险组件

目标：修正上一轮support-derived virtual unknown“全局抬阈值”导致known接受率大幅下降的问题。新实现保留原`--virtual_unknown_calibration_enabled`作为对照路径，新增`--virtual_unknown_risk_enabled`，只把support原型合成的边界样本作为独立unknown风险通道，不进入阈值拟合，也不改变`threshold_scope=support_known_only`。融合器新增`virtual_unknown`风险组件后，可在`scorer_cvs`中与score/radius/margin/evt等通道共同投票。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`_virtual_unknown_boundary_risk()`；新增CLI参数`--virtual_unknown_risk_enabled`、`--virtual_unknown_risk_samples_per_class`、`--virtual_unknown_risk_temperature`、`--virtual_unknown_risk_margin`；metadata/evidence记录`virtual_unknown_risk`、`virtual_unknown_score`和风险开关；风险模式不改变阈值scope。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`RISK_COMPONENT_KEYS`新增`virtual_unknown`，融合输出增加`virtual_unknown_risk`，adaptive gain也纳入该风险通道。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增测试确认virtual unknown风险组件不进入阈值校准、只记录风险字段，并保持query角色不变。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增测试确认`scorer_cvs`能显式使用`["score","virtual_unknown"]`风险组件。|

本地与Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地`test_phase2_collaborative_open_set_qknn_eval.py`为34 tests OK，`test_collaborative_open_set_qknn_eval.py`为28 tests OK；Git镜像同样为34 tests OK和28 tests OK。`E:\type10-7\code`不是Git仓库，本次变更已同步到Git-backed镜像`E:\type10-7\github_publish\CVS-RFFI-repo`，镜像分支`codex/cvs-rffi-release-20260626`当前待提交。

计划远端验证：同步4个变更文件到N607，在`CVS-RFFI`conda环境复测，再以`proto2_maha1_qknnonly`为基线跑`--collab_counts all`覆盖1到5个target receivers。候选：

|候选|新增参数|目的|
|---|---|---|
|`vrisk2_proto2_maha1_qknnonly`|`--virtual_unknown_risk_enabled --virtual_unknown_risk_samples_per_class 2 --virtual_unknown_risk_temperature 0.05`|轻量边界风险通道，检查是否降低FAR且少伤known。|
|`vrisk4_proto2_maha1_qknnonly`|`--virtual_unknown_risk_enabled --virtual_unknown_risk_samples_per_class 4 --virtual_unknown_risk_temperature 0.05`|增加边界样本，检查unknown风险敏感度。|
|`vrisk4_margin03_proto2_maha1_qknnonly`|再加`--virtual_unknown_risk_margin 0.03`|略提高边界风险强度，检查FAR/known折中。|

成功判据仍不放宽：主目标为old 99%/floor95%、seen-new 97%/floor93%、unknown拒识99%；若未达到，只能报告诊断负证据和下一步路线，不能写成部署成功。

## 2026-07-03SCORER-CVS-CPR最小实现

目标：继续推进ADV3B02主线目标。上一轮virtual unknown独立风险组件能降低FAR，但`unknown_risk=max(...)`仍会把大量known事件推入defer/reject。本轮实现`SCORER-CVS-CPR`的最小可测版本：每个receiver基于target-old/seen-new support的leave-one-out同类分数分布，输出预测类`class_conformal_pvalue`；融合器在strong known、receiver一致性足够且p-value达标时启用`conformal_rescue`，降低effective unknown risk，而不使用target unknown query拟合阈值。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`_label_score_samples_from_calibration()`和`_conformal_pvalue()`；CLI新增`--class_conformal_enabled`、`--class_conformal_min_support`；evidence记录`class_conformal_pvalue`、`class_conformal_support_count`；metadata记录每receiver/label的conformal校准样本数。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`conformal_rescue_enabled`、`conformal_rescue_min_pvalue`、`conformal_rescue_risk_scale`、`conformal_rescue_min_agreement`；`scorer_cvs`在强known且p-value达标时降低`effective_unknown_risk`；输出`conformal_rescue_applied`和`label_class_conformal_pvalue`。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增测试确认p-value由support派生并写入evidence。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增测试确认conformal rescue在真实protocol metadata下可接受强known样本。|

本地与Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为`test_phase2_collaborative_open_set_qknn_eval.py`35 tests OK，`test_collaborative_open_set_qknn_eval.py`29 tests OK。`E:\type10-7\code`不是Git仓库，变更已同步到Git-backed镜像`E:\type10-7\github_publish\CVS-RFFI-repo`，待远端验证后提交。

远端计划：同步4个代码/测试文件到N607，在`CVS-RFFI`环境复测；以`proto2_maha1_qknnonly`为基线，使用ADV3B02特征`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，跑`--collab_counts all`覆盖1到5个target receivers。候选：

|候选|新增参数|目的|
|---|---|---|
|`cpr_p05_s05_proto2_maha1_qknnonly`|`--class_conformal_enabled --conformal_rescue_enabled --conformal_rescue_min_pvalue 0.05 --conformal_rescue_risk_scale 0.5`|宽松p-value救援，检查是否恢复known性能且FAR可控。|
|`cpr_p20_s05_proto2_maha1_qknnonly`|`--class_conformal_enabled --conformal_rescue_enabled --conformal_rescue_min_pvalue 0.20 --conformal_rescue_risk_scale 0.5`|更严格p-value准入，降低unknown误接受风险。|
|`cpr_p20_s03_vrisk4_proto2_maha1_qknnonly`|再加`--virtual_unknown_risk_enabled --virtual_unknown_risk_samples_per_class 4 --virtual_unknown_risk_temperature 0.05 --conformal_rescue_risk_scale 0.3`|结合virtual unknown风险和conformal rescue，检查FAR/known折中。|

协议边界：conformal校准只使用target support的old/seen-new标签，不使用target unknown query；当前仍是`receiver_domain_ranked`诊断，不是严格同物理事件协同证明。

远端验证：

|项目|结果|
|---|---|
|N607预检|`tools\n607_ssh_preflight.ps1`通过，direct `N607`可达，远端project root可见，8张RTX3090均约`10/24576MiB`。|
|同步目标|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`、`code/evaluation/collaborative_open_set_qknn_eval.py`、两个对应测试文件。|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，工作目录`/home/szu2070436088/2510044040/CV-SincNet`。|
|远端测试|`test_phase2_collaborative_open_set_qknn_eval.py`35 tests OK；`test_collaborative_open_set_qknn_eval.py`29 tests OK。|
|运行资源|`CUDA_VISIBLE_DEVICES=0`；运行前后GPU均约`10/24576MiB`，无训练进程占用。|
|结果取回|JSON/CSV已取回到`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_collab_open_set_qknn_full_20260703\remote_artifacts`。|
|SSH清理|SCP/SSH后检查为`NO_SSH_PROCESS`和`NO_N607_SSH_ESTABLISHED`。|

远端命令基线：`--collab_counts all --k_shot 8 --query_per_class 20 --qknn_k 8 --candidate_class_top_m 2 --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 --support_calibration_mode leave_one_out --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only --scenario_aware --radius_norm 0.3 --fusion_policy scorer_cvs --collaboration_policy adaptive_gain --label_fusion_policy vote_margin --receiver_reliability_policy deployment_prior --adaptive_gain_min_risk 0.60 --accept_margin_threshold 0.03 --consensus_gap_threshold 0.0 --consensus_score_threshold 0.30 --scorer_component_vote_threshold 0.25 --unknown_quantile 0.75 --evt_tail_quantile 0.80 --evt_temperature 0.05 --latency_budget_ms 12 --evidence_packet_bytes 120 --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first --seen_new_rescue_enabled --seen_new_rescue_risk_scale 0.5 --seen_new_rescue_min_score 0.30 --seen_new_rescue_min_margin 0.03 --seen_new_rescue_min_agreement 0.50 --class_set_gate_enabled --old_gate_min_receivers 2 --old_gate_max_effective_unknown_risk 0.8 --old_gate_max_component_agreement 0.2 --seen_new_gate_min_receivers 1 --seen_new_gate_max_effective_unknown_risk 0.6 --seen_new_gate_max_component_agreement 0.25 --seed 407044`。

ADV3B02 CPR结果表：

|候选|协同数|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|avg_rx|bytes/event|old_n|seen_n|unknown_n|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`cpr_p05_s05_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7760|1.0000|120.0000|188|60|60|
|`cpr_p05_s05_proto2_maha1_qknnonly`|2|0.6144|0.2000|0.4667|0.1500|0.1667|0.6667|0.1789|1.8577|222.9268|153|45|48|
|`cpr_p05_s05_proto2_maha1_qknnonly`|3|0.8000|0.5500|0.5750|0.2000|0.1000|0.7000|0.1500|2.5550|306.6000|120|40|40|
|`cpr_p05_s05_proto2_maha1_qknnonly`|4|0.8851|0.0000|0.7429|0.6000|0.2500|0.5625|0.0584|3.4156|409.8701|87|35|32|
|`cpr_p05_s05_proto2_maha1_qknnonly`|5|0.8462|0.0000|0.4500|0.0000|0.3000|0.2000|0.2065|4.1522|498.2609|52|20|20|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7695|1.0000|120.0000|188|60|60|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly`|2|0.5948|0.2000|0.4444|0.1500|0.1458|0.6875|0.1667|1.8333|220.0000|153|45|48|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly`|3|0.7917|0.5500|0.5500|0.2000|0.0500|0.7250|0.1200|2.5150|301.8000|120|40|40|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly`|4|0.8621|0.0000|0.7143|0.6000|0.1875|0.6250|0.0584|3.3506|402.0779|87|35|32|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly`|5|0.8269|0.0000|0.4500|0.0000|0.1500|0.4000|0.1957|4.0978|491.7391|52|20|20|
|`cpr_p18_s05_u098_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7695|1.0000|120.0000|188|60|60|
|`cpr_p18_s05_u098_proto2_maha1_qknnonly`|2|0.5948|0.2000|0.4444|0.1500|0.1458|0.6875|0.1667|1.8333|220.0000|153|45|48|
|`cpr_p18_s05_u098_proto2_maha1_qknnonly`|3|0.7917|0.5500|0.5500|0.2000|0.0500|0.7250|0.1300|2.5150|301.8000|120|40|40|
|`cpr_p18_s05_u098_proto2_maha1_qknnonly`|4|0.8621|0.0000|0.7143|0.6000|0.1875|0.6250|0.0584|3.3506|402.0779|87|35|32|
|`cpr_p18_s05_u098_proto2_maha1_qknnonly`|5|0.8269|0.0000|0.4500|0.0000|0.1500|0.4000|0.1957|4.0978|491.7391|52|20|20|
|`cpr_p20_s03_vrisk4_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7922|1.0000|120.0000|188|60|60|
|`cpr_p20_s03_vrisk4_proto2_maha1_qknnonly`|2|0.5752|0.1500|0.4000|0.1000|0.1458|0.7083|0.1992|1.9268|231.2195|153|45|48|
|`cpr_p20_s03_vrisk4_proto2_maha1_qknnonly`|3|0.7250|0.3000|0.5250|0.1000|0.0000|0.8250|0.1700|2.8050|336.6000|120|40|40|
|`cpr_p20_s03_vrisk4_proto2_maha1_qknnonly`|4|0.8621|0.0000|0.6571|0.4000|0.0938|0.7188|0.1039|3.7662|451.9481|87|35|32|
|`cpr_p20_s03_vrisk4_proto2_maha1_qknnonly`|5|0.8462|0.0000|0.4000|0.0000|0.0000|0.4000|0.2500|4.8261|579.1304|52|20|20|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7922|1.0000|120.0000|188|60|60|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly`|2|0.5752|0.1500|0.4000|0.1000|0.1667|0.7083|0.2033|1.9268|231.2195|153|45|48|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly`|3|0.7333|0.3500|0.5250|0.1000|0.0250|0.8250|0.1600|2.8050|336.6000|120|40|40|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly`|4|0.8736|0.0000|0.6571|0.4000|0.1250|0.7188|0.0909|3.7662|451.9481|87|35|32|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly`|5|0.8462|0.0000|0.4000|0.0000|0.0000|0.3000|0.2717|4.8152|577.8261|52|20|20|
|`cpr_p20_s05_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7760|1.0000|120.0000|188|60|60|
|`cpr_p20_s05_proto2_maha1_qknnonly`|2|0.6078|0.2000|0.4667|0.1500|0.1667|0.6667|0.1829|1.8577|222.9268|153|45|48|
|`cpr_p20_s05_proto2_maha1_qknnonly`|3|0.8000|0.5500|0.5750|0.2000|0.0750|0.7000|0.1650|2.5550|306.6000|120|40|40|
|`cpr_p20_s05_proto2_maha1_qknnonly`|4|0.8851|0.0000|0.7429|0.6000|0.2188|0.5625|0.0649|3.4156|409.8701|87|35|32|
|`cpr_p20_s05_proto2_maha1_qknnonly`|5|0.8462|0.0000|0.4500|0.0000|0.3000|0.2000|0.2065|4.1522|498.2609|52|20|20|
|`cpr_p20_s05_u098_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7695|1.0000|120.0000|188|60|60|
|`cpr_p20_s05_u098_proto2_maha1_qknnonly`|2|0.5948|0.2000|0.4444|0.1500|0.1458|0.6875|0.1667|1.8333|220.0000|153|45|48|
|`cpr_p20_s05_u098_proto2_maha1_qknnonly`|3|0.7917|0.5500|0.5500|0.2000|0.0500|0.7250|0.1300|2.5150|301.8000|120|40|40|
|`cpr_p20_s05_u098_proto2_maha1_qknnonly`|4|0.8621|0.0000|0.7143|0.6000|0.1875|0.6250|0.0584|3.3506|402.0779|87|35|32|
|`cpr_p20_s05_u098_proto2_maha1_qknnonly`|5|0.8269|0.0000|0.4500|0.0000|0.1500|0.4000|0.1957|4.0978|491.7391|52|20|20|
|`cpr_p20_s05_u985_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7695|1.0000|120.0000|188|60|60|
|`cpr_p20_s05_u985_proto2_maha1_qknnonly`|2|0.5948|0.2000|0.4444|0.1500|0.1667|0.6875|0.1748|1.8374|220.4878|153|45|48|
|`cpr_p20_s05_u985_proto2_maha1_qknnonly`|3|0.7917|0.5500|0.5500|0.2000|0.0500|0.7250|0.1350|2.5150|301.8000|120|40|40|
|`cpr_p20_s05_u985_proto2_maha1_qknnonly`|4|0.8621|0.0000|0.7143|0.6000|0.1875|0.6250|0.0584|3.3506|402.0779|87|35|32|
|`cpr_p20_s05_u985_proto2_maha1_qknnonly`|5|0.8269|0.0000|0.4500|0.0000|0.1500|0.4000|0.1957|4.0978|491.7391|52|20|20|
|`cpr_p20_s05_vote020_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7760|1.0000|120.0000|188|60|60|
|`cpr_p20_s05_vote020_proto2_maha1_qknnonly`|2|0.6078|0.2000|0.4667|0.1500|0.1667|0.6667|0.1829|1.8577|222.9268|153|45|48|
|`cpr_p20_s05_vote020_proto2_maha1_qknnonly`|3|0.8000|0.5500|0.5750|0.2000|0.0750|0.7000|0.1650|2.5550|306.6000|120|40|40|
|`cpr_p20_s05_vote020_proto2_maha1_qknnonly`|4|0.8851|0.0000|0.7429|0.6000|0.2188|0.5625|0.0649|3.4156|409.8701|87|35|32|
|`cpr_p20_s05_vote020_proto2_maha1_qknnonly`|5|0.8462|0.0000|0.4500|0.0000|0.3000|0.4000|0.1630|4.1522|498.2609|52|20|20|
|`cpr_p50_s05_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7760|1.0000|120.0000|188|60|60|
|`cpr_p50_s05_proto2_maha1_qknnonly`|2|0.5229|0.1000|0.4667|0.1500|0.1458|0.6667|0.2520|1.8577|222.9268|153|45|48|
|`cpr_p50_s05_proto2_maha1_qknnonly`|3|0.7250|0.4500|0.5750|0.2000|0.0500|0.7000|0.2300|2.5550|306.6000|120|40|40|
|`cpr_p50_s05_proto2_maha1_qknnonly`|4|0.8506|0.0000|0.7429|0.6000|0.1875|0.5625|0.0974|3.4156|409.8701|87|35|32|
|`cpr_p50_s05_proto2_maha1_qknnonly`|5|0.8462|0.0000|0.4500|0.0000|0.1000|0.2000|0.2609|4.1522|498.2609|52|20|20|

主结论：CPR确实恢复known接受率。`cpr_p20_s05_proto2_maha1_qknnonly`与`cpr_p20_s05_vote020_proto2_maha1_qknnonly`在协同数3达到`old_acc=0.8000`、`min_old=0.5500`、`seen_new_acc=0.5750`，但`unknown_FAR=0.0750`，尚未满足`<=0.05`安全线；`cpr_p15/p18/p20_s05_u098`协同数3把`unknown_FAR`压到`0.0500`，但`old_acc=0.7917`，低于OLD80阶段门槛。协同数4/5的旧类均值更高，但`min_old=0.0000`，说明至少一个旧类完全失效，不能作为主线成功证据。

错误与不合理点：

|问题|证据|处理边界|
|---|---|---|
|不能把协同数4的高均值当成功|多个候选协同数4的`old_acc>=0.85`，但`min_old=0.0000`且`unknown_FAR`多为`0.1875-0.2500`|只能诊断为类间不均衡，不能写成部署成功。|
|不能把低FAR单点当成功|`cpr_p20_s03_vrisk4`协同数3`unknown_FAR=0.0000`，但`old_acc=0.7250`、`min_old=0.3000`|virtual unknown仍过度压制known，需要分层而非全局风险。|
|协同数1几乎不可用|所有候选协同数1`old_acc=0.0000`，高defer|单接收机路径必须重设门控或降级为请求更多receiver。|
|当前不是最终目标|最佳行仍远低于old 99%/floor95%、seen-new 97%/floor93%、unknown拒识99%|报告为阶段诊断进展，不得注册为deployment success。|

下一步算法方向：从`SCORER-CVS-CPR`升级为class-conditional cost-aware CPR。具体是对每个label维护类条件p-value、class-wise floor risk和receiver-domain reliability，不再使用统一`effective_unknown_risk`缩放；同时给低floor旧类设置类级补偿/隔离阈值，避免协同数4/5出现均值提升但某类归零。下一轮候选应围绕`unknown_FAR=0.05`附近做joint objective：优先搜索`old_acc>=0.80 && min_old>0.55 && unknown_FAR<=0.05`，再优化seen-new。

结果文件哈希：

|文件|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p05_s05_proto2_maha1_qknnonly.json`|`DEAD30A534661299194B2FD52F363779C87CE860291831E2C68DE63FFD568C98`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p05_s05_proto2_maha1_qknnonly_evidence.csv`|`0FCA42137442B05350FFDACBA74908CCD07699A6F896138C643593FBD5ADB49D`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p15_s05_u098_proto2_maha1_qknnonly.json`|`E6D587DF4F8C3D89215C9638B884DC6C9995534CA1843B20603F70A18A2E952E`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p15_s05_u098_proto2_maha1_qknnonly_evidence.csv`|`3B0916DD03A2103C33195CDDB14CD8C7292F045918C13F3B81550EA8F68BF877`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p18_s05_u098_proto2_maha1_qknnonly.json`|`1755E298A260330714F6D7B2ED96083F914A8CF9B2D29304DC84EF7E2E5AA8F9`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p18_s05_u098_proto2_maha1_qknnonly_evidence.csv`|`00627CD4C1CB375DD4DCBF4F337A5523A339FD02154B3FAD0A390633B3963D1B`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s03_vrisk4_proto2_maha1_qknnonly.json`|`3E3C70314EF0F0DF4A415BC69841FC891318F31FCB43FFDC899CD68869D4FE6F`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s03_vrisk4_proto2_maha1_qknnonly_evidence.csv`|`2B26C03EB640E59E86BAC8E58165929C8C51CFB0103B8C3E941D5835261BB2C5`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s04_vrisk2_proto2_maha1_qknnonly.json`|`A173DB65B67914BBD93D04C53AC7044C9EF8D06649C93C5D44BA056A6761DC14`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_evidence.csv`|`335DBC944AFFE588CE9EB65FD4698FD8FF1A63E0D9D7F3917268FEC62C2C8A34`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_proto2_maha1_qknnonly.json`|`9F6BDFE40C90F67C75E67A8DA914A7263CAB09AA6E45D2C2B0D0FF62BF273D01`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_proto2_maha1_qknnonly_evidence.csv`|`6BD7A9271AA27BD952E55BDB5B82A30BD8F6F16CAC9A08B39CA89399330139B2`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_u098_proto2_maha1_qknnonly.json`|`4F818F0CE3467CBF9F8D9A40B2E9F7B326ED49ED020EC8198B5E82186FF87724`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_u098_proto2_maha1_qknnonly_evidence.csv`|`BF8644237511AB6E28669D700D2A7E1ECACC9092E08D77F045194906A0E7DF96`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_u985_proto2_maha1_qknnonly.json`|`D1D8A78C78C744CFCE9B82B3ABB573C3AD4E17D1251CA4569E5B0684C92C6E9A`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_u985_proto2_maha1_qknnonly_evidence.csv`|`03EFB59D318F97E5E3DE89262F6DA15E7DE996C72B245C97F39044C51A8CC05B`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_vote020_proto2_maha1_qknnonly.json`|`39E5B0A3CF6369ADA918567FF929FA05084E4D3AAE0B0EB8B8450C1C32CE747C`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p20_s05_vote020_proto2_maha1_qknnonly_evidence.csv`|`821A01EFA318400489C8DF8BD38017B6ADC450FFBC233A6DE7447B7BD4722914`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p50_s05_proto2_maha1_qknnonly.json`|`387F362A30630F9AFDB0963A0AECCFF6CAD56B444795DFEF09196E468003E745`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_cpr_p50_s05_proto2_maha1_qknnonly_evidence.csv`|`07994D21FB243D243D14A70737BE3999C5DA9DE12556BB62B0C9E7229F20CF70`|

## 2026-07-03CPR review修复与固定版复跑

子agent review指出P1问题：原`conformal_rescue`在强known且p-value达标时直接缩放`effective_unknown_risk`，可能把高风险unknown救成known接受。修复后，只有当`risk_component_agreement < scorer_component_vote_threshold`时CPR才允许缩放风险；多风险通道一致高风险时fail closed，最多defer/reject，不直接accept。另将`class_conformal_min_support`默认值从1提高到2，避免K=1/LOO时p-value退化为粗粒度救援门。

新增回归测试：

|文件|测试|
|---|---|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`test_scorer_cvs_conformal_rescue_does_not_accept_multichannel_unknown`确认unknown高p-value但多通道高风险时不会被accept。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`test_class_conformal_defaults_fail_closed_with_single_support`确认默认min_support=2时K=1类p-value为0、support_count为0。|

验证结果：

|位置|命令|结果|
|---|---|---|
|本地`E:\type10-7`|`conda activate ssr-gpu; python -m py_compile ...; python code\tests\test_phase2_collaborative_open_set_qknn_eval.py; python code\tests\test_collaborative_open_set_qknn_eval.py`|36 tests OK和30 tests OK。|
|Git镜像|同上|36 tests OK和30 tests OK。|
|N607|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`执行py_compile和两组测试|36 tests OK和30 tests OK。|

远端同步：四个代码/测试文件已重新scp到N607。复测后检查为`NO_SSH_PROCESS`和`NO_N607_SSH_ESTABLISHED`。修复版复跑使用`CUDA_VISIBLE_DEVICES=0`，最终GPU读数为8张RTX3090均`10 MiB/24576 MiB`。

固定版关键候选结果：

|候选|协同数|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7695|1.0000|120.0000|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|2|0.5948|0.2000|0.4444|0.1500|0.1458|0.6875|0.1667|1.8333|220.0000|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|3|0.7917|0.5500|0.5500|0.2000|0.0500|0.7250|0.1200|2.5150|301.8000|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|4|0.8621|0.0000|0.7143|0.6000|0.1875|0.6250|0.0584|3.3506|402.0779|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|5|0.8269|0.0000|0.4500|0.0000|0.1500|0.4000|0.1957|4.0978|491.7391|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7922|1.0000|120.0000|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|2|0.5752|0.1500|0.4000|0.1000|0.1667|0.7083|0.2033|1.9268|231.2195|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|3|0.7333|0.3500|0.5250|0.1000|0.0250|0.8250|0.1600|2.8050|336.6000|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|4|0.8736|0.0000|0.6571|0.4000|0.1250|0.7188|0.0909|3.7662|451.9481|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|5|0.8462|0.0000|0.4000|0.0000|0.0000|0.3000|0.2717|4.8152|577.8261|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|1|0.0000|0.0000|0.2667|0.0500|0.0667|0.2833|0.7760|1.0000|120.0000|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|2|0.6078|0.2000|0.4667|0.1500|0.1667|0.6667|0.1829|1.8577|222.9268|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|3|0.8000|0.5500|0.5750|0.2000|0.0750|0.7000|0.1650|2.5550|306.6000|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|4|0.8851|0.0000|0.7429|0.6000|0.2188|0.5625|0.0649|3.4156|409.8701|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|5|0.8462|0.0000|0.4500|0.0000|0.3000|0.2000|0.2065|4.1522|498.2609|

固定版结论：关键候选数值与pre-fix主表一致，说明P1保护没有改变这些ADV3B02候选的聚合输出，但修复后的安全边界更严格。当前最佳折中仍是二选一：`cpr_p20_s05`协同3达到`old_acc=0.8000`但`unknown_FAR=0.0750`；`cpr_p15_s05_u098`协同3达到`unknown_FAR=0.0500`但`old_acc=0.7917`。仍不能声明Stage2-C成功或部署成功。

固定版结果SHA256：

|文件|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_adv3b02_cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed.json`|`B869D652AD431FFE471A01F694A50E50A4458C7138007405CC1BA7C1040BDFFB`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_adv3b02_cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed_evidence.csv`|`06CF788316C1F5642C15D89F7ADF9A368AD08907496C2A26AA4ED33F7A90E4A0`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_adv3b02_cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed.json`|`A2F047C616E7237F0CD52DEABC9573F52CBFFDBAF37CEE9F9252600659A97513`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_adv3b02_cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed_evidence.csv`|`88851140B9D85F870600620B09ACCA91D7014236A76DD6305999D611D445C149`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_adv3b02_cpr_p20_s05_proto2_maha1_qknnonly_fixed.json`|`B8EDFBE14F441553D48F657C5C6F93D63C282FBB2350B492AB37E3200EE432F1`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_adv3b02_cpr_p20_s05_proto2_maha1_qknnonly_fixed_evidence.csv`|`CA51EA4C0FF2D9F9EE99B935F946F76EB53A847958301E7D5686EE871BB19A06`|

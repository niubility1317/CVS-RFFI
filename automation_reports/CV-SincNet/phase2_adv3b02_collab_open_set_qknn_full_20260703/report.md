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

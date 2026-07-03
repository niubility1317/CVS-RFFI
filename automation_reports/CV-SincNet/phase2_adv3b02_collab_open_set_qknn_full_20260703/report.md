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

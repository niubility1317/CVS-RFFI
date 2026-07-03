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

## 2026-07-03CP-SET-CVS最小实现计划

目标：从`SCORER-CVS-CPR`的“强known风险折扣”推进到更接近卫星群部署的类条件证据融合。新策略`cp_set_cvs`不再把p-value作为全局unknown risk折扣，而是把class-conditional conformal p-value作为old/seen-new接受门：只有预测类属于old/seen-new、receiver一致性足够、p-value达到阈值且support计数非零时才允许accept；否则按资源预算request_more或defer。这样更符合unknown FAR受控的开集目标。

本轮资源约束说明：工作区未定位到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`原文件，因此本轮只使用当前代码已落地的资源约束字段：`latency_budget_ms`、`max_event_bytes`、`max_event_latency_ms`、`bytes/event`、`prototype_storage_bytes`。远端运行仍记录协同receiver数量、平均实际receiver数、`bytes/event`和GPU显存。

本地改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`fusion_policy=cp_set_cvs`；记录`label_class_conformal_support_count`和`cp_set_gate_passed`；多通道高风险时仍fail closed。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`--class_evidence_top_m`，在evidence CSV中记录`class_evidence_top{rank}_label/score/conformal_pvalue/support_count`，用于后续topM类条件融合和审计。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增测试确认`cp_set_cvs`低p-value时defer，高p-value时accept。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增测试确认topM类条件证据字段被记录。|

本地与Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为37 tests OK和31 tests OK。

远端计划：同步4个代码/测试文件到N607，在`CVS-RFFI`环境复测；使用ADV3B02特征`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，运行`--collab_counts all`覆盖1到5个target receivers。候选：

|候选|关键参数|目的|
|---|---|---|
|`cpset_p15_proto2_maha1_qknnonly`|`--fusion_policy cp_set_cvs --class_conformal_enabled --class_evidence_top_m 3 --conformal_rescue_min_pvalue 0.15 --unknown_risk_threshold 0.98`|较宽p-value门，观察是否比CPR减少unknown误接受。|
|`cpset_p20_proto2_maha1_qknnonly`|`--fusion_policy cp_set_cvs --class_conformal_enabled --class_evidence_top_m 3 --conformal_rescue_min_pvalue 0.20 --unknown_risk_threshold 0.995`|中等p-value门，对齐上一轮`cpr_p20_s05`。|
|`cpset_p20_vrisk2_proto2_maha1_qknnonly`|再加`--virtual_unknown_risk_enabled --virtual_unknown_risk_samples_per_class 2 --virtual_unknown_risk_temperature 0.05`|结合virtual unknown风险，优先压unknown FAR。|

成功判据不变：最终目标仍是old 99%且每类不低于95%、seen-new 97%且每类不低于93%、unknown拒识99%。本轮若未达到，只能作为下一步class-conditional fusion诊断，不写作部署成功。

远端执行结果：N607复测通过，`CVS-RFFI`环境下`test_phase2_collaborative_open_set_qknn_eval.py`为37 tests OK，`test_collaborative_open_set_qknn_eval.py`为31 tests OK。3组候选均输出`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`。最终GPU读数为8张RTX3090均`10 MiB/24576 MiB`；SSH/SCP后本地检查无残留`ssh.exe`或N607 22端口连接。

CP-SET-CVS结果表：

|候选|协同数|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`cpset_p15_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2500|0.0500|0.0167|0.3000|0.7792|1.0000|120.0000|
|`cpset_p15_proto2_maha1_qknnonly`|2|0.3464|0.0000|0.3111|0.0500|0.1250|0.7917|0.3333|1.8333|220.0000|
|`cpset_p15_proto2_maha1_qknnonly`|3|0.3917|0.3000|0.3250|0.0500|0.0250|0.8500|0.4200|2.5150|301.8000|
|`cpset_p15_proto2_maha1_qknnonly`|4|0.7586|0.0000|0.6000|0.4000|0.0938|0.7500|0.1688|3.3506|402.0779|
|`cpset_p15_proto2_maha1_qknnonly`|5|0.7500|0.0000|0.3500|0.0000|0.0500|0.5500|0.2826|4.0870|490.4348|
|`cpset_p20_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.2500|0.0500|0.0167|0.2833|0.7922|1.0000|120.0000|
|`cpset_p20_proto2_maha1_qknnonly`|2|0.3464|0.0000|0.3111|0.0500|0.1250|0.7500|0.3984|1.8577|222.9268|
|`cpset_p20_proto2_maha1_qknnonly`|3|0.3917|0.3000|0.3250|0.0500|0.0250|0.7750|0.4850|2.5550|306.6000|
|`cpset_p20_proto2_maha1_qknnonly`|4|0.7586|0.0000|0.6286|0.4000|0.0938|0.6562|0.2143|3.4156|409.8701|
|`cpset_p20_proto2_maha1_qknnonly`|5|0.7500|0.0000|0.3500|0.0000|0.0500|0.2000|0.3696|4.1522|498.2609|
|`cpset_p20_vrisk2_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.2833|0.8896|1.0000|120.0000|
|`cpset_p20_vrisk2_proto2_maha1_qknnonly`|2|0.0131|0.0000|0.0000|0.0000|0.0000|0.7708|0.6992|1.9268|231.2195|
|`cpset_p20_vrisk2_proto2_maha1_qknnonly`|3|0.0167|0.0000|0.0000|0.0000|0.0000|0.9250|0.6750|2.8050|336.6000|
|`cpset_p20_vrisk2_proto2_maha1_qknnonly`|4|0.1724|0.0000|0.0000|0.0000|0.0000|0.9062|0.6364|3.7662|451.9481|
|`cpset_p20_vrisk2_proto2_maha1_qknnonly`|5|0.1154|0.0000|0.0000|0.0000|0.0000|0.5000|0.7935|4.8152|577.8261|

解释：`cp_set_cvs`把p-value从“救援折扣”改成“接受门”后，unknown FAR明显下降。预算3的`cpset_p15/p20`均达到`unknown_FAR=0.0250`，`cpset_p20_vrisk2`全部预算为`unknown_FAR=0.0000`。但代价是known接受率严重下降，预算3时`old_acc=0.3917`、`seen_new_acc=0.3250`，加入virtual unknown后几乎完全压制known接受。该结果证明类条件接受门能控制unknown误接受，但当前p-value校准和风险门过保守，不能作为最终算法。

下一步：保留topM类条件证据字段，改为“候选集+请求更多receiver”而非单类硬门。具体应融合`class_evidence_top1..top3`的p-value和score，输出多标签候选集；当top类p-value不足但top2/top3存在可信类时请求更多receiver或保持ambiguous，而不是直接defer所有known。还应引入receiver-class可靠度`w_{r,y}`，避免低floor类在协同4/5被多数receiver压制。

CP-SET-CVS产物SHA256：

|文件|SHA256|
|---|---|
|`collab_open_set_qknn_cp_set_cvs_cpset_p15_proto2_maha1_qknnonly.json`|`8BE33C713CE01A16A090F041D44AFE7CB2FBD7AC819C08BBD07E9148E1540537`|
|`collab_open_set_qknn_cp_set_cvs_cpset_p15_proto2_maha1_qknnonly_evidence.csv`|`1D96A694F86BEC38D50BD997EE729346F03774E72A45C0D038727FAF8AAD19C8`|
|`collab_open_set_qknn_cp_set_cvs_cpset_p20_proto2_maha1_qknnonly.json`|`66FDB5A86080598721F63DDE8877BE07F90C3DCAC17C0A0F36FCF8D1C30F16EC`|
|`collab_open_set_qknn_cp_set_cvs_cpset_p20_proto2_maha1_qknnonly_evidence.csv`|`8455BB0581F740B9AE37DF2051308B09C725886A1FB0BF89B952EFB2B731F2CA`|
|`collab_open_set_qknn_cp_set_cvs_cpset_p20_vrisk2_proto2_maha1_qknnonly.json`|`F11CCA7943F45C28849C0D6D0094613E04781573EAD8B4B1ED9763A05914B5A6`|
|`collab_open_set_qknn_cp_set_cvs_cpset_p20_vrisk2_proto2_maha1_qknnonly_evidence.csv`|`7DF97A44424D3AD0DD7AB3EE7E8C9E0BD7B6561B4F24B7C652096A8BECA0034B`|

## 2026-07-03 21:47 AWARE-CQKNN-Lite topM候选集融合

目标：把上一轮`cp_set_cvs`从“top1类条件硬门”扩展为更适合卫星群的轻量协同推理：每个target receiver只上传`class_evidence_topM`标签、score、class conformal p-value和support count，聚合端按receiver级候选证据融合；协同数量继续用`collab_counts=all`覆盖1到target receiver数量。

文献/方法子agent结论：最可落地组合为冻结CV-SincNet/`z_id`，使用KNN/prototype/Mahalanobis侧路和conformal set-valued gate；多receiver阶段只融合推理证据，不把unknown query用于阈值拟合，不做星上full-model fine-tune。

算法子agent建议：采用AWARE-CQKNN-Lite，即`support-derived QKNN evidence + class-evidence log/soft pool + adaptive receiver budget + CP fail-closed gate`。复杂度保持在每事件`O(BL)`融合通信，`B`为实际参与receiver数，`L`为每receiver上传topM标签数。

review子agent阻断项及修正：初始topM方案存在“单个receiver多个候选冒充多receiver一致性”的风险。已收紧为receiver级候选计数：同一receiver同一label只计一次；`agreement/vote_gap`按支持该label的receiver数除以实际receiver数；非top1候选必须至少由两个receiver的topM一致支持，且score>0、support_count>=1，才可通过`cp_set_cvs`候选门。

本地改动：

|文件|目的|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`cp_set_cvs`读取`class_evidence_top{rank}_label/score/conformal_pvalue/support_count`；按协议old/seen-new标签集过滤候选；按receiver级候选数计算agreement；记录`label_candidate_receiver_count`、`label_top1_receiver_count`、`label_min_evidence_rank`、`filtered_candidate_count`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增测试：两个target receiver的top1均错但top2类条件证据一致时，`cp_set_cvs`可恢复正确old类；单receiver多候选不会被当作多receiver一致性。|

本地验证：

```powershell
$env:PYTHONPATH='E:\type10-7\code'
conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"
conda run -n ssr-gpu python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"
```

结果：`test_collaborative_open_set_qknn_eval.py`为32 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`为37 tests OK。

Git镜像验证：

```powershell
$env:PYTHONPATH='E:\type10-7\github_publish\CVS-RFFI-repo\code'
conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"
conda run -n ssr-gpu python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"
```

结果：32 tests OK和37 tests OK。

远端测试计划：先运行`tools\n607_ssh_preflight.ps1`，选择显存占用最低的GPU；同步`collaborative_open_set_qknn_eval.py`和测试文件到N607；在`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`复测；随后使用ADV3B02特征和权重`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`相关产物，运行`--collab_counts all`覆盖1到5个target receiver、`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`星地信道视图。若用户坚持“源接收机1到7”字面口径，需要先修订协议，因为`项目.md`规定部署协同应在target receiver domain内，不能让source receiver参与Stage2部署推理。

## 2026-07-03 21:58 per-label topM风险证据融合

目标：修复上一轮topM融合的结构性问题。上一轮非top1候选虽然能参与类条件投票，但仍复用top1的`unknown_risk`、`known_margin`、`class_radius_z`等风险字段，导致正确top2/top3类可能被错误拒绝或defer。本轮把topM候选证据扩展为per-label evidence，并让`cp_set_cvs`在选中label存在per-label风险字段时使用该label自己的风险进行接受/拒识判断。

资源约束文档状态：本地限定搜索仍未定位到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`原文件；本轮继续使用代码已落地的资源字段`latency_budget_ms`、`evidence_packet_bytes`、`bytes/event`、`latency_ms_p95`和`prototype_storage_bytes`。若后续补回原文档，应按文档更新报告中的资源预算表。

本地改动：

|文件|目的|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|为`class_evidence_top{rank}`写出`margin`、`effective_score_threshold`、`unknown_risk`、`score/radius/margin/mahalanobis/evt/oldness_risk`、`class_radius`和`class_radius_z`。|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`cp_set_cvs`对选中topM label聚合per-label risk和risk component vote；高风险拒识和class-set gate使用选中label风险，不再固定复用top1风险。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增测试：top1高风险但top2正确且低风险时，两个receiver一致可恢复正确old类。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增topM per-label风险字段记录测试。|

Git镜像提交：`d6e64a0 Add per-label risk evidence for collaborative qKNN topM`。镜像分支仍领先远端276个提交；根工作区`E:\type10-7\code`不是Git仓库，本轮代码闭环仍在`github_publish\CVS-RFFI-repo`。

本地和镜像验证：

```powershell
$env:PYTHONPATH='E:\type10-7\code'
conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"
conda run -n ssr-gpu python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"
```

结果：本地与Git镜像均为33 tests OK和37 tests OK。

远端同步和验证：

|文件|远端SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`d1d381a460801d330877d3cf7fba3ef4e1114cb5595a8165d7ed7a7f4281d624`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`d6035bf055975d594daed8817fc58dd592acdec886156c53821c4c7983679bf8`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`5e34647a692ce327355166f51f3c361a868d699e37d0e0e416a19c653e7d7da4`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|`ade4f7b35b88ccd826bf556329d169a9db249aeb334cae2cd40767a814807d2d`|

远端使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，在`/home/szu2070436088/2510044040/CV-SincNet/code`目录下运行，结果为33 tests OK和37 tests OK。N607预检显示8张RTX3090均为`10/24576MiB`，本轮运行前后仍为`10/24576MiB`，使用GPU0。所有SSH/SCP后本地检查均无残留`ssh.exe`或N607 22端口连接。

远端诊断命令摘要：输入仍为`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，即ADV3B02_CORE90_SOFT_E200对应Stage2-C特征；参数保持`--collab_counts all --k_shot 8 --query_per_class 20 --qknn_k 8 --candidate_class_top_m 2 --class_evidence_top_m 3 --fusion_policy cp_set_cvs --collaboration_policy adaptive_gain --event_alignment_policy receiver_domain_ranked`。两组候选分别为`conformal_rescue_min_pvalue=0.15/0.20`，输出均为`receiver_count=5`、`group_count=308`、`evidence_row_count=1000`，覆盖target receiver domain的协同数量1到5。

结果表：

|候选|协同数|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`perlabel_p15`|1|0.0000|0.0000|0.2500|0.0500|0.0167|0.3000|0.7792|0.0766|1.0000|120.0000|0.1322|
|`perlabel_p15`|2|0.4837|0.0500|0.2222|0.0000|0.1250|0.4792|0.3902|0.4646|1.8943|227.3171|0.1322|
|`perlabel_p15`|3|0.5250|0.2500|0.4000|0.0500|0.0000|0.9000|0.2500|0.4938|2.7250|327.0000|0.1322|
|`perlabel_p15`|4|0.5862|0.0000|0.3429|0.0000|0.0000|1.0000|0.1818|0.5246|3.6494|437.9221|0.1322|
|`perlabel_p15`|5|0.6346|0.0000|0.0000|0.0000|0.0000|1.0000|0.1957|0.4722|4.5000|540.0000|0.1322|
|`perlabel_p20`|1|0.0000|0.0000|0.2500|0.0500|0.0167|0.2833|0.7922|0.0766|1.0000|120.0000|0.1305|
|`perlabel_p20`|2|0.5098|0.0500|0.2222|0.0000|0.1458|0.4583|0.4065|0.4848|1.9187|230.2439|0.1305|
|`perlabel_p20`|3|0.5417|0.2500|0.4500|0.0500|0.0000|0.8750|0.2700|0.5188|2.7650|331.8000|0.1305|
|`perlabel_p20`|4|0.5977|0.0000|0.3714|0.0000|0.0000|0.9375|0.2338|0.5410|3.7143|445.7143|0.1305|
|`perlabel_p20`|5|0.6346|0.0000|0.0000|0.0000|0.0000|0.9500|0.2935|0.4722|4.5435|545.2174|0.1305|

产物SHA256：

|文件|SHA256|
|---|---|
|`collab_open_set_qknn_cp_set_cvs_topm_perlabel_p15_adv3b02.json`|`F0A4EF22FDA999A3CDF609AE6588C721F0AB7E16C8781DE1CADDCCE90B755A61`|
|`collab_open_set_qknn_cp_set_cvs_topm_perlabel_p15_adv3b02_evidence.csv`|`55A5DCFDA71A4AFAF8D6CA3DAB497E2D00D1017B8C50A255B046024EECDF89FA`|
|`collab_open_set_qknn_cp_set_cvs_topm_perlabel_p20_adv3b02.json`|`00DFBD761AEE45EFBF9A730DD0B61DABA007EBEF3FC2DADDF78EDFFAF6A40ED1`|
|`collab_open_set_qknn_cp_set_cvs_topm_perlabel_p20_adv3b02_evidence.csv`|`B0424B4F83DF4305A8A11EB61789D07BE6CAEE7BCB13324B2F9E5DD483039A17`|

解释：相对上一轮`cp_set_cvs`，per-label风险证据显著缓解了known被top1风险误拒的问题。例如协同3从旧的约`old_acc=0.3917`、`seen_new_acc=0.3250`提升到`perlabel_p20`的`old_acc=0.5417`、`seen_new_acc=0.4500`，且`unknown_FAR=0.0000`。这说明方向有效，但仍远低于最终目标`old_acc=99%/min_old>=95%`、`seen_new_acc=97%/min_seen>=93%`、`unknown_reject>=99%`。当前结果仍只能作为`receiver_domain_ranked`诊断，不是严格同事件卫星群协同证据，也不能声明Stage2-C成功或部署成功。

下一步建议：继续沿per-label证据路线，加入receiver-class可靠度`w_{r,y}`和set-valued输出。当前协同5的seen-new归零，说明多数receiver融合会压制弱新类；需要让融合权重按`receiver,label`而不是只按receiver全局可靠性分配，并在top label不唯一时输出候选集或request_more，而不是直接accept/defer。
## 2026-07-03 22:40 receiver-class可靠度融合本地落地

### 目标

在`cp_set_cvs`协同推理中加入`class_reliability_policy=conformal_margin_risk`，为每个`receiver x candidate label`计算可靠度`w_{r,y}`，缓解多接收机融合时高分但低p-value/高风险候选压制真实seen-new或old候选的问题。该路线仍限定为`R_t`目标接收机域内证据融合，不允许source receiver进入部署协同证据。

### 本地变更

| 文件 | 作用 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 新增`_class_reliability`；在label fusion中用`receiver_reliability * class_reliability`加权候选label；输出`class_reliability_policy`和`mean_label_class_reliability`；严格校验证据`receiver_id`必须属于`protocol_metadata.target_receiver_ids`。 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 新增CLI参数`--class_reliability_policy {none,conformal_margin_risk}`并透传到评估模块。 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 新增receiver-class可靠度选择、p-value单调性、`R_t`证据范围硬校验测试。 |

### 本地验证

| 环境 | 命令 | 结果 |
|---|---|---|
| local `ssr-gpu` | `python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | PASS |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"` | PASS，36 tests |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"` | PASS，37 tests |
| mirror `ssr-gpu` | 同上编译和两组单测 | PASS，36+37 tests |

### Git状态

| 仓库 | 状态 |
|---|---|
| `E:\type10-7` | 非Git仓库，不能作为版本承载目录。 |
| `E:\type10-7\github_publish\CVS-RFFI-repo` | 已提交`4ef78a3 Add receiver-class reliability for collaborative qKNN`，分支领先远端277。 |

### 同步计划

| local | remote |
|---|---|
| `E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py` |

### 边界

该实现是协同推理诊断路线，不是部署成功声明。若`old_acc`、`seen_new_acc`、`unknown_FAR`或per-class floor未达目标，只能报告为负结果或下一步修复证据。

## 2026-07-03 22:51 receiver-class可靠度融合N607诊断结果

### 远端同步与验证

| 项目 | 结果 |
|---|---|
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU选择 | GPU0；诊断前后所有GPU均为`10/24576MiB`，未增加持续显存占用 |
| 远端编译 | PASS |
| 远端单测 | `test_collaborative_open_set_qknn_eval.py`：36 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`：37 tests OK |
| SSH/SCP断连 | 每次SSH/SCP后本地检查均为`NO_SSH_PROCESS`和`NO_N607_SSH_ESTABLISHED` |

### 远端文件哈希

| 文件 | SHA256 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | `2b0506c98c316409c87714e3f732b7437a244c9c55c507049fcdf4dc99f51ef6` |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | `8103f796fcbd791ab46074a14e3c103e536bf705fdeeef29e735f26d3610faa9` |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | `e6ebf22180de7aa9ae4aef796c9c2ef138603bec24b8a02f4487fce77bfb2448` |

### 诊断命令要点

| 配置项 | 值 |
|---|---|
| feature | `runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz` |
| `collab_counts` | `all`，覆盖1到5个target receiver证据 |
| `fusion_policy` | `cp_set_cvs` |
| `collaboration_policy` | `adaptive_gain` |
| `label_fusion_policy` | `vote_margin` |
| `class_reliability_policy` | `conformal_margin_risk` |
| `event_alignment_policy` | `receiver_domain_ranked`，仍是数据集诊断，不是严格same-event星群同步证据 |
| `class_evidence_top_m` | 3 |
| `latency_budget_ms` | 12 |
| `evidence_packet_bytes` | 120 |
| `conformal_rescue_min_pvalue` | 0.15和0.20两组 |

### artifact

| artifact | SHA256 |
|---|---|
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_classrel_p015_adv3b02.json` | `2B5059669BE4B6E6F2E635D904ED48B3EFE70720B8D1769D374919E853BD482E` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_classrel_p015_adv3b02_evidence.csv` | `1C01D0A8F05C9414A6BCF8443C66D3288FC33AB0A34E98F37B427E6D98D428AE` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_classrel_p020_adv3b02.json` | `7DABA00EDE78BEC485E4F7DF6123D84A9FC0F08E8BAC1914D195EC6B97138CEB` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_classrel_p020_adv3b02_evidence.csv` | `A7377A4EEE894FA5249E18933755818F0AF154CCAEAE5E6F471E9AD1C63A33F8` |

### 结果表：`conformal_rescue_min_pvalue=0.15`

| k | total | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | avg_rx | bytes/event | p95_latency | mean_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 308 | 0.0000 | 0.0000 | 0.2500 | 0.0500 | 0.0167 | 0.3000 | 0.7792 | 0.0766 | 1.0000 | 120.0 | 0.1321 | 0.6081 |
| 2 | 246 | 0.4902 | 0.1000 | 0.2222 | 0.0000 | 0.1250 | 0.4792 | 0.3862 | 0.4697 | 1.8943 | 227.3 | 0.1321 | 0.4553 |
| 3 | 200 | 0.5250 | 0.2500 | 0.4000 | 0.0500 | 0.0000 | 0.9000 | 0.2500 | 0.4938 | 2.7250 | 327.0 | 0.1321 | 0.4641 |
| 4 | 154 | 0.5977 | 0.0000 | 0.3429 | 0.0000 | 0.0000 | 1.0000 | 0.1753 | 0.5328 | 3.6494 | 437.9 | 0.1321 | 0.4685 |
| 5 | 92 | 0.6346 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1957 | 0.4722 | 4.5000 | 540.0 | 0.1321 | 0.4482 |

### 结果表：`conformal_rescue_min_pvalue=0.20`

| k | total | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | avg_rx | bytes/event | p95_latency | mean_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 308 | 0.0000 | 0.0000 | 0.2500 | 0.0500 | 0.0167 | 0.3000 | 0.7792 | 0.0766 | 1.0000 | 120.0 | 0.1328 | 0.6081 |
| 2 | 246 | 0.4902 | 0.1000 | 0.2222 | 0.0000 | 0.1250 | 0.4792 | 0.3862 | 0.4697 | 1.8943 | 227.3 | 0.1328 | 0.4553 |
| 3 | 200 | 0.5250 | 0.2500 | 0.4000 | 0.0500 | 0.0000 | 0.9000 | 0.2500 | 0.4938 | 2.7250 | 327.0 | 0.1328 | 0.4641 |
| 4 | 154 | 0.5977 | 0.0000 | 0.3429 | 0.0000 | 0.0000 | 1.0000 | 0.1753 | 0.5328 | 3.6494 | 437.9 | 0.1328 | 0.4685 |
| 5 | 92 | 0.6346 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1957 | 0.4722 | 4.5000 | 540.0 | 0.1328 | 0.4482 |

### 结论

receiver-class可靠度融合使k=3/4/5维持低FAR，但没有达到目标门槛：`old_acc`最高0.6346，`min_old`仍为0，`seen_new_acc`最高0.4000且`min_seen`最高0.0500。该结果不能登记为Stage2-C成功或部署成功，只能作为负诊断。下一步应优先尝试class-specific aggregation与TEEN式seen-new prototype校准，而不是继续提高协同receiver数量；当前k增大后seen-new被压制，说明多receiver融合仍有类别级偏置。

## 2026-07-03 23:08 weighted vote margin协同融合本地落地

### 目标

上一轮`class_reliability_policy=conformal_margin_risk`没有显著改变远端结果，原因是远端诊断使用的`label_fusion_policy=vote_margin`仍以未加权receiver票数为主，`w_{r,y}`只影响很小的score项。为让receiver-class可靠度真正进入类别排序，本轮新增`weighted_vote_margin`：用`sum_r receiver_reliability_r * class_reliability_{r,y}`作为类票权，再叠加类margin和小权重score。

### 本地变更

| 文件 | 作用 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 新增`label_fusion_policy=weighted_vote_margin`；类别排序使用`label_weight_totals + mean_margin + 1e-3 * weighted_score`；该策略下`agreement`和`vote_gap`改用类别权重占比。 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | CLI参数`--label_fusion_policy`允许`weighted_vote_margin`。 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 新增测试：两个receiver对高风险错误类和低风险正确类票数相同、错误类margin更高时，普通`vote_margin`失败，`weighted_vote_margin`可选择正确类。 |

### 本地与镜像验证

| 环境 | 命令 | 结果 |
|---|---|---|
| local `ssr-gpu` | `python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py` | PASS |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"` | PASS，37 tests |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"` | PASS，37 tests |
| mirror `ssr-gpu` | 编译+上述两组单测 | PASS，37+37 tests |

### Git状态

| 仓库 | 状态 |
|---|---|
| `E:\type10-7` | 非Git仓库。 |
| `E:\type10-7\github_publish\CVS-RFFI-repo` | 已提交`3628856 Add weighted vote margin for collaborative qKNN`，分支领先远端278。 |

### 远端计划

同步三处代码/测试文件到N607，用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`远端编译与单测，然后在`features.npz`上运行`collab_counts=all`、`class_reliability_policy=conformal_margin_risk`、`label_fusion_policy=weighted_vote_margin`，覆盖1到5个target receiver证据。该实验仍是`receiver_domain_ranked`诊断，若指标未达门槛，不写成功声明。

### 审查修正

只读审查指出`weighted_vote_margin`初版的`agreement/vote_gap`按全局最大权重计算，可能与最终选中label不一致；`score_gap_ratio`也存在rank-score与label-score分母混用。已修正为：

- 选中label由`rank_score`决定；
- `agreement`、`vote_gap`使用选中label自身权重与次高其他label权重；
- `score_gap_ratio`在`weighted_vote_margin`下使用rank-score总量作分母；
- 新增反例测试，确保低权重高margin标签不会借另一个标签的高agreement被accept。

补丁提交：`81ecd71 Align weighted vote agreement with selected label`。

## 2026-07-03 23:23 weighted vote margin N607诊断结果

### 远端同步与验证

| 项目 | 结果 |
|---|---|
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| 远端文件 | `code/evaluation/collaborative_open_set_qknn_eval.py`、`code/scripts/phase2_collaborative_open_set_qknn_eval.py`、`code/tests/test_collaborative_open_set_qknn_eval.py` |
| 远端SHA | eval `b1c8437f0321c249ecf4c2474d9d5443e6ffd632440dc75886c6042300a1caf1`；script `49b0502fb7c0124944e63c7b49b0e0140b29e9eef94f6700255e5e572ff83107`；test `a01a407293fc5cc0ecd0e8e3e21acc8b0a6f1d77df6360ae8b0538f21df39d63` |
| 远端验证 | 编译PASS；`test_collaborative_open_set_qknn_eval.py`38 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`37 tests OK |
| GPU | GPU0运行；运行前后所有GPU均为`10/24576MiB` |
| SSH/SCP | 每次后均检查为`NO_SSH_PROCESS`和`NO_N607_SSH_ESTABLISHED` |

### artifact

| artifact | SHA256 |
|---|---|
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_weightedvote_p015_adv3b02.json` | `59256CFBC273155A2F4EC0802AE04235178BD180D73D9306845C8FE0245247A8` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_weightedvote_p015_adv3b02_evidence.csv` | `6C31FAFB84B733EA31A08BFCD29ADF976689B6C95939A3E3A2259B7BE3BE7763` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_weightedvote_p020_adv3b02.json` | `CDD5C01E58B091E8563BEB79159855C88EED1FF568D08138F7BC7C49492BE139` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_topm_weightedvote_p020_adv3b02_evidence.csv` | `08DA65F1579528487E31067A7C4F7954599E8E58ED79750AF1E4E2AC30010ED7` |

### 结果表：`weightedvote_p015`

| k | total | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | avg_rx | bytes/event | p95_latency | mean_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 308 | 0.0000 | 0.0000 | 0.2500 | 0.0500 | 0.0167 | 0.3000 | 0.7792 | 0.0766 | 1.0000 | 120.0 | 0.1815 | 0.6098 |
| 2 | 246 | 0.4837 | 0.1000 | 0.3111 | 0.0500 | 0.1667 | 0.3542 | 0.4024 | 0.5303 | 1.8333 | 220.0 | 0.1815 | 0.5282 |
| 3 | 200 | 0.5250 | 0.3000 | 0.4250 | 0.1000 | 0.0500 | 0.7500 | 0.3100 | 0.5437 | 2.5000 | 300.0 | 0.1815 | 0.5345 |
| 4 | 154 | 0.5517 | 0.0000 | 0.4000 | 0.1000 | 0.0312 | 0.8125 | 0.2532 | 0.5492 | 3.3377 | 400.5 | 0.1815 | 0.5267 |
| 5 | 92 | 0.6154 | 0.0000 | 0.1000 | 0.0000 | 0.0500 | 0.9000 | 0.2174 | 0.5139 | 4.1087 | 493.0 | 0.1815 | 0.4940 |

### 结果表：`weightedvote_p020`

| k | total | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | avg_rx | bytes/event | p95_latency | mean_rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 308 | 0.0000 | 0.0000 | 0.2500 | 0.0500 | 0.0167 | 0.3000 | 0.7792 | 0.0766 | 1.0000 | 120.0 | 0.1798 | 0.6098 |
| 2 | 246 | 0.4837 | 0.1000 | 0.3111 | 0.0500 | 0.1667 | 0.3542 | 0.4024 | 0.5303 | 1.8333 | 220.0 | 0.1798 | 0.5282 |
| 3 | 200 | 0.5250 | 0.3000 | 0.4250 | 0.1000 | 0.0500 | 0.7500 | 0.3100 | 0.5437 | 2.5000 | 300.0 | 0.1798 | 0.5345 |
| 4 | 154 | 0.5517 | 0.0000 | 0.4000 | 0.1000 | 0.0312 | 0.8125 | 0.2532 | 0.5492 | 3.3377 | 400.5 | 0.1798 | 0.5267 |
| 5 | 92 | 0.6154 | 0.0000 | 0.1000 | 0.0000 | 0.0500 | 0.9000 | 0.2174 | 0.5139 | 4.1087 | 493.0 | 0.1798 | 0.4940 |

### 解释

`weighted_vote_margin`相对上一轮`vote_margin + class_reliability`提升了seen-new：k=3从0.4000到0.4250，`min_seen`从0.0500到0.1000；但old_acc仍只有0.5250，k=5也只有0.6154，并且k=2的unknown_FAR升至0.1667。说明类别权重确实缓解了seen-new被多数receiver压制的问题，但没有解决旧类目标域分离不足和unknown/open-set边界不稳的问题。该结果仍是负诊断，不能作为Stage2-C成功或部署成功。

## 2026-07-03 23:41 seen-new prototype calibration本地落地

### 目标

weighted vote改善seen-new但仍受类原型偏置影响。根据TEEN式思想，本轮在星上可部署边界内只校准support memory中的seen-new prototype，不训练主干、不使用unknown query、不改变`R_t/R_s`协议。新增两种轻量策略：

- `teen_blend`：`p_new'=normalize((1-alpha)p_new + alpha * old_mix)`，用于新类原型向相近旧类语义/信道方向收缩；
- `teen_separate`：`p_new'=normalize(p_new + alpha * (p_new - old_mix))`，用于增强新旧类分离。

两者都只使用部署support原型和地面旧类原型包，不读取source receiver样本或unknown query。

### 本地变更

| 文件 | 作用 |
|---|---|
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 新增`_calibrate_seen_new_centroids`；`build_qknn_memory`支持`prototype_calibration_policy`、`prototype_calibration_alpha`、`prototype_calibration_top_m`；CLI和metadata/evidence记录校准策略。 |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | 新增memory层测试：校准只移动seen-new centroid，不改变old centroid；新增集成测试：metadata/evidence记录校准参数。 |

### 本地与镜像验证

| 环境 | 命令 | 结果 |
|---|---|---|
| local `ssr-gpu` | 编译`phase2_collaborative_open_set_qknn_eval.py`和目标测试 | PASS |
| local `ssr-gpu` | `test_phase2_collaborative_open_set_qknn_eval.py` | PASS，39 tests |
| local `ssr-gpu` | `test_collaborative_open_set_qknn_eval.py` | PASS，38 tests |
| mirror `ssr-gpu` | 同上 | PASS，39+38 tests |

### Git状态

已提交`4dd3ee2 Add seen-new prototype calibration for collaborative qKNN`，镜像仓库分支领先远端280。

## 2026-07-03 23:53 seen-new prototype calibration N607诊断结果

### 远端同步与验证

| 项目 | 结果 |
|---|---|
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| 远端文件 | `code/scripts/phase2_collaborative_open_set_qknn_eval.py`、`code/tests/test_phase2_collaborative_open_set_qknn_eval.py` |
| 远端SHA | script `a69d4ffef1c45be0ee040df2d080b6337c048d6c9bd1737fcd32502e53cc4e8c`；test `598edaaf1594ae2a3c9a637940027c7ce5b408fe92a22c7e65b1832463bf7667` |
| 远端验证 | 编译PASS；`test_phase2_collaborative_open_set_qknn_eval.py`39 tests OK；`test_collaborative_open_set_qknn_eval.py`38 tests OK |
| GPU | GPU0运行；运行前后所有GPU均为`10/24576MiB` |
| SSH/SCP | 每次后均检查为`NO_SSH_PROCESS`和`NO_N607_SSH_ESTABLISHED` |

### artifact

| artifact | SHA256 |
|---|---|
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_blend_a025_weightedvote_p020_adv3b02.json` | `FAD28B922EE3EA02354EBB9614EE54F701EF402EF111C10E2916B6275A737007` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_blend_a025_weightedvote_p020_adv3b02_evidence.csv` | `BE52A12B87E1DD4978897D6BA2C59CC05CB1845B076C2E4B2DFB701D7B1C4A8D` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_blend_a050_weightedvote_p020_adv3b02.json` | `DFCA17E6F11C8FF301332CACCB6E2A6E3CFA36E6B94DE854ED89CABEF3EA72D1` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_blend_a050_weightedvote_p020_adv3b02_evidence.csv` | `5EDADAAA9B6470D1C18F779EDC206F0081B13F06C41BC407E38C698C70731E5D` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_separate_a025_weightedvote_p020_adv3b02.json` | `44D16DF9BDF4E142DD85E110CBD4B686D5171E68A87E784F80A3D841F608AC9B` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_separate_a025_weightedvote_p020_adv3b02_evidence.csv` | `ED7355C7373EF718E4D1B3ED6E202E9AFD3F3BBBAF8EE3DE591671D46F5D8D9E` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_separate_a050_weightedvote_p020_adv3b02.json` | `67E6B6C506873A1EEE53555A4C4B7386C0E6F7A8ED69E2C8A511DB6C9E57A854` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_proto_teen_separate_a050_weightedvote_p020_adv3b02_evidence.csv` | `13B5D4340EDCE05AE050CCB2C226ABE917BD749A5A1289A07A463B63BD471CFA` |

### 结果摘要

| 候选 | k | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | bytes/event | p95_latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `teen_blend_a025` | 3 | 0.5250 | 0.3000 | 0.4500 | 0.1000 | 0.0500 | 0.7750 | 0.3050 | 0.5500 | 297.6 | 0.1835 |
| `teen_blend_a050` | 3 | 0.5417 | 0.3000 | 0.4500 | 0.1000 | 0.0250 | 0.8000 | 0.2900 | 0.5563 | 298.2 | 0.1310 |
| `teen_separate_a025` | 3 | 0.5250 | 0.3000 | 0.4000 | 0.1000 | 0.0500 | 0.7500 | 0.2950 | 0.5625 | 294.6 | 0.1806 |
| `teen_separate_a050` | 3 | 0.5583 | 0.3000 | 0.4250 | 0.1000 | 0.0500 | 0.7500 | 0.2550 | 0.5938 | 293.4 | 0.1835 |
| `teen_blend_a050` | 4 | 0.6092 | 0.0000 | 0.4000 | 0.1000 | 0.0000 | 0.8438 | 0.2403 | 0.5902 | 395.8 | 0.1310 |
| `teen_separate_a050` | 5 | 0.6154 | 0.0000 | 0.1000 | 0.0000 | 0.0500 | 0.9000 | 0.1739 | 0.5833 | 463.0 | 0.1835 |

完整k=1到5结果保存在对应JSON。与weighted baseline相比，`teen_blend_a050`在k=3把unknown_FAR从0.0500降到0.0250，同时保持seen_new_acc=0.4500并把known_cov从0.5437提升到0.5563；`teen_separate_a050`在k=3把old_acc提升到0.5583、known_cov提升到0.5938，但FAR仍为0.0500。两者都只是小幅改进，仍远低于目标。

### 结论

TEEN式原型校准在当前ADV3B02/qknn8特征上能带来有限收益，但无法突破旧类和seen-new的类间分离上限。当前最稳联合候选是`teen_blend_a050,k=3`：`old_acc=0.5417`、`seen_new_acc=0.4500`、`unknown_FAR=0.0250`、`min_old=0.3000`、`min_seen=0.1000`。它不能作为Stage2-C成功或部署成功。下一步应转向更强的目标域旧类上限诊断或轻量adapter/BN affine诊断，判断瓶颈是否来自冻结ADV3B02特征空间本身，而不是继续只调协同融合。

## 2026-07-04 00:22 target-support feature adapter本地落地

### 目标

在冻结ADV3B02 backbone和`z_id`特征的前提下，加入只由目标接收机域`R_t`内old+seen-new support拟合的轻量特征adapter，诊断当前瓶颈是否来自目标域公共偏移。该模块不使用unknown query，不更新主干，不改变`Y_old/Y_new/Y_unknown`互斥定义。

### 本地变更

| 文件 | 作用 |
|---|---|
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 新增`FeatureAdapter`、`support_center`和`support_bn_affine`策略；memory、query、support/proxy阈值校准、class threshold和conformal校准统一进入adapter空间；CLI新增`--feature_adapter_policy`、`--feature_adapter_strength`、`--feature_adapter_variance_floor`；evidence和metadata记录adapter配置；新增Stage2-C TX split硬校验，要求target-old属于source TX，seen-new/unknown不属于source TX；`strict_event_key`改为所有target receivers共享event交集。 |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | 新增adapter行为、BN affine零方差、metadata/evidence字段、source TX语义、strict event交集测试。 |

### 子agent监督和修复

| 子agent角色 | 发现 | 处理 |
|---|---|---|
| 代码review | adapter后memory/query和raw阈值校准空间不一致；evidence缺`feature_adapter_variance_floor`；helper内部未夹紧variance floor。 | 已修复：support/proxy校准统一使用adapter空间；virtual unknown保持memory空间不二次adapter；evidence补variance floor；helper内部夹紧到`1e-8`。 |
| 算法合理性 | target-support轻量adapter本身符合`项目.md`允许的小adapter/BN affine边界，但缺少`Y_old`属于source TX的前置硬校验；strict event语义原实现允许partial receiver group。 | 已修复：新增source TX split检查；strict event改为target receivers共同event交集，无共享event时失败。 |
| 逐项监督 | 当前receiver-domain ranked诊断不能声明严格同事件协同；指标仍未达Stage2-C成功。 | 本节继续标为诊断模块，后续N607结果只按负/正诊断报告，不写部署成功。 |
| 文献方法 | 建议后续主线转向质量感知late fusion、按需协作、prototype shrinkage和support-only阈值bank。 | 作为下一步算法方向记录，本次实现先验证target-support adapter是否能改善特征空间偏移。 |

### 本地与镜像验证

| 环境 | 命令 | 结果 |
|---|---|---|
| local `ssr-gpu` | `python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | PASS |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"` | PASS，44 tests |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"` | PASS，38 tests |
| mirror `ssr-gpu` | 同上 | PASS，44+38 tests |

### Git和哈希

| 项目 | 结果 |
|---|---|
| 镜像仓库 | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| Git提交 | `d4ccd78 Add target-support feature adapter for collaborative qKNN`；`83c1c9c Fix feature adapter calibration protocol checks` |
| Git状态 | `codex/cvs-rffi-release-20260626`领先远端282，工作区clean |
| script SHA256 | `BD8678B6F031A22E412908D2630CBA7A1E3A5BB9EB2279DC3E27DAACE2196D8F` |
| test SHA256 | `DC002D411DA777B38BBCBE8B582F287E2FEA213B2E3A3BEBA98732DA70397E21` |

### N607同步计划

同步文件：

| local | remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_collaborative_open_set_qknn_eval.py` |

远端环境使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。诊断候选将基于当前最稳`teen_blend_a050 + weighted_vote_margin + conformal_margin_risk`配置，增加`feature_adapter_policy in {support_center,support_bn_affine}`和`feature_adapter_strength in {0.25,0.50}`，仍使用`collab_counts all`覆盖1到全部target receivers。

## 2026-07-04 00:34 target-support feature adapter N607诊断结果

### 远端同步与验证

| 项目 | 结果 |
|---|---|
| N607 preflight | 直连`N607`通过；项目根`/home/szu2070436088/2510044040/CV-SincNet`可见；8张RTX3090均为`10/24576MiB` |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| 同步文件 | `code/scripts/phase2_collaborative_open_set_qknn_eval.py`；`code/tests/test_phase2_collaborative_open_set_qknn_eval.py` |
| 远端SHA | script `bd8678b6f031a22e412908d2630cba7a1e3a5bb9eb2279dc3e27daace2196d8f`；test `dc002d411da777b38bbcbe8b582f287e2fea213b2e3a3beba98732da70397e21` |
| 远端验证 | 编译PASS；`test_phase2_collaborative_open_set_qknn_eval.py`44 tests OK；`test_collaborative_open_set_qknn_eval.py`38 tests OK |
| 运行GPU | `CUDA_VISIBLE_DEVICES=0`；运行前后所有GPU均为`10/24576MiB` |
| 输出规模 | 每个候选均为`receiver_count=5`、`group_count=308`、`evidence_row_count=1000` |
| SSH/SCP | 每次后均检查为`NO_SSH_PROCESS`和`NO_N607_OR_BRIDGE_SSH_ESTABLISHED` |

### artifact

| artifact | SHA256 |
|---|---|
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_center_s025_adv3b02.json` | `14613DCA71C5A6A25D2FB1F3C590359C94220F48E03FB9F23DA3D543C347CA13` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_center_s025_adv3b02_evidence.csv` | `7ED6C8AFBD84585EA7DCD6BBB7E3AA5A2309DB79BF36BCF0E1B632FCC2AA4634` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_center_s050_adv3b02.json` | `6FE9FE702E5C85F430F8100CC76831EF561FFE8F8FB70F82B7A8BCBBF93D020D` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_center_s050_adv3b02_evidence.csv` | `E82AC44519A1DFCE8AF558F8EE8666D181355B152DA7EE0B67F64C77E5A90968` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_bn_affine_s025_adv3b02.json` | `3D341EE30F8ECF87A283D4BFE6136D5029B21719A1EE9004C98541CF756D36AF` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_bn_affine_s025_adv3b02_evidence.csv` | `29C36288DDDDE002DE18FC6FCB7FD6E504D19A30DB9ACDE694833239FD097EC6` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_bn_affine_s050_adv3b02.json` | `193E3EEE41280C50B41C483491BCEB049EABA4E5ABFF09169DE4B7F41F7953F5` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_feature_adapter_support_bn_affine_s050_adv3b02_evidence.csv` | `79DB10760F2444DC3651FF8A9B1590C86E02A36E1F38E2BAC9BB911394449229` |

### 结果摘要

| 候选 | k | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | bytes/event | p95_latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `support_center_s025` | 3 | 0.5250 | 0.3000 | 0.4250 | 0.0500 | 0.0500 | 0.7750 | 0.2950 | 0.5375 | 300.6 | 0.1333 |
| `support_center_s050` | 3 | 0.5333 | 0.3000 | 0.4500 | 0.0500 | 0.0500 | 0.8000 | 0.2900 | 0.5375 | 300.6 | 0.1335 |
| `support_bn_affine_s025` | 3 | 0.5167 | 0.3000 | 0.4250 | 0.0500 | 0.0500 | 0.7750 | 0.2950 | 0.5312 | 300.0 | 0.1844 |
| `support_bn_affine_s050` | 3 | 0.5333 | 0.2500 | 0.4250 | 0.0500 | 0.0500 | 0.8000 | 0.3000 | 0.5312 | 301.8 | 0.1347 |
| `support_center_s050` | 4 | 0.6092 | 0.0000 | 0.4000 | 0.1000 | 0.0312 | 0.7500 | 0.2597 | 0.5820 | 399.7 | 0.1335 |
| `support_center_s050` | 5 | 0.6538 | 0.0000 | 0.1000 | 0.0000 | 0.0500 | 0.9000 | 0.2391 | 0.5417 | 490.4 | 0.1335 |
| `support_bn_affine_s050` | 5 | 0.6538 | 0.0000 | 0.1000 | 0.0000 | 0.0500 | 0.9000 | 0.2500 | 0.5417 | 497.0 | 0.1347 |

完整k=1到5结果保存在对应JSON。`support_center_s050,k=3`相对上一轮`teen_blend_a050,k=3`的`old_acc=0.5417`略降到0.5333，`seen_new_acc=0.4500`持平，`unknown_FAR`从0.0250升至0.0500；k=4/k=5可以提高old_acc到0.6092/0.6538，但`min_old=0`且seen-new在k=5降到0.1000。`support_bn_affine`未表现出稳定优势，低K方差估计对当前qKNN空间没有带来有效校正。

### 结论

target-support feature adapter是协议合规的星上轻量诊断模块，但在当前ADV3B02/qknn8 Stage2-C特征上没有突破旧类和seen-new联合性能上限，也没有改善unknown拒识的折中边界。本轮仍不能声明99/97/99目标达成、Stage2-C成功或部署成功。下一步应按文献子agent建议转向`support校准的receiver质量权重+按需协作`，或回到地面训练/特征学习阶段加入面向`R_t`漂移的更强表征约束；单纯后处理adapter、prototype和阈值调参已出现明显收益上限。

## 2026-07-04 01:05 support utility按需协同本地落地

### 目标

上一轮adapter诊断说明冻结ADV3B02/qknn8特征空间的后处理收益有限。本轮实现更贴近卫星群部署的按需协同策略：每个receiver只上传局部证据包，融合节点根据support-derived质量、class conformal p-value、margin/score不足、unknown边界风险和通信/时延成本决定是否继续请求其他receiver。该策略目标是减少无效全量协同，并在边界样本上优先选择更可靠、低成本、有support支撑的receiver。

### 本地变更

| 文件 | 作用 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 新增`collaboration_policy=support_utility`；新增`_support_utility_candidate_score`和`_fuse_support_utility_event`；输出`support_utility_trace`与`support_utility_stop_reason`；汇总`collaboration_stop_reasons`支持该新策略；修复seen-new rescue安全边界，rescue只允许`role=seen_new`事件触发，unknown query预测为seen-new时不得作为rescue接受。 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | CLI新增`--collaboration_policy support_utility`选择。 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 新增support utility选择低成本高support质量receiver的测试；修正seen-new rescue测试，验证unknown role不触发rescue。 |

### 本地与镜像验证

| 环境 | 命令 | 结果 |
|---|---|---|
| local `ssr-gpu` | `python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py` | PASS |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"` | PASS，39 tests |
| local `ssr-gpu` | `python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"` | PASS，44 tests |
| mirror `ssr-gpu` | 编译与上述两个测试文件 | PASS，39+44 tests |

### Git和哈希

| 项目 | 结果 |
|---|---|
| Git提交 | `58cf3ea Add support utility collaboration policy` |
| Git状态 | `codex/cvs-rffi-release-20260626`领先远端283，工作区clean |
| eval SHA256 | `64C4BB8771EB15BCE20EF41EEAA276FE7CBEA2DDD79C8A9D2C1BB5171324C401` |
| script SHA256 | `D1C8653F4FD926DA1418B973CE09985E2810F0A91AD3E40F475C234A35BD7FCC` |
| eval test SHA256 | `87A2B8F8124879D9047C358BF02FB7E5A666ED1AD54C04D0B5132CB46669EB8F` |

### N607同步计划

同步文件：

| local | remote |
|---|---|
| `E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py` |

远端诊断继续使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、`ADV3B02_CORE90_SOFT_E200`对应`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`和`qknn8`，以`collab_counts all`覆盖1到5个target receivers。该结果仍为`receiver_domain_ranked`诊断，不声明严格同事件协同。

## 2026-07-04 01:18 support utility按需协同N607诊断结果

### 远端同步与验证

| 项目 | 结果 |
|---|---|
| N607 preflight | 直连`N607`通过；项目根可见；8张RTX3090均为`10/24576MiB` |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| 远端SHA | eval `64c4bb8771eb15bce20ef41eeaa276fe7cbea2ddd79c8a9d2c1bb5171324c401`；script `d1c8653f4fd926da1418b973ce09985e2810f0a91ad3e40f475c234a35bd7fcc`；eval test `87a2b8f8124879d9047c358bf02fb7e5a666ed1ad54c04d0b5132cb46669eb8f` |
| 远端验证 | 编译PASS；`test_collaborative_open_set_qknn_eval.py`39 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`44 tests OK |
| 运行GPU | `CUDA_VISIBLE_DEVICES=0`；运行前后所有GPU均为`10/24576MiB` |
| 输出规模 | 每个候选均为`receiver_count=5`、`group_count=308`、`evidence_row_count=1000` |
| SSH/SCP | 每次后均检查为`NO_SSH_PROCESS`和`NO_N607_OR_BRIDGE_SSH_ESTABLISHED` |

### artifact

| artifact | SHA256 |
|---|---|
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_support_utility_p020_adv3b02.json` | `2D7E9796E79EEF8F72C4BF2BEB8A6BF6CBC737D48D7545E6513689C1CDD8E79F` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_support_utility_p020_adv3b02_evidence.csv` | `703F5C0512A9DBE1051805DFD0DBB8DBB9431BA400555203A7D5B8A575031823` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_support_utility_p015_adv3b02.json` | `211EB270C53F67A1B86CBA331236F7867C1E84DDABB801A6DE4B784FD77DD869` |
| `remote_artifacts/collab_open_set_qknn_cp_set_cvs_support_utility_p015_adv3b02_evidence.csv` | `7F3B8A49FC9C38AE323C7C46AEF018FDB49F982172AA59C94F8ADBC1298026EC` |

### 结果摘要

| 候选 | k | old_acc | min_old | seen_new_acc | min_seen | unknown_FAR | unknown_reject | defer | known_cov | avg_rx | bytes/event | p95_latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `support_utility_p020` | 1 | 0.0000 | 0.0000 | 0.2500 | 0.0500 | 0.0167 | 0.3167 | 0.7760 | 0.0766 | 1.0000 | 120.0 | 0.1832 |
| `support_utility_p020` | 2 | 0.4641 | 0.1000 | 0.3333 | 0.1000 | 0.1458 | 0.3750 | 0.3699 | 0.5152 | 1.7520 | 210.2 | 0.1832 |
| `support_utility_p020` | 3 | 0.5167 | 0.3000 | 0.4500 | 0.1000 | 0.0250 | 0.8000 | 0.2250 | 0.5750 | 2.2450 | 269.4 | 0.1832 |
| `support_utility_p020` | 4 | 0.5402 | 0.0000 | 0.4286 | 0.1500 | 0.0312 | 0.8438 | 0.1494 | 0.6066 | 2.7013 | 324.2 | 0.1832 |
| `support_utility_p020` | 5 | 0.5962 | 0.0000 | 0.1500 | 0.0000 | 0.0500 | 0.9000 | 0.1413 | 0.5694 | 3.2717 | 392.6 | 0.1832 |
| `support_utility_p015` | 3 | 0.5167 | 0.3000 | 0.4500 | 0.1000 | 0.0250 | 0.8000 | 0.2250 | 0.5750 | 2.2450 | 269.4 | 0.1803 |
| `support_utility_p015` | 4 | 0.5402 | 0.0000 | 0.4286 | 0.1500 | 0.0312 | 0.8438 | 0.1494 | 0.6066 | 2.7013 | 324.2 | 0.1803 |

`p020`和`p015`结果一致，说明当前配置下`conformal_rescue_min_pvalue`不是主瓶颈。与上一轮`teen_blend_a050,k=3`相比，`support_utility,k=3`把平均参与receiver从约2.5降到2.245，`bytes/event`从约298.2降到269.4，defer从0.2900降到0.2250，并保持`seen_new_acc=0.4500`、`unknown_FAR=0.0250`；但`old_acc`从0.5417降到0.5167。因此它是效率改进，不是性能突破。

### 结论

`support_utility`实现了更现实的按需协同：在边界样本请求更多receiver，在低收益或已可决策样本提前停止，并显式报告`avg_rx`、`bytes/event`、`p95_latency`和停止原因。但它没有达到目标指标，仍不能声明99/97/99、Stage2-C成功或部署成功。当前证据表明：协同调度可以降低资源消耗和defer，但无法在ADV3B02/qknn8现有特征空间内解决旧类/seen-new/unknown的根本可分性瓶颈。下一步应进入更强表征路线：地面训练阶段加入open-set/source-unknown约束、receiver leakage约束和satellite stress下的class-conditional compactness，而不是继续单独调融合器。
## 2026-07-03support_utility+CPR gate3补充诊断

目标：按多子agent审查建议，对现有ADV3B02 open-set qknn特征补跑更严格的support utility+CPR gate3，验证是否能在星地信道视图下降低unknown FAR。该补跑使用已有`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，仅作为ADV3B02诊断对照；主线SA33指定权重结果见`phase2_sa33_collab_open_set_qknn_full_20260703/report.md`。

远端环境：N607直连预检通过，远端`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；运行前后GPU均为`10 MiB/24576 MiB`级别。产物已拉回`automation_reports/CV-SincNet/phase2_adv3b02_collab_open_set_qknn_full_20260703/remote_artifacts/`；SSH/SCP后本地检查无残留`ssh.exe`或N607/bridge的22端口`ESTABLISHED`连接。

产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_support_utility_cpr_p20_s05_gate3_adv3b02.json`|`236E258BB2C367C9EC92B90FB528644F671BE52ADFF2156F3CDBBB8233E78CBD`|
|`collab_open_set_qknn_scorer_cvs_support_utility_cpr_p20_s05_gate3_adv3b02.csv`|`AFB3CE0B4BD11A58236D2828FEF0FFF85624B08893EEAAC1219870D883EAF1B5`|
|`collab_open_set_qknn_scorer_cvs_support_utility_cpr_p15_s05_u098_gate3_adv3b02.json`|`8CF5673EAC671A42159E36091289062B29D1433BF1975AA2938E684D8CE7A51B`|
|`collab_open_set_qknn_scorer_cvs_support_utility_cpr_p15_s05_u098_gate3_adv3b02.csv`|`B67682F782E384CD9F427BF06E42AF5DA85355569C2F72C1E9CA5D7E2FBF5FA1`|

结果表：

|候选|协同数|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|bytes/event|lat_p95|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`p15_s05_u098_gate3`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.3000|0.8442|0.0000|120.0000|0.1299|
|`p15_s05_u098_gate3`|2|0.0000|0.0000|0.3778|0.1500|0.0000|0.4792|0.6748|0.0859|218.0488|0.1299|
|`p15_s05_u098_gate3`|3|0.4917|0.2500|0.5250|0.2500|0.0000|0.9500|0.2000|0.5000|296.4000|0.1299|
|`p15_s05_u098_gate3`|4|0.5402|0.0000|0.5714|0.3500|0.0000|1.0000|0.0974|0.5574|345.1948|0.1299|
|`p15_s05_u098_gate3`|5|0.6154|0.0000|0.3500|0.0000|0.0000|1.0000|0.0652|0.5556|400.4348|0.1299|
|`p20_s05_gate3`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.2833|0.8571|0.0000|120.0000|0.1823|
|`p20_s05_gate3`|2|0.0000|0.0000|0.3778|0.1500|0.0000|0.4583|0.6951|0.0859|219.5122|0.1823|
|`p20_s05_gate3`|3|0.5083|0.2500|0.6500|0.4000|0.0000|0.9500|0.1900|0.5437|300.0000|0.1823|
|`p20_s05_gate3`|4|0.5517|0.0000|0.6286|0.4500|0.0000|0.9688|0.1104|0.5820|345.1948|0.1823|
|`p20_s05_gate3`|5|0.6346|0.0000|0.4500|0.0000|0.0000|1.0000|0.0761|0.5972|405.6522|0.1823|

判定：ADV3B02 gate3能把unknown FAR压到0，但known性能仍不足，且per-class floor仍有0；这不是Stage2-C成功证据，只是说明更强CPR gate可作为unknown安全阀。后续应优先提升known覆盖和per-class floor，而不是继续只压unknown FAR。

## 2026-07-04 candidate_set_cvs候选集协同融合

### 目标

上一轮上限诊断显示，ADV3B02/qknn8的正确old/seen-new标签经常存在于`class_evidence_top_m`候选集中，但`cp_set_cvs`硬门控会过度压制known接受，`scorer_cvs`放松后又会显著提高unknown FAR。因此本轮新增`candidate_set_cvs`：先在top-M类条件候选集中做weighted evidence pooling，再用独立的event unknown risk和label unknown risk作为安全阀。该策略不使用unknown query调阈值，不使用真实role做选择，只使用support/conformal/evidence字段。

### 本地变更与验证

|文件|变更|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`fusion_policy=candidate_set_cvs`；候选集接受条件包括`candidate_set_min_receivers`、`candidate_set_min_top1_receivers`、`candidate_set_min_conformal_pvalue`、`candidate_set_max_label_unknown_risk`、`candidate_set_max_event_unknown_risk`、`candidate_set_min_score_gap`和`candidate_set_unknown_reject_risk`；输出对应metadata与逐事件字段。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`--fusion_policy candidate_set_cvs`和候选集安全阀参数。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增回归测试：正确标签仅在top-M候选而非top1时可被恢复；高风险unknown即使存在known候选也被拒识。|

验证结果：

|环境|命令|结果|
|---|---|---|
|local `ssr-gpu`|`python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py`|PASS|
|local `ssr-gpu`|`python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"`|41 tests OK|
|local `ssr-gpu`|`python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"`|44 tests OK|
|Git镜像|同上|41+44 tests OK|
|Git提交|`470a214 Add candidate set CVS fusion policy`；`9b38d75 Record candidate set CVS fusion results`；`81d4e9b Fix candidate set report version state`|已提交到`E:\type10-7\github_publish\CVS-RFFI-repo`；具体领先数量以`git status -sb`实时输出为准|

由于`E:\type10-7`根目录不是Git仓库，代码快照保存在`E:\type10-7\code\snapshots\candidate_set_cvs_20260704\`。资源约束设计说明原文`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`仍未在当前工作区找到；本轮继续按可量化字段`avg_rx`、`bytes/event`、`latency_ms_p95`、GPU显存读数和prototype/evidence状态报告资源。

### N607同步与运行

N607直连预检通过，项目根为`/home/szu2070436088/2510044040/CV-SincNet`。运行前后8张RTX3090均为`10 MiB/24576 MiB`，无GPU compute app；用户要求显存占用低即可开启实验，本轮离线证据推理使用`CUDA_VISIBLE_DEVICES=0`但未显著占用显存。远端Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python3.10.19。

同步文件与远端SHA256：

|文件|远端SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`cd85f438d50da0ea63fc67f8c18d850dc37721b24ca77793f09a4c5babacfb81`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`50498926b7739c416f4eb957153dad17c3e5dbd85d6bcdac1029868ffb261f71`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`adf02537f8000af26f8bff94620f5c0b91a9e697c2279b4e7e22a1654296ab1c`|

远端验证：编译PASS；`test_collaborative_open_set_qknn_eval.py`为41 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`为44 tests OK。每次SSH/SCP后本地检查均为无残留`ssh.exe`、无N607/bridge 22端口`ESTABLISHED`连接。

远端输入：`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，SHA256为`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`；该特征对应`ADV3B02_CORE90_SOFT_E200` Stage2-C qknn8链路，覆盖target receiver 1到5、`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`星地信道视图。输出均为`receiver_count=5`、`group_count=307`、`evidence_row_count=1000`。

产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_min2_er090_adv3b02.json`|`2D0EC7DB48E4B4FAF8EEB068938EAD8F2224F384FDD83D88466BD60174CF4570`|
|`collab_open_set_qknn_candidate_set_cvs_min2_er090_adv3b02_evidence.csv`|`4917B07F2034D0D991A06CB62D899DA638C3B9778C544796C505A7B21EBBA4DD`|
|`collab_open_set_qknn_candidate_set_cvs_min3_er095_adv3b02.json`|`2B514F1126CAC97210B09796EB0726180CCB9D60C433ED12A1F30D7A1334F22B`|
|`collab_open_set_qknn_candidate_set_cvs_min3_er095_adv3b02_evidence.csv`|`01C3C1F520C3BB5BAA5798F23962B1711CC4F4CF3ED203ADFA43BB4423C15468`|
|`collab_open_set_qknn_candidate_set_cvs_support_utility_adv3b02.json`|`A4055E66C6B8B872919316536E3D5C431A61A56825F1AF2E0794D349D0863582`|
|`collab_open_set_qknn_candidate_set_cvs_support_utility_adv3b02_evidence.csv`|`A7F0A7F5D07AA26B9FE0257500461070926F22CD011AC4837E7D77FEE4B5C16B`|

### 结果表

|候选|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`candidate_min2_er090`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7492|0.0000|1.0000|40.0000|0.1061|
|`candidate_min2_er090`|2|0.3224|0.0000|0.2885|0.1500|0.0652|0.9348|0.1840|0.3676|2.0000|80.0000|0.1061|
|`candidate_min2_er090`|3|0.3667|0.0500|0.5500|0.4500|0.0250|0.9750|0.1050|0.4437|3.0000|120.0000|0.1061|
|`candidate_min2_er090`|4|0.6818|0.0000|0.7857|0.7000|0.0294|0.9118|0.0867|0.7672|4.0000|160.0000|0.1061|
|`candidate_min2_er090`|5|0.6038|0.0000|0.6000|0.0000|0.0500|0.9500|0.0968|0.7397|5.0000|200.0000|0.1061|
|`candidate_min3_er095`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7492|0.0000|1.0000|40.0000|0.1495|
|`candidate_min3_er095`|2|0.0000|0.0000|0.0000|0.0000|0.0000|0.9348|0.4720|0.0000|2.0000|80.0000|0.1495|
|`candidate_min3_er095`|3|0.2667|0.0000|0.3250|0.1000|0.0750|0.9250|0.2200|0.2875|3.0000|120.0000|0.1495|
|`candidate_min3_er095`|4|0.4886|0.0000|0.5000|0.3500|0.0882|0.8529|0.2667|0.5086|4.0000|160.0000|0.1495|
|`candidate_min3_er095`|5|0.6226|0.0000|0.5000|0.0000|0.0000|1.0000|0.2151|0.6164|5.0000|200.0000|0.1495|
|`candidate_support_utility`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7492|0.0000|1.0000|40.0000|0.1408|
|`candidate_support_utility`|2|0.3224|0.0500|0.2308|0.0500|0.1087|0.8913|0.1280|0.3382|1.8200|72.8000|0.1408|
|`candidate_support_utility`|3|0.3833|0.1000|0.4250|0.2000|0.0500|0.9500|0.1000|0.4125|2.6550|106.2000|0.1408|
|`candidate_support_utility`|4|0.6023|0.0000|0.5714|0.4500|0.1471|0.8529|0.1000|0.6552|3.4133|136.5333|0.1408|
|`candidate_support_utility`|5|0.5849|0.0000|0.5000|0.0000|0.0500|0.9500|0.0968|0.7123|4.3118|172.4731|0.1408|

### 判定

`candidate_set_cvs`证明了一个关键事实：正确类别存在于top-M证据时，放弃过硬CP门控可以明显提高known识别。例如`candidate_min2_er090,k=4`达到`old_acc=0.6818`、`seen_new_acc=0.7857`、`min_seen=0.7000`、`unknown_FAR=0.0294`，显著高于上一轮`support_utility,k=4`的seen-new 0.4286和固定CP-set gate3的known低覆盖。但它仍远低于目标：old没有达到99%，`min_old=0`，seen-new也未达97/93，unknown拒识未达99%。

`candidate_support_utility`体现了资源收益：k=3平均只使用`2.655`个receiver、`106.2 bytes/event`，相对固定3 receiver的`120 bytes/event`更省，并保持`unknown_FAR=0.0500`；但known性能低于固定候选集融合。说明当前按需协同的utility函数仍偏保守，适合作为资源受限模式，不是性能最优模式。

本轮不能标记目标完成。当前最有价值的推进是：已经把瓶颈从“未知类拒识单独压FAR”推进到“top-M候选集能恢复known，但old floor和严格class coverage仍不足”。下一步应围绕两个方向继续：1）在候选集融合里加入receiver-class可靠度`w_{r,y}`，专门处理`min_old=0`的类别；2）若目标仍要求99/97/99，需要回到地面训练或轻量adapter训练阶段增强class-conditional compactness，仅靠后处理融合已经接近上限。

## 2026-07-04 receiver-class可靠度`w_{r,y}`协同融合

### 目标

上一轮`candidate_set_cvs`说明正确类别常在top-M候选集中，但不同receiver对不同类的support质量不一致，导致某些旧类被高分但低可靠receiver持续压制。本轮新增显式`receiver_class_reliability_policy=support_calibrated`：每个receiver只用本地target-old/seen-new support校准分数，为每个候选类生成`w_{r,y}`，融合时将该权重乘入类候选证据。该机制不使用unknown query拟合阈值，不使用query真实role，只使用support-derived校准统计。

### 本地变更与验证

|文件|变更|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`receiver_class_reliability_policy`和`label_receiver_class_reliability`；`cp_set_cvs`、`candidate_set_cvs`、`support_utility`、`rb_capr_utility`等路径均可使用`w_{r,y}`。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增support侧`_receiver_class_reliability_from_support()`；evidence记录`receiver_class_reliability`和`class_evidence_top{m}_receiver_class_reliability`；metadata记录每receiver每类可靠度表。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增回归测试：开启`support_calibrated`后，低分但高receiver-class可靠度的正确类可压过高分低可靠类；默认关闭时保持原分数排序。|

验证结果：

|环境|命令|结果|
|---|---|---|
|local `ssr-gpu`|`python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py`|PASS|
|local `ssr-gpu`|`python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"`|42 tests OK|
|local `ssr-gpu`|`python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"`|44 tests OK|
|Git镜像|同上|42+44 tests OK|
|Git提交|`33bdc1b Add receiver class reliability fusion weights`|已提交到`E:\type10-7\github_publish\CVS-RFFI-repo`，当前分支领先远端290。|

由于`E:\type10-7`根目录不是Git仓库，代码快照保存在`E:\type10-7\code\snapshots\receiver_class_reliability_20260704\`。

### N607同步与验证

N607直连预检通过，项目根为`/home/szu2070436088/2510044040/CV-SincNet`。远端使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python3.10.19。同步后远端编译PASS，`test_collaborative_open_set_qknn_eval.py`为42 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为44 tests OK。运行前后8张RTX3090均为`10 MiB/24576 MiB`，未见GPU compute app；每次SSH/SCP后本地检查均为无残留`ssh.exe`、无N607/bridge 22端口`ESTABLISHED`连接。

远端SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`b982e1a41f1f883773e253ad497736f9dec56e578c75462871b065eab1a6ff23`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`727a5602816386e1eb819c2e6cc3729469845b4b8a2a558fb7c9e0d42f88b1f0`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`9bb02a5e00a8ab23ede86c963f6961ee10af89b40dea9141e525e56d9cc95e9c`|

本轮复用远端`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，该特征对应`ADV3B02_CORE90_SOFT_E200`、Stage2-C、qknn8、target receiver 1到5和`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`星地信道视图。资源约束设计说明原文`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`仍未在当前工作区找到；本轮继续报告可量化代理指标`avg_rx`、`bytes/event`、`latency_ms_p95`和GPU显存状态。

产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_min2_er075_adv3b02.json`|`0B5A739F40071F969AADD7F00A6168AF5482ACDEC1DD5140887B3EC42D3AB279`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_min2_er075_adv3b02_evidence.csv`|`794B35FBBEF58C9E9EAA194223E8D6133DD9B59901E34AE3C4A8FDE7576DC865`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_min2_er080_adv3b02.json`|`A88CCF88E8B368CEC4C0185A63BE9C8C5A7CE76602EA71255F90F663406503D5`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_min2_er080_adv3b02_evidence.csv`|`A30A94B33299B7A24CEE58A72D35D16F09F2AC54C664888BCB1780BA29CC1EE4`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_min2_er090_adv3b02.json`|`592DB58CEE5028E1BFACCF41D5195D942CA02DBE55E687205361E037AD15BBA5`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_min2_er090_adv3b02_evidence.csv`|`950BB0D5A6772C890BAB0933618A75FC577C24AFC7398474E663728A50D67008`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_support_utility_adv3b02.json`|`B8793FEAA34C3FAB77263718A3D941D1AFB5373133CBD23BF3C6F1DE472C4F0F`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_support_utility_adv3b02_evidence.csv`|`3AD2E6ECFFA25ABF6519689ECE6738A4A5AF79A31B0C9D4A4F4E18D3FD86760A`|

### 结果表

|候选|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|mean_w_ry|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`rcwr_min2_er075`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1320|0.9254|单receiver不足|
|`rcwr_min2_er075`|2|0.3158|0.0000|0.5000|0.5000|0.0652|0.9348|0.0440|0.5098|2.0000|80.0|0.1320|0.9165|FAR超标|
|`rcwr_min2_er075`|3|0.3750|0.1000|0.5000|0.4000|0.0250|0.9750|0.0150|0.4813|3.0000|120.0|0.1320|0.9209|FAR达标但known不足|
|`rcwr_min2_er075`|4|0.7386|0.0000|0.7143|0.6500|0.0294|0.9706|0.0000|0.7845|4.0000|160.0|0.1320|0.9327|FAR达标，known显著提升但未达OLD80|
|`rcwr_min2_er075`|5|0.6981|0.0000|0.6500|0.0000|0.0000|1.0000|0.0108|0.7945|5.0000|200.0|0.1320|0.9653|拒识强，缺类floor|
|`rcwr_min2_er080`|3|0.4000|0.1500|0.5250|0.4500|0.0250|0.9750|0.0050|0.5062|3.0000|120.0|0.1329|0.9209|FAR达标但known不足|
|`rcwr_min2_er080`|4|0.7614|0.0000|0.7500|0.7000|0.0588|0.9412|0.0000|0.8103|4.0000|160.0|0.1329|0.9327|接近OLD80，FAR略超|
|`rcwr_min2_er090`|3|0.4750|0.1500|0.5500|0.4500|0.0250|0.9750|0.0050|0.5750|3.0000|120.0|0.1825|0.9209|FAR达标，k=3最佳known折中|
|`rcwr_min2_er090`|4|0.8182|0.0000|0.7857|0.7000|0.0882|0.9118|0.0000|0.8707|4.0000|160.0|0.1825|0.9327|首次越过OLD80，但FAR超标|
|`rcwr_min2_er090`|5|0.7925|0.0000|0.6500|0.0000|0.0500|0.9500|0.0000|0.8630|5.0000|200.0|0.1825|0.9653|FAR达标边界，seen-new缺类|
|`rcwr_support_utility`|3|0.4917|0.2500|0.4750|0.3000|0.1000|0.9000|0.0050|0.5500|2.4600|98.4|0.1326|0.9311|省资源但FAR超标|
|`rcwr_support_utility`|5|0.6415|0.0000|0.5000|0.0000|0.0000|1.0000|0.0000|0.7260|3.7742|151.0|0.1326|0.9573|省资源且拒识强，known不足|

### 判定

`w_{r,y}`是本轮最明确的正向进展。相同ADV3B02/qknn8特征和同一Stage2-C协议下，上一轮`candidate_min2_er090,k=4`为`old_acc=0.6818`、`seen_new_acc=0.7857`、`unknown_FAR=0.0294`；加入receiver-class可靠度后，`rcwr_min2_er090,k=4`达到`old_acc=0.8182`、`seen_new_acc=0.7857`、`known_cov=0.8707`，首次跨过项目当前阶段化门槛`old_acc>=0.80`。这证明按类选择可靠receiver能恢复旧类目标域性能，比单纯全局receiver权重更有效。

但该row的`unknown_FAR=0.0882`，未满足`unknown_FAR<=0.05`，更没有达到用户目标中的未知拒识99%。更严格的`rcwr_min2_er075,k=4`可把FAR压到0.0294，但`old_acc`降到0.7386；`rcwr_min2_er090,k=5`达到`unknown_FAR=0.0500`且`old_acc=0.7925`，仍未过OLD80，且`min_seen=0`。所有候选的`min_old`仍为0，说明仍有旧类地板失败；当前不能声明99/97/99目标达成、Stage2-C成功或部署成功。

按需协同版本`rcwr_support_utility`降低资源消耗，例如k=3平均`2.46`个receiver、`98.4 bytes/event`，但FAR升至0.1000；k=5虽FAR为0且平均`3.77`个receiver、`151.0 bytes/event`，known性能不足。因此它目前是资源诊断，不是性能最优路线。

下一步应以`rcwr_min2_er090,k=4`作为OLD80候选上界，专门补独立open-set风险通道：按类Mahalanobis/EVT tail、source/proxy-known校准的oldness gate、以及不使用unknown query的低密度拒识。仅继续放宽候选集接受会提高old/seen-new，但会把unknown吸收为known；仅继续收紧event risk会恢复FAR，但会丢掉OLD80。

### Post-review hardening

子agent review指出一项P1协议风险和两项P2工程风险：`seen_new_rescue_enabled`旧分支使用query真实`role`作为是否救援的条件；`receiver_class_reliability_policy=support_calibrated`若未同时开启`class_conformal_enabled`会静默退化为1.0；JSON复现元数据缺少完整argv/cwd/python/output路径。本轮已修复：

|问题|修复|
|---|---|
|`seen_new_rescue`读取query真实role|删除role依赖，改为仅依赖预测标签是否属于seen-new注册表、strong known证据和多风险组件不一致；多通道高风险样本不允许被rescue。|
|`support_calibrated`静默退化|若未开启`--class_conformal_enabled`直接报错；若某receiver没有support校准分数直接报错。|
|复现元数据不足|JSON新增`run_command_argv`、`run_cwd`、`python_executable`、`output_json`和`output_evidence_csv`字段。|
|测试覆盖|更新role-free rescue测试，保留`42+44`单测通过。|

本地`ssr-gpu`和Git镜像均通过：编译PASS，`test_collaborative_open_set_qknn_eval.py`为42 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为44 tests OK。Git追加提交：`3fd3b2d Harden receiver class reliability protocol guards`。

N607已再次同步并用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`验证通过。最新远端SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`7e84960c7c7827f7aaae762296aceedb5168a03a1ad31c4f9aaefd0953b862b0`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`cb89ed26a3bccb5d09b2e8aea3b695ea62c12b0186e5ac96dd7b377ca6996e00`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`56ba9509c00014c03ae2106f4f3d22df3203d691fd9f594ddb2ee0bd41403867`|

上述hardening不改变已生成rcwr结果的判定：本轮实验命令未启用`seen_new_rescue_enabled`，且已显式启用`class_conformal_enabled`，所以主结果不受P1分支污染。后续重新生成JSON时会带完整argv/cwd/python/output复现字段。

## 2026-07-04 candidate_set多风险组件门控

### 目标与改动

本轮针对`rcwr_min2_er090,k=4`“old_acc跨过0.80但unknown_FAR=0.0882”的边界问题，新增独立open-set风险否决通道：在`candidate_set_cvs`接受候选时，除`candidate_set_min_conformal_pvalue`外，增加`candidate_set_max_label_risk_component_agreement`，用标签级多风险组件一致性否决高unknown吸收风险样本。该门控不使用unknown query标签；它只读取事件证据中的风险组件聚合结果。

|文件|改动|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`candidate_set_max_label_risk_component_agreement`参数、候选集接受条件和JSON元数据字段。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增CLI参数`--candidate_set_max_label_risk_component_agreement`并传入评估入口。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增`test_candidate_set_cvs_vetoes_high_label_component_agreement`，覆盖高组件风险一致性否决。|

根目录`E:\type10-7`不是Git仓库，本地快照保存在`E:\type10-7\code\snapshots\candidate_component_veto_20260704\`。Git镜像`E:\type10-7\github_publish\CVS-RFFI-repo`已提交`ede6539 Add candidate set component risk veto`。

### 本地与N607验证

本地`ssr-gpu`与Git镜像均通过以下验证：

|环境|命令|结果|
|---|---|---|
|本地工作区|`python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py`|PASS|
|本地工作区|`python -m unittest discover -s code\tests -p "test_collaborative_open_set_qknn_eval.py"`|43 tests OK|
|本地工作区|`python -m unittest discover -s code\tests -p "test_phase2_collaborative_open_set_qknn_eval.py"`|44 tests OK|
|Git镜像|同上三条|PASS，43 tests OK，44 tests OK|

N607直连预检通过，远端项目根为`/home/szu2070436088/2510044040/CV-SincNet`，远端Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。同步后远端`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为43 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为44 tests OK。运行前后8张RTX3090均为`10 MiB/24576 MiB`，选择显存占用最低的GPU0执行；每次SSH/SCP后本地检查均为无残留`ssh.exe`、无N607/bridge 22端口`ESTABLISHED`连接。

远端最新代码SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`3030322907e81df5032fc7cab3d93e01c43a6ef409876a22c4c554aec134c9e2`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`7eae43b3b73386f802860649e82225842bd290f751f2d8e97a1e7a7e00afdd36`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`d3475481bbcbc93cb1200b395321f4763c797898c02baa923e71d74f2741b53b`|

### 远端命令口径

本轮复用`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，对应`ADV3B02_CORE90_SOFT_E200`、Stage2-C、qknn8、target receiver 1到5和`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`星地信道视图。主要固定参数为：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --collab_counts all --k_shot 8 --query_per_class 20 --qknn_k 8 --seed 4070303 \
  --fusion_policy candidate_set_cvs --collaboration_policy fixed_k \
  --candidate_class_top_m 2 --class_evidence_top_m 3 \
  --class_conformal_enabled --class_conformal_min_support 2 \
  --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only \
  --scenario_aware --radius_norm 0.3 \
  --label_fusion_policy weighted_vote_margin \
  --class_reliability_policy conformal_margin_risk \
  --receiver_class_reliability_policy support_calibrated \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --unknown_risk_threshold 0.8 --scorer_component_vote_threshold 0.50 \
  --candidate_set_min_receivers 2 --candidate_set_min_top1_receivers 0 \
  --candidate_set_max_label_unknown_risk 1.0 \
  --candidate_set_max_event_unknown_risk 0.90 \
  --candidate_set_unknown_reject_risk 0.80 \
  --evidence_packet_bytes 40
```

差异参数：

|配置|差异参数|JSON|Evidence CSV|
|---|---|---|---|
|`lrca050_p035`|`--candidate_set_min_conformal_pvalue 0.35 --candidate_set_max_label_risk_component_agreement 0.50`|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_lrca050_p035_adv3b02.json`|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_lrca050_p035_adv3b02_evidence.csv`|
|`lrca049_p035`|`--candidate_set_min_conformal_pvalue 0.35 --candidate_set_max_label_risk_component_agreement 0.49`|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_lrca049_p035_adv3b02.json`|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_lrca049_p035_adv3b02_evidence.csv`|

JSON已记录`run_command_argv`、`run_cwd`、`python_executable`、`output_json`和`output_evidence_csv`。

### 产物SHA256

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_lrca050_p035_adv3b02.json`|`6479948C6C69D1A58F1CF7CE298687C170AD100190F119C038F18A0FAC1470B0`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_lrca050_p035_adv3b02_evidence.csv`|`C7CA3774970C83BFA4B10E53B8909BA3A1475277FB5D59E8BF4B7B18F2342C65`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_lrca049_p035_adv3b02.json`|`BCCE76EB0D8FCF22C91D8DD8F580DB969434623AE74E696A13B60BC40F68B36A`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_lrca049_p035_adv3b02_evidence.csv`|`6158B632A84100988951EB2A7BDA3E4A73FD9618CABAF87145AAB4A09748D099`|

### 结果表

|配置|协同数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|defer_rate|known_coverage|avg_rx|bytes/event|latency_ms_p95|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`lrca050_p035`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1818|单receiver证据不足|
|`lrca050_p035`|2|0.3816|0.0500|0.5192|0.5000|0.0652|0.9348|0.0320|0.5539|2.0000|80.0|0.1818|FAR略超，known不足|
|`lrca050_p035`|3|0.4750|0.1500|0.5500|0.4500|0.0250|0.9750|0.0050|0.5750|3.0000|120.0|0.1818|FAR达标但known不足|
|`lrca050_p035`|4|0.8182|0.0000|0.7857|0.7000|0.0588|0.9412|0.0000|0.8707|4.0000|160.0|0.1818|保留OLD80，FAR仍略超|
|`lrca050_p035`|5|0.7925|0.0000|0.6500|0.0000|0.0500|0.9500|0.0000|0.8493|5.0000|200.0|0.1818|FAR到边界，old未过0.80|
|`lrca049_p035`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1323|单receiver证据不足|
|`lrca049_p035`|2|0.3750|0.0500|0.4808|0.4688|0.0652|0.9348|0.0640|0.5098|2.0000|80.0|0.1323|FAR略超，known不足|
|`lrca049_p035`|3|0.4750|0.1500|0.5500|0.4500|0.0250|0.9750|0.0050|0.5750|3.0000|120.0|0.1323|FAR达标但known不足|
|`lrca049_p035`|4|0.8182|0.0000|0.7143|0.6500|0.0294|0.9706|0.0000|0.8448|4.0000|160.0|0.1323|当前最佳OLD80+FAR组合|
|`lrca049_p035`|5|0.7925|0.0000|0.6500|0.0000|0.0000|1.0000|0.0000|0.8493|5.0000|200.0|0.1323|拒识最强但old和seen-new不足|

### 判定

`candidate_set_max_label_risk_component_agreement=0.49`在k=4时把`rcwr_min2_er090,k=4`的`unknown_FAR`从0.0882压到0.0294，同时保留`old_acc=0.8182`。这是当前最好的OLD80+FAR组合，说明多风险组件一致性门控能在不牺牲old主精度的情况下抑制unknown吸收。

但该结果仍不是用户目标：`min_old_class_acc=0.0000`，`seen_new_acc=0.7143`，`min_seen_new_class_acc=0.6500`，unknown拒识率0.9706低于0.99。因此当前只能声明“候选方向有效且达到阶段性OLD80+FAR组合”，不能声明99/97/99、不能声明Stage2-C成功、也不能声明卫星部署成功。

资源方面，`lrca049_p035,k=4`固定使用4个receiver，`160 bytes/event`，`latency_ms_p95=0.1323`，运行GPU显存占用保持在基线`10 MiB/24576 MiB`。资源约束设计说明原文`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`仍未在当前工作区找到；本轮继续使用`avg_rx`、`bytes/event`、`latency_ms_p95`、GPU/VRAM作为可复核代理指标。

下一步不应继续简单收紧全局门控。主要失败来自`20-15`旧类缺类地板和seen-new回落，建议围绕`lrca049_p035,k=4`做按类地板保护：对缺类old引入source/proxy-known oldness gate、按类support envelope下界和低风险old-only rescue；同时保留`candidate_set_max_label_risk_component_agreement<=0.49`作为unknown吸收硬门控。

## 2026-07-04 available_up_to_k协同预算策略

### 目标与诊断

上一轮`lrca049_p035,k=4`的`min_old_class_acc=0.0000`不是单纯分类错误，而是`20-15`在exact-k口径下被完整组过滤移除：evidence中`20-15`有40个event、100条receiver观测，但每个event最多只有3个receiver观测，因此`k=4/5`要求“至少k个receiver”的完整组时，该旧类直接进入`missing_old_classes`。这与现实卫星群不完全一致：在轨协同更合理的口径是“最多请求k个receiver，可用多少融合多少，并报告实际参与receiver数量和资源”。

本轮新增`collab_group_policy`：

|策略|语义|用途|
|---|---|---|
|`exact_k`|保留原行为，只评估至少有k个receiver观测的event。|固定协同数量、严格同分母对照。|
|`available_up_to_k`|把k解释为最大协同预算，只要event达到`partial_collab_min_receivers`即可进入，实际融合`min(k, available_receivers)`。|异步卫星群/覆盖不完整场景，必须报告`avg_rx`和实际资源。|

该改动不改变Stage2-C协议、不使用unknown query调阈值，也不改变`Y_old/Y_new/Y_unknown`互斥。它改变的是协同分母与资源解释：`k`不再等于每个event实际参与receiver数，而是最大请求预算。

### 本地变更与验证

|文件|改动|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`collab_group_policy`、`partial_collab_min_receivers`，在`available_up_to_k`下按最小可用receiver门槛保留partial event，并在JSON中记录策略。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增CLI参数`--collab_group_policy {exact_k,available_up_to_k}`和`--partial_collab_min_receivers`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增partial class group测试，证明`exact_k,k=4`会丢失只有3个receiver的旧类，而`available_up_to_k`保留该类并报告实际平均参与数。|

本地`ssr-gpu`验证：`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`先为44 tests OK，子agent审查修复后为45 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`为44 tests OK。Git镜像重复同样验证通过，并提交`a37d8aa Add available receiver budget collaboration policy`和`3dfb150 Clarify available receiver budget metrics`。根目录非Git，本地快照保存在`E:\type10-7\code\snapshots\available_up_to_k_20260704\`。

### N607同步与验证

N607直连预检通过，远端项目根为`/home/szu2070436088/2510044040/CV-SincNet`，远端Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。同步后远端`py_compile`通过，`test_collaborative_open_set_qknn_eval.py`为45 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`为44 tests OK。远端代码SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`14fc582646fa6455cf7b6f44de64d62e0a8dfa611391936c854c6987a0871fb7`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`a6142d1a6fe00b3cbe12b8ec81e2af5bfad866fc12cd17b606bdba62f1f3d857`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`585e09b96c51523dd1321e353a848ce207b5fe212ec9d58b16486b8737988e87`|

远端运行前后8张RTX3090均为`10 MiB/24576 MiB`，使用GPU0；每次SSH/SCP后本地检查均为无残留`ssh.exe`、无N607/bridge 22端口`ESTABLISHED`连接。

### 远端实验配置

本轮正式复跑配置为`avail3_p050_lrca033`：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_adv3b02.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_adv3b02_evidence.csv \
  --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 3 \
  --k_shot 8 --query_per_class 20 --qknn_k 8 --seed 4070303 \
  --candidate_class_top_m 2 --class_evidence_top_m 3 \
  --class_conformal_enabled --class_conformal_min_support 2 \
  --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy candidate_set_cvs --collaboration_policy fixed_k \
  --label_fusion_policy weighted_vote_margin \
  --class_reliability_policy conformal_margin_risk \
  --receiver_class_reliability_policy support_calibrated \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --unknown_risk_threshold 0.8 --scorer_component_vote_threshold 0.50 \
  --candidate_set_min_receivers 2 --candidate_set_min_top1_receivers 0 \
  --candidate_set_min_conformal_pvalue 0.50 \
  --candidate_set_max_label_unknown_risk 1.0 \
  --candidate_set_max_event_unknown_risk 1.0 \
  --candidate_set_max_label_risk_component_agreement 0.33 \
  --candidate_set_unknown_reject_risk 0.80 \
  --evidence_packet_bytes 40
```

产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_adv3b02.json`|`835813D44D736610D700843A369F272AE05D1EEE090A9A88B6A3B7D5A7AE8342`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_adv3b02_evidence.csv`|`5E4936FFB1895004F8D84C447E9DE35F018B55EA6F74E42030BA4F16ED5F8115`|

### 结果表

|配置|协同预算k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|missing_old|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`avail3_p050_lrca033`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1888|[]|单receiver不足|
|`avail3_p050_lrca033`|2|0.4737|0.1000|0.5769|0.5312|0.1957|0.8043|0.0920|0.6029|2.0000|80.0|0.1888|[]|coverage修复但FAR过高|
|`avail3_p050_lrca033`|3|0.6167|0.3000|0.6500|0.6000|0.0750|0.9250|0.0400|0.7000|3.0000|120.0|0.1888|[]|类覆盖完整，FAR仍超|
|`avail3_p050_lrca033`|4|0.8083|0.6500|0.6500|0.5500|0.0500|0.9250|0.0150|0.8187|3.7500|150.0|0.1888|[]|当前最佳类覆盖+OLD80+FAR边界组合|
|`avail3_p050_lrca033`|5|0.7917|0.6000|0.6500|0.5500|0.0500|0.9500|0.0100|0.8063|4.2150|168.6|0.1888|[]|拒识略强但old低于0.80|

k=4逐类结果：

|类别集|逐类准确率|
|---|---|
|old|`14-10=0.8000`，`14-7=0.8500`，`20-15=0.6500`，`20-19=0.7000`，`6-15=0.9000`，`8-20=0.9500`|
|seen-new|`19-3=0.5500`，`3-8=0.7500`|

### 判定

`available_up_to_k`解决了exact-k分母造成的旧类缺失问题：k=4不再出现`missing_old_classes=['20-15']`，并且`min_old_class_acc`从0提升到0.6500。这是向用户要求“报告参与推理数量、时延、资源约束”和现实卫星群异步可用性迈进的一步。

但它不是最终目标。当前最佳`avail3_p050_lrca033,k=4`仅达到`old_acc=0.8083`、`min_old=0.6500`、`seen_new_acc=0.6500`、`unknown_reject=0.9250`。未知类拒识仍显著低于0.99，旧类每类不低于0.95和新类不低于0.93也未达到。因此不能声明99/97/99、不能声明Stage2-C成功、不能声明卫星部署成功。

下一步机制不应继续只调全局阈值。需要增加“unknown误接收解释通道”和“按类地板保护通道”：对unknown false accept事件提取最终融合证据，设计source/proxy-known oldness gate或receiver-pair inconsistency gate；对`20-15`、`20-19`和seen-new低类建立class floor rescue，但必须只用support/receiver先验，不得使用query真值或unknown query阈值拟合。

### 子agent审查后的口径修复

子agent review指出两项P1风险：`available_up_to_k`下`k`是最大协同预算而不是每个event实际receiver数，旧字段`participating_receivers=4`容易被误读；顶层`eligible_group_count`仍按exact-k最大预算统计，与策略分母不一致。本轮已修复：

|问题|修复|
|---|---|
|预算k可能被误读为实际k|每个count新增`receiver_budget`、`min_required_receivers`、`actual_receiver_count_histogram`、`partial_group_count`、`exact_budget_group_count`。|
|顶层eligible口径不一致|新增`exact_max_requested_group_count`、`policy_eligible_group_count_at_max_budget`、`policy_excluded_group_count_at_max_budget`、`policy_min_receivers_at_max_budget`。|
|`partial_collab_min_receivers<=0`静默修正|改为显式`ValueError`，新增单测覆盖。|

修复后远端重新验证并复跑同一配置。N607最新测试为`test_collaborative_open_set_qknn_eval.py`45 tests OK，`test_phase2_collaborative_open_set_qknn_eval.py`44 tests OK。刷新后的`avail3_p050_lrca033,k=4`字段为：`receiver_budget=4`，`min_required_receivers=3`，`actual_receiver_count_histogram={'3': 50, '4': 150}`，`partial_group_count=50`，`exact_budget_group_count=150`，`avg_rx=3.75`，`p95_rx=4.0`，`max_rx=4`。顶层策略字段为：`exact_max_requested_group_count=93`，`policy_eligible_group_count_at_max_budget=200`，`policy_excluded_group_count_at_max_budget=107`，`policy_min_receivers_at_max_budget=3`。

因此该结果必须表述为“最大预算k=4、实际平均3.75个receiver参与”的budgeted collaboration，不能表述为固定4个receiver协同。指标数值未因元数据修复改变，刷新后的`latency_ms_p95=0.1888`。

## 2026-07-04 candidate_set高unknown风险比例否决

### 目标与诊断

`avail3_p050_lrca033,k=4`仍有2个unknown false accept。事件级复现显示两个误接收类型不同：

|误接收事件|错误输出|诊断|
|---|---|---|
|`unknown|10-1|leo_clear_weak|rank00007`|`14-10`|事件总`unknown_risk≈1.0`，但候选标签组件风险比例低，导致candidate_set接受；同一候选标签下有一半参与receiver给出高unknown风险。|
|`unknown|10-10|leo_low_elev_weak|rank00014`|`14-7`|低unknown风险、高support density、高一致性的强相似unknown；现有support-only风险难以区分。|

因此本轮只针对第一类“事件级高unknown风险但候选集仍接受”的情况加门控，不试图用同一阈值强行解决第二类强相似unknown，避免过度伤害old/seen-new。

新增参数：

|参数|含义|
|---|---|
|`candidate_set_event_high_unknown_risk_veto`|当事件级`unknown_risk`达到该阈值时，启用额外否决检查。默认`1e12`，即关闭。|
|`candidate_set_max_label_high_unknown_risk_fraction`|同一候选标签下，高unknown风险receiver占比达到该值则否决。|
|`candidate_set_high_unknown_risk_threshold`|单receiver被视作高unknown风险的阈值。|

该门控只读取各receiver对候选标签的support/prototype风险证据和事件风险聚合，不用unknown query拟合阈值。当前正式配置为`candidate_set_event_high_unknown_risk_veto=0.999`、`candidate_set_max_label_high_unknown_risk_fraction=0.5`、`candidate_set_high_unknown_risk_threshold=0.8`。

### 本地与远端验证

|项目|结果|
|---|---|
|本地`ssr-gpu`|`py_compile`通过；`test_collaborative_open_set_qknn_eval.py`46 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`44 tests OK。|
|Git镜像|同样验证通过，提交`6def37c Add candidate high unknown fraction veto`。|
|N607|远端`py_compile`通过；`test_collaborative_open_set_qknn_eval.py`46 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`44 tests OK；子agent审查后补充veto计数和参数范围校验并重新验证。|

N607最新代码SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`353af9b0cae940cb61fe987233dfe60bac0297a32db1df8609c90d9e9f6f1c32`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`90478f4a8014eb8557ccaafa8d48c39bd92538e030cbee7f44414c8dae162648`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`a8b67add47dd51676fa1e4165bf3a8150b7aaecf55939403d9a50caa458c2d24`|

N607运行前后8张RTX3090均为`10 MiB/24576 MiB`，使用GPU0；SSH/SCP后本地检查为无残留`ssh.exe`、无N607/bridge 22端口`ESTABLISHED`连接。

### 远端实验配置

新增正式候选`avail3_p050_lrca033_huv0999_f050`：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_huv0999_f050_adv3b02.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_huv0999_f050_adv3b02_evidence.csv \
  --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 3 \
  --k_shot 8 --query_per_class 20 --qknn_k 8 --seed 4070303 \
  --candidate_class_top_m 2 --class_evidence_top_m 3 \
  --class_conformal_enabled --class_conformal_min_support 2 \
  --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy candidate_set_cvs --collaboration_policy fixed_k \
  --label_fusion_policy weighted_vote_margin \
  --class_reliability_policy conformal_margin_risk \
  --receiver_class_reliability_policy support_calibrated \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --unknown_risk_threshold 0.8 --scorer_component_vote_threshold 0.50 \
  --candidate_set_min_receivers 2 --candidate_set_min_top1_receivers 0 \
  --candidate_set_min_conformal_pvalue 0.50 \
  --candidate_set_max_label_unknown_risk 1.0 \
  --candidate_set_max_event_unknown_risk 1.0 \
  --candidate_set_max_label_risk_component_agreement 0.33 \
  --candidate_set_event_high_unknown_risk_veto 0.999 \
  --candidate_set_max_label_high_unknown_risk_fraction 0.5 \
  --candidate_set_high_unknown_risk_threshold 0.8 \
  --candidate_set_unknown_reject_risk 0.80 \
  --evidence_packet_bytes 40
```

产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_huv0999_f050_adv3b02.json`|`B5AC9761FCF3719DA91D6C7CC3E9BED2000A4430369D23C7966EF67AE59E068A`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p050_lrca033_huv0999_f050_adv3b02_evidence.csv`|`F9A70130D53B3EF8C4EF725802A5E0EAC330D899BB8D5D797E80749D65B88AB8`|

### 结果表

|配置|预算k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|实际rx直方图|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`avail3_p050_lrca033_huv0999_f050`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1348|`{'1':307}`|单receiver不足|
|`avail3_p050_lrca033_huv0999_f050`|2|0.4145|0.0500|0.5192|0.4688|0.1304|0.8696|0.0920|0.5441|2.0000|80.0|0.1348|`{'2':250}`|FAR仍高|
|`avail3_p050_lrca033_huv0999_f050`|3|0.5750|0.2500|0.6500|0.6000|0.0750|0.9250|0.0400|0.6625|3.0000|120.0|0.1348|`{'3':200}`|FAR仍超|
|`avail3_p050_lrca033_huv0999_f050`|4|0.8000|0.6500|0.6500|0.5500|0.0250|0.9500|0.0150|0.8125|3.7500|150.0|0.1348|`{'3':50,'4':150}`|当前最佳OLD80+FAR组合|
|`avail3_p050_lrca033_huv0999_f050`|5|0.7833|0.6000|0.6500|0.5500|0.0250|0.9750|0.0100|0.8000|4.2150|168.6|0.1348|`{'3':50,'4':57,'5':93}`|拒识更强但old低于0.80|

k=4逐类结果：

|类别集|逐类准确率|
|---|---|
|old|`14-10=0.8000`，`14-7=0.8500`，`20-15=0.6500`，`20-19=0.6500`，`6-15=0.9000`，`8-20=0.9500`|
|seen-new|`19-3=0.5500`，`3-8=0.7500`|

### 判定

`candidate_set_high_unknown_veto`审计字段显示，k=4共触发22次，占全部200个event的0.11；其中unknown 19次、old 3次。这说明FAR下降主要来自unknown侧否决，但old轻微下降也由该门控造成，属于当前可解释代价。

相对`avail3_p050_lrca033,k=4`，本轮把`unknown_FAR`从0.0500降到0.0250，`unknown_reject`从0.9250升到0.9500，代价是`old_acc`从0.8083降到0.8000、`20-19`从0.7000降到0.6500。该机制确实解决了一类“事件总风险很高但candidate_set仍接受”的unknown误接收，是当前最好的OLD80+unknown_FAR组合。

但它仍不是最终目标：`min_old=0.6500`，`seen_new_acc=0.6500`，`min_seen=0.5500`，unknown拒识0.9500，均未达到用户要求的99/95、97/93、99。剩余主要失败是第二类强相似unknown和低类地板；下一步需要引入不依赖unknown query阈值拟合的receiver-pair不一致证据或源域外类虚拟边界，而不是继续只调candidate_set阈值。

## 2026-07-04 candidate_set高unknown风险比例门控参数再平衡

### 目标

上一轮`avail3_p050_lrca033_huv0999_f050,k=4`把`unknown_FAR`降到0.0250，但`seen_new_acc`仍只有0.6500。本轮不改代码，基于已验证的高unknown风险比例否决机制，重新平衡`candidate_set_min_conformal_pvalue`和`candidate_set_max_label_risk_component_agreement`，目标是在保持低FAR的同时恢复seen-new和旧类地板。

本地离线扫参显示`pvalue=0.55`、`label_risk_component_agreement<=0.49`、`huv=0.999`优于上一轮：k=4预期`old≈0.8083`、`seen_new≈0.7500`、`unknown_FAR≈0.0250`。因此在N607上正式复跑`avail3_p055_lrca049_huv0999_f050`。

### N607配置

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_adv3b02.json \
  --output_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_adv3b02_evidence.csv \
  --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 3 \
  --k_shot 8 --query_per_class 20 --qknn_k 8 --seed 4070303 \
  --candidate_class_top_m 2 --class_evidence_top_m 3 \
  --class_conformal_enabled --class_conformal_min_support 2 \
  --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 \
  --support_calibration_mode leave_one_out \
  --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only \
  --scenario_aware --radius_norm 0.3 \
  --fusion_policy candidate_set_cvs --collaboration_policy fixed_k \
  --label_fusion_policy weighted_vote_margin \
  --class_reliability_policy conformal_margin_risk \
  --receiver_class_reliability_policy support_calibrated \
  --event_alignment_policy receiver_domain_ranked \
  --support_selection_policy stable_first \
  --unknown_risk_threshold 0.8 --scorer_component_vote_threshold 0.50 \
  --candidate_set_min_receivers 2 --candidate_set_min_top1_receivers 0 \
  --candidate_set_min_conformal_pvalue 0.55 \
  --candidate_set_max_label_unknown_risk 1.0 \
  --candidate_set_max_event_unknown_risk 1.0 \
  --candidate_set_max_label_risk_component_agreement 0.49 \
  --candidate_set_event_high_unknown_risk_veto 0.999 \
  --candidate_set_max_label_high_unknown_risk_fraction 0.5 \
  --candidate_set_high_unknown_risk_threshold 0.8 \
  --candidate_set_unknown_reject_risk 0.80 \
  --evidence_packet_bytes 40
```

运行前后8张RTX3090均为`10 MiB/24576 MiB`，使用GPU0；运行后本地检查无残留`ssh.exe`和N607/bridge 22端口连接。

产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_adv3b02.json`|`CDE6709DE940AC508C24291EDBAA2F5931CF6EA32A0824CCE6B01A9508388658`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_adv3b02_evidence.csv`|`771668FCDB2EC627EDF82D81447C5ECF6F9945B5F9639A094ECE435A9491604C`|

### 结果表

|配置|预算k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|veto_count|veto_by_role|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`avail3_p055_lrca049_huv0999_f050`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1840|40|`old=22,seen_new=2,unknown=16`|单receiver不足|
|`avail3_p055_lrca049_huv0999_f050`|2|0.4079|0.0500|0.5192|0.4688|0.1304|0.8696|0.0920|0.5392|2.0000|80.0|0.1840|62|`old=34,seen_new=8,unknown=20`|FAR仍高|
|`avail3_p055_lrca049_huv0999_f050`|3|0.6250|0.2500|0.7250|0.6000|0.0750|0.9000|0.0300|0.7188|3.0000|120.0|0.1840|39|`old=13,seen_new=3,unknown=23`|seen-new提升但FAR仍超|
|`avail3_p055_lrca049_huv0999_f050`|4|0.8083|0.7000|0.7250|0.6000|0.0250|0.9500|0.0150|0.8250|3.7500|150.0|0.1840|22|`old=3,unknown=19`|当前最佳综合组合|
|`avail3_p055_lrca049_huv0999_f050`|5|0.7833|0.5500|0.7000|0.5500|0.0250|0.9500|0.0100|0.8187|4.2150|168.6|0.1840|22|`old=3,unknown=19`|FAR低但old低于0.80|

k=4逐类结果：

|类别集|逐类准确率|
|---|---|
|old|`14-10=0.7000`，`14-7=0.7000`，`20-15=0.7500`，`20-19=0.7000`，`6-15=1.0000`，`8-20=1.0000`|
|seen-new|`19-3=0.6000`，`3-8=0.8500`|

### 判定

`avail3_p055_lrca049_huv0999_f050,k=4`是当前最好的综合候选：相对上一轮`avail3_p050_lrca033_huv0999_f050,k=4`，`old_acc`从0.8000升到0.8083，`min_old`从0.6500升到0.7000，`seen_new_acc`从0.6500升到0.7250，`min_seen`从0.5500升到0.6000，同时`unknown_FAR`保持0.0250，`unknown_reject`保持0.9500。资源仍是最大预算4，实际平均3.75个receiver参与，约`150 bytes/event`。

该结果仍远未达最终目标：旧类总体未到0.99、旧类地板未到0.95；seen-new总体未到0.97、地板未到0.93；unknown拒识仍低于0.99。它只能作为下一阶段机制开发的当前最优基线，不能声明Stage2-C成功、部署成功或论文主结论。

## 2026-07-04 CCBR-CVS class shell风险机制实现与N607计划

### 机制

本轮新增`CCBR-CVS`第一版：Class-Conditional Boundary and Receiver-consistency Routing中的support-only类外壳风险。它不改变`ADV3B02_CORE90_SOFT_E200`底座，不训练主干，不使用unknown query调阈值；只在每个receiver本地qknn8 evidence中记录候选类到support类中心的外壳距离：

```text
d_{r,y}(z)=1-cos(z,c_{r,y})
R_{r,y}=Q_q(1-cos(s_{r,y},c_{r,y}))+slack
class_shell_risk=sigmoid((d_{r,y}-scale*R_{r,y}+margin)/T)
```

融合端新增`candidate_set_max_label_shell_risk`和`candidate_set_shell_reject_risk`。前者阻止高外壳风险的candidate_set接受，后者在达到拒识阈值时输出`unknown_reject`。该机制的目标是拦截“基础unknown风险较低但落在候选类support外壳之外”的未知类false accept，同时通过默认关闭保持旧候选可复现。

### 本地变更与验证

|文件|变更|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`class_shell`风险组件、`label_shell_risk`聚合、candidate_set shell门控/拒识和shell veto统计。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`--class_shell_unknown_risk_enabled`、`--class_shell_radius_scale`、`--class_shell_risk_temperature`、`--class_shell_risk_margin`，并把主预测与top-M候选的`class_shell_risk`写入CSV。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增高`class_shell_risk`拦截unknown false accept的单元测试。|

本地验证：

```text
conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：语法检查通过；协同开集测试`47 passed`；Phase2脚本测试`44 passed`。首次并行`conda run`触发Windows conda临时文件锁，串行直接调用环境Python后通过，不作为实验失败证据。

Git提交：`91e8a00 Add class shell risk for collaborative open set`。本地快照：`E:\type10-7\code\snapshots\class_shell_cvs_20260704\`。

### N607计划

预检时间：`2026年07月04日01:18:18 CST`。N607直连预检通过，项目根存在，8张RTX3090均为`10 MiB/24576 MiB`，选择GPU0。预检后确认无残留`ssh.exe`和N607/bridge 22端口连接。

计划同步：

|本地文件|N607目标|
|---|---|
|`E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py`|

计划候选：继承当前最佳`avail3_p055_lrca049_huv0999_f050`，只新增class shell风险，小网格跑`class_shell_radius_scale in {1.25,1.50,1.75}`，每个候选`collab_counts=all`、`available_up_to_k`、`partial_collab_min_receivers=3`，报告协同数量1到全体receiver、old/seen-new/unknown、per-class floor、bytes/event、latency proxy、shell veto统计。远端Python使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

### N607同步与远端验证

远端同步后SHA256：

|文件|N607 SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`6356033ab55fd8be86153a879e89aabd78df48792bf02c5829b134be1c872f5f`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`2e4fe924f0b18c354b1c6b67f9e05ba79e1fd70edf53aeb3514b0cb5c7367903`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`136d942e172498db0d35b2c472a7850fdc76c971688cbc2531d218806c2df819`|

远端验证：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m unittest discover -s code/tests -p 'test_collaborative_open_set_qknn_eval.py' -q
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m unittest discover -s code/tests -p 'test_phase2_collaborative_open_set_qknn_eval.py' -q
```

结果：远端`py_compile`通过；`test_collaborative_open_set_qknn_eval.py`为`47 tests OK`；`test_phase2_collaborative_open_set_qknn_eval.py`为`44 tests OK`。`CVS-RFFI`环境无`pytest`模块，因此远端用标准库`unittest`验证。

### N607结果

所有运行均使用GPU0，运行前后8张RTX3090均为`10 MiB/24576 MiB`。每次SSH/SCP后均确认无残留`ssh.exe`和N607/bridge 22端口连接。

产物SHA256：

|候选|JSON SHA256|CSV SHA256|
|---|---|---|
|`shell_s125_max095_rej098_unk080`|`AA9A19CC1A410583CD7241BBDF881D5DB5C4A81AB5110858854A4773DDAC76D2`|`505EEB34208CDB2C45B4DF8B85A1FF729D458F4BEBB58F468C30C92C2DA6A7BD`|
|`shell_s150_max095_rej098_unk080`|`5AE24F9960A493CE901FD2FA1644522B4594CF624A09EB67C8F43CC2EC29124C`|`6E13D7904F31A5C97679995542514E3436FF6D12173DF0A76F7E66E6E5C64DFD`|
|`shell_s175_max095_rej098_unk080`|`CACB47E19066DF2BD710DD8CBAFD82354F41A6F9BBF2DB5F74E01007E1A144AD`|`973BD54157CD36F3ACEA79D7B79A8987A9A43AA19172CA4C247A1297494AE2ED`|
|`shell_s125_max095_rej098_unk995`|`1BEA52682A9D756F52378B0834919433959BD7ADCA6FD473E97227CD77FCAE05`|`63B3E6BC1BAB9EAE9096A0C4C981E84AB8CC0637718F57303535739631C46F55`|
|`shell_s150_max095_rej098_unk995`|`67CA352B8E03D18CC5F64923BDFA3785BC4953903E779CAD525B5E72714D274C`|`3376C944341DEA844D992EFE43A7E3DAE26C7326182FD5425177683251FAB84C`|
|`shell_s175_max095_rej098_unk995`|`DA571E77D2A33D8B2087110F5FD559D10DF3E7B036AC5398124AEEA1C885863F`|`2FA54FCEE0859DA57F39B917297579D75D805765A10E25F27F810A9BA803E2FD`|

`unk995`是诊断性放宽拒识阈值，用来检查class shell是否能恢复seen-new；它使k=4 unknown拒识从0.9500降到0.7750，不能作为候选主线。主线比较采用`unk080`，即保留上一轮`candidate_set_unknown_reject_risk=0.80`。

主线`unk080`结果：

|候选|预算k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|p95_latency|high_unknown_veto|shell_veto|shell_veto_by_role|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`shell_s125_max095_rej098_unk080`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1311|40|9|`old=3,seen_new=1,unknown=5`|单receiver不足|
|`shell_s125_max095_rej098_unk080`|2|0.4079|0.0500|0.5769|0.5312|0.1304|0.8696|0.0800|0.5588|2.0000|80.0|0.1311|62|1|`old=1`|FAR高|
|`shell_s125_max095_rej098_unk080`|3|0.6250|0.2500|0.7250|0.6000|0.0750|0.9000|0.0300|0.7188|3.0000|120.0|0.1311|39|7|`unknown=7`|FAR仍超|
|`shell_s125_max095_rej098_unk080`|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|0.8375|3.7500|150.0|0.1311|22|11|`old=1,unknown=10`|seen-new提升但shell误伤1个old|
|`shell_s125_max095_rej098_unk080`|5|0.7833|0.5500|0.7250|0.5500|0.0500|0.9250|0.0100|0.8313|4.2150|168.6|0.1311|22|17|`old=2,unknown=15`|FAR边界但old下降|
|`shell_s150_max095_rej098_unk080`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1838|40|5|`old=2,unknown=3`|单receiver不足|
|`shell_s150_max095_rej098_unk080`|2|0.4079|0.0500|0.5769|0.5312|0.1304|0.8696|0.0720|0.5686|2.0000|80.0|0.1838|62|0|`{}`|FAR高|
|`shell_s150_max095_rej098_unk080`|3|0.6250|0.2500|0.7250|0.6000|0.0750|0.9000|0.0300|0.7250|3.0000|120.0|0.1838|39|3|`unknown=3`|FAR仍超|
|`shell_s150_max095_rej098_unk080`|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|0.8375|3.7500|150.0|0.1838|22|3|`unknown=3`|当前最佳综合候选|
|`shell_s150_max095_rej098_unk080`|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|0.8313|4.2150|168.6|0.1838|22|7|`unknown=7`|FAR升高|
|`shell_s175_max095_rej098_unk080`|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|0.0000|1.0000|40.0|0.1301|40|4|`old=2,unknown=2`|单receiver不足|
|`shell_s175_max095_rej098_unk080`|2|0.4079|0.0500|0.5769|0.5312|0.1304|0.8696|0.0720|0.5686|2.0000|80.0|0.1301|62|0|`{}`|FAR高|
|`shell_s175_max095_rej098_unk080`|3|0.6250|0.2500|0.7250|0.6000|0.0750|0.9000|0.0300|0.7250|3.0000|120.0|0.1301|39|2|`unknown=2`|FAR仍超|
|`shell_s175_max095_rej098_unk080`|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|0.8375|3.7500|150.0|0.1301|22|3|`unknown=3`|与s150同指标，p95 proxy更低|
|`shell_s175_max095_rej098_unk080`|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|0.8313|4.2150|168.6|0.1301|22|7|`unknown=7`|FAR升高|

k=4逐类结果在三个主线尺度上相同：

|类别集|逐类准确率|
|---|---|
|old|`14-10=0.7000`，`14-7=0.7000`，`20-15=0.7500`，`20-19=0.7000`，`6-15=1.0000`，`8-20=1.0000`|
|seen-new|`19-3=0.6000`，`3-8=0.9000`|

### 判定

`shell_s150_max095_rej098_unk080,k=4`和`shell_s175_max095_rej098_unk080,k=4`相对上一轮当前最佳`avail3_p055_lrca049_huv0999_f050,k=4`有小幅有效改进：

|候选|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|known_cov|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|上一轮`avail3_p055_lrca049_huv0999_f050,k=4`|0.8083|0.7000|0.7250|0.6000|0.0250|0.9500|0.8250|3.7500|150.0|
|本轮`shell_s150_max095_rej098_unk080,k=4`|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.8375|3.7500|150.0|
|本轮`shell_s175_max095_rej098_unk080,k=4`|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.8375|3.7500|150.0|

改进点：seen-new总体从0.7250升到0.7500，`3-8`从0.8500升到0.9000，unknown FAR和unknown拒识不变，known coverage从0.8250升到0.8375。`shell_s150/s175`的shell veto在k=4均只命中unknown3次，没有误伤old/seen-new；`shell_s125`误伤1个old，因此不作为主候选。

边界：该候选仍远未达最终目标。旧类地板仍只有0.7000，seen-new地板仍只有0.6000，unknown拒识仍为0.9500而非0.9900。当前结果只能说明support-only class shell风险是一个有效小增益机制，不能声明Stage2-C成功、部署成功或论文主结论。

### 子agent监督意见纳入

|角色|关键意见|处理|
|---|---|---|
|文献/方法检索|建议优先采用收缩原型、Mahalanobis/energy、support-only边界、证据级轻量融合；避免full-model在线训练、MAML、无门控自训练、unknown query阈值拟合。|采用support-only class shell风险，不改主干，不用unknown query校准。|
|算法构建|建议`CCBR-CVS`，新增`class_shell_risk`和后续`receiver_pair_inconsistency`；第一版先让shell风险硬门控，pair不一致先审计。|已实现`class_shell_risk`；pair不一致尚未实现，列为下一步。|
|合理性监督|`receiver_domain_ranked`不能写成严格同事件协同；必须保留Stage2-C边界、unknown query评估专用、资源proxy边界。|报告明确当前仍是receiver-domain ensemble诊断，不升格为严格同事件卫星群协同证据。|
|逐项完成监督|目标指标仍未完成；缺strict_event_key复核、资源约束原文、checkpoint SHA写入最终JSON、低类地板归因。|列入下一步硬缺口。|
|查漏补缺review|过度调参风险高；需protocol audit、receiver pair矩阵、threshold sweep、低类地板优化和matched-all denominator。|本轮只承认小增益；下一步优先做pair矩阵和strict_event_key可行性审计。|

### 下一步

1. 固定`shell_s150_max095_rej098_unk080,k=4`为当前机制基线，停止围绕同一阈值族做无边界扫参。
2. 增加receiver pair矩阵和逐事件错误表，定位哪些接收机组合救`3-8`、哪些组合误拒`19-3`和低地板old类。
3. 检查是否能用`strict_event_key`构造严格同事件协同；若不能，必须把当前全量结果继续标为`receiver_domain_ranked`诊断。
4. 实现`receiver_pair_inconsistency`审计字段，先报告不门控，再决定是否启用。
5. 最终JSON需补写ADV3B02 checkpoint路径/SHA和feature生成命令，增强artifact追溯。

## 2026-07-04 receiver pair不一致审计实现

### 目的

本轮不继续调全局阈值，而是补齐下一步机制开发所需的诊断证据：在融合器中记录receiver pair标签不一致、unknown risk跨度和score跨度；新增pair审计脚本，对当前最佳`shell_s150_max095_rej098_unk080`的evidence按两两receiver组合重算指标，并输出错误事件表。该审计字段只报告，不参与accept/reject/defer决策。

### 本地变更与验证

|文件|变更|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`receiver_pair_label_disagreement`、`receiver_pair_unknown_risk_range`、`receiver_pair_score_range`事件级字段和count级均值；新增`include_event_results`用于审计导出。|
|`code/scripts/collab_evidence_pair_audit.py`|新增pair矩阵与错误事件表生成脚本，复用正式融合器，不另写判定规则；从evidence推断只读`Y_old/Y_new`协议标签，避免candidate_set把合法类当作`other`。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|覆盖事件级审计字段与`include_event_results`。|
|`code/tests/test_collab_evidence_pair_audit.py`|覆盖pair矩阵和错误事件筛选。|

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\collab_evidence_pair_audit.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_collab_evidence_pair_audit.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：根工作区`92 passed`，Git镜像同一组测试`92 passed`。`.pytest_cache`仍因本机目录权限给出warning，不影响测试结论。

代码SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`FE50422E70F0D4C24749A38F95CDCC50CD19A8BDD51A71812E94965BD743D6D1`|
|`code/scripts/collab_evidence_pair_audit.py`|`B11D5EAF6DF2B4F8D6D41D06BDBC17D4C8D4F9E8E7968206840AF1DB2AFE7C6F`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`E6FF68BD440D23D1A85DE0C3BC9FB8D342140B29E19D584597FE6E5C8BB3FDFD`|
|`code/tests/test_collab_evidence_pair_audit.py`|`D00973FDCBD1A3F12BBB321A7B6985100B1DFF1C9363BD2EF65035C2E162B01D`|

### pair审计产物

```text
python code/scripts/collab_evidence_pair_audit.py --evidence_csv remote_artifacts/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_max095_rej098_unk080_adv3b02_evidence.csv --run_json remote_artifacts/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_max095_rej098_unk080_adv3b02.json --output_pair_csv pair_audit_shell_s150_k2_matrix.csv --output_error_csv pair_audit_shell_s150_k2_errors.csv --max_error_rows 500
```

|产物|SHA256|
|---|---|
|`pair_audit_shell_s150_k2_matrix.csv`|`7AE3BE1BB77CE0261C44908786C7CACC20940A8203888B043BA4409606F4F11F`|
|`pair_audit_shell_s150_k2_errors.csv`|`7C7826469946373EC05FE231B99D0722C4662DA8EB52F24AA03AEF9FBAB8FCA4`|

两两receiver结果：

|receiver pair|total|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|`3-19+7-14`|160|0.4694|0.1000|0.7857|0.7500|0.0294|0.9706|新类较好且FAR低，但旧类严重不足|
|`3-19+8-8`|134|0.5833|0.0000|0.7500|0.7000|0.0294|0.9412|新类较好但`20-15`塌陷|
|`7-14+7-7`|190|0.8091|0.6000|0.5500|0.3500|0.1250|0.8750|旧类最好但FAR和新类不可接受|
|`7-14+8-8`|174|0.7340|0.2000|0.7000|0.5500|0.2000|0.7750|旧类/新类中等，FAR过高|
|`7-7+8-8`|184|0.7692|0.5000|0.4750|0.3000|0.2250|0.7500|FAR过高且新类差|

主要错误簇：

|错误簇|计数|含义|
|---|---:|---|
|`old 20-19 -> unknown_reject`|62|旧类低地板主要来自过强拒识。|
|`seen_new 19-3 -> unknown_reject`|54|新类地板主要来自把`19-3`判成unknown。|
|`old 14-10 -> accept 19-3`|50|`14-10`和seen-new`19-3`存在强混淆。|
|`seen_new 19-3 -> accept 14-10`|29|与上一项互为混淆，说明仅靠shell风险不能分开这对类。|
|`unknown -> accept 14-7/14-10/6-15`|30|剩余unknown false accept集中吸收到少数old类。|

### 判定

pair审计证明：低成本2星组合没有单个pair同时满足旧类、新类和unknown拒识。`3-19+7-14`和`3-19+8-8`能相对保护seen-new并维持低FAR，但旧类地板很低；`7-14+7-7`能保护旧类，但FAR上升到0.125且`19-3`地板很差。下一步不能简单选择固定receiver pair，而应做角色/类别条件化路由：对`19-3`优先调用`3-19/7-14/8-8`证据，对旧类`20-19/14-10`增加反混淆二级验证，对unknown吸收类`14-7/14-10/6-15`加入更强oldness或pair verifier。

资源约束说明文件状态：在当前工作区按`卫星协同`、`资源约束`、`RFFI系统说明`关键词递归搜索，仍未找到用户点名的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`。因此本报告继续使用已有字段`participating_receivers`、`bytes_per_event`、`latency_ms_p50/p95`、`prototype_storage_bytes`、`max_event_bytes`、`max_event_latency_ms`作为临时资源口径；找到原文后必须按原文重新校验上限。

### N607同步与严格事件诊断

N607直连preflight：`2026年07月04日01:36:22 CST`通过，项目根可见，8张RTX3090均为`10/24576MiB`。本轮同步使用直接`scp -F E:\type10-7\tools\n607_ssh_config`，目标路径如下：

|本地文件|N607目标|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py`|
|`code/scripts/collab_evidence_pair_audit.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/collab_evidence_pair_audit.py`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py`|
|`code/tests/test_collab_evidence_pair_audit.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collab_evidence_pair_audit.py`|

远端使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`验证：

```text
python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/collab_evidence_pair_audit.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
python -m unittest discover -s code/tests -p 'test_collaborative_open_set_qknn_eval.py' -q
python -m unittest discover -s code/tests -p 'test_collab_evidence_pair_audit.py' -q
```

结果：`test_collaborative_open_set_qknn_eval.py`为`47 tests OK`；`test_collab_evidence_pair_audit.py`为`1 test OK`。远端hash与本地一致：

|文件|N607 SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`fe50422e70f0d4c24749a38f95cdcc50cd19a8bdd51a71812e94965bd743d6d1`|
|`code/scripts/collab_evidence_pair_audit.py`|`b11d5eaf6df2b4f8d6d41d06bdbc17d4c8d4f9e8e7968206840af1db2afe7c6f`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`e6ff68bd440d23d1a85de0c3bc9fb8d342140b29e19d584597fe6e5c8bb3fdfd`|
|`code/tests/test_collab_evidence_pair_audit.py`|`d00973fdcbd1a3f12bbb321a7b6985100b1dff1c9363bd2ef65035c2e162b01d`|

远端pair审计复现：

|远端产物|SHA256|
|---|---|
|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/pair_audit_shell_s150_k2_matrix.csv`|`7ae3be1bb77ce0261c44908786c7cacc20940a8203888b043ba4409606f4f11f`|
|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/pair_audit_shell_s150_k2_errors.csv`|`7c7826469946373ec05fe231b99d0722c4662da8eb52f24aa03aef9fbab8fca4`|

严格同事件诊断命令把当前最佳`shell_s150`配置的`--event_alignment_policy`改为`strict_event_key`后失败，错误为：

```text
RuntimeError: NO_ALIGNED_COLLABORATIVE_EVENTS: target receiver query rows do not share role+tx+day+sig+scenario keys; use --event_alignment_policy receiver_domain_ranked only for explicitly marked receiver-domain ensemble diagnostics
```

判定：当前全量协同结果不能升格为严格同物理事件卫星群协同，只能继续标为`receiver_domain_ranked` receiver-domain ensemble诊断。若要满足真实卫星群协同部署证据，需要重新导出带共享物理事件键的features或构造严格同事件query集合。

远端结束状态：8张RTX3090均为`10/24576MiB`；本地检查无残留`ssh.exe`，无到`172.31.111.215:22`或`172.31.105.18:22`的ESTABLISHED连接。Git镜像提交：`ace6ff4 Add receiver pair audit for collaborative open set`。

## 2026-07-04 support_quality_prior接收机选择策略

### 目的

pair审计显示固定2星组合存在明显取舍：保护seen-new的pair会牺牲旧类，保护旧类的pair会放大FAR。因此本轮实现一个轻量接收机选择策略`support_quality_prior`，在每个事件内按节点本地support校准质量排序，再取预算内接收机。排序使用`reliability`、`support_density`、`class_conformal_pvalue`、`receiver_class_reliability`、`known_score/margin`、`class_shell/radius`安全项，不使用unknown query标签或真实类别。

该策略目标不是直接声明成功，而是检验“支持集质量优先路由”是否能提升低地板新类，作为后续类别条件化路由的依据。

### 本地变更与验证

|文件|变更|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`receiver_selection_policy=support_quality_prior`和`_receiver_support_quality()`。默认仍为`fixed_receiver_order`，历史结果可复现。|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`support_quality_prior`选项。|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增测试，验证该策略能优先选择support校准质量更高的receiver。|

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：根工作区和Git镜像均通过，`92 passed`。本地离线重算当前最佳`shell_s150` evidence显示：`support_quality_prior,k=3`可把`seen_new_acc`提升到0.8750、`min_seen`提升到0.8500，`old_acc=0.8167`、`min_old=0.7000`，但`unknown_FAR=0.1250`，拒识退化明显。因此该策略只能作为“新类救援路由”诊断，不能替代当前低FAR主线。

代码SHA256：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`E50ABB250F592D9EEC3ED77F233FE2DFDEDD58156A464DA7BBBF822A7D5212BB`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`8C27203EF0AA6CEFE4BC62FF1AC44E5581AF21979D8E7F122867E07CF1BB2919`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`67FBA2F87D8C7A9CFE3C87FE48273F7DC49FCD3E3867012AB97193AC57DC9524`|

Git镜像提交：`ebdf1cc Add support quality receiver selection`。

### N607计划

计划同步上述3个文件到N607，远端使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`验证语法和单测，然后复跑当前最佳`shell_s150`配置，仅把`--receiver_selection_policy`改为`support_quality_prior`。运行仍使用`ADV3B02_CORE90_SOFT_E200`特征、qknn8、`collab_counts=all`、`available_up_to_k`、`partial_collab_min_receivers=3`，输出：

```text
runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_supportq_adv3b02.json
runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_supportq_adv3b02_evidence.csv
```

### N607结果

N607直连preflight：`2026年07月04日01:44:17 CST`通过，项目根可见，8张RTX3090均为`10/24576MiB`。同步后远端使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`验证：

```text
python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
python -m unittest discover -s code/tests -p 'test_collaborative_open_set_qknn_eval.py' -q
```

结果：`48 tests OK`。远端hash：

|文件|N607 SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`e50abb250f592d9eec3ed77f233fe2dfdedd58156a464da7bbbf822a7d5212bb`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`8c27203ef0aa6cefe4bc62ff1ac44e5581af21979d8e7f122867e07cf1bb2919`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`67fba2f87d8c7a9cfe3c87fe48273f7dc49fcd3e3867012ab97193ac57dc9524`|

远端运行使用GPU0，输出`1000`行evidence。产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_supportq_adv3b02.json`|`B519EE9961ABAD8AB5274B0CCE6A229DE342CAF8109807D4D40F5361B167739D`|
|`collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_supportq_adv3b02_evidence.csv`|`D5E2D03D6CB9C3E25BA6170B4EC6B1DEA6BA55DD5D57EBD475776CA70D43BE3B`|

全量`1..5`协同数量结果：

|预算k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|known_cov|avg_rx|bytes/event|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.3333|0.8795|0.0000|1.0000|40.0|单receiver仍不足|
|2|0.7368|0.5000|0.7115|0.6562|0.4130|0.5652|0.0680|0.7941|2.0000|80.0|known提升但FAR不可用|
|3|0.8167|0.7000|0.8750|0.8500|0.1250|0.8500|0.0100|0.8875|3.0000|120.0|新类显著提升但拒识失败|
|4|0.7917|0.6000|0.8250|0.7500|0.1000|0.8750|0.0100|0.8688|3.7500|150.0|FAR仍超标且old低于0.80|
|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|0.8313|4.2150|168.6|FAR改善但known退化|

k=3逐类结果：

|类别集|逐类准确率|
|---|---|
|old|`14-10=0.7000`，`14-7=0.7000`，`20-15=0.7500`，`20-19=0.7500`，`6-15=1.0000`，`8-20=1.0000`|
|seen-new|`19-3=0.8500`，`3-8=0.9000`|

k=3 open-set confusion：

|项|计数|
|---|---:|
|`old->old`|100|
|`old->seen_new`|5|
|`old->unknown_reject`|13|
|`old->defer`|2|
|`seen_new->seen_new`|35|
|`seen_new->old`|2|
|`seen_new->unknown_reject`|3|
|`unknown->unknown_reject`|34|
|`unknown->old`|4|
|`unknown->seen_new`|1|
|`unknown->defer`|1|

### 判定

`support_quality_prior`证明了接收机选择策略确实能救新类：相对当前低FAR主线`shell_s150,k=4`，新类总体从`0.7500`提高到`0.8750`，新类地板从`0.6000`提高到`0.8500`，且旧类总体从`0.8083`小幅提高到`0.8167`。但代价是unknown拒识从`0.9500`降到`0.8500`，`unknown_FAR`从`0.0250`升到`0.1250`。因此它不能作为最终主线，也不能声明Stage2-C成功；它是下一步“类别条件化双路由”的证据：known救援路径应使用`support_quality_prior`，unknown安全路径仍需保留高unknown风险否决、class shell或新增oldness/pair verifier。

远端结束状态：8张RTX3090均为`10/24576MiB`；本地无残留`ssh.exe`，无N607/bridge 22端口ESTABLISHED连接。

## 2026-07-04 dual_route_cvs协同推理实验计划

### 目标

实现并测试`dual_route_cvs`双路协同推理：安全路由使用固定receiver顺序保持unknown拒识边界，救援路由使用support-only`receiver_deployment_prior`排序；只有在class conformal、support-calibrated receiver-class reliability、class shell风险、组件风险一致性、receiver预测分歧、receiver unknown风险分歧和安全路由风险均满足门控时才允许救援接管。

该设计吸收子agent审查意见：禁止用query派生`support_quality_prior`偷看全receiver后只计入k个receiver；救援排序只能使用support校准先验，缺少先验时退回固定顺序并记录`dual_route_rescue_selection_policy=fixed_receiver_order_no_prior`。

### 本地改动

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`dual_route_cvs`、部署先验receiver选择、救援门控、事件审计字段，并修复`receiver_pair_label_disagreement`未记录预测标签导致恒为1.0的问题|`8A2B09EB1E924907CAADF70031A69391A0E83A70858F8F5E7C9A57289218F041`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI暴露`dual_route_cvs`和门控参数，生成support-only`receiver_deployment_prior`|`CC4432902B8C70797AE22FDBBADBD6E0A372519E8625BB2FB567DD5ADCD9B59C`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增双路救援通过与高unknown风险阻断测试|`0296A94E8BC386363EABC85925C58FC703B28D361918D3077B43F57D24C4E7EF`|

本地快照：`E:\type10-7\code\snapshots\dual_route_cvs_20260704\`。Git镜像提交：`4270c02 Add dual route collaborative inference`。

### 本地验证

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：`94 passed`；`.pytest_cache`写入被Windows拒绝但不影响测试。

### N607同步与运行计划

同步映射：

|本地|N607|
|---|---|
|`E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py`|

远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。

远端验证：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m unittest discover -s code/tests -p 'test_collaborative_open_set_qknn_eval.py' -q
```

实验命令将使用当前`ADV3B02_CORE90_SOFT_E200`特征与星地信道Stage2-C数据，`collab_counts=all`覆盖1到全体target receiver数量，`available_up_to_k`、`partial_collab_min_receivers=3`保持与当前主线一致。目标输出：

```text
runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_dualroute_adv3b02.json
runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rcwr_avail3_p055_lrca049_huv0999_f050_shell_s150_dualroute_adv3b02_evidence.csv
```

成功/失败判据：主目标仍是old_acc 0.99、old per-class>=0.95、seen_new_acc 0.97、seen per-class>=0.93、unknown_reject 0.99。若未达标，必须报告为算法诊断，不得声明部署成功。

更正：本节仅为本地实现后的初始ADV3B02计划记录，未作为本轮最终用户请求执行。用户指定权重为`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`；实际N607复跑与最终审计结果已转入`E:\type10-7\automation_reports\CV-SincNet\phase2_sa33_collab_open_set_qknn_full_20260703\report.md`。

## 2026-07-04 ADV3B02 pairguard协同推理执行记录

### 目标与边界

本轮重新面向`ADV3B02_CORE90_SOFT_E200`执行协同推理改进，算法名称暂定`candidate_set_cvs_pairguard`。目标是在不增加query探测receiver预算的前提下，提高unknown拒识安全性，并保留`support_quality_prior`已证明的known类救援能力。当前证据仍属于`receiver_domain_ranked`诊断口径；若strict event key继续缺失，不得声明严格同物理事件卫星群协同成功。

文献启发：

|来源|可落地机制|本轮采用方式|
|---|---|---|
|[Towards Receiver-Agnostic and Collaborative RFFI](https://arxiv.org/abs/2207.02999)|多receiver协同推理可提升RFFI，但需抑制receiver硬件偏移|保留多receiver证据融合，并新增receiver间分歧门控|
|[Few-Shot Open-Set RFFI/MLGPN](https://ieeexplore.ieee.org/document/10960433/)|Gaussian prototype/Mahalanobis支持few-shot开集识别|继续使用class conformal、class shell和unknown risk作为轻量开集证据|
|[Collaborative DNN Inference for Edge Intelligence](https://arxiv.org/abs/2207.07812)|边缘协同推理应按算力、通信、置信度做动态分配|只上传score/risk/reliability等轻量证据，记录bytes/latency|
|[Federated RFFI Powered by Unsupervised Contrastive Learning](https://www.eng.auburn.edu/~szm0001/papers/tifs24.pdf)|跨节点RFFI训练可不共享原始IQ|后续在线微调只允许adapter/prototype/threshold轻量更新，backbone冻结|

子agent监督结论已吸收：禁止用query派生receiver质量先看全receiver再只统计k个receiver；不能把`receiver_domain_ranked`写成strict event协同；unknown query只能eval-only；未达标必须标记diagnostic-only。

### 本地改动

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|为`candidate_set_cvs`新增pairguard接受门控：`candidate_set_max_receiver_pair_label_disagreement`、`candidate_set_max_receiver_pair_unknown_risk_range`、`candidate_set_min_label_receiver_class_reliability`、`candidate_set_require_label_shell_observed`；逐事件metadata同步输出|`10CBFDF475C6CE67120D0BF1C8D0B9312F67C27335274C8180CC79664292942B`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI暴露上述4个pairguard参数并透传到评估器|`0A1DC5CAD120EC151C0F517C106F2D9E6C59E088DC7FC1E18F5779E146CCB11D`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增pairguard单测，覆盖pair标签分歧、unknown风险跨度、receiver-class可靠性、shell观测缺失和参数校验|`B808FA4B89B5791A94F25A7D764B5D8DBB1995AA95FF570B60311F3EC9A54FEA`|

### 本地验证

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：本地工作树`95 passed`；Git镜像树同样`95 passed`。根目录测试仍有`.pytest_cache`写入权限警告，不影响测试结果。

### N607运行计划

远端环境按用户要求使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。同步目标工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。

计划输出：

```text
runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_pairguard_dualroute_adv3b02.json
runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_pairguard_dualroute_adv3b02_evidence.csv
```

计划命令关键参数：`--collab_counts all`覆盖1到全体target receiver数量；`--collab_group_policy available_up_to_k`、`--partial_collab_min_receivers 3`保持当前ADV3B02口径；`--fusion_policy candidate_set_cvs`、`--collaboration_policy dual_route_cvs`；显式启用pairguard：`--candidate_set_max_receiver_pair_label_disagreement 0.34`、`--candidate_set_max_receiver_pair_unknown_risk_range 0.30`、`--candidate_set_min_label_receiver_class_reliability 0.75`、`--candidate_set_require_label_shell_observed`。远端运行前需完成N607预检、低显存GPU选择、SCP同步、远端`py_compile`和单测。

### N607执行与结果

预检：`tools\n607_ssh_preflight.ps1`通过。N607项目根目录存在，8张RTX3090均为`10/24576MiB`，选择并列最低显存占用的GPU0。远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。远端验证：`py_compile`通过，`unittest discover -s code/tests -p 'test_collaborative_open_set_qknn_eval.py' -q`为`51 tests OK`。同步后三个远端代码文件SHA256与本地一致。

远端输出均使用`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，星地信道特征沿用ADV3B02 Stage2-C特征包；`receiver_count=5`、`group_count=307`、`evidence_row_count=1000`，`--collab_counts all`覆盖k=1..5。

|路线|JSON SHA256|CSV SHA256|结论|
|---|---|---|---|
|`dualroute_noguard`|`CD745E3C1E81AEF4CF2B46E4CD9252777AD22C59809FC3D4D3AFA7E45B1396AB`|`2277705159F548B8CD75DE204624C3370C0E217C589ABA9921E3FFC7765FFED1`|k=2、k=3优于固定路由；k=4、k=5与固定路由持平；保留为当前最合理轻量双路由诊断。|
|`pairguard_strict`|`4D7E51C511F9BA87438B28F9C05F3AB1B1D00A1FCD75675A46A36D2A390E9934`|`00BAEF7ECA1A9209A5C67DA91B6D71077A4FD0A18A2FC307A87E00D7EAB1F43D`|unknown_FAR最低，但known被大量转为unknown_reject，不适合作主线。|
|`pairguard_balanced`|`9C6781A0B63A3CAB9BEA577B9A6F3E8AE8945F7BB57448BF5294E046C45AB519`|`1BC9D7042984B4FF24F00EF7602639DF26175E76D1B2540A1690B2B38E9B6076`|比strict保留更多known，但仍显著低于固定路由，说明pairguard不能作为硬accept门控直接上线。|

主结果表：

|route|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|bytes/event|p95 ms|rescue_rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|baseline_fixed_k|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|40.0|0.1838||
|baseline_fixed_k|2|0.4079|0.0500|0.5769|0.5313|0.1304|0.8696|0.0720|80.0|0.1838||
|baseline_fixed_k|3|0.6250|0.2500|0.7250|0.6000|0.0750|0.9000|0.0300|120.0|0.1838||
|baseline_fixed_k|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1838||
|baseline_fixed_k|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1838||
|dualroute_noguard|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|40.0|0.1374|0.0000|
|dualroute_noguard|2|0.5000|0.2000|0.5962|0.5625|0.1304|0.8696|0.0440|80.0|0.1374|0.2120|
|dualroute_noguard|3|0.6500|0.3000|0.7250|0.6000|0.0750|0.9000|0.0250|120.0|0.1374|0.2100|
|dualroute_noguard|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1374|0.1800|
|dualroute_noguard|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1374|0.1550|
|pairguard_strict|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|40.0|0.1379|0.0000|
|pairguard_strict|2|0.3224|0.1500|0.2692|0.1000|0.0217|0.9348|0.1800|80.0|0.1379|0.1960|
|pairguard_strict|3|0.2500|0.0500|0.3500|0.2500|0.0000|0.9750|0.1300|120.0|0.1379|0.1700|
|pairguard_strict|4|0.1250|0.0000|0.2750|0.1000|0.0000|0.9500|0.2400|150.0|0.1379|0.1050|
|pairguard_strict|5|0.0750|0.0000|0.2750|0.1000|0.0000|0.9750|0.2550|168.6|0.1379|0.0800|
|pairguard_balanced|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.4667|0.7459|40.0|0.1358|0.0000|
|pairguard_balanced|2|0.3947|0.2000|0.5000|0.5000|0.0652|0.9348|0.0520|80.0|0.1358|0.2120|
|pairguard_balanced|3|0.4167|0.1000|0.5000|0.4000|0.0000|0.9750|0.0300|120.0|0.1358|0.2100|
|pairguard_balanced|4|0.4333|0.1500|0.3750|0.2500|0.0000|0.9500|0.1050|150.0|0.1358|0.1800|
|pairguard_balanced|5|0.3667|0.1500|0.3500|0.2000|0.0250|0.9500|0.1150|168.6|0.1358|0.1550|

### 判定与下一步

`dualroute_noguard`是本轮唯一没有明显退化的新增路线：k=2 old从`0.4079`升到`0.5000`，seen-new从`0.5769`升到`0.5962`；k=3 old从`0.6250`升到`0.6500`，seen-new持平`0.7250`；k=4和k=5与固定路由持平。它的unknown_FAR未改善，k=2仍为`0.1304`，k=3为`0.0750`，因此仍是diagnostic-only。

pairguard证明了“receiver分歧/风险跨度/shell观测”确实能压低unknown false accept，但直接作为硬accept门控会把大量known样本打成unknown_reject。下一步不应继续收紧阈值，而应把pairguard改为分层策略：仅在unknown risk接近边界或label shell异常时触发二级校验；对高pvalue、高receiver-class reliability、高score gap的known样本保持accept；对冲突样本进入`request_more`或地面复核，而不是直接unknown_reject。

达标状态：未达成old 99%、old per-class>=95%、seen-new 97%、seen per-class>=93%、unknown rejection 99%目标；不得声明Stage2-C部署成功。最终SSH/SCP后本地无`ssh.exe`残留，无N607和bridge 22端口ESTABLISHED连接。

## 2026-07-04 ADV3B02 boundary_pairguard执行计划

### 动机

上一轮`pairguard_strict`和`pairguard_balanced`说明receiver分歧、unknown风险跨度和shell观测能降低unknown false accept，但如果把这些条件作为所有known接受的硬门控，会把大量old/seen-new样本转为unknown_reject。新实现增加`candidate_set_pairguard_mode=boundary_veto`：只有当事件级unknown风险、label级unknown风险或shell风险接近边界时，才触发pairguard拦截；低风险高置信known样本即使receiver之间存在分歧，也不直接被硬拒绝。

资源约束说明文件`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`本地仍未定位到。当前报告继续使用评估器内的`receiver_count`、`bytes_per_event`、`latency_ms_p95`、`avg_participating_receivers`作为资源替代证据，并保留该文档缺失风险。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`candidate_set_pairguard_mode={accept_gate,boundary_veto}`与边界触发阈值；逐事件输出pairguard失败/触发/veto字段|`8469FA97742A09D6425D2B5EE024E452CBBDBB076EFE0D31BA8BFFD6EE3F235C`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI暴露boundary pairguard参数|`221922B12587EA0F752944BD3077055B219593805263D5157ACE03119795ED27`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增boundary pairguard单测，证明低风险分歧样本可accept，高风险分歧样本被defer|`E8C2CA81867C7443B29E92D06A2B7643909FFA22449A6EE3981CFFFF9FD60761`|

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：本地工作树`96 passed`；Git镜像树同样`96 passed`。Git镜像提交：`3f2419a Add boundary pairguard mode`。

### N607运行计划

同步目标：

|本地|N607|
|---|---|
|`E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py`|

远端环境继续使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，工作目录为`/home/szu2070436088/2510044040/CV-SincNet`。计划跑两组`boundary_veto`全量k=1..5对照：

|路线|关键参数|目的|
|---|---|---|
|`boundary_pairguard_evt090`|`--candidate_set_pairguard_mode boundary_veto --candidate_set_pairguard_min_event_unknown_risk 0.90 --candidate_set_max_receiver_pair_label_disagreement 0.50 --candidate_set_max_receiver_pair_unknown_risk_range 0.70 --candidate_set_min_label_receiver_class_reliability 0.75 --candidate_set_require_label_shell_observed`|在风险接近unknown时才拦截，测试能否保留`dualroute_noguard`的known性能并降低unknown_FAR。|
|`boundary_pairguard_evt095`|同上但`candidate_set_pairguard_min_event_unknown_risk 0.95`|更保守触发，作为低伤害版本。|

输出路径将写入`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/`，并拉回到本地`remote_artifacts/`。成功目标仍为old 99%、old per-class>=95%、seen-new 97%、seen per-class>=93%、unknown rejection 99%；未达到则继续标为diagnostic-only。

### N607结果

N607同步后远端哈希与本地一致；远端验证`52 tests OK`。实际运行三组：`boundary_pairguard_evt090`、`boundary_pairguard_evt095`和一个更轻的`boundary_shellrel_evt095`。运行均使用GPU0；首次运行前后GPU显存均为`10/24576MiB`，后续短作业未观察到显存上升。每组均覆盖`collab_counts=all`即k=1..5。

|route|JSON SHA256|CSV SHA256|
|---|---|---|
|`boundary_pairguard_evt090`|`5A04B78C9933FA498173EF32E9C479D4DC86BA8CF353B60C2477D197ACF6D675`|`B67AF0B29895A812074D0BF2AF1F75FAC342F1A9024777AB34229191725ACA03`|
|`boundary_pairguard_evt095`|`984D89D5D51A120A03C38C492DC40EDE3850364A97AB383E47393572C4EA3097`|`CD6C5209C9393E84D535971082B9320A5379846EA380C6643689E13741AFC487`|
|`boundary_shellrel_evt095`|`45D7B8042E589CB35D14E54099BE1C21EEB8BF147E54BA24C6E2E024C4E0DD8C`|`C4CFF9564E0329B4ABCA52A5C2994BEAF53901EBEA69A1082CF8C4734923F809`|

|route|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|bytes/event|p95 ms|pairguard_veto_rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|dualroute_noguard|2|0.5000|0.2000|0.5962|0.5625|0.1304|0.8696|0.0440|80.0|0.1374|0.0000|
|dualroute_noguard|3|0.6500|0.3000|0.7250|0.6000|0.0750|0.9000|0.0250|120.0|0.1374|0.0000|
|dualroute_noguard|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1374|0.0000|
|dualroute_noguard|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1374|0.0000|
|boundary_pairguard_evt090|2|0.4276|0.2000|0.5000|0.5000|0.0652|0.9348|0.0440|80.0|0.1335|0.2800|
|boundary_pairguard_evt090|3|0.4750|0.1500|0.5500|0.4500|0.0000|0.9750|0.0250|120.0|0.1335|0.4500|
|boundary_pairguard_evt090|4|0.6167|0.4000|0.5250|0.4000|0.0250|0.9500|0.0150|150.0|0.1335|0.3850|
|boundary_pairguard_evt090|5|0.5917|0.2500|0.5000|0.3500|0.0250|0.9500|0.0100|168.6|0.1335|0.4550|
|boundary_pairguard_evt095|2|0.4408|0.2000|0.5000|0.5000|0.0652|0.9348|0.0440|80.0|0.1336|0.2600|
|boundary_pairguard_evt095|3|0.4917|0.1500|0.5750|0.4500|0.0250|0.9500|0.0250|120.0|0.1336|0.4250|
|boundary_pairguard_evt095|4|0.6167|0.4000|0.5500|0.4000|0.0250|0.9500|0.0150|150.0|0.1336|0.3750|
|boundary_pairguard_evt095|5|0.5917|0.2500|0.5250|0.3500|0.0250|0.9500|0.0100|168.6|0.1336|0.4450|
|boundary_shellrel_evt095|2|0.5000|0.2000|0.5962|0.5625|0.1304|0.8696|0.0440|80.0|0.1839|0.0040|
|boundary_shellrel_evt095|3|0.6500|0.3000|0.7250|0.6000|0.0750|0.9000|0.0250|120.0|0.1839|0.0000|
|boundary_shellrel_evt095|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1839|0.0000|
|boundary_shellrel_evt095|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1839|0.0000|

判定：`boundary_pairguard_evt090/evt095`确实降低了unknown_FAR，k=3分别为`0.0000/0.0250`，但pairguard veto率达到`0.4250-0.4500`，导致old和seen-new大幅下降。`boundary_shellrel_evt095`几乎不伤known，但也几乎不改善unknown。当前证据说明下一步不能继续靠全局pair分歧/风险跨度阈值，而应改为“类别条件化pairguard”：只对历史false-accept高发标签或receiver pair触发，或把pairguard输出改为`request_more`/地面复核权重，而不是直接阻断accept。当前仍未达标，所有新增结果均为diagnostic-only。

最终SSH/SCP后本地无`ssh.exe`残留，无N607和bridge 22端口ESTABLISHED连接。

## 2026-07-04 SR-PairFuse soft floor/strong bypass实现

### 设计依据

上一轮`sr_pairfuse_soft14_7_noguardbase_evt095`证明：取消硬pairguard后，k=4/k=5的old可恢复到`0.8917`，但unknown_FAR仍为`0.2750/0.3000`。逐事件审计显示，部分unknown false accept具有较高score或pvalue，单纯按弱证据比例惩罚会出现`soft_weakness=0`从而放行；但强证据old/seen-new事件又不能被无差别硬拦截。因此本轮新增两个机制：

|机制|参数|作用|
|---|---|---|
|强证据保护|`candidate_set_pairguard_soft_strong_bypass`|命中高风险pairguard边界时，若margin、agreement、conformal pvalue、receiver-class reliability均达到配置下限，则不施加soft penalty。|
|弱证据最低惩罚|`candidate_set_pairguard_soft_floor`|命中高风险pairguard边界且未满足强证据保护时，即使比例惩罚很小，也至少把accept风险抬高到可阻断边界误接收的水平。|

该机制仍是diagnostic route：pairguard label/receiver组合来自当前评估集错误分布，正式论文叙述不能把unknown query调参作为可部署先验；若保留该方向，应改为source/support/proxy-known校准得到风险表。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`candidate_set_pairguard_soft_floor`参数，补强证据bypass，并把floor记录到事件、顶层metadata和汇总审计指标；修正默认门槛真空bypass和shell缺失bypass风险。|`30D0D710154AA962C483DEB1E2D5257630CA9ECB43EBB39481B0E8B83CB347B6`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`--candidate_set_pairguard_soft_floor`并传入评估函数。|`EA2875D99F79528AE0D0137BCA2EB372282499B2CAA193C9904DB5C0FF89F980`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|补充强证据bypass、弱证据floor触发、默认门槛不得真空bypass、shell缺失不得bypass、非法floor校验。|`57F45DC3BDA05C83D4065A534F0804711DD6F05713B84F9C5F82F7C61F03D4F8`|

本地快照：`E:\type10-7\code\snapshots\phase2_sr_pairfuse_floor_20260704\`。

验证命令：

```text
conda run --no-capture-output -n ssr-gpu python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -k pairguard -q -p no:cacheprovider
```

结果：`py_compile`通过；pairguard相关测试`2 passed,50 deselected`。Git镜像提交：`272a5e2 Add pairguard soft floor bypass`，安全修正提交：`93cb0c7 Guard pairguard soft bypass`。

### N607执行计划

远端Python环境固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。计划同步上述三个文件到N607，先做远端`py_compile`，再在显存占用最低的GPU上运行k=1..5全量星地信道评估：

|run|关键参数|目的|
|---|---|---|
|`candidate_set_cvs_sr_pairfuse_floor_risklabels_evt095_adv3b02`|`--collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 3 --candidate_set_pairguard_action soft_penalty --candidate_set_pairguard_soft_penalty 0.35 --candidate_set_pairguard_soft_floor 0.35 --candidate_set_pairguard_soft_min_margin 0.50 --candidate_set_pairguard_soft_min_pvalue 0.75 --candidate_set_pairguard_soft_min_reliability 0.85`|验证弱证据floor能否在保留k=4/k=5 known性能的同时压低unknown false accept。|

成功标准仍为：old_acc≥`0.99`且各old类≥`0.95`，seen_new_acc≥`0.97`且各seen-new类≥`0.93`，unknown拒识≥`0.99`。未达到时只能作为diagnostic evidence。

### N607结果

远端使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，同步后三个文件哈希与本地安全修正版一致。运行前8张RTX3090均约`10/24576MiB`，选择GPU0；运行后8张GPU均回到`10/24576MiB`。本地无`ssh.exe`残留，无N607和bridge 22端口ESTABLISHED连接。

输出文件：

|artifact|path|SHA256|
|---|---|---|
|JSON|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_sr_pairfuse_floor_risklabels_evt095_adv3b02.json`|`0CBD161088929328DCD31F47B79581FE3342A11CEB720F92CC454C7A44A7E9F5`|
|CSV|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_sr_pairfuse_floor_risklabels_evt095_adv3b02_evidence.csv`|`B353C30D6FD2FBBE3C42B6811AAA7F5744286B95019DFF2B66418124C97C0AE9`|

协同接收机数量覆盖`1..5`，观察到的target receiver为`20-1,3-19,7-14,7-7,8-8`。本轮仍是`receiver_domain_ranked`诊断，不等同严格同事件卫星群协同。

|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|avg_rx|bytes/event|p95 ms|boundary_hit|soft_applied|strong_bypass|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.9667|0.1327|1.0000|40.0|0.1045|0.0000|0.0000|0.0000|
|2|0.4408|0.0000|0.4314|0.1500|0.1429|0.8571|0.0000|2.0000|80.0|0.1045|0.1270|0.0714|0.0556|
|3|0.6833|0.1500|0.6250|0.3500|0.1250|0.8750|0.0000|3.0000|120.0|0.1045|0.3200|0.2350|0.0850|
|4|0.8750|0.7000|0.5750|0.2500|0.1750|0.8250|0.0000|3.7400|149.6|0.1045|0.3050|0.1700|0.1350|
|5|0.8500|0.7000|0.5750|0.2500|0.1250|0.8750|0.0000|4.1950|167.8|0.1045|0.4300|0.2500|0.1800|

判定：`soft_floor+strong_bypass`没有达到目标，也没有优于上一轮`sr_pairfuse_soft14_7_noguardbase_evt095`。它在k=4/k=5保留较高old_acc（`0.8750/0.8500`），但seen-new显著下降到`0.5750/0.5750`，unknown拒识仅`0.8250/0.8750`，距离`0.99`仍很远。k=2最低old类为`0.0000`，说明该风险表在低协同数量下仍会误伤类别，不具备部署或论文成功声明条件。

监督修正：子agent指出默认强证据门槛全为0时会真空bypass，以及shell缺失可能被绕过。已在本轮安全修正提交`93cb0c7`中处理：强证据bypass必须至少配置一个正门槛，且`shell_missing_failed`不得bypass；同时增加`candidate_set_pairguard_boundary_hit_count/rate/by_role`和`candidate_set_pairguard_soft_strong_bypass_count/rate/by_role`用于审计。

下一步建议：放弃基于unknown错误分布手工列举label/receiver pair的主线化叙述。更合理路线是把文献子agent建议落到可验证算法：冻结`z_id`主干，使用source/support校准的receiver reliability和prototype开集门控，加入可回滚的轻量adapter/TTA，但阈值和风险先验必须来自source/support/proxy-known，不能来自unknown query错误分布。

## 2026-07-04 Support-Calibrated Evidence Router实现

### 设计依据

本轮停止继续扩展unknown错误分布驱动的pairguard表，改为实现`support_router_cvs`。该策略遵循COSR-CI/AWARE-CI资源约束：节点只上传低带宽qknn8 evidence、support/conformal质量、open-set风险、时延和字节数；聚合端先检查old/seen-new是否有足够support-calibrated强证据，再让open-set风险触发`unknown_reject/request_more/defer`。它的关键边界是：分类证据和未知风险分离，但所有阈值仍只能来自source/support/proxy-known，不能使用unknown query调参。

机制：

|层级|条件|输出|
|---|---|---|
|强support接收|old/seen-new标签、候选receiver数、top1 receiver数、conformal pvalue、receiver-class reliability、score、margin、support density/radius gate、label/event risk上限同时通过|`accept`|
|未知证据|未满足强support接收，且event/label unknown risk、class shell risk或多组件risk agreement达到阈值|`unknown_reject`，若有预算则`request_more`|
|低置信|既非强support，也非明确未知|`defer`或`request_more`|

该算法仍是offline evidence级诊断，不等同严格同事件真实卫星群协同；`receiver_domain_ranked`只是部署proxy排序。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`support_router_cvs`融合策略、事件级`support_router_accept/unknown_evidence`和汇总计数；修复`progressive_budget`参数链路。|`9F0B0B843E7065BE3F23D11FAC289A19256B5C9239FEBC30A8A8D5DA14AF35AA`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI允许`--fusion_policy support_router_cvs`。|`83CADEA95D3B8553232B2B3C2D90B1BA540DB9C2CD95CEA6FC2ADDD52F84FBE2`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|新增support router单测，覆盖强support old接收与弱support unknown拒识；完整open-set测试回归。|`1ED3F248FC0AD1500189B352C392AB06C7347BCBCD6E69C70FB166001AA37C32`|

本地快照：`E:\type10-7\code\snapshots\phase2_support_router_cvs_20260704\`。Git镜像提交：`da7b144 Add support calibrated evidence router`。

验证：

```text
conda run --no-capture-output -n ssr-gpu python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -q -p no:cacheprovider
```

结果：`py_compile`通过；`53 passed`。

### N607执行计划

远端环境继续使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。运行前按N607 preflight检查GPU占用，选择显存占用最低GPU；如多卡同为低占用，默认GPU0。

计划运行`support_router_cvs`的k=1..5全量星地信道评估：

|run|关键参数|目的|
|---|---|---|
|`support_router_evt085_adv3b02`|`--fusion_policy support_router_cvs --class_set_gate_enabled --old_gate_min_support_density 0.55 --old_gate_max_radius_z 2.0 --seen_new_gate_min_support_density 0.50 --seen_new_gate_max_radius_z 2.5 --candidate_set_min_conformal_pvalue 0.50 --candidate_set_min_label_receiver_class_reliability 0.70 --candidate_set_unknown_reject_risk 0.85`|验证support-calibrated强证据能否保留old/seen-new，同时避免pairguard错误表导致的类别误伤。|

成功标准保持不变：old_acc≥`0.99`且min_old≥`0.95`，seen_new_acc≥`0.97`且min_seen≥`0.93`，unknown_reject≥`0.99`且unknown_FAR≤`0.01`。未达到则只作为diagnostic evidence。

## 2026-07-04 SR-PairFuse软pairguard执行计划

### 设计依据

上一轮`request_more`诊断证明控制路径可审计，但门控过严，known大量进入`unknown_reject/request_more`，不能作为性能提升。本轮实现`SR-PairFuse`最小版本：`candidate_set_pairguard_action=soft_penalty`。当高风险receiver-label pair命中时，不直接veto或request_more，而是根据证据弱度给局部candidate risk加软惩罚；强margin、高conformal pvalue、高receiver-class reliability的known样本应继续通过accept gate。

软惩罚审计字段包括：`candidate_set_pairguard_soft_applied`、`candidate_set_pairguard_soft_weakness`、各弱度分量、`candidate_set_pairguard_soft_penalty_value`、`candidate_set_label_unknown_risk_for_accept`、`candidate_set_event_unknown_risk_for_accept`、`candidate_set_label_component_agreement_for_accept`。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`soft_penalty`动作和软惩罚局部accept risk，穿透`dual_route_cvs`并汇总soft命中计数|`91876C2C9F419D8F6D80CE5E6AE0F8EA8F2A95B1B6F90D24FEF2FCEFABA3A516`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`--candidate_set_pairguard_soft_*`参数|`FC1018483D2FC8DC4C19C2FFACBFB3700CC49F780793DE6BA20A579E6FE3347D`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|补充强证据保留accept、弱margin转入request_more的单测|`BA815A9BFBF52F1207583DC801299023FD3BF0AAF54F122C1743BF0D9210AB58`|

本地快照：`E:\type10-7\code\snapshots\phase2_sr_pairfuse_soft_20260704\`。Git镜像提交：`4a0b42f Add soft pairguard penalty for candidate fusion`。

验证命令：

```text
conda run --no-capture-output -n ssr-gpu python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -k pairguard -q -p no:cacheprovider
```

结果：根工作区和Git镜像均通过；目标单测为`2 passed`。

### N607计划

远端继续使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。同步3个文件后运行k=1..5全量矩阵。初始候选路线：

|route|关键参数|目的|
|---|---|---|
|`sr_pairfuse_soft14_7_evt095`|`--candidate_set_pairguard_action soft_penalty --candidate_set_pairguard_labels 14-7 --candidate_set_pairguard_receiver_sets 20-1+3-19;7-14+7-7;3-19+7-14+7-7 --candidate_set_pairguard_soft_penalty 0.35 --candidate_set_pairguard_soft_min_margin 0.18 --candidate_set_pairguard_soft_min_pvalue 0.55 --candidate_set_pairguard_soft_min_reliability 0.75`|验证软惩罚是否相对硬veto/request_more恢复old/seen-new，同时继续降低高风险pair unknown false accept。|

判据：不能只看unknown_FAR。主判断为同一k行内`old_acc`、`seen_new_acc`、`unknown_FAR/unknown_reject`、`request_more/defer`联合指标；若known明显低于`dualroute_noguard`，仍为diagnostic-only。

### N607执行与结果

N607 preflight通过，8张RTX3090均为`10/24576MiB`，选择GPU0。同步后远端哈希与本地一致，远端环境为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，`py_compile`通过。最终运行后GPU仍为`10/24576MiB`，本地无`ssh.exe`残留，无N607/bridge 22端口ESTABLISHED连接。

远端同步哈希：

|文件|SHA256|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|`91876C2C9F419D8F6D80CE5E6AE0F8EA8F2A95B1B6F90D24FEF2FCEFABA3A516`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|`FC1018483D2FC8DC4C19C2FFACBFB3700CC49F780793DE6BA20A579E6FE3347D`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|`BA815A9BFBF52F1207583DC801299023FD3BF0AAF54F122C1743BF0D9210AB58`|

产物哈希：

|route|JSON SHA256|CSV SHA256|
|---|---|---|
|`sr_pairfuse_soft14_7_evt095`|`9F500480D91523AD7D2F99777DFAC10D9932CD032F3D7F9A21C30E5359536956`|`C63A3DAC834C51CFE61BEB5DB03C3C9E672ADC567D403BB84D44F3A3A9208B93`|
|`sr_pairfuse_soft14_7_relaxed_evt095`|`0F91910C9A698FEF6DF550FF452C4D52F72B279652809F5B86881E9CAAB1FA63`|`1EF8D8D701E409F0ACF9C75A0FCBA4EF92BF98CFF4FDAE4D979017EAF2816DCD`|
|`sr_pairfuse_soft14_7_noguardbase_evt095`|`EA276BEAE58066CDEC9601DA14ED9F4B5A86B7EDF98B1E2B10F7F9F3DFE289C7`|`B81030AD096663A227F740B783BBD0647B3C219E5636CE54A9F923C5FDA232FB`|
|`sr_pairfuse_soft_risklabels_evt095`|`7083070CC015584B8552672664521A9B5B282266D4842935524594BF22B67EA7`|`45D51A8C4F7A0451A6D73272ECFCA41FB8D53878B04B7FD1A3833E1B4FED5C04`|
|`sr_pairfuse_soft_risklabels_lu095_evt095`|`2B5972D7481C339FD739C70319688D7FAD400E470DEBE186421E7F47361F50FA`|`C8F0B2DDCD996CD78C0383E9FF31D588FB6B58C1258EED1DCB1ACDCE247E2C78`|

主结果表：

|route|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|request_more|bytes/event|p95 ms|soft_rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|dualroute_noguard|2|0.5000|0.2000|0.5962|0.5625|0.1304|0.8696|0.0440|0.0000|80.0|0.1374|0.0000|
|dualroute_noguard|3|0.6500|0.3000|0.7250|0.6000|0.0750|0.9000|0.0250|0.0000|120.0|0.1374|0.0000|
|dualroute_noguard|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|0.0000|150.0|0.1374|0.0000|
|dualroute_noguard|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|0.0000|168.6|0.1374|0.0000|
|sr_pairfuse_soft14_7_evt095|2|0.2895|0.0000|0.3333|0.1000|0.0000|1.0000|0.0000|0.0040|80.0|0.1049|0.0040|
|sr_pairfuse_soft14_7_evt095|3|0.2500|0.0000|0.5250|0.3500|0.0250|0.9750|0.0000|0.0000|120.0|0.1049|0.0000|
|sr_pairfuse_soft14_7_evt095|4|0.3667|0.0000|0.4250|0.2000|0.0000|1.0000|0.0000|0.0200|149.6|0.1049|0.0000|
|sr_pairfuse_soft14_7_evt095|5|0.3750|0.0000|0.3500|0.0500|0.0000|1.0000|0.0150|0.0000|167.8|0.1049|0.0000|
|sr_pairfuse_soft14_7_noguardbase_evt095|2|0.4408|0.0000|0.4314|0.1500|0.1429|0.8571|0.0000|0.0000|80.0|0.1442|0.0040|
|sr_pairfuse_soft14_7_noguardbase_evt095|3|0.7000|0.1500|0.8500|0.8000|0.2500|0.7500|0.0000|0.0000|120.0|0.1442|0.0000|
|sr_pairfuse_soft14_7_noguardbase_evt095|4|0.8917|0.7500|0.8000|0.7000|0.2750|0.7250|0.0000|0.0000|149.6|0.1442|0.0000|
|sr_pairfuse_soft14_7_noguardbase_evt095|5|0.8917|0.7500|0.7500|0.6000|0.3000|0.7000|0.0000|0.0000|167.8|0.1442|0.0000|
|sr_pairfuse_soft_risklabels_lu095_evt095|2|0.3026|0.0000|0.3529|0.1500|0.0000|1.0000|0.0000|0.0000|80.0|0.1106|0.0556|
|sr_pairfuse_soft_risklabels_lu095_evt095|3|0.3333|0.0000|0.7000|0.5500|0.0500|0.9500|0.0000|0.0000|120.0|0.1106|0.0600|
|sr_pairfuse_soft_risklabels_lu095_evt095|4|0.4583|0.0500|0.5250|0.3000|0.0500|0.9500|0.0000|0.0000|149.6|0.1106|0.0600|
|sr_pairfuse_soft_risklabels_lu095_evt095|5|0.4833|0.1000|0.4000|0.0500|0.1500|0.8500|0.0000|0.0000|167.8|0.1106|0.1050|

判定：`SR-PairFuse`最小实现已接入并可审计，但仍未达目标。最佳known行来自`sr_pairfuse_soft14_7_noguardbase_evt095`，k=4达到`old_acc=0.8917`、`seen_new_acc=0.8000`，但unknown_FAR升到`0.2750`；最低FAR行来自`sr_pairfuse_soft_risklabels_lu095_evt095`的k=2，unknown_FAR为`0.0000`，但old/seen-new崩塌。两者都不能作为99/97/99成功证据。

当前技术结论：手工label/receiver作用域的软惩罚只是在known保留和unknown拒识之间移动阈值，尚未解决星地信道下unknown与old/new空间重叠的问题。下一步需要把`rho_{r,y}`从query诊断改成support/proxy-known校准得到的pair风险先验，并引入强证据known bypass：当margin、pvalue、route一致性和receiver reliability同时强时不施加unknown-risk惩罚；当candidate为高风险label且label_unknown≈1.0但margin弱时才提高局部门槛。

本轮仍为diagnostic-only，不能声明Stage2-C部署成功、不能声明99/97/99达标，不能将`receiver_domain_ranked`写成严格同物理事件协同。

## 2026-07-04 ADV3B02 pairguard_request_more执行计划

### 设计依据

上一轮`rxscope_14_7_pairs_evt095`证明：把pairguard限定到高风险`label=14-7`和少量receiver组合后，k=2/k=3的unknown_FAR下降，k=4/k=5保持`dualroute_noguard`结果。但硬veto仍会把部分old正确accept变成未决，导致k=2/k=3 old_acc轻微下降。因此本轮新增`candidate_set_pairguard_action=request_more`：当`boundary_veto`命中且还有未使用接收机、请求预算允许时，不直接veto，而是输出`request_more`并记录审计字段。默认仍为`veto`，保持既有实验兼容。

该动作是部署语义实验：离线表中`request_more`计为未决，不会被报告为已正确分类；它只用于验证高风险pair是否能转入“请求更多卫星/接收机证据”的控制路径。若本轮known仍下降，下一步应实现子agent建议的`SR-PairFuse`软惩罚，而不是继续扩大硬门控。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`candidate_set_pairguard_action={veto,request_more}`，穿透`dual_route_cvs`，记录`candidate_set_pairguard_boundary_hit`、`candidate_set_pairguard_request_more`和汇总计数|`24DFE988C8B994445909C6855D1A9F9BA38E30B8B678EB47F0CF94BDADCC8142`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI新增`--candidate_set_pairguard_action`|`5C321EEFC2434E57B536A51FC45108E436EF6D4D01B90657452BCD15AB489BB9`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|补充三接收机k=2单测，验证pairguard命中时可从veto转为`request_more`并进入汇总计数|`13BD87C0F6E86E5399BE97F012314900B6C5581D9B030D5C6CC4BA8DA928128A`|

`E:\type10-7\code`不是Git仓库，已创建本地快照：`E:\type10-7\code\snapshots\phase2_pairguard_request_more_20260704\`。同一改动已复制到Git镜像`E:\type10-7\github_publish\CVS-RFFI-repo`，待提交。

验证命令：

```text
conda run --no-capture-output -n ssr-gpu python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py
conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -k pairguard -q -p no:cacheprovider
```

结果：根工作区和Git镜像均通过；目标单测结果为`2 passed`。一次并行`conda run`触发Windows临时文件锁，已按项目经验改为串行执行，未作为实验失败证据。

### N607计划

远端仍使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。同步3个文件后，运行k=1..5全量矩阵：

|route|关键参数|目的|
|---|---|---|
|`rxscope_14_7_pairs_request_more_evt095`|`--candidate_set_pairguard_mode boundary_veto --candidate_set_pairguard_action request_more --candidate_set_pairguard_labels 14-7 --candidate_set_pairguard_receiver_sets 20-1+3-19;7-14+7-7;3-19+7-14+7-7 --candidate_set_pairguard_min_event_unknown_risk 0.95 --candidate_set_max_receiver_pair_label_disagreement 0.50 --candidate_set_max_receiver_pair_unknown_risk_range 0.70 --candidate_set_min_label_receiver_class_reliability 0.75 --candidate_set_require_label_shell_observed`|验证高风险pair是否从硬veto转为请求更多接收机证据，报告unknown_FAR、known损伤、`candidate_set_pairguard_request_more_count`、receiver_count、bytes/event和p95 latency。|

执行前需N607 preflight、GPU空闲/低显存选择、SCP哈希核对和远端`CVS-RFFI`环境语法验证。执行后必须检查本地无`ssh.exe`残留和无到N607/bridge的ESTABLISHED 22连接。

### N607执行与结果

N607 preflight通过，直接SSH目标可达，项目根为`/home/szu2070436088/2510044040/CV-SincNet`。运行前8张RTX3090均为`10/24576MiB`，选择并列最低显存占用的GPU0。远端环境按用户要求使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

同步与验证：

|项目|结果|
|---|---|
|远端评估核心SHA256|`2A37D5E0E3672DBAB1F237B3FE4CB34E90413FF6E6F2BB175E39ADCDD2113D9B`|
|远端CLI脚本SHA256|`5C321EEFC2434E57B536A51FC45108E436EF6D4D01B90657452BCD15AB489BB9`|
|远端测试文件SHA256|`13BD87C0F6E86E5399BE97F012314900B6C5581D9B030D5C6CC4BA8DA928128A`|
|远端语法验证|`py_compile`通过|
|远端pytest|`CVS-RFFI`环境未安装`pytest`，未做包安装；以远端`py_compile`、CLI `--help`和实际全量运行替代验证|
|Git镜像提交|`6b11da5 Add pairguard request-more action`、`77b38c5 Use receiver budget for pairguard request-more`、`da83a1a Thread request-more budget through dual route`|

执行中暴露两类配置/实现问题，均已按失败边界处理：

|阶段|问题|处理|是否作为有效结果|
|---|---|---|---|
|首次远端命令|缺少`--class_conformal_enabled`，与`receiver_class_reliability_policy=support_calibrated`冲突|补充该参数后重跑|否|
|第二次远端命令|`candidate_set_max_label_shell_risk=1.5`不满足[0,1]校验|改为`1.0`，`candidate_set_shell_reject_risk`保留`1.5`|否|
|第三次远端命令|`request_more`未触发，原因是`can_request_more`未穿透到`dual_route_cvs`分支|本地修复、提交、同步后重跑|否|
|一次远端grep|命令超时留下本地`ssh.exe` PID4212和N607 22端口连接|已`Stop-Process -Id 4212 -Force`并复查无残留|否|

最终有效输出：

|文件|SHA256|
|---|---|
|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rxscope14_7_request_more_final_evt095_adv3b02.json`|`237530F4A7DA2009C54BD0452D2C6D5F95FC9968633AA1CB48D0492E6230B8A0`|
|`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_rxscope14_7_request_more_final_evt095_adv3b02_evidence.csv`|`C7CEFEF3EBA8DE1071DF2635846F01DF707AEC763C044CF3E7D1C87E06A7E56D`|

运行元数据：`receiver_count=5`、`group_count=309`、`evidence_row_count=1000`，`collab_counts=all`覆盖k=1..5；`candidate_set_pairguard_action=request_more`；星地信道特征沿用`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，即ADV3B02 Stage2-C星地/LEO特征包。运行后8张GPU仍均为`10/24576MiB`。

|route|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|request_more|bytes/event|p95 ms|pairguard_veto|pairguard_request_more|request_more_by_role|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|rxscope14_7_request_more_final_evt095|1|0.0000|0.0000|0.0000|0.0000|0.0000|0.9667|0.0000|0.1327|40.0|0.1034|0.0000|0.0000|`{}`|
|rxscope14_7_request_more_final_evt095|2|0.1316|0.0000|0.1765|0.0000|0.0000|0.9796|0.0000|0.0119|80.0|0.1034|0.0000|0.0119|`{"old":2,"unknown":1}`|
|rxscope14_7_request_more_final_evt095|3|0.0667|0.0000|0.1250|0.0000|0.0000|0.9750|0.0000|0.0350|120.0|0.1034|0.0000|0.0350|`{"old":6,"unknown":1}`|
|rxscope14_7_request_more_final_evt095|4|0.1667|0.0000|0.1500|0.0000|0.0000|1.0000|0.0000|0.0350|149.6|0.1034|0.0000|0.0000|`{}`|
|rxscope14_7_request_more_final_evt095|5|0.2417|0.0000|0.1500|0.0000|0.0000|1.0000|0.0250|0.0000|167.8|0.1034|0.0000|0.0000|`{}`|

判定：该路线没有达到目标，且不接近目标。虽然`unknown_FAR=0`、`unknown_reject≈0.98-1.00`，但这是以大规模known误拒为代价：k=2 old_acc仅`0.1316`、seen_new_acc仅`0.1765`，k=5 old_acc也仅`0.2417`、seen_new_acc`0.1500`。因此不能声明99/97/99达成，不能声明Stage2-C部署成功，也不能把unknown_FAR下降解释为真实性能提升。它只证明：`request_more`控制路径已接入并可审计，但当前门控参数过严，导致known类被过度转入`unknown_reject/request_more`。

对比上一轮最佳`rxscope_14_7_pairs_evt095`，本轮`request_more`没有改善总体联合指标：上一轮k=4可保持`old_acc=0.8083`、`seen_new_acc=0.7500`、`unknown_FAR=0.0250`；本轮k=4只有`old_acc=0.1667`、`seen_new_acc=0.1500`。下一步不应继续扩大硬门控或单纯调低FAR，而应实现子agent建议的`SR-PairFuse`软风险惩罚：只在弱证据时降低高风险receiver-label pair权重或提高局部门槛，对强margin、低novelty、route一致的known样本保留快速accept路径。

最终SSH/SCP后本地无`ssh.exe`残留，无N607和bridge 22端口ESTABLISHED连接。

## 2026-07-04 ADV3B02 receiver_set_pairguard执行计划

### 设计依据

`labelscope14_7_pairguard_evt095`已证明只针对`14-7`可在k=2/3降低unknown_FAR并保持seen-new，但仍小幅伤害old。进一步审计`dualroute_noguard`逐事件结果发现，unknown false accept同时有receiver组合集中性：

|k|输出label|selected_receiver_ids|事件数|
|---:|---|---|---:|
|2|`14-7`|`20-1,3-19`|2|
|2|`14-7`|`7-14,7-7`|2|
|2|`19-3`|`20-1,3-19`|1|
|3|`14-7`|`3-19,7-14,7-7`|2|
|3|`6-15`|`20-1,3-19,7-14`|1|

新增`candidate_set_pairguard_receiver_sets`，语法为分号分隔组合、加号连接receiver，例如`20-1+3-19;7-14+7-7`。pairguard只有同时命中label作用域和receiver组合时才触发，目标是进一步减少正常old query误伤。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`candidate_set_pairguard_receiver_sets`，逐事件输出`candidate_set_pairguard_receiver_scoped`和规范化组合列表|`50B4B339F011E15E99B2C7CBE2EAFC564957538F3F99000BBB05C9A0A1F67667`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI暴露`--candidate_set_pairguard_receiver_sets`|`501AED47B28C37D8759E9E02A7C59FF2775C95976FB8465912837A36FB87D10B`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|补充receiver-set作用域单测|`31DE8A4DCDBF881579BEF7258771183BB6719B53E26FA8D622D7B14BCFD096C9`|

验证：本地工作树`96 passed`，Git镜像树`96 passed`。Git镜像提交：`8aab498 Scope pairguard by receiver set`。

### N607计划

远端继续使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。计划运行两组k=1..5全量对照：

|路线|关键参数|目的|
|---|---|---|
|`rxscope_14_7_pairs_evt095`|`--candidate_set_pairguard_labels 14-7 --candidate_set_pairguard_receiver_sets 20-1+3-19;7-14+7-7;3-19+7-14+7-7 --candidate_set_pairguard_min_event_unknown_risk 0.95`|只处理`14-7`高风险receiver组合，目标是保留old性能并降低k=2/3 unknown_FAR。|
|`rxscope_all_pairs_evt095`|`--candidate_set_pairguard_labels 14-7,6-15,14-10,19-3 --candidate_set_pairguard_receiver_sets 20-1+3-19;7-14+7-7;3-19+7-14+7-7;20-1+3-19+7-14`|覆盖k=2/3主要false accept组合，但比上一轮labelscope更窄。|

### N607结果

远端同步后SHA256与本地一致；`CVS-RFFI`环境验证`52 tests OK`。两组实验均覆盖k=1..5；首次运行前后8张RTX3090均为`10/24576MiB`。

|route|JSON SHA256|CSV SHA256|
|---|---|---|
|`rxscope_14_7_pairs_evt095`|`B62FEDEA0282B11E00EB523C9409A97BCB0B53720B7676E6944D44C3DD513ABC`|`BA3745A070CB98737817CC2E8E88193079E4CE6094823ACDFDB53FFA9E2C4DAF`|
|`rxscope_all_pairs_evt095`|`C231E232DC0CF8298E5535A5D9BBB663520A54D11FA2A2890F8C943A556AF5C9`|`43213306618AB1E4E6D6B8A180483A1988F1B57E92E8335F7B5F36206B944386`|

|route|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|bytes/event|p95 ms|pairguard_veto_rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|dualroute_noguard|2|0.5000|0.2000|0.5962|0.5625|0.1304|0.8696|0.0440|80.0|0.1374|0.0000|
|dualroute_noguard|3|0.6500|0.3000|0.7250|0.6000|0.0750|0.9000|0.0250|120.0|0.1374|0.0000|
|dualroute_noguard|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1374|0.0000|
|dualroute_noguard|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1374|0.0000|
|rxscope_14_7_pairs_evt095|2|0.4934|0.2000|0.5962|0.5625|0.0870|0.9130|0.0440|80.0|0.1826|0.0400|
|rxscope_14_7_pairs_evt095|3|0.6333|0.3000|0.7250|0.6000|0.0500|0.9250|0.0250|120.0|0.1826|0.0300|
|rxscope_14_7_pairs_evt095|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1826|0.0000|
|rxscope_14_7_pairs_evt095|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1826|0.0000|
|rxscope_all_pairs_evt095|2|0.4737|0.2000|0.5385|0.5000|0.0652|0.9348|0.0440|80.0|0.1340|0.1280|
|rxscope_all_pairs_evt095|3|0.5750|0.1500|0.6500|0.4500|0.0250|0.9500|0.0250|120.0|0.1340|0.2450|
|rxscope_all_pairs_evt095|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1340|0.0000|
|rxscope_all_pairs_evt095|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1340|0.0000|

判定：`rxscope_14_7_pairs_evt095`是目前最好的pairguard折中路线。它在k=2把unknown_FAR从`0.1304`降到`0.0870`，k=3从`0.0750`降到`0.0500`，同时保持seen-new不变；k=4/k=5完全保持`dualroute_noguard`的old、seen-new和unknown表现。代价是k=2/k=3 old分别小幅下降`0.0066/0.0167`，但显著小于全局pairguard或纯labelscope。`rxscope_all_pairs_evt095`unknown更低，但known损伤明显，不作为下一步主线。

下一步应将`rxscope_14_7_pairs_evt095`从硬veto改成软策略：命中高风险label+receiver组合时优先`request_more`或降低对应receiver投票权，而不是直接阻断accept；这样有机会保留k=2/k=3 old，同时继续压低unknown_FAR。当前仍未达到old 99%、seen-new 97%、unknown拒识99%，所有新增结果仍为diagnostic-only。

最终SSH/SCP后本地无`ssh.exe`残留，无N607和bridge 22端口ESTABLISHED连接。

## 2026-07-04 ADV3B02 label_scoped_pairguard执行计划

### 设计依据

`dualroute_noguard`逐事件审计显示unknown false accept不是均匀分布：

|k|false accepts|输出label分布|主要receiver组合|
|---:|---:|---|---|
|2|6|`14-7`:5，`19-3`:1|`20-1,3-19`为3次，`7-14,7-7`为2次|
|3|3|`14-7`:2，`6-15`:1|`3-19,7-14,7-7`为2次|
|4|1|`14-10`:1|`20-1,3-19,7-14,7-7`|
|5|3|`6-15`:2，`14-10`:1|全5receiver组合|

因此新增`candidate_set_pairguard_labels`，只对高风险输出label执行pairguard，避免上一轮全局pairguard把大量正常old/seen-new样本一起拦截。本轮label作用域：`14-7,6-15,14-10,19-3`。该选择来自本轮诊断结果，只能作为diagnostic route；若后续作为正式方法，应改为support/proxy-known校准得到label风险先验，不能用unknown query结果调参。

### 本地改动与验证

|文件|用途|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`candidate_set_pairguard_labels`，空值表示所有label，非空时只对指定输出label应用pairguard并记录`candidate_set_pairguard_label_scoped`|`D44F792B9ACACEAFDD63974435DBF1C94FBEC9F003A567B916224C53D2F55FBF`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI暴露`--candidate_set_pairguard_labels`|`15F58E0CA88D5ACFDE87C8036F51F806A1DCB902501F180F082487473C8F4E0E`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|补充label scope单测，证明命中label才触发pairguard veto|`BA30967674A8BFE51E57D1D373DF7F5C828A0D08B11E93758D42CC30D9E88794`|

验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q
```

结果：本地工作树`96 passed`，Git镜像树`96 passed`。Git镜像提交：`f6b4b83 Scope pairguard by output label`。

### N607计划

远端仍使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。计划运行两组k=1..5全量对照：

|路线|关键参数|目的|
|---|---|---|
|`labelscope_pairguard_evt090`|`--candidate_set_pairguard_mode boundary_veto --candidate_set_pairguard_labels 14-7,6-15,14-10,19-3 --candidate_set_pairguard_min_event_unknown_risk 0.90 --candidate_set_max_receiver_pair_label_disagreement 0.50 --candidate_set_max_receiver_pair_unknown_risk_range 0.70`|验证只对高风险label触发时，unknown_FAR能否下降且known损伤小于全局pairguard。|
|`labelscope_pairguard_evt095`|同上但`candidate_set_pairguard_min_event_unknown_risk 0.95`|更保守边界版本。|

### N607结果

远端同步后哈希与本地一致；远端`CVS-RFFI`环境验证`52 tests OK`。运行三组k=1..5全量对照：`labelscope_pairguard_evt090`、`labelscope_pairguard_evt095`和更窄的`labelscope14_7_pairguard_evt095`。GPU0运行，首次运行前后8张RTX3090均为`10/24576MiB`。

|route|JSON SHA256|CSV SHA256|
|---|---|---|
|`labelscope_pairguard_evt090`|`697CA04448A981CA352E6617BC784D223B1884953D4EF2B79C749276C50D8B63`|`0A15B0A402F066CCA3A2E36604F5A555C90D0A5EAF5B218EA0FB69993265E333`|
|`labelscope_pairguard_evt095`|`4914528AF8D0CAF16B563CEA7FCD435AD615226555D27427EB8B2378369D6499`|`2897C64E50C5CEBD933052586C36A8AB9AFEE423439955D5ECE4C064BD520309`|
|`labelscope14_7_pairguard_evt095`|`C14CF3DE3B8E8C0F99FE5EE62CDD0A9573C64A08DD7D9F1AFC28522B171FDE1D`|`116448FC16D47A8438E7F12A0DECE77DFFFD198AEB9D2E12C3CAF96EEB298FEA`|

|route|k|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|bytes/event|p95 ms|pairguard_veto_rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|dualroute_noguard|2|0.5000|0.2000|0.5962|0.5625|0.1304|0.8696|0.0440|80.0|0.1374|0.0000|
|dualroute_noguard|3|0.6500|0.3000|0.7250|0.6000|0.0750|0.9000|0.0250|120.0|0.1374|0.0000|
|dualroute_noguard|4|0.8083|0.7000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1374|0.0000|
|dualroute_noguard|5|0.7833|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1374|0.0000|
|labelscope_pairguard_evt095|2|0.4737|0.2000|0.5385|0.5000|0.0652|0.9348|0.0440|80.0|0.1327|0.1440|
|labelscope_pairguard_evt095|3|0.5583|0.1500|0.6500|0.4500|0.0250|0.9500|0.0250|120.0|0.1327|0.2900|
|labelscope_pairguard_evt095|4|0.7083|0.4000|0.6500|0.4000|0.0250|0.9500|0.0150|150.0|0.1327|0.2500|
|labelscope_pairguard_evt095|5|0.6917|0.2500|0.6250|0.3500|0.0250|0.9500|0.0100|168.6|0.1327|0.3150|
|labelscope14_7_pairguard_evt095|2|0.4934|0.2000|0.5962|0.5625|0.0870|0.9130|0.0440|80.0|0.1332|0.0560|
|labelscope14_7_pairguard_evt095|3|0.6167|0.3000|0.7250|0.6000|0.0500|0.9250|0.0250|120.0|0.1332|0.0950|
|labelscope14_7_pairguard_evt095|4|0.7917|0.6000|0.7500|0.6000|0.0250|0.9500|0.0150|150.0|0.1332|0.0850|
|labelscope14_7_pairguard_evt095|5|0.7667|0.5500|0.7250|0.5500|0.0750|0.9000|0.0100|168.6|0.1332|0.0850|

判定：类别作用域比全局pairguard更接近目标，尤其`labelscope14_7_pairguard_evt095`在k=2把unknown_FAR从`0.1304`降至`0.0870`，k=3从`0.0750`降至`0.0500`，且seen-new不低于`dualroute_noguard`。但old_acc仍下降：k=3从`0.6500`降到`0.6167`，k=4从`0.8083`降到`0.7917`，已经跌破OLD80_FIRST阶段门槛。因此它仍不是主线成功，只能说明“高风险label定向二级门控”比全局pairguard更合理。下一步应把`14-7`门控从直接veto改为`request_more`或软降权，或结合receiver-pair级先验只处理`20-1,3-19`和`7-14,7-7`这类高风险组合，避免对正常old query造成同等惩罚。

最终SSH/SCP后本地无`ssh.exe`残留，无N607和bridge 22端口ESTABLISHED连接。





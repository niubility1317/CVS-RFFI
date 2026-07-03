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

# V218历史U1/head_only分叉证据边界审计

日期：2026-09-11。范围：本地已有文件只读；全文解析两行各200条epoch JSONL及完整stdout，读取旧配置、激活报告、训练完成receipt、旧代码和本次8步回放结果。未访问远端、未训练、未评分、未加载checkpoint。本文新增一个审计文档，不修改历史日志或原结论数值。

## 结论

**VERIFIED：历史两行最迟在E1结束时已呈现基础loss及源验证准确率差异；无法定位第一次不同的训练step、算子或参数，也无法证明根因。**已有记录的最小观察粒度是epoch。V218的“历史首次分叉原因明确”应标为`UNRESOLVED_HISTORICAL_FIRST_DIVERGENCE`，不能被V2独立辅助事务修复、负测通过或合成8步等价替代。

**VERIFIED：旧正式两行均AMP开启、max_grad_norm=0；没有两行AMP开关或全局clip阈值不匹配的证据。**本次旧V1合成回放则为FP32、8步、seed392002，不能覆盖历史AMP、seed392005、9800次尝试及真实源IQ。

**VERIFIED：旧共享optimizer/scaler/有限性检查路径确实允许辅助异常影响主干跳步；这是已定位的控制面耦合路径。UNKNOWN：它是否造成历史E1最早差异。**E1两行均49/49成功，首次非有限梯度跳步均到E4才出现，因此不能把历史已记录的跳步当作E1差异的直接起因。

## 已读证据

- [旧完整报告](E:/type10-7/automation_reports/CV-SincNet/core90_cross_response_s392005_20260911_r1/CORE90_CROSS_RESPONSE_FULL_REPORT_20260911.md)：§5、§7、§9、§10及附录配置。
- [U1完整epoch日志](E:/type10-7/automation_reports/CV-SincNet/core90_cross_response_s392005_20260911_r1/completed_detail/U1/metrics_epoch.jsonl)、[head_only完整epoch日志](E:/type10-7/automation_reports/CV-SincNet/core90_cross_response_s392005_20260911_r1/completed_detail/head_only/metrics_epoch.jsonl)：各200行，完整解析。
- [U1 stdout](E:/type10-7/automation_reports/CV-SincNet/core90_cross_response_s392005_20260911_r1/completed_detail/U1/U1.log)、[head_only stdout](E:/type10-7/automation_reports/CV-SincNet/core90_cross_response_s392005_20260911_r1/completed_detail/head_only/head_only.log)：完整读取；两行均`init=scratch`、`seed=392005`、`amp=1`、`max_grad_norm=0.00`。
- 两行同目录的`cross_response_activation.json`、`phase1_training_completion_receipt.json`及`phase1_terminal_status.json`；未加载权重。
- [V2实施计划§5](CORE90_CROSS_RESPONSE_V2_IMPLEMENTATION_PLAN_20260911.md)：明确要求从相同状态、ticket、各类RNG复制，并逐阶段定位第一次差异；这是待验收要求，不是已完成根因报告。
- [旧V1本次FP32主轨迹回放](E:/type10-7/logs/core90_v2_acceptance_20260911/legacy_fp32_01/trajectory_main_reanalysis.json)：`EXACT_MAIN_TRAJECTORY_PARITY`，两分支各40事件、8步、atol=rtol=0、无AMP溢出；比较主optimizer状态时排除预期存在的辅助参数组/矩状态。

## 匹配到了什么，尚未匹配什么

|项目|可证实状态|边界|
|---|---|---|
|训练seed、data_seed|均392005；cross config除3个预期开关外一致|相同seed不等于已比较初始化张量/RNG全过程|
|数据划分|两行`source_split_receipt`完整字典相同；L/U/V=6300/56700/27000；源RX1/3/4/6/8、day1/2/3一致|receipt中的既有索引摘要可支持划分一致，不能证明每一步读取/增强张量一致|
|基础配置|stdout及epoch共有配置字段未发现非路径性的基础超参差异；均AdamW、lr=0.0002、weight_decay=0.0001、E200、130+70|未存逐step已解析argv/输入快照；不把共有epoch字段比较冒充完整运行状态逐值核验|
|响应配置|仅`response_enabled:false→true`、`head_only:false→true`、`predictor_mode:additive→bilinear`不同|U1没有响应头；head_only有双线性辅助头，这是预期实验差异|
|AMP/clip|两行全部200轮`amp=True`，`train_grad_clip_active=0`、`train_grad_clip_limit=0`|全局clip机制存在于旧代码，但历史两行没有启用；不能作为已发生的历史clip根因|
|采样|两行128候选、均匀采样、9800批、完整曝光6300唯一物理记录；loader_workers=0|最终覆盖450/626和唯一曝光总数不是逐step ticket等价证据|
|实际角色成功计数|U1各rotation为9796/9788/9792/9784；head_only为9796/9796/9788/9788|成功更新数不同会影响此汇总；不能仅由汇总逆推出最早输入差异|
|本地逐step证据|检查的历史completed_detail目录只有epoch/stdout、激活、终态及评分产物，没有训练ticket/增强张量/logits/逐参数梯度/scaler逐步轨迹|最终prediction_manifest是推理query记录，不能替代训练ticket；本文不复算其评分|

旧代码已有独立loader generator，以及用`torch.random.fork_rng`和`seed+913`隔离辅助头初始化的实现，见[旧integration.py:109](E:/type10-7/code/snapshots/core90_cross_response_20260911_wt/code/cvsrffi/cross_response/integration.py:109)、[旧integration.py:140](E:/type10-7/code/snapshots/core90_cross_response_20260911_wt/code/cvsrffi/cross_response/integration.py:140)。因此不能无证据声称“忘记seed”或“初始化必然消耗了主RNG”。也未取得足以把差异归因于CUDA并发/非确定算子的算子级证据。

## 历史最早可观察差异

|E1字段|U1|head_only|
|---|---:|---:|
|train_loss_tx_labeled|12.585292952401298|12.585053599610621|
|train_loss_domain_labeled|2.710446309070198|2.710280160514676|
|train_loss_orth_labeled|0.0005079832632446244|0.0005049469906893768|
|源val_tx_acc（%）|27.977777777777778|27.637037037037036|
|optimizer_step_applied|1.0|1.0|
|skipped_nonfinite_grad / loss|0 / 0|0 / 0|

这里引用基础loss而非加入辅助响应后的总loss；总目标不同本身不能证明主干分叉。E1基础项和源验证结果不同，支持轨迹/状态未保持完全一致，但缺初始模型及逐step快照，无法区分差异始于初始化、输入/增强、前向、反向还是更新，也不能给出第几步及首个参数名。

完整200轮非有限梯度跳步记录如下；每个列出的epoch均1/49跳步：

- U1：E4、5、8、21、63、80、119、126、141、168；成功9790/9800。
- head_only：E4、5、7、21、63、80、102、121；成功9792/9800。

这些记录证明后续数值事件不同。它们没有保存第几个batch、触发的参数名、辅助/主干来源或当时scale，因而不能把“10次对8次跳步”解释为已经锁定辅助头溢出的历史因果。最早基础差异已在E1，早于首次已记录跳步。

## 旧报告为何提示追查

旧完整报告给出U1/head_only最终clean=79.3389%/77.6712%、三LEO均值=63.1215%/62.7311%，以及不同的成功步数和源训练轨迹。head_only按定义不应直接给主干响应梯度，这些差异足以提出控制面与轨迹审计问题。原报告同时明确“相同seed不保证……逐位相同轨迹”，未给出首次不同step或根因。V2计划将其具体化为应完成的首次分叉定位，不能把计划中的排查目标改写成历史已经证明的原因。

## 已定位的旧代码耦合与本次验收的正确口径

[旧train_ssdg.py:7601](E:/type10-7/code/snapshots/core90_cross_response_20260911_wt/code/SSDG/train_ssdg.py:7601)把响应辅助参数加入主optimizer。其[10398附近反向/10402之后更新路径](E:/type10-7/code/snapshots/core90_cross_response_20260911_wt/code/SSDG/train_ssdg.py:10398)依次调用统一backward、`scaler.unscale_(optimizer)`，检查主模型和`cr_runtime.finite_gradients()`；任一非有限导致整个主更新不执行，随后共用`scaler.update()`。只有`max_grad_norm>0`时才把主模型与辅助参数一起clip。本次两行此开关为0。

因此可分别验收：①共享事务路径存在，V2已实施隔离并可由NaN/Inf负测验证；②本次真实CORE90合成短回放主轨迹相等；③历史E200第一次分叉和具体根因仍不可重建。尤其旧V1 FP32也能8步精确相等，说明不能宣称“旧实现一定从第一步就分叉、V2才使其首次相等”。

若未来已有授权允许进一步历史重建，需要匹配历史AMP、真实源batch/角色/增强和初始状态，或取得历史已保存逐step证据；应从现有证据继续，不以当前target结果选择重跑条件。本文不要求新审批，不启动补实验，不降低V218原验收标准：**历史根因项保留未完成，已实现的可观测性和隔离修复单独记为通过。**

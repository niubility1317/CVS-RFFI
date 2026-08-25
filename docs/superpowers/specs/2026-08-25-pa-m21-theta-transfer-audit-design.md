# PA-M2.1 Theta Transfer Audit Design

## 1.目标与结论边界

本设计实现用户批准的《PA-M2.1独立因子复审与后续优化实施计划（修订版）》。本轮冻结Core90和旧C4 challenge encoder，不修改当前q，不重复或覆盖既有A/B实验。核心问题是：当前PA sidecar的theta是否在独立support bank和权重独立审计条件下，具有超出q-only的跨packet、跨receiver和跨day TX增量；只有该问题通过后，才评价truth-blind条件专家能否把oracle空间转化为安全分类增益。

本轮只能声明`V_audit_retro`对新C1′/C4′权重独立。原C4架构、损失和超参数已经受历史完整`V_select`结果影响，因此不能称为研究历史完全未见集或最终确认集。未来完整路线必须在方法设计前永久预留`V_final`。

## 2.两阶段状态机

阶段A为`M2.1A_THETA_TRANSFER_AUDIT`，执行C1′/C4′独立replay、多fold F0–F9、q条件probe、M0精确检索、近重复审计和leave-one-TX-out residual审计。

- `A_PASS`：C4′跨receiver四项条件全部通过，且head/candidate/satellite敏感性不反转。
- `A_PARTIAL`：C1′、C4′均显示PA迁移，但C4′相对C1′条件化增量不足。
- `A_FAIL`：C4′ F3不优于F0或其他必要条件失败。

阶段B为`M2.1B_TRUTH_BLIND_EXPERT_GATE`，只在`A_PASS`后运行。`A_PARTIAL`或`A_FAIL`必须产生合法分析artifact并将阶段B标为`NOT_RUN_A_GATE`，不能被误记为系统技术失败。

## 3.数据角色与权重独立性

保持项目协议的`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`不变。只在方法内部把原`V_select`按capture block拆为：

- `V_select_fit`：约65%，仅用于C1′/C4′选epoch和外部support bank；
- `V_audit_retro`：约35%，只用于一次阶段A/B审计；
- guard blocks：不进入任一子角色。

分组键为`(TX,RX,day,eq,capture_block)`，其中`capture_block=floor(sig_i/B)`。在`B∈{10,20,25}`中选择能让每个主要TX×RX×day cell保留多个block且满足覆盖要求的最小可行值；选择只依赖元数据分布，不读取q、theta、标签性能或审计指标。默认目标比例为65/35，若cell覆盖不足可固定为70/30，并在预登记报告中记录。

同一capture block只能属于一个子角色。相邻角色block之间保留一个guard block。`sig_i`只表示同一TX/RX/day/eq组合内的时序代理，禁止解释为跨receiver同步物理发射ID。

## 4.近重复审计

运行时收集但不发布样本级`base_index`、`sig_i`、TX/RX/day/eq、capture block、规范化IQ精确摘要和固定随机投影摘要。只发布聚合结果：

- role间`base_index`交集和exact duplicate数量；
- 近重复相似度分位数；
- 相似度大于0.999和0.995的比例；
- 跨role最近`sig_i`间隔；
- guard block覆盖和丢弃数量。

样本级q、theta、embedding、原始IQ和逐样本预测流不得进入Git交付。

## 5.C1′与C4′独立replay

Core90和旧C4 challenge encoder完全冻结。从同一固定随机初始化模板构造C1′和C4′的response、pool、operator classifier与holdout predictor。

|候选|q条件|挑战匹配|DiD|Holdout|目的|
|---|---:|---:|---:|---:|---|
|C1′|常量|否|否|否|同容量非条件控制|
|C4′|当前冻结q|是|是|是|当前完整条件侧路|

二者使用相同参数量、训练step、训练seed、`L_s`训练输入和`V_select_fit`选模规则。首轮保留旧C4的operator source accuracy选模语义，只改变选模数据范围；synthetic LEO operator accuracy、oracle gain、holdout NMSE和pair coverage只读记录，不参与选模。M0关系本轮只审计，不参与训练。

## 6.Sidecar V3契约

新输出schema为`cvs.phase1.ccoi_pa_sidecar.v3`。`architecture_config`必须包含：

```json
{
  "input_length": 256,
  "token_length": 64,
  "stride": 16,
  "q_dim": 32,
  "challenge_hidden_dim": 64,
  "codebook_size": 48,
  "response_dim": 64,
  "operator_dim": 64,
  "pa_channels": 64,
  "num_classes": 6,
  "num_domains": 7,
  "holdout_anchor_policy": "all_nonoverlap_folds",
  "conditioned": true,
  "pa_map_contract": "core90_pa_token_map_v1"
}
```

加载时同时验证schema、必填配置、state shape、input length、token length、stride、PA map contract、类别/域数量并执行strict state load。旧V2只允许在`legacy_migration_mode=true`时读取challenge encoder并立即生成V3配置，不能直接作为新实验输出。

## 7.四fold holdout

审计覆盖全部4个非重叠anchor fold。每个审计单位为`(base_index,fold_id)`。每个fold分别构造support theta、holdout q和PA target，验证raw support/holdout无重叠。主指标为4个fold NMSE的macro平均，同时保留逐fold结果和显式position ID。任何只运行fold0的正式路径必须失败。

## 8.独立support bank与F0–F9

目标只来自`V_audit_retro`，F2–F6 support只能来自`V_select_fit_support_bank`。关系先由metadata硬约束，候选再由固定seed和稳定键确定；主结果禁止fallback，选择过程不得读取q。

|Row|输入与关系|用途|
|---|---|---|
|F0|audit q，theta置零|q-only|
|F1|同packet非重叠support theta|同包上限|
|F2|同TX/RX/day、不同packet|跨packet|
|F3|同TX、跨RX、同day|跨receiver|
|F4|同TX/RX、跨day|same-receiver跨day|
|F5|异TX、同RX/day|TX特异性负控制|
|F6|同TX、跨RX，以固定PA统计匹配|简单物理challenge匹配|
|F7|真实同步同challenge|当前`UNAVAILABLE`|
|F8|正确theta配显式异TX/异packet q|q-response对应关系|
|F9|训练目标均值|无条件基线|

F6使用RMS、PAPR、四阶/六阶幅度矩、包络差分、幅度一阶自相关和正则化memory-polynomial条件数，不使用学习q。每个关系运行3个candidate mapping seed。

核心比较统一限制在`F2∩F3∩F5`共同anchor集合，同时报告all-valid、common-anchor、每cell样本数和invalid比例。

## 9.阶段A判据

C4′ F3必须同时满足：

1. 相对F0误差下降至少5%，TX×RX×day分组bootstrap 95%CI下界大于0；
2. 相对F5误差下降至少5%，CI下界大于0；
3. 相对F2退化不超过10%；
4. F3覆盖至少80%，每个TX至少覆盖两个跨receiver关系，主要TX×目标RX cell达到预登记最小样本数。

C4′相对C1′的F3改善至少3%且CI下界大于0，才可声明当前challenge conditioning具有独立价值。若C1′、C4′均通过但差异不足，则结论为`KEEP_PA_OPERATOR/STOP_CURRENT_CHALLENGE_CONDITIONING`。

首轮sidecar主seed为1个，F head seed为3个，candidate mapping seed为3个，synthetic satellite使用主seed和1个敏感性seed。至少2/3 head seed、2/3 mapping seed方向一致，两个satellite seed不得发生结论反转。

## 10.q、M0与码本诊断

q probe增加固定RX/day内TX、固定TX/day内RX、固定TX/RX内day、token-shuffle sequence、permutation-invariant DeepSets和ordered sequence MLP。报告ordered减shuffled差异。

M0只使用同一物理source样本的clean/satellite精确对，候选池限制为同TX、同RX、同day、同fold。报告Recall@1、Recall@5、median rank、MRR、exact-vs-other距离AUC、clean-satellite theta距离和exact-pair margin。

码本继续报告token hard occupancy、packet dominant occupancy、position occupancy、transition matrix和soft effective codes，但不优化hard均衡，也不作为晋级条件。

## 11.Leave-one-TX-out residual

公共模型按6个TX整类留出：训练折只含其他TX，对被留出TX预测公共响应。报告HR相对F0、residual TX/RX/day probe、between-TX距离、same-TX cross-RX距离和每fold训练TX集合。任何被留出TX出现在common model训练折都必须失败。HR不要求超过同包F1。

## 12.阶段B安全gate

阶段B采用有界残差融合：

```text
logits_final = logits_base + g(x) * eta * Clip(s*logits_operator - logits_base)
eta ∈ {0.05,0.10,0.20}
```

gate只读部署可用的base/operator margin与entropy、JS divergence、top1分歧、RMS、PAPR、PA条件数、spectral null ratio、clipping ratio、SNR proxy、残余CFO/相位不稳定度和challenge coverage。禁止true TX、true receiver、day和审计标签作为输入。

在`V_cal`上按TX×RX×day×capture block group-CV拟合低容量多项logistic regression，预测rescue和harm概率，以`p_R-lambda_h*p_H>tau`启用有界纠正。固定`eta/tau/lambda_h/clip`后，在`V_audit_retro`只评估一次。

阶段B通过条件：三个source synthetic LEO平均提升至少0.20pp且分组bootstrap CI下界大于0；clean下降不超过0.10pp；最差receiver下降不超过0.05pp；selected weighted utility为正；coverage达到预登记下限；leave-one-source-receiver CV多数receiver效用为正。

## 13.协议、安全与产物

- 全流程source-only，`target_or_query_access=false`。
- A/B旧run、08d3提交和旧artifact只读，不重启、不覆盖、不复现。
- 新run ID和output root不可覆盖；一个run只有一个launch owner。
- 本地TDD、真实checkpoint无query smoke、一次独立P0/P1审查和最小预登记完成后立即发布，不增加白名单外gate。
- 保留一次release归档本地/远端SHA比较，不增加逐文件SHA、seal或receipt链。
- 阶段A低性能是科学结论，不是技术停止理由。
- 完成后提交聚合JSON和最终中文报告，不提交样本级特征、IQ或预测流。

## 14.聚合artifact

正式输出包括`split_manifest.json`、`sidecar_architecture_c1p.json`、`sidecar_architecture_c4p.json`、`sidecar_training_summary.json`、`duplicate_audit.json`、`q_conditional_probe.json`、`m0_exact_pair_retrieval.json`、`factor_matrix_c1p.json`、`factor_matrix_c4p.json`、`loto_residual_audit.json`、`gate_calibration_summary.json`、`gate_audit_summary.json`、`decision_manifest.json`和`final_report.md`。阶段B未运行时，两个gate文件仍输出明确的`NOT_RUN_A_GATE`状态。

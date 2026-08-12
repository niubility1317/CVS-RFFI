# Phase1 CLIC target LEO-weak 12臂预测v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_target_prediction_20260812_v1`。
- 当前状态：`LOCAL_VERIFYING_G_V3_BINDING / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：从既有target confirmation v2缓存唯一派生VALIDATED_ONCE收据和known-test配置，封装一个IQ-only目标包，并使用predictor artifacts v2的F1—F6×C／G共12个冻结predictor逐行零适配前向，生成12份不可变prediction。
- 用户边界：不要求与ADV3B02共用同一封存目标包、物理行、received-IQ字节或seed；只要求训练数据配置和known-test数据配置相同。C／G本身仍使用同一个本轮IQ-only package。

## 固定测试数据与语义

- cache manifest：`runs/phase1_clic_target_confirmation_20260812_v2/cache/cache_set.json`，production loader已验证总3120行，每scene1040，registered-known240、unknown800，三scene物理ID互斥、single observation、finite、零clean/query/truth/fit/update/selection访问。
- scope=`phase1_clic_target_confirmation`，协议=`p2_min_v1`；receiver=`20-1`；days=`0,1,2`；registered-known为共同6类池，formal评分按每fold predictor封存local4，inactive2显式排除且不得转unknown。
- test semantics：`code/configs/phase1_clic_target_test_semantics_20260812_v1.json`。其`input_len=256`和`float32`与缓存实物一致；三scene、信道模型来源和采样参数来自目标缓存builder实际使用的`training_controls.SAT_CHANNEL_SCENARIO_CONFIGS`／`simplified_leo_residual_weak_v1`；`zero_adapt=true`。
- 指标口径预注册为：target-known含unknown／defer为错误；按scene／receiver／class／day及三scene等权／sample-pooled报告域泛化；unknown显式拒识分子只计`decision=unknown`，defer不计；同时报告AUROC-unknown、AUPR-out和FPR95。

## 输入、输出与访问边界

- C predictors：`runs/phase1_clic_predictor_artifacts_20260812_v2/F{1..6}C_CLIC12/c_predictor_state.json`；这些工件已在v2系统性G失败前成功封存，v2报告保留其SHA和技术证据。
- G predictors：`runs/phase1_clic_g_bundles_20260812_v3_safe_pack/F{1..6}G_CLIC12/g_deployment_bundle.zip`。G-only v3在相同training v5／clean v4／source-LEO v4和相同F1G—F6G矩阵上，仅把模型状态封装从旧Torch／NumPy ABI桥改为严格有界的连续CPU tensor字节复制；N607正式串行run已6／6 exit0，6份bundle均经production verify和单行reload通过。
- 输出根：`runs/phase1_clic_target_prediction_20260812_v1`；日志根：`logs/phase1_clic_target_prediction_20260812_v1`；启动前必须不存在且不可覆盖。
- validation输出：`validation/{known_test_config.json,validator_receipt.json}`；package输出：`sealed_target/iq_only_package/{manifest.json,received_iq.npz}`；truth仅在`sealed_target/truth_sidecar.json`，不得传给publisher。
- predictions：`predictions/F{1..6}{C,G}_CLIC12.prediction.json`共12份；每份应有3120行且forward_count=3120，C／G绑定相同package SHA，分别绑定独立predictor SHA、source rule、local4顺序和训练配置SHA。
- publisher只能接收predictor-state路径、IQ-only package路径和output路径；禁止target fit／update／retry／selection，四项计数必须全0。

## 运行、停止与后续评分

- 运行将分两段：先CPU执行validation和IQ-only package封存；再启动6个CPU fold worker，每个worker严格依次执行同fold C、G，线程上限为2。冻结runtime当前明确使用CPU，因此不虚构GPU映射；正式launcher唯一调用，retry=`NO`。
- 启动后核对outer／worker PID、CWD／cmdline／run-root和日志增长；至少2fold出现同一确定性异常且未产完整prediction，或发生协议访问、错误hash／checkout、覆盖风险时，只停止本run精确进程并保留证据；不得按性能值停止。
- prediction完整后，单独truth-side scorer才可首次打开truth sidecar；评分必须同时给出target-known DG、unknown拒识和三scene域泛化。ADV3B02对比只接受训练／known-test配置逐字段等价且分层crossed证据完整的不可变原件；当前未找到该原件时，先封存12份prediction，不伪造非劣结论。
- 本地launcher：`code/scripts/launch_phase1_clic_target_prediction12_v1_20260812.sh`，SHA-256=`1FC5E465CFB414F8B37254BF6BB6B7F9BDF3CB677D1B144D2BCB8A74DE043067`；测试语义JSON SHA-256=`416371DB57C08E6877F2DA49E73C62A241F857304D52480C884D8F6F86A84A04`。`bash -n`和dry-run通过且精确14行，即validation1＋package1＋prediction C6／G6，C6仅绑定predictor artifacts v2、G6仅绑定G-only safe-pack v3；禁止truth／ADV／score／fit／update／role／query／selection／retry参数为0；launcher专测`1／1`通过。
- G-only v3已技术闭合：F1—F6 bundle bytes分别为4605944、4606470、4604494、4605507、4605656、4605261；全部`state_origin=checkpoint_model_exact`、state rebuild／reload通过、source-only且zero clean／query fit，无target成员。本target run只消费这些冻结工件，不重复封装或临时配置。
- 待完成：更新后的C-v2／G-v3双根绑定fresh P0／P1审查；N607唯一启动、12份预测工件QA及报告回填。

## 预期工件技术表

|候选|方法类别|训练配置|测试配置|target LEO-weak|unknown rejection|域泛化|适配／反馈|当前结论|
|---|---|---|---|---|---|---|---|---|
|F1C—F6C|C：raw phase control|各fold封存source local4，training v5|共同confirmation v2，RX20-1／day0,1,2／三scene|每臂3120行，逐行一次forward|prediction封`decision`与`e_unknown`，待truth-side score|待按scene／RX／class／day、三scene等权与sample-pooled评分|fit/update/retry/selection全0|待运行|
|F1G—F6G|G：complex local invariant curvature|各fold封存source local4，training v5|与C完全相同的同一IQ-only package|每臂3120行，逐行一次forward|同上|同上|同上|待运行|

本表只预注册输出合同，不含任何性能数值；old／seen-new／unknown、coverage／defer、DG等只有在12份prediction闭合且独立scorer合法打开truth后才填写。当前无匹配ADV3B02原件，不以伪造baseline阻断本轮prediction生成。

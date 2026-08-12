# Phase1 CLIC target LEO-weak 12臂预测v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_target_prediction_20260812_v1`。
- 当前状态：`ARTIFACTS_COMPLETE / FORMAL_INVOCATION=1 / RETRY=NO / NO_PERFORMANCE_RESULT`。
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
- 已完成：C-v2／G-v3双根绑定静态审查、N607唯一正式启动、12份预测工件技术QA及本报告回填；本run不打开truth、不执行评分、不读取性能字段。

## 预期工件技术表

|候选|方法类别|训练配置|测试配置|target LEO-weak|unknown rejection|域泛化|适配／反馈|当前结论|
|---|---|---|---|---|---|---|---|---|
|F1C—F6C|C：raw phase control|各fold封存source local4，training v5|共同confirmation v2，RX20-1／day0,1,2／三scene|每臂3120行，逐行一次forward|prediction封`decision`与`e_unknown`，未打开truth|性能评分N/A（本run不读性能）|fit/update/retry/selection全0|技术闭合6/6|
|F1G—F6G|G：complex local invariant curvature|各fold封存source local4，training v5|与C完全相同的同一IQ-only package|每臂3120行，逐行一次forward|同上，未打开truth|性能评分N/A（本run不读性能）|同上|技术闭合6/6|

本表只预注册输出合同，不含任何性能数值；old／seen-new／unknown、coverage／defer、DG等只有在12份prediction闭合且独立scorer合法打开truth后才填写。当前无匹配ADV3B02原件，不以伪造baseline阻断本轮prediction生成。

## N607落地与静态证据

- 冻结Git：`610df85421376063a45cfd5aeecf3bf716f48eef`；归档未包含共享`conversation_index/`。归档bytes=`267653120`，SHA-256=`28CAEBC7504369FC0DFF7A4D4C89F4EA962C76CDB58234897AADD7141518782B`；SCP恰1次，远端bytes/SHA闭合，随后原子切换到`/home/szu2070436088/2510044040/releases/phase1_clic_target_prediction_20260812_v1_610df854`。
- 远端物理归一化：launcher raw/canonical=`1FC5E465CFB414F8B37254BF6BB6B7F9BDF3CB677D1B144D2BCB8A74DE043067`；evaluator raw=`BD703E16A6D9DF9BFD7C638B7427F8AE2A8DBD8932FD2B64A781709431C4C323`、canonical=`035DB2FF5BBCFF3DCFE021A62ADA6F25EAF71F7FD8AE4BC399F89EBBE1DB4EE4`；target helper raw=`6CCFF2A9334178B7F17CA52B6A4A6748BC8B1670A794E41D75DD5B986CB5C50B`、canonical=`AC60E085B1A397F3F2215476EE74463E18BEEFD18965AE5891546D78B5B0A2E6`；top-level CLI raw=`9416EF19826CFE5AE622AD69CEFBAFE9E4EAC9DB6B2983FCCC7F640C55FFE64`、canonical=`D24CF3F007260775F05B206BB565EE33A92A52B2A08DC661C7F0DBE567A0F6DB`；G exporter raw=`B60484233F1946C504D7470CC444084A6437FA3D62D966D07DE2F5685A73B400`、canonical=`463FE8416FF68EA71F59F6433E6B1D79CCE91C038F7944A9962BE92D67569C47`；semantics raw=`130D2D7C0EC15FC1E2054477C2C79D5C216D81548F5DD73AF4F0D28ADE43D1E0`、canonical=`416371DB57C08E6877F2DA49E73C62A241F857304D52480C884D8F6F86A84A04`。
- 静态检查：4个Python入口`py_compile`通过；evaluator、top-level target CLI、module target CLI、G exporter `--help`通过；launcher `bash -n`通过；semantics JSON解析通过；`bash launcher --dry-run`精确14行（validation1/package1/C6/G6）；禁止truth/ADV/score/fit/update/role/query/selection/retry参数计数均为0。
- 输入绑定：target cache manifest为`cvs_leo_weak_iq_cache_set_v2`；C descriptor与G safe-pack v3均按F1—F6逐项SHA固定并在启动前production loader/verify抽查F1通过。正式命令唯一一次：`nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_target_prediction_20260812_v1_610df854/code/scripts/launch_phase1_clic_target_prediction12_v1_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_target_prediction_20260812_v1_outer.out 2>&1 &`；outer PID=`2796861`，CPU-only，CUDA未占用，retry=`NO`。

## 运行工件与技术QA

- validation闭合：`known_test_config.json` 1490 bytes、`validator_receipt.json` 1487 bytes。IQ-only package仅含`manifest.json`（2362 bytes）与`received_iq.npz`（6198753 bytes），package SHA singleton=`64373f39e45cb7dcfc5b5d0a989e43ba17f3ff16d85577631434e65e6235fff5`；rows=3120，三scene均1040，IQ shape=`(3120,2,256)`且finite，physical scene/row闭合。publisher未接收truth sidecar路径，`truth_sidecar_opened=false`。
- 12份prediction均技术闭合：每份rows=3120、`forward_count=3120`、`target_fit_rows=0`、`update_rows=0`、`retry_count=0`、`selection_count=0`，package raw/logical SHA、predictor/source local4/train-config bindings均通过，未执行production forward重跑，未读取或解释`decision`／`e_unknown`。

| fold/arm | prediction bytes | prediction SHA-256 | rows | forward_count | package/predictor binding | truth opened | fit/update/retry/selection |
|---|---:|---|---:|---:|---|---|---|
|F1C|851822|`5e81ecee60cc773a1be7331a19b348f38c4b371af80e2b8adf24df8e56fc3647`|3120|3120|PASS|false|0/0/0/0|
|F1G|851623|`7fcd98ec85b847248fa6789111b7123908b310419fd433b2279040ce47eef6d9`|3120|3120|PASS|false|0/0/0/0|
|F2C|851674|`f5e5aa021ab3b428efdb88189c377c2722b589b2182abfc51b3861efbf32d125`|3120|3120|PASS|false|0/0/0/0|
|F2G|851774|`066b994ac8319d6cb8b05eec91faea8d8277ab5178cbd9e803df980d4d58d94b`|3120|3120|PASS|false|0/0/0/0|
|F3C|851667|`eef3f633249be7964050dfef7fb6ae27ac226f2d394aa93ed9728f57f94b951b`|3120|3120|PASS|false|0/0/0/0|
|F3G|851806|`c135236616e2af50ac90b34e0fdaa4c81c33ffc7561ec231d71531ed3916de9f`|3120|3120|PASS|false|0/0/0/0|
|F4C|851485|`b0567a2546c9203cd32a47586c3f618782d7a4dc01ad5c62b3c2cd3852070013`|3120|3120|PASS|false|0/0/0/0|
|F4G|851772|`8bb541db2842f610b8c191a4a929febd8c33f8685278dcb9362e034b94f1cf2f`|3120|3120|PASS|false|0/0/0/0|
|F5C|851487|`854ee862833b7554c726e12cf632fb78c883c3fc06e63e3f02c4c94bdc6ca92f`|3120|3120|PASS|false|0/0/0/0|
|F5G|851832|`90e8095eaf4a6f88faa1fb9ae9d621c53a8ca8f4c7d17f16389ced6dd3786af0`|3120|3120|PASS|false|0/0/0/0|
|F6C|851558|`06db03ab3bd2fdf520d1d51a455e3900da9a64f217469892c7c971831311d582`|3120|3120|PASS|false|0/0/0/0|
|F6G|851520|`9dd3acc421d4cfa3ad660f714a2f186458016f07af646ab93f7f85c7081ccb50`|3120|3120|PASS|false|0/0/0/0|

- 运行收尾：`pids_target_prediction6.tsv`保留6个fold worker PID（2797006、2797007、2797009、2797011、2797013、2797014），均已退出；6份fold日志各302 bytes，validation/package日志分别599/286 bytes；outer文件0 bytes、SHA=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。最终无run-owned进程，GPU0—7均0%利用率/1 MiB，SSH/SCP/TCP22客户端清零。

## 最终封存

`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。本run只证明IQ-only package、12份逐行prediction及其C/G、package、source local4、train-config技术绑定和零适配计数闭合；未打开truth sidecar，未执行scorer，未读取、比较或报告任何性能字段。后续如需评分，必须由独立truth-side scorer在新的受控步骤中进行。

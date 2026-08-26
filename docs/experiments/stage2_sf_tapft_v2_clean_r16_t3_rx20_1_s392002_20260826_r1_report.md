# SF-TAPFT V2 R0实验报告

## 当前结论

数据与模型输入已经准备完成，真实checkpoint无query smoke和4折OOF+全support重训smoke均通过。当前最高状态为`LOCAL_VERIFIED / LANDED / SMOKE_PASS`，下一步立即启动完整R0性能验证；尚无完整性能结果，不进入R1。

## 预登记运行合同

|字段|值|
|---|---|
|run ID|`stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r1`|
|候选|SF-TAPFT V2 R0 clean reference|
|Git提交|`ebfb63571b25a1eea23f50f202b3381f294b99b2`|
|模型|`ADV3B02_CORE90_SOFT_E200`|
|checkpoint SHA-256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|Phase1 deployment binding|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_core90_bundle_20260827_r1_techfix1/final/deployment_binding.json`|
|deployment binding SHA-256|`685745d0b213a0f6a164ee763815f4a6b840c95d0abcddabee1002a05670832c`|
|target support|receiver`20-1`、seed`713101`、旧类6类、K=10、共60条|
|support包|`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2b_sclba_a_t5t25_s713101_20260824_v1/input/support_rx20_1_k10_clear_smoke.npz`|
|capsule|`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`|
|split|`stage2b-rx20-1-seed713101-before-support-prefix`|
|协议|`p2_min_v1 / VALIDATED_ONCE`|
|方法预算|rank16；A/B/C阶段步数`500/1500/2500`；4折OOF；选择后从fresh checkpoint对全60条support做4500步final-step refit|
|GPU|物理GPU0，命令内`CUDA_VISIBLE_DEVICES=0`、运行设备`cuda:0`|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v2_clean_r16_t3_20260827_ebfb6357`|
|release SHA-256|`a28ec33946f4bf9bc0d3e2137aa499e4d0cc5390aaa1a35ca8b9fd316cb08265`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r1`|
|log|`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r1.out`|
|预期artifact|`sf_tapft_clean_single_bundle.pt`、`selection.json`、4折OOF指标和全support refit证据|
|停止规则|checkpoint/bundle谱系不一致、错误capsule/split/K/类序、输出覆盖、协议越权、确定性执行故障或无法产生合法bundle；不得因中途性能低而停止|

## 已完成证据

- 正式CORE90 Phase1 bundle状态为`FORMAL_PHASE2_ELIGIBLE`，模型195个tensor精确加载，missing、unexpected、shape mismatch均为0。
- 复用旧Phase2 support包，不重建、不重验；其协议状态、capsule、split、receiver、K和物理样本划分均未改变。
- release归档本地到N607 SHA-256一致，远端核心脚本编译通过。
- 直接3步无query smoke：60条support、15个许可更新参数、总步数3，source/query/truth均未打开。
- 4折OOF+全support重训smoke：每类10条、`fold0_as_final=false`、最终角色为`clean_single_full_support_refit`，合法V2 bundle已生成；source/query/truth均未打开。
- 3步smoke诊断值：frozen OOF BA=`0.6041667163`，adapted OOF BA=`0.7777778506`，adapted NLL=`0.8774776086`；仅证明链路闭合，不作为完整R0性能结论。

## 完整运行命令

在远端release checkout中，以`CVS-RFFI`环境运行`run_target_only_progressive_nested.py`，加载正式配置`stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826.json`，输出到上述不可覆盖run root，设备为`cuda:0`，折数为4。

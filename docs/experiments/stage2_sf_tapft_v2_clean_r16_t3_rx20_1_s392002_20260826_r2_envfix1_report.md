# SF-TAPFT V2 R0环境修复重启报告

## 当前结论

前一run在进入Python前因解释器路径错误退出且未读取数据。本run使用已现场验证的`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`重新启动；模型、数据、方法、seed和阶段预算均不变。

## 预登记运行合同

|字段|值|
|---|---|
|run ID|`stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r2_envfix1`|
|候选|SF-TAPFT V2 R0 clean reference|
|代码与配置提交|`ebfb63571b25a1eea23f50f202b3381f294b99b2`|
|解释器|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch`2.1.0+cu121`|
|target support|receiver`20-1`、seed`713101`、旧类6类、K=10、共60条|
|协议|`p2_min_v1 / VALIDATED_ONCE`；capsule和split不变|
|方法预算|rank16；A/B/C阶段`500/1500/2500`步；4折OOF；全60条support做4500步refit|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v2_clean_r16_t3_20260827_ebfb6357`|
|GPU|物理GPU0，运行设备`cuda:0`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r2_envfix1`|
|log|`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r2_envfix1.out`|
|预期artifact|`sf_tapft_clean_single_bundle.pt`、`selection.json`、4折OOF指标和全support refit证据|
|停止规则|数据/query越权、错误绑定或split、输出覆盖、确定性执行故障、无法产生合法bundle；不得因中途性能低而停止|

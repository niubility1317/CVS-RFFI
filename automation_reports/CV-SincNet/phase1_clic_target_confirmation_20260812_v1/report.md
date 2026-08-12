# P1-CLIC目标域LEO盲态确认v1

## 当前状态

- 实验ID：`phase1_clic_target_confirmation_20260812_v1`
- 日期：2026-08-12
- 操作者：Codex主控；N607唯一runner待委派
- 状态：`LOCAL_VERIFICATION_IN_PROGRESS / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`
- 目标：一次性构建共同的`p2_min_v1`目标域LEO weak registered-known／unknown缓存，随后让F1—F6的C／G共用同一IQ-only包完成零适配推理；隔离scorer报告域泛化、真实unknown拒识和配置匹配ADV3B02比较。

## 固定数据与矩阵

|字段|固定值|
|---|---|
|receiver|`20-1`|
|registered-known union|`14-10,14-7,20-15,20-19,6-15,8-20`|
|unknown|`1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6`|
|days|`0,1,2`|
|物理样本|每TX 120条，三scene各40条；每physical ID只产生一份received IQ|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|seed|dataset=`713101`；scene=`7131010/7131011/7131012`|
|目标预测|F1—F6×C／G，共12份；同一target package；每样本恰一次forward|
|更新权限|target端训练／适配／阈值拟合／温度校准／选择／重试均为0|

共同union6只定义共享缓存。每fold的正式registered-known计分宇宙严格取对应source-policy local4；inactive2显式排除且不得转为unknown。unknown20与registered union6互斥。

## 指标与门

每个fold×arm×scene同row保存：registered-known overall、macro、min-class、min-receiver、min-day、accepted accuracy、false reject、defer；unknown保存AUROC、AUPR-out、FPR95、显式拒识率、误接收率、安全处置率及TX／receiver／day最差切片。全体unknown及每个scene的`decision=unknown`比例均须至少70%，`defer`单列且不进入分子。

ADV3B02无需与CLIC共享capsule、物理样本、received-IQ或seed；只要求训练数据配置与known测试数据配置相同。baseline与candidate均按相同fold local4、receiver、day和scene语义cell比较；禁止union6边际替代local4切片。

## 本地实现与验证

- cache spec：`code/configs/phase1_clic_target_confirmation_20260812_v1.json`
- cache launcher：`code/scripts/launch_phase1_clic_target_cache_20260812.sh`
- source clean v3 launcher：`code/scripts/launch_phase1_clic_postfreeze_source12_v3_20260812.sh`
- target evaluator：`code/evaluate_phase1_clic_target_leo.py`
- C predictor sealer/runtime：`code/cvsrffi/phase1_clic_target_leo.py`
- G bundle：`code/export_phase1_clic_deployment_bundle.py`

已完成的窄验证：新cache scope／精确角色／三scene物理互斥／NPZ TOCTOU为5项GREEN；真实`CLEAN.export→bundle`闭环GREEN；target相关34项GREEN；C／G real forward与cache builder禁用`Tensor.numpy`／`torch.from_numpy`后4项GREEN；69 unknown+31 defer按207／300写入失败门而不误计defer。完整回归和独立终审仍在执行，未达到发布门。

## N607预注册

- 普通账号直连preflight；8卡可见后才落地。
- 所有代码先形成干净Git commit和不可覆盖release；SCP后逐文件SHA闭合。
- cache阶段正式入口：`bash <release>/code/scripts/launch_phase1_clic_target_cache_20260812.sh`；GPU0；日志`logs/phase1_clic_target_confirmation_20260812_v1/target_cache.out`。
- clean v3入口：`bash <release>/code/scripts/launch_phase1_clic_postfreeze_source12_v3_20260812.sh`；12进程按GPU表固定分配。
- 预期cache工件：3个scene NPZ、`cache_set.json`、PID表和日志；总行数26×120=3120，每scene1040，registered 6×40=240，unknown 20×40=800，三scenephysical ID两两不交。
- 技术停止：错误checkout/hash、覆盖风险、协议越权或至少两个独立row在prediction前出现同一确定性异常指纹时停止精确run-owned进程；不因任何性能值停止或选择。
- retry：`NO`。技术故障使用新run ID修复，不覆盖本run。

## 风险与待闭合项

- N607的Torch2.1+NumPy2旧桥曾导致原生崩溃；当前直接cache、C forward、G reload路径均已有禁桥回归，但仍需真实F1烟测。
- 当前N607无已证实匹配的新scope cache，也无满足新crossed local4 RX/day合同的ADV3B02 immutable reference；必须新建cache，并为正式非劣评分补齐匹配reference。缺reference不得阻塞IQ-only预测封存，但不得发布非劣结论。
- 本报告当前不含性能数值，不作晋级结论。

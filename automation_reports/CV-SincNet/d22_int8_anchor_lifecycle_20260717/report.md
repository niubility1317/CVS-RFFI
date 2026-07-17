# D22 int8锚定prototype生命周期与快速适配

## 启动记录

- experiment ID：`d22_int8_anchor_lifecycle_20260717/support_screen_v1`
- 时间：2026-07-17，operator：Codex
- 目标：在不打开query的前提下，使用用户特许的历史Phase1 int8聚合质心，对B0–B4候选执行K10/new5三场景support-fold筛选；同时评估旧类域适应、新类注册、逐类floor和遗忘代理。
- 假设：固定received-IQ的`z_id160+FFT96+RF32`轻头负责域适应与新类注册，int8旧类身份方向与同IQ direct logits只通过`max_old`保持算子执行旧类内部纠错，可改善旧类floor而不改变old/new组边界。
- 对比：B0 identity target prototype、B1 int8 max-old、B2 int8+direct max-old、B3合法单IQ轻头、B4轻头+max-old。

## 版本与本地验证

- Git分支：`codex/cvs-rffi-release-20260626`。
- 资源协议提交：`c4aa43fd`；正式档80k/30epoch/50step/256KB，探索档120k/45epoch/75step/384KB。
- runner SHA256：`7e46db1e99ac40f4e9d7679dcb7f668553d928a0672a7bcf07022383949c8553`。
- CIAF模块SHA256：`f46c5007cb1c0279bf2b27169ad79989eba908f32658c5a4d7f819916381aeb1`。
- Phase2合同模块本地SHA256：`3b65707f91eb7012b5cd67bd572aa7d786c07ef4414b4b4a25139732a71a0b7b`；远端旧SHA256为`b9e236014aa3cfefe2d37d10def25133fd18b6f0923fcb9edf2059a41c6515e3`，须从本地Git工作树同步至`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/phase2_runtime_contract.py`。
- SOMP-H runtime本地SHA256：`343a0ddcdb200351a8099cde2c2a9bbdd6a4eb661e54d322ecbd6901a5e720ee`；远端旧SHA256为`9565ded5b2511c173ce313fd31eb2b2b313be329a5b0c57143634daac57076b6`，须同步至`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/somph_predictor_runtime.py`。
- class binding SHA256：`bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f`。
- 历史int8组件NPZ SHA256：`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest SHA256：`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。
- 本地`ssr-gpu`验证：`tests/test_phase1_int8_prototype_bundle.py tests/test_stage2_ciaf.py tests/test_run_d19_support_only_ciaf.py`共28项PASS。
- N607对应runner/CIAF哈希与本地一致；GPU0～7启动前无实验计算进程。

## 数据与权限

- receiver：`20-1`；开发seed：`713101`；K=10；旧6类；真实seen-new5类；场景为`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- 输入为D18已密封的before/after `enrollment_only` support package；只读取registered support IQ/labels、同IQ固定FFT/RF表征、同runtime direct logits和不可变int8聚合组件。
- `query_opened=false`，不生成query prediction或正式accuracy，不读取truth/scorer，不允许role、真实批类别数、quota或global assignment。
- 历史int8组件仅用于`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`；若选出正路线，仍需重建checkpoint+int8共同bundle和method lock。

## N607启动信息

- 工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（PyTorch2.1.0+cu121）
- GPU：0
- launcher：`code/scripts/launch_d22_int8_support_screen_20260717.sh`
- 远端launcher：`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_d22_int8_support_screen_20260717.sh`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d20_int8_maxold_fftrf_20260717/support_screen_v1.log`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d20_int8_maxold_fftrf_20260717/output/support_screen_v1`
- PID：`3303460`；最终状态：`LOCAL_PROTOCOL_REPAIR_REQUIRED`，support打开前执行兼容失败。
- 启动后GPU0进程曾落地；PID现已退出，未形成目标output。
- 预期产物：`RECEIPT.json`、`selection.json`、`support_audit.json`、`training_log.jsonl`、`resource_audit.json`。

## 成功与停止条件

- support门必须在15个场景×fold单元上逐类非劣于B0，并严格改善最差旧类floor；B4还须相对B3保持新类逐类完全不变并改善旧类floor。
- 若所有候选失败，回退B0并记录负证据，不打开query。
- 若候选通过，只获得下一步共同bundle重建资格，不构成正式性能或125矩阵成功。

## 风险与完成后检查

- 当前历史组件只有int8质心和scale，没有radius/offset；不得伪造source半径。
- 检查完整training log、各fold逐类floor、旧类score列锁定、B4 new预测不变、状态/MAC/显存和`query_opened=false`。
- 任务结束后更新本报告的PID、状态、结果表、解释与下一实验。

## PID3303460完成核验：support打开前执行兼容失败

- 核验时间：2026-07-17 16:22–16:25 CST；核验方式：N607 direct只读短连接。
- PID3303460已经退出，但不是runner完成或artifact-complete。
- 失败发生在support materialization的manifest预检阶段；精确错误为`SOMP-H bundle manifest exact schema mismatch`。
- 预期output路径不存在`training_log.jsonl`和`selection.json`，B0–B4全部未开始，不得解释为性能负结果。
- 进一步比对确认D18密封manifest具有当前47字段；`somph_predictor_bundle.py`远端与本地一致，但它导入的`phase2_runtime_contract.py`远端仍是12字段旧版，导致manifest总合同仅33字段。同步当前26字段合同后，manifest严格schema应恢复47字段；这不是放宽schema校验。
- 另一个历史`attempt2`来自冻结source副本，进入support提取后因NumPy2/PyTorch2.1的`torch.from_numpy`ABI不兼容失败；当前Git版runner已用DLPack桥替代该路径，重跑前须完成本地窄测试。

|候选|support执行状态|场景/fold结果|逐类结果|结论|
|---|---|---|---|---|
|B0|未开始|无|无|不可评价|
|B1|未开始|无|无|不可评价|
|B2|未开始|无|无|不可评价|
|B3|未开始|无|无|不可评价|
|B4|未开始|无|无|不可评价|

重跑使用`support_screen_v2`独立output/log/PID，并为该run设置独立`PYTHONPYCACHEPREFIX`，从当前源码重新编译模块；不删除历史cache、不覆盖v1失败日志。重跑前必须验证远端新cache进程看到47字段校验器，且继续保持query、clean/source与成员allowlist边界不变。

### support_screen_v2结果与v3修复

- v2 PID`3310020`同样在support打开前退出，manifest严格47字段校验已通过，随后失败于`SOMP-H method lock contract failed: ['phase2_contract']`。
- 只读核验显示密封method lock自身恰好含26字段且与当前合同逐项一致；失败来自远端`somph_predictor_runtime.py`仍是旧SHA`9565…`，其expected lock使用旧合同。
- formal policy声明的三项code closure中，`somph_predictor_bundle.py`与`somph_runtime_trust.py`远端已匹配；`stage2_predictor_bundle.py`受既有签名authority约束，不在没有重建authority的情况下追随本地新版本。
- v3仅同步当前已提交的`somph_predictor_runtime.py`和独立v3 launcher，再做无IQ method-lock原子预检；不修改D18 package、seal、policy或authorization。

### support_screen_v3结果与v4闭包补齐

- v3 PID`3312866`通过manifest与method-lock严格校验，随后在support打开前失败于`signed policy envelope binding drift`。
- 逐字段比较显示11个envelope绑定字段中仅`code_closure_sha256`不一致：签名envelope要求`b0b7f2c2…9606f`，远端闭包为`47096709…cd06`。
- 本地当前三成员闭包使用`stage2_predictor_bundle.py` SHA256`bb27beaa…44aa9`，精确重建签名要求的`b0b7f2c2…9606f`；远端旧成员SHA256`8bf20101…bc05`是唯一差异。
- v4同步该已签名闭包成员及独立v4 launcher，不修改、重签或绕过package/seal/policy/authorization；先核验闭包SHA精确等于envelope，再启动support-only筛选。

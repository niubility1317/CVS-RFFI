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
- 预期产物：`RECEIPT.json`、`selection.json`、`support_audit.json`、`training_log.jsonl`、`resource_audit.json`。

## 成功与停止条件

- support门必须在15个场景×fold单元上逐类非劣于B0，并严格改善最差旧类floor；B4还须相对B3保持新类逐类完全不变并改善旧类floor。
- 若所有候选失败，回退B0并记录负证据，不打开query。
- 若候选通过，只获得下一步共同bundle重建资格，不构成正式性能或125矩阵成功。

## 风险与完成后检查

- 当前历史组件只有int8质心和scale，没有radius/offset；不得伪造source半径。
- 检查完整training log、各fold逐类floor、旧类score列锁定、B4 new预测不变、状态/MAC/显存和`query_opened=false`。
- 任务结束后更新本报告的PID、状态、结果表、解释与下一实验。

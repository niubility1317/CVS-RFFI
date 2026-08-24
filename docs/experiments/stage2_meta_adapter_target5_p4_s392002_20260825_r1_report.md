# CVS_META_ADAPTER_TRI_R4_V1 P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_p4_s392002_20260825_r1`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 实现提交：`8d07f752e5093766f31edab7fdc97159c60d70f1`
- Phase1闭合提交：`26ad71643b35d5fbcab5f98308bb11ea24d19c65`
- 固定本次计划与预登记提交：`c489dc8df100ea6c7cd79ad135f9a0f07725d2d0`

## 候选与矩阵

- 候选：Phase1 r5选择的P4 FOMAML+Meta-SGD bundle；不新增、训练或持久化D92式协方差、LDA或分类头。
- Target5固定receiver：`20-1`；单seed：`392002`。
- 五个同row operating point：`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每个包含`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，共15个prediction row。
- Phase2仅使用`p2_min_v1`、`VALIDATED_ONCE`固定LEO received IQ和合法target support标签；不读取source／clean样本、source cache、query真值或query角色。query只用于独立逐样本推理，不更新模型状态。
- 原编码器内实际可训练参数8670／1058341，占比0.8192%；正式更新3步。

## 固定输入

- P4 bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P4/selected_meta_bundle.pt`
- 冻结原型：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P4/frozen_prototypes.npz`
- 工厂计划：`configs/stage2_meta_adapter_target5_p4_s392002_20260825_r1.json`
- 复用既有`VALIDATED_ONCE`数据，不因adapter、bundle、checkpoint或资源预算变化重验。

## N607执行预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/checkout`
- 复用已验证release归档：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_a465e329.tar.gz`；本地／远端SHA256均为`8becfa7e4a8e68aa3bae1c8668c81b2b8f6bb6af47617c0cc1b9349203e2c349`，相关生产入口远端编译已通过。
- 工厂输出root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_p4_s392002_20260825_r1`
- prediction输出root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_p4_s392002_20260825_r1.out`
- expected artifacts：工厂级`matrix_config.json`和`factory_receipt.json`；每row的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`和`receipt.json`；truth-last `score.json`；矩阵级`target5_summary.json`。
- 技术停止规则：仅在协议越权、query或source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 验证与审查

- 与本次候选直接相关的Phase1真实入口36项、Meta-Adapter Phase1／Phase2邻近宽回归228项均通过。
- 一次定点P0/P1审查未发现会导致真实实验跑错、越权、覆盖输出、不能启动或不能产生合法prediction的问题。
- 工厂会强制核对`protocol_schema`、`phase2_data_status`、`capsule_id`、`split_id`、receiver、K和三类场景，并拒绝query truth／role及source访问标记。
- launcher第一步为P4真实checkpoint无query smoke；只有receipt确认`query_opened=false`、`source_opened=false`、`backward_count=3`、严格checkpoint加载和≤1%参数预算后，才立即继续15-row矩阵。

## 科学停止与晋级

prediction完整后才由独立scorer连接truth。按15个同row score聚合`DA1_REG0-DA0_REG0`：旧类均值至少+1.0pp且旧类floor至少+0.5pp才晋级Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并推进下一少层候选。

## 实际闭合

- Target工厂成功生成15个truth-free row，`factory_receipt.json`状态为`TARGET_INPUTS_COMPLETE`，并确认`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- P4真实checkpoint无query smoke在读取support IQ后的NumPy→Torch转换处失败：N607现有NumPy2.2.5与Torch2.1.0组合中，`torch.from_numpy`报`TypeError: expected np.ndarray (got numpy.ndarray)`。
- 失败发生在query打开和prediction产生之前；smoke output root与prediction output root均不存在，GPU无残留计算进程，15-row矩阵从未启动。因此本run没有性能结果，也不存在query泄漏或输出覆盖。
- 本地RED测试以相同错误指纹稳定复现；GREEN实现改用`torch.frombuffer`处理IQ、整数标签和冻结原型，并用有界Python值桥接prediction输出，避免同一ABI在后续写盘处再次失败。
- 直接相关69项Stage2工厂／runner／matrix／handoff／scorer／row export回归与199项Meta-Adapter Phase1／Phase2邻近回归通过。r1保持技术失败封存，修复后使用新不可覆盖run继续，不复用失败smoke或prediction root。

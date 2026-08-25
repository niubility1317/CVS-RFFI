# Time-only Rank-8 Prototype-Aligned Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`
- 状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- Phase1：`phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1`，状态`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- 候选保持time-only rank8、P4 FOMAML+Meta-SGD和正式3步support更新；Phase1及Phase2均在冻结原型余弦CE空间使用`support_logit_scale=16.0`，最终判决仍为同一冻结原型余弦argmax。
- 正式bundle严格回读预算5458／1055125=0.517285%；10个可训练张量只含原编码器`id/dom_backbone.meta_adapter_time`，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`；receiver=`20-1`；operating point为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- 配置：`configs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1.json`。相对已闭合time-only rank8 Target5计划，15个row完全不变，只替换`bundle_id`、`checkpoint_path`和`prototype_path`。

## Phase2权限边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`固定received IQ及原capsule／split；候选变化不触发数据重验。
- Phase2仅读取合法target support IQ和support标签、正式bundle、冻结原型；不读取source／clean样本、source cache、query真值或query角色。
- query只在3步support反向传播结束并冻结模型后逐样本推理，不更新模型、原型、归一化统计或其他状态。
- 同一冻结判决规则比较`DA0_REG0`与`DA1_REG0`；REG0新类指标为N/A，不引入D92式分类头。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1/checkout`
- Target工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1.out`
- 工厂命令：`python code/scripts/build_stage2_meta_adapter_target_matrix.py --plan configs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1.json --output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`。
- smoke命令：`python code/scripts/smoke_stage2_meta_adapter_no_query.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1/smoke_config_no_query.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1_smoke --device cuda`。
- prediction命令：`python code/scripts/run_stage2_meta_adapter_matrix.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1/matrix_config.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1 --device cuda`。
- expected artifacts：`factory_receipt.json`、`smoke_receipt.json`、15个row各自的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`、`receipt.json`和truth-last `score.json`，矩阵级`matrix_receipt.json`和`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 审查与科学门槛

- prototype-aligned候选唯一一次独立P0/P1审查结论为P0无、P1无；Stage2仅复用已验证factory／runner／scorer，不进行重复审查。
- 15个truth-free prediction row全部闭合后，才由独立scorer连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一少层候选。

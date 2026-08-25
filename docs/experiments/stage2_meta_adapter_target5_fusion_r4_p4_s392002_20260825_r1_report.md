# Fusion-only Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1`
- 状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- Phase1结果提交：`fb20a11ea379f5320220ed963891c2d2f672ebd4`
- 本次计划提交：`7bc9dd59a6d6e1e76e44f4e9b4cb6dd938f6d9b4`

## 候选与同row矩阵

- 候选：P4 FOMAML+Meta-SGD的fusion-only版本；正式bundle：`phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2:P4`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`；Phase2可训练集合仅含原编码器`id_backbone.meta_adapter_fusion`和`dom_backbone.meta_adapter_fusion`。两支共同进入真实反向传播，identity分支接收任务梯度；未被identity目标使用的domain分支通过精确零梯度保持bitwise不变。
- 单seed：`392002`；Target5 receiver：`20-1`。
- operating point：`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- 15个row的manifest、support、query、receiver、seed、K、new-count、场景、capsule和split与已闭合P4 Target5完全相同；本次只替换正式fusion r2 bundle和冻结原型。
- 数据沿用同一`p2_min_v1`、`VALIDATED_ONCE`固定received IQ，不因候选bundle变化重验。

## Phase2权限与资源边界

- Phase2仅读取固定received IQ、合法target support标签、正式fusion bundle和冻结原型；不读取source／clean样本、source cache、query真值或query角色。
- query仅在3步support更新完成并冻结模型后逐样本推理，不更新模型、原型、归一化统计或任何其他状态。
- 不新增或训练D92式协方差、LDA或持久分类头；沿用冻结原型余弦判决规则比较`DA0_REG0`与`DA1_REG0`。
- Phase2可训练参数2890／1052557，占0.274569%，远低于1%；正式更新固定3步，低于40步上限。真实smoke以运行时audit为准。

## 本地与发布预登记

- 本次配置：`configs/stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1.json`；只改bundle ID、checkpoint和prototype路径，所有Target5数据行保持不变。
- 精确计划对比通过：5个entries与原P4 Target5逐项相等，变更键严格为`bundle_id`、`checkpoint_path`、`prototype_path`。
- 69项Stage2工厂／runner／matrix／handoff／scorer／适配回归通过；9个相关生产入口本地编译通过。
- 测试环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；工作目录：本分支worktree根目录。
- 独立P0/P1审查已在同一fusion-only候选实现阶段完成，并对唯一P1做过定点复审；按每候选最多一次审查规则不再增加重复审查门。
- N607账户：普通用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1/checkout`
- smoke output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1_smoke`
- prediction output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1.out`
- expected artifacts：`smoke_receipt.json`；15个row各自的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`、`receipt.json`和truth-last `score.json`；矩阵级`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 科学晋级规则

15个truth-free prediction row全部闭合后，才由独立scorer连接truth。聚合`DA1_REG0-DA0_REG0`：旧类均值至少+1.0pp且旧类floor至少+0.5pp才进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一少层候选。

## N607发布与真实checkpoint smoke

- release固定提交：`a55ac12cb293dd92901db9d806573840872198ee`；归档：`E:\type10-7\release_archives\stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1_release.tar.gz`。
- 本地与远端归档SHA256均为`89f9eb12031e9d78e47281563d188ee20f89eb05153993708bc1e159eac7b6f2`；11个相关生产入口远端编译通过。
- 发布前独立确认release、Target工厂输出、smoke、prediction和stdout目标均不存在；同名进程不存在；GPU0～GPU7均无计算进程；项目盘剩余7.3TiB。
- Target工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1`；15／15 truth-free row闭合，`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- 首次smoke在任何checkpoint、support或query读取前因派生配置仍含`query_path`而被严格allowlist拒绝，未创建smoke输出。只删除smoke专用配置的`query_path`后原位重跑；正式15-row矩阵未改变。
- 真实fusion r2 checkpoint无query smoke通过：`status=REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`、`checkpoint_load_strict=true`、`backward_count=3`、`trainable_fraction=0.0027456945324576248`（0.274569%）、`query_opened=false`、`source_opened=false`、`query_state_update_count=0`、`performance_result=null`。
- 当前最高状态：`LANDED`。下一步只启动一次15-row truth-free prediction矩阵，并立即核对PID、CWD、cmdline、GPU和日志增长。

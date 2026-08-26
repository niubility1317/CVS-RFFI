# SF-TAPFT V1目标域内部性能筛选预登记

## 实验身份

- run ID：`stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 候选：`SF_TAPFT_V1_REPORT_DEFAULT`
- 权限：`DIAGNOSTIC_NON_FORMAL`
- Git commit：`1023d70b37bccc7f5144e018b9045aad68ebd013`
- 数据绑定：`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id=d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`、`split_id=stage2b-rx20-1-seed713101-before-support-prefix`

## 可证伪矩阵与停止规则

- 单seed：`392002`。
- 单目标接收机：`rx20-1`。
- `K=10`旧类target support，4-fold target-inner选择。
- frozen与SF-TAPFT在相同OOF fold上比较balanced accuracy、NLL和true-class margin。
- 只有多数fold不下降、平均NLL改善且accuracy或margin改善时选择`adapted`；否则选择`zero_adapt`并停止该候选，不进入query性能验证。
- 技术停止仅限协议/query越界、错误checkpoint/split/GPU、输出碰撞、错误checkout、确定性重复异常、无法产生`selection.json`或进程归属不清；不得因中途指标低而停止。

## 版本与命令

- 本地Git工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`
- N607 release目标：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_fix3`
- N607 CWD：上述release目录。
- smoke命令：`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 code/scripts/run_target_only_progressive_adapt.py --config configs/stage2_sf_tapft_v1_rx20_1_clear_smoke_s392002_20260826.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_smoke --device cuda:0`
- 性能筛选命令：`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 code/scripts/run_target_only_progressive_nested.py --config configs/stage2_sf_tapft_v1_report_default_s392002_20260826.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1 --device cuda:0 --folds 4`

## 环境、输入与输出

- N607 GPU：物理GPU0；preflight时利用率0%、显存1MiB。
- 服务器解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch`2.1.0+cu121`，CUDA可用。服务器不存在`ssr-gpu`环境，本次不伪造环境名。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- support：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2b_sclba_a_t5t25_s713101_20260824_v1/input/support_rx20_1_k10_clear_smoke.npz`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_smoke`
- 性能筛选输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1.log`
- 预期artifact：smoke的`sf_tapft_bundle.pt/smoke.json`；性能筛选的`selection.json`，仅当选择`adapted`时另有`sf_tapft_bundle.pt`。

## 声明边界

该实验只产生target-inner OOF筛选证据，不读取query，不产生正式Phase2 prediction，也不连接truth。`RUNNING`、smoke通过或OOF选择均不得表述为正式最终性能；只有后续独立query prediction和truth-last scorer闭合后才可声明query性能。

## 启动记录

- 首次真实checkpoint smoke在模型更新前停止：既有已验证support NPZ仅含`received_iq/support_labels`，runner错误要求内嵌`support_physical_ids`，因此报`target support NPZ allowlist mismatch`；smoke/run输出均未创建，无性能结果。
- 定点修复提交为`9f8bf87bc590ad7c53240e6ae052c79562e8755e`：对于匹配`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`的既有两字段support，runner生成仅用于inner-fold行隔离的稳定opaque row ID，并在receipt中记录`physical_id_origin=validated_support_row_index`；不把它声明为新数据验证或真实物理ID证据。
- 修复后19项SF-TAPFT聚焦测试通过；等待新release落地并重新执行同一run ID的首次有效smoke。原失败发生在输出创建和参数更新前，因此不会覆盖或混合artifact。
- 原release目录保持不变作为失败证据；修复版使用新的`..._fix1`release目录，不覆盖旧release。
- `fix1` smoke在support tensor构造时因N607的PyTorch2.1/NumPy桥接异常停止：`torch.from_numpy`拒绝实际`numpy.ndarray`；仍未创建smoke/run输出。
- `fix2`提交`1aad719c2f513fc8b9d85f1fe0586d4184fc2df7`保留正常`from_numpy`快路径，仅在`TypeError`时对小型support数组使用列表构造tensor；故障注入聚焦测试通过。新的`..._fix2`release不覆盖前两版。
- `fix2`随后进入适配器，但N607 PyTorch2.1缺少`torch.amp.GradScaler`，仍在首步更新前停止且未创建输出。`fix3`提交`1023d70b37bccc7f5144e018b9045aad68ebd013`在新API存在时使用新入口，否则回退`torch.cuda.amp.GradScaler`；20项聚焦测试通过。

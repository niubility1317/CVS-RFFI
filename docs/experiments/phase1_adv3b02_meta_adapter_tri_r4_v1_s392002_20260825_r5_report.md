# CVS_META_ADAPTER_TRI_R4_V1 r5最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5`
- 状态：`LOCAL_VERIFIED / NOT_LAUNCHED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 固定代码与配置提交：`8d07f752e5093766f31edab7fdc97159c60d70f1`

## 候选与矩阵

P0为冻结base控制；P1为随机adapter；P2为source监督adapter；P3为FOMAML固定LR；P4为FOMAML+Meta-SGD。P1～P4顺序运行，科学失败不阻断后续候选。Phase1只使用source角色训练和选择，最终checkpoint分别评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。r5与r4配置除不可覆盖run ID外完全相同；r4因包含已知Torch／NumPy原型写入缺陷而保持`NOT_LAUNCHED`。

## 本地验证与定点P0/P1审查

- RED测试模拟`Tensor.numpy()`返回不兼容数组，旧实现稳定在冻结原型artifact路径失败；GREEN实现仅对6类冻结原型使用有界列表桥接，NPZ数组由当前NumPy创建。
- checkpoint bundle继续保存原Torch张量，训练、adapter、source选择、四场景评价和Phase2 query路径均未改变。
- Phase1真实入口整文件36项通过；Meta-Adapter Phase1／Phase2邻近宽回归228项通过。N607同一`CVS-RFFI`环境内存复现确认桥接后数组类型身份正确，NPZ写入和无pickle回读成功。
- 定点P0/P1审查范围仅为原失败路径及其直接消费者：未发现会导致run启动错误、输出覆盖、协议越权、缺少合法prediction或错误修改checkpoint原型类型的问题。

## N607最小预登记

- N607账户：普通`N607`用户`szu2070436088`
- 环境：现有`CVS-RFFI`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：0；启动前再次核对占用，且不得超过每GPU两个训练进程。
- release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_a465e329.tar.gz`
- release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_a465e329.tar.gz`
- 本地release归档固定提交为`a465e3292a2c4c60069eb8d4dbffb685fcb2b709`，35491571字节、5007个条目，包含r5配置和报告；SHA256=`8becfa7e4a8e68aa3bae1c8668c81b2b8f6bb6af47617c0cc1b9349203e2c349`。
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5.out`
- 启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5 --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu 0`
- expected artifacts：每个P1～P4子目录的`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`source_adaptation_curve.json`、`run_summary.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`、`frozen_prototypes.npz`，以及矩阵级`candidate_matrix_summary.json`。
- 技术停止规则：仅在协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、launcher-wide故障，或至少两个候选出现相同确定性pre-prediction异常时停止；不得因低准确率停止。

## 当前边界

- r3已封为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；r4未同步、未启动且不再使用。
- r5本地release已生成但尚未同步、未启动、没有性能结果。远端SHA、远端编译、资源／路径preflight和启动健康检查完成前，状态不得提升为`RUNNING`。

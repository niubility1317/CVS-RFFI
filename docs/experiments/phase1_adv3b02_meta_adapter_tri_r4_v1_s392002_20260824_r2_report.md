# CVS_META_ADAPTER_TRI_R4_V1 r2 N607预登记报告

- run ID：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2`
- 状态：`LOCAL_VERIFIED/NOT_LAUNCHED`
- 时间：2026-08-25（Asia/Hong_Kong）
- 修复提交：`c7e5e3b384b4d0522df39a841e539e39af1d2dc2`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`

## 候选与矩阵

P0为冻结base控制；P1为随机adapter；P2为source监督adapter；P3为FOMAML固定LR；P4为FOMAML+Meta-SGD。P1～P4顺序运行，任一科学失败返回0并继续下一候选。Phase1训练与选择仍只用source角色，最终checkpoint分别评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地修复验证

- 根因：release checkout会把配置相对输入路径解析到checkout内部，无法读取项目根目录的冻结checkpoint和ManySig。
- 修复：launcher新增`--base-checkpoint`和`--wisig-pkl`显式只读覆盖，不改变训练算法、候选或协议。
- RED：新增回归测试在旧launcher上因参数不识别而失败。
- GREEN：Phase1入口30项、邻近trainer／Phase2 runner／scorer68项，共98项通过；`r2`配置加载与六个生产入口`py_compile`通过。

## N607最小预登记

- N607账户：普通`N607`用户`szu2070436088`
- GPU：0；每GPU并发训练数计划为1。
- release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2_c7e5e3b3.tar.gz`
- release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2_c7e5e3b3.tar.gz`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r2 --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu 0`
- expected artifacts：每个P1～P4子目录的`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`source_adaptation_curve.json`、`run_summary.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`、`frozen_prototypes.npz`，以及矩阵级`candidate_matrix_summary.json`。
- 技术停止规则：仅在协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、launcher-wide故障，或至少两个候选出现相同确定性pre-artifact异常时停止；不得因低准确率停止。

`r1`已封为技术失败且不再使用。`r2`在归档SHA核对、远端编译和启动健康检查完成前不声明`RUNNING`，当前没有性能结果。

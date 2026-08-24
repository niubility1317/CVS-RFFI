# CVS_META_ADAPTER_TRI_R4_V1 r4后备run最小预登记

- run ID：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4`
- 状态：`LOCAL_VERIFIED / NOT_LAUNCHED`
- 启动条件：仅当r3按既有预登记规则进入`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`后，才允许发布和启动r4；r3仍运行时禁止同步、启动或并行替代。
- 固定代码与配置提交：`70961b7a9e9f952cec6160036b6b09ea0db5e415`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`

## 候选与矩阵

P0为冻结base控制；P1为随机adapter；P2为source监督adapter；P3为FOMAML固定LR；P4为FOMAML+Meta-SGD。P1～P4顺序运行，科学失败不阻断后续候选。Phase1只使用source角色训练和选择，最终checkpoint分别评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。候选、seed、数据角色、训练步和评价规则与r3完全相同，仅run ID和后备快路径实现不同。

## 本地验证

- r4配置通过生产配置加载器；与r3逐字段比较只有`run_id`不同，共5个候选。
- 后备快路径包含三项等价优化：episode ref只读WiSig索引、candidate plan和class/spec pool按冻结refs复用、四场景评价按128条流式累计。
- Phase1真实入口35项、入口/episode采样器/trainer联合88项及Phase1/Phase2 Meta-Adapter 14文件宽回归265项通过；仅有既存AMP API弃用警告。

## N607最小预登记

- N607账户：普通`N607`用户`szu2070436088`
- 环境：现有`CVS-RFFI`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：0；r3结束并完成只读资源preflight前不得占用。
- 预定release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4_70961b7a.tar.gz`
- 预定release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4_70961b7a.tar.gz`
- 预定远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 预定output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4`
- 预定stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4.out`
- 预定启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r4 --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu 0`
- expected artifacts：每个P1～P4子目录的`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`source_adaptation_curve.json`、`run_summary.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`、`frozen_prototypes.npz`，以及矩阵级`candidate_matrix_summary.json`。
- 技术停止规则：仅在协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、launcher-wide故障，或至少两个候选出现相同确定性pre-artifact异常时停止；不得因低准确率停止。

## 尚未执行

- 尚未创建release归档，尚未进行本地到远端SHA比较、远端编译、资源/路径preflight、同步、启动或启动健康检查。
- 以上步骤不是缺失授权，而是受“r3只读监控且不重复启动”约束而有意等待；r3若正常完成，r4保持`NOT_LAUNCHED`并不再使用。

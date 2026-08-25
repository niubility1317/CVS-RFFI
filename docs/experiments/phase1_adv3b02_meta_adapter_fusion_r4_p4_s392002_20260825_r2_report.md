# CVS_META_ADAPTER_FUSION_R4_P4_V1 r2最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2`
- 当前状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 固定代码与配置提交：`391795c19096de643dfb626bc2ac95ec7f4fd141`

## 候选、修复与停止规则

r2保持fusion-only、rank=4、P4 FOMAML+Meta-SGD、source-only元训练和正式3步适配，不改checkpoint、seed、数据、模型判决或Phase2门槛。它只修复r1已证实的两个问题：bundle校验允许已登记的fusion-only profile；训练不再重复4个`Q_SAME_DOMAIN`batch，而是使用20个确定性episode池，按8／4／3／3／2覆盖`Q_SAME_DOMAIN`、`Q_RX_HOLDOUT`、`Q_DAY_CHANNEL_HOLDOUT`、`Q_CLEAN_TO_LEO`和`Q_LEO_CROSS`，每步轮换4个batch。

Phase1只读取receiver0～6、day0～1的source角色样本训练和选择；最终checkpoint必须评价clean和三类LEO weak。Phase1合法完成后才进入同row单seed Target5；仅当`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp时进入Target25。

## 本地验证与审查边界

- RED复现fusion bundle保存失败；GREEN验证fusion-only单／双分支bundle均可保存并严格回载，未登记的其他授权站点子集仍拒绝。
- RED复现固定batch不能覆盖声明任务；GREEN验证20个池按8／4／3／3／2精确覆盖五类query任务，前5步消费整个池并确定性轮换。
- Meta-Adapter Phase1/Phase2聚焦及邻近回归261项通过；fusion-only参数预算仍为2,890/1,052,557，占0.2746%，正式3步。
- 同一候选的唯一独立P0/P1审查及允许的一次定点复审已在r1完成；按项目穷尽式最小流程不增加第二轮审查。r1失败根因和完整日志诊断已写入r1报告。

## N607最小预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- GPU：0；启动前核对资源，不超过每GPU两个训练进程。
- 本地release归档：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2_release.tar.gz`
- 远端release归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2_release.tar.gz`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2.out`

冻结训练命令如下：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2.json --meta_adapter_rank 4 --meta_adapter_sites fusion --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r2 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

预期artifact：`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`run_summary.json`、`config_snapshot.json`、`source_adaptation_curve.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`和`frozen_prototypes.npz`。

技术停止只允许协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、确定性重复异常或进程归属不清；不得因中间性能低而停止。当前尚未证明N607落地、真实checkpoint smoke、训练完成或目标域正向收益。

# PHASE1 SHORT195_S3 z_id特征空间验证计划

生成时间：2026-06-30 00:30 CST

## 基线锚点

本轮地面训练验证以`PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3`为唯一基线锚点。该实验来自`phase1_gpu0_jointsafe36_queue_20260629_0930`，入口是`code/SSDG/train_ssdg.py`，协议为Safe-SSDG-CVS-R01，训练阶段只使用source receivers，不使用target receiver、Stage2 support/query或unknown query。

远端只读核对结果：

| 项 | 值 |
|---|---|
| best epoch | E200 |
| best_score | 84.9483 |
| val_tx | 98.51% |
| test_overall_tx | 90.34% |
| strict_udu | 84.14% |
| receiver_floor | 76.24% |
| satellite_mean | 76.62% |
| satellite_floor | 75.39% |
| satellite_strict_mean | 70.45% |
| satellite_strict_floor | 69.24% |
| pseudo precision | 100.00% |
| checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3/best_joint_safe_ssdg.pth` |

## 本地桥接边界

`code/SSDG/train_ssdg.py`已新增默认关闭的`z_id`特征空间桥接：

| 能力 | 本地复用点 | SSDG新增参数 | 默认行为 |
|---|---|---|---|
| 训练期原型记忆 | `code/cvsrffi/losses.py::PrototypeMemoryBank` | `--use_proto_memory`、`--lambda_proto`、`--proto_*` | 默认关闭 |
| 开放世界角度几何 | `code/cvsrffi/losses.py::open_world_feature_space_loss` | `--lambda_open_world_feat`、`--ow_feat_*` | 默认关闭 |
| Phase2原型导出 | `code/cvsrffi/phase2_prototypes.py::export_phase2_prototypes` | `--phase2_export_prototypes`、`--phase2_export_*` | 默认关闭 |
| dense-tail测试调度 | `code/training_test_eval.py::should_run_training_test` | `--test_eval_policy interval_final --test_eval_interval 10 --test_eval_final_window 20 --test_eval_final_interval 2` | 默认仍every_epoch |

旧的`--lambda_tx_proto`、`--lambda_rx_proto`、`--lambda_mask_aux`、`--lambda_txrx_rect`仍是未接线audit权重；非零时继续报错，避免把旧审计字段静默解释成新的loss。`open_world_feature_space_loss`的训练主项使用cosine-margin hinge，角度值只作为detach后的日志指标，避免真实batch上`acos`反传导致非有限梯度。

## 候选矩阵

脚本：`code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh`

run_id：`phase1_short195s3_zidbridge_20260630_0030`

所有候选继承SHORT195_S3的损失、伪标签、星地增强和joint-safe设置，但不强制固定200 epoch。默认先跑E160快速筛选，即`SCREEN_EPOCHS=160`、`SCREEN_LABEL_EPOCHS=150`、`SCREEN_PSEUDO_EPOCHS=10`；脚本允许用环境变量覆写epoch设置。E160结果只作为机制筛选和稳定性证据，不直接等同于E200基线的最终性能证据。筛选通过的最多2个候选再扩展到E200/E220确认。

根据当前验证目标，GPU1-GPU7全部用于联合机制探索，不再启动控制组或单机制消融。历史SHORT195_S3 E200结果作为唯一外部基线。每个候选都同时启用训练期原型记忆、open-world角度特征空间约束、域对齐项和Phase2 `z_id`原型导出，只改变联合强度。

| ID | GPU | 联合机制强度 | 新增主动项 |
|---|---:|---|---|
| `PHASE1_SHORT195S3_ZIDJOINT_C0_CONSERVE_E160` | 1 | 保守联合 | `--use_proto_memory true --lambda_proto 0.003 --lambda_open_world_feat 0.003 --proto_domain_align_weight 0.15 --ow_feat_domain_align_weight 0.01` |
| `PHASE1_SHORT195S3_ZIDJOINT_C1_LOW_E160` | 2 | 低强度均衡联合 | `--lambda_proto 0.004 --lambda_open_world_feat 0.004 --proto_domain_align_weight 0.25 --ow_feat_domain_align_weight 0.01` |
| `PHASE1_SHORT195S3_ZIDJOINT_C2_GEOM_E160` | 3 | 特征几何偏强 | `--lambda_proto 0.004 --lambda_open_world_feat 0.006 --proto_domain_align_weight 0.25 --ow_feat_domain_align_weight 0.02` |
| `PHASE1_SHORT195S3_ZIDJOINT_C3_PROTO_E160` | 4 | 原型约束偏强 | `--lambda_proto 0.006 --lambda_open_world_feat 0.004 --proto_domain_align_weight 0.35 --ow_feat_domain_align_weight 0.01` |
| `PHASE1_SHORT195S3_ZIDJOINT_C4_BALANCED_E160` | 5 | 中强度均衡联合 | `--lambda_proto 0.006 --lambda_open_world_feat 0.006 --proto_domain_align_weight 0.25 --ow_feat_domain_align_weight 0.02` |
| `PHASE1_SHORT195S3_ZIDJOINT_C5_DOMAIN_E160` | 6 | 域对齐偏强联合 | `--lambda_proto 0.004 --lambda_open_world_feat 0.005 --proto_domain_align_weight 0.50 --ow_feat_domain_align_weight 0.03` |
| `PHASE1_SHORT195S3_ZIDJOINT_C6_STRONG_E160` | 7 | 强联合压力测试 | `--lambda_proto 0.008 --lambda_open_world_feat 0.006 --proto_domain_align_weight 0.35 --ow_feat_domain_align_weight 0.03` |

## 验证标准

| 门槛 | 要求 |
|---|---|
| source DG | E160阶段看趋势和稳定性；正式E200/E220确认时，相对SHORT195_S3，overall、strict UDU、receiver_floor任一下降超过2pp则不晋级 |
| satellite | satellite_mean和satellite_floor下降超过1pp则人工复核 |
| 训练稳定性 | 无Traceback、CUDA OOM、NaN loss、非有限梯度连续跳过 |
| 原型记忆 | `train_proto_active_classes`稳定大于0，`train_w_loss_proto_labeled`非零且有限 |
| open-world几何 | `train_ow_feat_active_classes>=2`，`train_w_loss_open_world_feat`非零且有限 |
| Phase2导出 | 每行产出`phase2_zid_prototypes.pt`和`.json`，包含`prototypes`、`tx_domain_prototypes`、radii和geometry |
| 测试调度 | named test和satellite eval只在每10轮、最后20轮每2轮、最终epoch运行 |

## 与Meta-SSL的关系

本轮按用户指定的SSDG基线做同入口地面验证。Meta-SSL和few-shot适配属于后续支持集/部署适配链路，不能在Phase1 source-only地面训练中读取target receiver、Stage2 support/query或unknown query。当前安排先让全部GPU1-GPU7候选产出联合优化后的`z_id`表示和Phase2原型包；通过稳定性筛选后，再把晋级候选接入Stage2-B的few-shot/meta-learning验证，避免把地面训练证据和部署适配证据混成同一级别。

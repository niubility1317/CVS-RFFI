# qKNN Stage2-C effective8源域适配实验报告

|字段|内容|
|---|---|
|实验ID|`qknn_ground_effective8_r16_e12_leoonly_20260715_v13`|
|预注册时间|2026-07-15 12:45 CST|
|operator|Codex`/root`|
|当前状态|`PRELAUNCH_SOURCE_ONLY`|
|目标|在不读取任何target support/query、clean或proxy数据的前提下，训练并验证一个≤50k参数、≤20epoch的ADV3B02 effective-feature LoRA；只有source receiver holdout全部PASS后才允许构建candidate lock|
|比较对象|同一ADV3B02 checkpoint的无adapter source holdout；source nested-K identity mean head|

## 假设与方法

历史875个配对运行显示K1遗忘不能靠单独延长epoch解决。该候选只更新8个会进入160维`feat_joint/z_id`的后段Linear：`t_proj`、`f_proj`、`pa_proj.0`、`fuse.0`、`cls_head.id_proj.0`、`cls_head.pa_proj.0`、`cls_head.id_gate.0`、`cls_head.joint_proj.0`。rank16 LoRA共44,048参数，地面训练12epoch；目标端不做梯度训练，只在candidate lock之后以3个LEO support View闭式拟合统一K头。

训练只使用ManySig source TX index`0,1,2,3,4,5`和source receiver index`0,1,2,3,4,5`。receiver index`6`完整留给source validation；正式target receiver index`7,8,9,10,11`禁止进入训练和source validation。训练和验证场景固定为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，clean样本计数必须为0。

## 版本与本地验证

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`。实现主提交=`9765654`，source-only无target cell补丁=`e56eaba`。根`E:\type10-7`不是Git仓库；根`项目.md`和本报告为控制面源文件，Git镜像报告单独提交。

本地`ssr-gpu`完成Python静态编译和63项完整定向测试；source-only CLI补丁另有6项测试PASS，`git diff --check` PASS。关键本地文件及同步目标如下：

|本地文件|N607目标|SHA256|
|---|---|---|
|`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`|同相对路径|`11e4be9705b99369ebb2f6c7eabfc59c3e756617ec7bbe39e8fb7ae5b6fe41c8`|
|`paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py`|同相对路径|`fb51583d237d57e0aad2e20483d2e6ea4502cfe71565a61e60b527b0c9aaadc2`|
|`paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py`|同相对路径|`018652df67b6f17e44ebfab1b317ccced15ee0c7d9ce39fe2332eeb58999b6da`|
|`paper_reproduction/cvs_aligned/k1_symmetric_head.py`|同相对路径|`16b591645ee45d5441bc910ba264022f39737f075ed5dae63914763539259d3b`|
|`paper_reproduction/scripts/benchmark_cvs_adaptive_rxlight_tta.py`|同相对路径|`3bf75a52209b8e0650ccb6c06c47860339c1490ccc1b098854a30f9e2a291db1`|
|`paper_reproduction/scripts/build_cvs_stage2c_candidate_lock.py`|同相对路径|`040cc58512b73b61a33ac089c111056e4eaba09116ae51951e3daa2d61c87a23`|
|`paper_reproduction/scripts/summarize_cvs_stage2c_locked_matrix.py`|同相对路径|`f0cb1d2633d0da82df0fbd4d3f4d8b17ba1fbf3d84677ec5c3189cdc1437f3d3`|

同步前远端重叠文件已快照到`E:\type10-7\code\snapshots\qknn_ground_effective8_v13_remote_before_sync_20260715_1240`。N607直连preflight PASS；checkpoint、ManySig、ManyTx存在，项目盘可用7.6TB，目标run/log根不存在。12:40 inventory显示既有875任务仍在运行，不干预、不终止；GPU分配在启动前重新检查，优先当前空闲GPU4，且不得超过每卡2个训练进程。

## 精确训练配置

|参数|值|
|---|---:|
|checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|adapter scope/rank/alpha|`effective_feature`/16/16|
|trainable params|44,048|
|epoch/batch/seed|12/256/4070391|
|optimizer|AdamW，`lr=5e-4`，`weight_decay=1e-4`，`grad_clip=1`|
|feature trust region|SmoothL1=1，cos=2，same-LEO identity=22，feature margin=4.5，same-LEO margin=7.5，teacher logit=0.16|
|K与几何损失|prototype CE=0.2，View一致性=0.25，relation Gram=0.5，prototype Gram=0.25，worst-K=0.5，K=`1,2,5,10,20`|
|proxy/unknown loss|全部0|
|输入|`input_adapter=false`，`input_repair=raw`，LEO-only|
|预计状态|LoRA FP16 88,096B；连同26类统一头和门限≤103,796B|

远端工作目录=`/home/szu2070436088/2510044040/CV-SincNet`，Python环境=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，训练日志=`logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/train.log`，PID文件=`logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/train.pid`。精确训练命令为：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/train_apply_phase1_iq_preadapter_20260703.py --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --wisig_pkl Dataset_WigSig/ManySig.pkl --new_wisig_pkl Dataset_WigSig/ManyTx.pkl --runs_root runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13 --source_tx_ids 0,1,2,3,4,5 --source_rxs 0,1,2,3,4,5 --max_source_samples_per_tx 1000 --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --num_old_classes 6 --feature_name z_id --sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --star_ground_channel_impl simplified_leo_residual --batch_size 256 --epochs 12 --no-input_adapter_enabled --model_adapter_mode lora_effective_feature --lora_rank 16 --lora_alpha 16 --adapter_state_out runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/effective8_adapter_fp16.pt --adapter_manifest_out runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/training_manifest.json --source_only_ground_lora --input_repair raw --lr 5e-4 --weight_decay 1e-4 --mse_weight 1 --cos_weight 2 --proto_ce_weight 0.2 --logit_ce_weight 0 --clean_identity_weight 22 --clean_cos_weight 1 --feature_margin_weight 4.5 --clean_feature_margin_weight 7.5 --feature_margin_tolerance 0.01 --proxy_unknown_separation_weight 0 --proxy_unknown_supcon_weight 0 --proxy_unknown_proto_ce_weight 0 --proxy_unknown_pair_margin_weight 0 --proxy_unknown_old_margin_weight 0 --proxy_unknown_hard_pair_margin_weight 0 --proxy_unknown_hard_old_margin_weight 0 --teacher_logit_distill_weight 0.16 --multiview_consistency_weight 0.25 --relation_preservation_weight 0.5 --prototype_gram_weight 0.25 --prototype_gram_max_cosine 0.65 --worst_k_risk_weight 0.5 --worst_k_values 1,2,5,10,20 --worst_k_tau 0.2 --worst_k_proto_temperature 0.07 --distill_temperature 2 --residual_weight 0 --proto_temperature 0.07 --grad_clip 1 --log_every 1 --device cuda:0 --seed 4070391
```

训练自然完成后先分析完整日志和manifest，不直接读取target。只有无Traceback/OOM/nan/inf、12个连续epoch、44,048参数、88,096B、proxy/clean/target计数均为0且artifact哈希一致，才允许执行source validation。预注册validation输出=`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/source_validation`；必须核验receiver split、fixed1/floor、5-view、自适应平均View、extra-view触发、统一头每个K不劣于identity以及全部gate。任一gate失败则停止，不构建candidate lock，不读target support/query。

## 预期输出与结束条件

|输出|路径|
|---|---|
|FP16 LoRA|`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/effective8_adapter_fp16.pt`|
|训练manifest|同run根`training_manifest.json`|
|完整训练日志|`logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v13/train.log`|
|source统计|`source_validation/source_joint_feature_stats_fp32.npz`|
|source验证|`source_validation/source_validation.json`|
|promotion manifest|`source_validation/promotion_manifest.json`|

风险包括既有875任务动态占用GPU、远端import路径遮蔽、ManySig大文件I/O、source holdout未通过以及多View不触发。启动前重新核验GPU和目标路径；任务只新增自己的run/log，不覆盖数据、checkpoint或其它实验输出。

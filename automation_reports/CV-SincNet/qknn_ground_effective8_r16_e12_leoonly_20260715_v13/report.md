# qKNN Stage2-C effective8源域适配实验报告

|字段|内容|
|---|---|
|实验ID|`qknn_ground_effective8_r16_e12_leoonly_20260715_v13`|
|预注册时间|2026-07-15 12:45 CST|
|operator|Codex`/root`|
|当前状态|`LOCAL_PROTOCOL_REPAIR_REQUIRED`；2026-07-15新版`LEO_weak-only`硬门生效后，旧命令禁止启动|
|目标|在不读取任何target support/query、clean或proxy数据的前提下，训练并验证一个≤50k参数、≤20epoch的ADV3B02 effective-feature LoRA；只有source receiver holdout全部PASS后才允许构建candidate lock|
|比较对象|同一ADV3B02 checkpoint的无adapter source holdout；source nested-K identity mean head|

## 2026-07-15新版协议追踪表

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|P2-LW-01|`项目.md`7.1、8.5、9|Phase2全部support/query、适配、选择与评估输入在进入Phase2前已经叠加三个允许的`leo_*_weak`场景之一|训练入口、target cache loader、benchmark|pending|待补负例与manifest审计|不得在Phase2读取raw/clean IQ后现场生成LEO|
|P2-LW-02|`项目.md`7.1、9|禁止clean派生feature、logit、prototype、loss anchor、TTA门限或promotion信号|训练损失、source validator、candidate lock|pending|待做clean可达性反向搜索和运行时硬门|当前命令中的`clean_*`与teacher分支必须删除、替换或严格隔离|
|P2-LW-03|`项目.md`7.1|每个launchable row/artifact记录policy、`clean_sample_access=false`、scenario、satellite seed和sample-level provenance|cache manifest、training manifest、candidate lock、metrics|pending|待schema正负例测试|缺字段必须fail closed|
|P2-LW-04|`项目.md`5、7.1、9|Phase1地面artifact生成与Phase2部署运行分离，Phase2不重新读取ManySig/ManyTx clean/raw入口|地面cache builder、adapter trainer、Phase2 loader|pending|待入口级测试|Phase2只允许加载锁定adapter和预叠加LEO缓存|
|P2-LW-05|`项目.md`8.5|自适应1→3→5-view来自同一物理LEO观测后的接收侧View，不重新采样信道|benchmark、TTA manifest|pending|待physical sample ID与overlay ID一致性测试|报告平均/P95前向数及1/3/5触发率|
|P2-LW-06|`项目.md`7.3、8.5、9|禁止old/new角色Oracle、类别quota、query标签拟合、排序/分块和Hungarian|统一head、benchmark、summarizer|implemented|现有定向测试已覆盖部分路径，待新版全链路反审|所有注册类同规则|
|P2-LW-07|`项目.md`9|开发seed仅在K10选统一candidate；锁定后K1/K5/K10/K20嵌套sample ID、真实TX集合、head和TTA门限|candidate lock、cross-K summarizer|pending|待跨K/跨场景锁定测试|K1还需比较直接ADV3B02并报告配对CI|
|P2-LW-08|`项目.md`7.2、9|adapter≤50k、适配≤20epoch、状态≤256KB、无dense query图，报告MAC/延迟/显存/状态Pareto|adapter manifest、resource audit|implemented|44,048参数与88,096B已实体审计；待端到端MAC/延迟/显存|参数结构可保留，运行入口仍不合规|
|P2-LW-09|`项目.md`9；`AGENTS.md`实验报告|完成5receiver×≥5确认seed×3场景×5/10/20真实新类及逐类、逐receiver、完整日志、独立矩阵证据|matrix、metrics、report|pending|尚未启动合法确认实验|不得以旧875诊断替代|
|P2-LW-10|`项目.md`7.1|封存当前读取`ManySig.pkl --input_repair raw`且携带`clean_*`损失的旧预启动命令|本报告、旧远端同步文件|rejected|目标run/log目录不存在，无训练启动|代码可作为待修草稿，但禁止runner执行|

当前最高风险是P2-LW-01至P2-LW-04：输入路径若只在字段上声明`LEO_weak-only`，实际仍从clean/raw构造特征，则所有后续性能与资源结果均不具备Phase2资格。本报告在这些项目验证前保持`LOCAL_PROTOCOL_REPAIR_REQUIRED`。

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

## 12:48上线审计

7个远端文件SHA256与本报告逐项一致，远端`py_compile` PASS。真实checkpoint实体审计为195个tensor、`missing=0/unexpected=0/skipped=0`、`num_domains=14`；rank16 effective8注入精确44,048参数、88,096B、8个白名单Linear，原checkpoint可训参数和梯度更新均为0。首次实体审计因远端同名顶层`cvsrffi`包遮蔽而在import前退出；把项目`code`固定在`sys.path[0]`后审计PASS，该异常未读取数据、未创建run且不是模型失败。

12:47:59实时inventory显示875任务已自然结束，`active_training_processes=[]`、`gpu_compute=[]`，8张GPU均为10MiB空闲状态；目标run/log根再次确认不存在。本run固定使用物理GPU4，启动后只允许该卡出现1个新增训练进程。

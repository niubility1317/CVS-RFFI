# Fusion-only Rank-8 Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1`
- 状态：`ANALYZED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 基线：冻结`ADV3B02_CORE90_SOFT_E200`checkpoint。
- 本次计划提交：`5670ebfd1726b3582e38cd78102003a0a460af00`。

## 候选与唯一机制变量

- 上一fusion-only rank-4候选已在同row Target5闭合为`mean_delta_pp=0.0`、`floor_delta_pp=0.0`，15／15 row无决策变化。
- 新候选保持同一fusion层位、P4 FOMAML+Meta-SGD、20-episode加权任务池、200个meta train step、3步inner update、Phase1-C主干低学习率外循环、数据划分和冻结原型规则，只把bottleneck rank从4提高到8。
- 双分支可训练参数5458／1055125，占0.517285%，低于1%；可训练名仍仅限`id_backbone.meta_adapter_fusion`和`dom_backbone.meta_adapter_fusion`，不含分类头、协方差、LDA或持久新头。

## Phase1数据与训练边界

- 只读取WiSig source receiver 0～6、source days 0～1；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- clean测试固定days 2～3，与训练／选择物理样本不相交；训练中不读取任何Phase2 target support、target query、query真值／角色或target receiver字段。
- 任务池固定20个episode，权重为`Q_SAME_DOMAIN=40%`、`Q_RX_HOLDOUT=20%`、`Q_DAY_CHANNEL_HOLDOUT=15%`、`Q_CLEAN_TO_LEO=15%`、`Q_LEO_CROSS=10%`；K取1／2／5／10。
- 训练完成后必须保存正式bundle、冻结原型、训练曲线和日志，并在selected final checkpoint上评价clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。

## 本地验证与审查

- RED：rank-8 fusion配置和bundle旧路径4项负测稳定失败。
- GREEN：只注册`rank=8+sites=fusion`；`rank=8+time,freq,fusion`仍拒绝；历史rank-4 tri／fusion保持兼容。
- 283项Meta-Adapter Phase1／Phase2邻近回归通过，相关生产入口编译通过。
- 独立P0/P1审查发现1个P1：预登记必须显式冻结`--meta_adapter_rank 8 --meta_adapter_sites fusion`，否则`train.py`会回退rank-4 tri默认值并被入口拒绝。报告已加入完整命令；唯一一次定点复审确认原P1解除，无P0/P1残留。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- 配置：`configs/phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1.json`。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1.out`
- 冻结命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1.json --meta_adapter_rank 8 --meta_adapter_sites fusion --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

- expected artifacts：`selected_meta_bundle.pt`、`frozen_prototypes.npz`、`logs.jsonl`、`metrics.csv`、`source_adaptation_curve.json`、P0／final四场景评价、`run_summary.json`和config snapshot。
- 技术停止规则：只在协议越权、错误checkout／数据split、输出覆盖、launcher-wide故障、无训练进展或重复确定性异常时停止；不得因中间性能低停止。

Phase1闭合后，仅在source-only选择规则允许时进入完全相同的Target5最小矩阵；仍以`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp作为Target25门槛。

## N607发布与smoke证据

- release固定提交：`89ef0d2d899af067c6db24a6d9aceec66aaab2f6`；归档：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1_release.tar.gz`。
- 本地与远端归档SHA256均为`a68d04cbf5d7371bf124dd02f90706ecb59e4716b87ec4a66784ef201300684b`；7个相关生产入口远端编译通过。
- 启动前独立确认release、output root和stdout目标原先均不存在，同名进程不存在；GPU0～GPU7均无计算进程。
- 冻结真实`ADV3B02_CORE90_SOFT_E200`checkpoint smoke通过：`REAL_FUSION_R8_CHECKPOINT_BUNDLE_NO_QUERY_SMOKE_PASS`；迁移路径为`rank0_legacy_shell`，真实双分支fusion inner参数5458／1055125，占0.517285%，rank=8、正式3步、`query_read=false`、`target_read=false`。
- smoke把临时bundle严格保存并回读，`bundle_trainable_fraction=0.005172846819097263`，bundle大小4289426字节；没有加载Phase2数据或产生性能结果。
- smoke完成时状态为`LANDED`，随后按冻结命令只启动了一次Phase1训练。

## 启动健康证据

- 唯一启动于N607时间2026-08-25 09:44:29；shell owner PID=`2775067`，训练子进程PID=`2775068`，没有重复启动。
- CWD严格为release checkout；训练cmdline逐项绑定rank-8 config、`--meta_adapter_rank 8 --meta_adapter_sites fusion`、冻结checkpoint、ManySig、独立output root、seed392002和GPU0。
- GPU0回读训练PID`2775068`、显存488MiB；stdout从0增长至965字节并进入WiSig初始化，未出现异常、OOM、Killed、NaN或Inf。
- 首次启动SSH因后台shell继承连接未自动退出；只关闭了该已知本地SSH客户端。独立回读确认远端shell与训练子进程继续运行，未重启、未终止任何远端进程。
- 启动健康检查时最高状态为`RUNNING`；随后任务自然完成，未执行kill、restart或覆盖。

## Phase1完成与本地闭合

- N607时间2026-08-25 10:04:29自然完成；训练PID和父shell均退出，GPU0不再有该run计算进程。stdout明确记录`status=ARTIFACTS_COMPLETE`，无Traceback、OOM、Killed、NaN或Inf。
- 9个规定artifact全部非空：正式bundle4289426字节、训练日志174681字节、训练指标5044字节、训练曲线130836字节、最终评价3343字节、P0评价3405字节、冻结原型4412字节、summary3367字节、config snapshot2362字节。
- 本地证据目录：`E:\type10-7\local_artifacts\meta_adapter_recovery\phase1_fusion_r8_r1_complete_20260825`。严格bundle回读成功，只含`id_backbone.meta_adapter_fusion`和`dom_backbone.meta_adapter_fusion`的10个张量；无分类头；实际预算5458／1055125，占0.517285%。
- `metrics.csv`完整覆盖step0～199；`logs.jsonl`为800个episode、每步4个、正式`inner_steps=3`。任务分布严格为`Q_SAME_DOMAIN=320`、`Q_RX_HOLDOUT=160`、`Q_DAY_CHANNEL_HOLDOUT=120`、`Q_CLEAN_TO_LEO=120`、`Q_LEO_CROSS=80`；K分布严格为K1=280、K2=200、K5=160、K10=160。outer loss和三类episode loss全部有限。
- `run_summary.json`给出`SOURCE_SELECTION_ELIGIBLE`；config不含target字段，Phase1保持source-only。

## Clean与三类LEO weak结果

|场景|P0均值|Final均值|均值变化|P0 floor|Final floor|floor变化|
|---|---:|---:|---:|---:|---:|---:|
|clean|92.0131%|91.7512%|-0.2619pp|87.9143%|86.3929%|-1.5214pp|
|leo_clear_weak|79.1738%|78.5655%|-0.6083pp|52.2643%|55.6500%|+3.3857pp|
|leo_low_elev_weak|75.1393%|74.4512%|-0.6881pp|45.3143%|48.2000%|+2.8857pp|
|leo_rain_weak|74.9202%|74.3619%|-0.5583pp|43.9214%|46.8643%|+2.9429pp|

## 结论与下一步

- rank-8 fusion在Phase1呈现明确的“提高LEO旧类floor、牺牲平均准确率”权衡：三类LEO floor均提高约2.89～3.39pp，但三类LEO均值均下降约0.56～0.69pp，clean均值和floor也下降。该结果不能表述为目标域正向收益。
- 工程、协议、资源和source-only选择均闭合，满足进入原预登记同row单seed Target5最小可证伪矩阵的条件。Target5仍必须truth-blind运行后由独立scorer连接truth；只有`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25，否则记录科学失败并继续下一个少层候选。

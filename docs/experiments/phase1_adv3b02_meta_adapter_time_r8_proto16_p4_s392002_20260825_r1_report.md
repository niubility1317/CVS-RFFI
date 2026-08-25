# Time-only Rank-8 Prototype-Aligned Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1`
- 状态：`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 冻结代码／配置提交：`6eefefa29c3dca85944d0a8e0deae0fcc351ea62`；首次push后独立回读远端分支OID一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`checkpoint。
- 科学锚点：time+fusion rank4与time-only rank8在Target5均只改变余弦分数而无决策变化；fusion rank8产生6个决策变化但low-elev净负。

## 候选与唯一机制变量

- 保持time-only rank8、P4 FOMAML+Meta-SGD、200个outer step、相同source episode池和正式3步更新；不再扩大rank或增加adapter位置。
- 唯一机制变化是把Phase1内层、外层和源选择全部从冻结旧分类头对齐到Phase2实际使用的`frozen_prototype_cosine_ce_v1`空间，并固定`support_logit_scale=16.0`。温度只放大support／meta-query交叉熵梯度，不改变余弦argmax判决，不新增或训练分类头、协方差、LDA或持久新头。
- 双分支预计可训练参数仍为5458／1055125，占0.517285%；10个可训练张量只含`id/dom_backbone.meta_adapter_time`，低于1%。

## Phase1数据与训练边界

- 只读取WiSig source receiver0～6、source days0～1；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- clean测试固定days2～3，与训练／选择物理样本不相交；训练中不读取任何Phase2 target support、target query、query真值／角色或target receiver字段。
- 任务池固定20个episode，权重为`Q_SAME_DOMAIN=40%`、`Q_RX_HOLDOUT=20%`、`Q_DAY_CHANNEL_HOLDOUT=15%`、`Q_CLEAN_TO_LEO=15%`、`Q_LEO_CROSS=10%`；K取1／2／5／10。
- 最终必须评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地验证

- RED：新增6项部署对齐测试中5项按预期失败，分别覆盖bundle元数据、Phase1训练器、Phase2温度使用和Phase1配置；非正温度负测原本已被旧严格键规则拒绝。
- GREEN：6项新增测试全部通过；额外证明改变冻结分类头不会改变prototype-aligned源曲线或Meta-SGD更新。
- 269项Meta-Adapter Phase1／Phase2邻近回归和11个相关生产入口编译通过。
- 候选唯一一次独立P0/P1审查结论为P0无、P1无；审查另用真实`lite_d dual`临时bundle严格回读确认objective／scale为`frozen_prototype_cosine_ce_v1`／16.0，实际预算5458／1055125=0.517285%，runner与smoke均从同一bundle审计读取该设置，query仍在适配冻结后才打开。
- 配置：`configs/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1.json`。相对time-only rank8只允许`run_id`、`adapter.adaptation_objective`和`adapter.support_logit_scale`变化。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1.out`
- 冻结命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1.json --meta_adapter_rank 8 --meta_adapter_sites time --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

- expected artifacts：正式bundle、冻结原型、`logs.jsonl`、`metrics.csv`、训练曲线、P0／final四场景评价、`run_summary.json`和config snapshot。
- 技术停止规则：只在协议越权、错误checkout／数据split、输出覆盖、launcher-wide故障、无训练进展或重复确定性异常时停止；不得因中间性能低停止。

## Release与真实checkpoint smoke

- 最终release提交：`9f7c2fad9eda66bad8aa0b43c4f4f60dd5ffc0e2`；独立回读远端分支OID一致。
- release归档：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1_release.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1_release.tar.gz`；唯一一次本地／远端SHA256均为`a5c3927da8f0ca693d7ffaebae0329b41ac72bcf760bcd54d72ce8f533ac1b82`。
- N607只读核对确认release、output root、stdout和同名进程原先均不存在；冻结checkpoint与WiSig数据路径存在；GPU0预检为空闲。
- run专属checkout解压后8个相关入口远端编译通过。
- 真实`ADV3B02_CORE90_SOFT_E200`checkpoint无query smoke通过：只存在`id/dom_backbone.meta_adapter_time`，可训练参数5458／1055125=0.517285%，正式3步；bundle严格回读`frozen_prototype_cosine_ce_v1`／16.0，`query_read=false`、`target_read=false`。
- 2026-08-25 12:47:46 HKT由唯一owner按冻结命令启动，主PID=`2866002`；一次启动检查确认PPID=1、CWD／cmdline／run root均与预登记一致，GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`，进程显存488MiB，stdout已产生启动记录。

## 后续门槛

Phase1只有在source-only选择规则允许且9个artifact完整时进入同row单seed Target5。Target5仍以`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp作为Target25门槛；失败则记录科学失败并继续下一少层候选。

## Phase1完成与独立回读

- 进程于2026-08-25 13:08:25 HKT自然完成，stdout状态为`ARTIFACTS_COMPLETE`；无Traceback、OOM、NaN或Inf。9／9项预期artifact均非空，并已下载到`E:\type10-7\local_artifacts\meta_adapter_recovery\phase1_time_r8_proto16_p4_r1_complete_20260825`独立回读。
- 训练闭合为200／200个outer step、800／800个episode，每步4个episode；任务类型计数为320／160／120／120／80，K-shot计数为280／200／160／160，全部3步更新，所有loss有限。
- 正式bundle严格回读：只含10个`id/dom_backbone.meta_adapter_time`张量，可训练参数5458／1055125=0.517285%，无head／cls／LDA／cov；objective／scale为`frozen_prototype_cosine_ce_v1`／16.0。
- `V_select`两个holdout的A0→A3分别为100%→100%和100%→91.6667%，worst delta=-8.3333pp；runner仍按冻结source-only规则给出`SOURCE_SELECTION_ELIGIBLE`。这只允许进入Target5，不代表目标域收益。

| 场景 | P0均值 | final均值 | 均值变化 | P0 floor | final floor | floor变化 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 92.0464% | 92.0488% | +0.0024pp | 87.8286% | 87.7214% | -0.1071pp |
| leo_clear_weak | 79.2167% | 78.9833% | -0.2333pp | 52.2500% | 53.3643% | +1.1143pp |
| leo_low_elev_weak | 75.1821% | 74.8429% | -0.3393pp | 45.2786% | 46.1286% | +0.8500pp |
| leo_rain_weak | 74.9262% | 74.6262% | -0.3000pp | 43.8571% | 44.5214% | +0.6643pp |

Phase1结论：prototype-aligned训练把三类LEO的尾类floor提高0.66～1.11pp，但牺牲约0.23～0.34pp总体均值。其目标域快速适配价值必须由同row Target5的`DA1_REG0-DA0_REG0`直接证伪；不得把Phase1 source结果宣传为域适应成功。

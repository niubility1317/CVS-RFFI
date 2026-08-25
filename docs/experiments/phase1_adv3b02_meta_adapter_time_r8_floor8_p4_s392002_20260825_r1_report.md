# Time-only Rank-8 Class-Floor Scale-8 Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1`
- 状态：`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`a2f2461620f518b23d6a7506fd7a493c08928854`；push后独立回读远端OID一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`checkpoint。
- 科学锚点：普通prototype CE的scale1未改变判决；scale8聚合旧类均值-0.1667pp、floor 0pp；scale16聚合均值+0.1111pp但clear floor下降5pp。因此停止继续搜索固定scale，改测support弱类方向约束。

## 候选与唯一机制变量

- 保持time-only rank8、P4 FOMAML+Meta-SGD、200个outer step、相同source episode池、scale8和正式3步更新；adapter位置、rank、冻结原型和最终余弦argmax判决均不变。
- 唯一机制变量为`frozen_prototype_class_floor_ce_v1`：每步先计算普通support平均CE和各support类平均CE，再以固定tau=0.2计算减去`tau*log(类数)`的归一化smooth-max，最终损失为二者各50%。各类同难时严格退化为普通CE，弱类更难时才提高其梯度权重。
- 该损失只读取当前合法support IQ／标签和冻结原型，不读取query、source cache或持久分类状态。双分支可训练参数仍为5458／1055125=0.517285%；10个张量只含`id/dom_backbone.meta_adapter_time`，不含分类头、协方差、LDA或持久新头。

## Phase1数据与矩阵

- 只读取WiSig source receiver0～6、source days0～1；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- clean测试固定days2～3，与训练／选择物理样本不相交；训练中不读取Phase2 target support、target query、query真值／角色或target receiver字段。
- episode池固定20个，权重为`Q_SAME_DOMAIN=40%`、`Q_RX_HOLDOUT=20%`、`Q_DAY_CHANNEL_HOLDOUT=15%`、`Q_CLEAN_TO_LEO=15%`、`Q_LEO_CROSS=10%`；K取1／2／5／10。
- 最终checkpoint评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地RED→GREEN与独立审查

- 配置：`configs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1.json`。
- RED：新增测试首先因新objective常量不存在而在收集阶段失败。GREEN：5个聚焦文件175项通过；完整邻近回归274项通过；11个生产入口编译通过；配置独立解析为rank8／time-only／3步／class-floor／scale8。
- 唯一一次独立P0/P1审查结论为P0无、P1无。审查确认Phase1→bundle→Phase2实际贯通同一objective／scale，只读support，适配冻结后才打开query，参数比例0.517285%，无新增head／LDA／cov／source／clean／cache路径。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1`
- stdout：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1.out`
- 冻结命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1.json --meta_adapter_rank 8 --meta_adapter_sites time --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

- expected artifacts：正式bundle、冻结原型、`logs.jsonl`、`metrics.csv`、训练曲线、P0／final四场景评价、`run_summary.json`和config snapshot。
- 技术停止规则：只在协议越权、错误checkout／数据split、输出覆盖、launcher-wide故障、无训练进展或重复确定性异常时停止；不得因中间性能低停止。

## 后续门槛

Phase1仅在source-only选择规则允许且9个artifact完整时进入同row单seed Target5。Target5仍以`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp作为Target25门槛；失败则记录科学失败并继续下一少层候选。

## Release与真实checkpoint无query smoke

- release提交：`1eb9e0bcd6f9c867c6a6c7161d59e2cd30e4f844`；远端分支OID独立回读一致。归档本地／远端唯一一次SHA256均为`057f76d0864db761453b0f38eee9292ce7530fcff0b3eda45207be4b70b461ad`；run专属checkout内11个生产入口远端编译通过，配置回读为目标run ID、`frozen_prototype_class_floor_ce_v1`和scale8。
- 发布前只读核对确认release／run／stdout／smoke目标原先不存在，无同名训练进程；冻结checkpoint、WiSig、既有VALIDATED_ONCE support和冻结原型存在，GPU0空闲。
- 以真实已训练scale8 meta bundle仅替换objective元数据作为pre-Phase1运行时探针；严格bundle回读5458／1055125=0.517285%，随后在既有合法Target5首row support上完成3次真实反向传播。receipt为`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`，`frozen_prototype_class_floor_ce_v1`／8.0，`query_opened=false`、`source_opened=false`、`query_state_update_count=0`，不产生性能结论。
- 2026-08-25 14:22:41 HKT由唯一owner按冻结命令启动，主PID=`2913629`；一次启动检查确认PPID=1、CWD／cmdline／run root均与预登记一致，GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`、显存486MiB，stdout已增长并进入训练初始化。

## Phase1完成与独立回读

- 进程自然完成，stdout状态为`ARTIFACTS_COMPLETE`；无Traceback、RuntimeError、CUDA OOM、NaN或Inf。9／9项正式artifact均非空，完整下载至`E:\type10-7\local_artifacts\meta_adapter_recovery\phase1_time_r8_floor8_p4_r1_complete_20260825`独立回读。
- 训练闭合为200／200个outer step、800／800个episode，每步4个episode且全部3步更新，所有loss有限。正式bundle严格回读`frozen_prototype_class_floor_ce_v1`／8.0、5458／1055125=0.517285%，10个可训练张量均为id/dom time adapter，无head／cls／LDA／cov。
- `V_select`两个holdout的A0→A3分别为100%→100%和100%→91.6667%，worst delta=-8.3333pp；冻结source-only规则给出`SOURCE_SELECTION_ELIGIBLE`，仅允许进入Target5。

| 场景 | P0均值 | final均值 | 均值变化 | P0 floor | final floor | floor变化 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 92.0464% | 92.1429% | +0.0964pp | 87.8286% | 88.2357% | +0.4071pp |
| `leo_clear_weak` | 79.2167% | 78.9345% | -0.2821pp | 52.2500% | 54.1571% | +1.9071pp |
| `leo_low_elev_weak` | 75.1821% | 74.7726% | -0.4095pp | 45.2786% | 47.1429% | +1.8643pp |
| `leo_rain_weak` | 74.9262% | 74.5155% | -0.4107pp | 43.8571% | 45.3786% | +1.5214pp |

Phase1结论：class-floor机制在source四场景稳定抬高弱类floor，但LEO均值仍小幅下降。该证据只允许进入同row Target5，不能替代目标域`DA1_REG0-DA0_REG0`判定。

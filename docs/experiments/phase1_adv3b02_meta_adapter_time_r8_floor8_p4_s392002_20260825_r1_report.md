# Time-only Rank-8 Class-Floor Scale-8 Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1`
- 状态：`LOCAL_VERIFIED`
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

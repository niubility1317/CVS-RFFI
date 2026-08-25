# Time-only Rank-8 Prototype-Aligned Scale-8 Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1`
- 状态：`RUNNING`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`2a7435bef0fc1292d31fe3ac3082bada76caeefd`；首次push后独立回读远端分支OID一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`checkpoint。
- 科学锚点：scale16在Target5产生52个决策变化和聚合均值+0.1111pp，但clear weak floor下降5pp；旧scale1则0个决策变化。

## 候选与唯一变量

- 保持time-only rank8、P4 FOMAML+Meta-SGD、200个outer step、相同source episode池和正式3步更新；不改变adapter位置、rank或判决规则。
- 唯一候选变量是把Phase1内层、外层、源选择及后续Phase2共同使用的`support_logit_scale`从16.0降到8.0；objective仍为`frozen_prototype_cosine_ce_v1`。目的是在scale1不动与scale16过冲之间寻找可证伪中点。
- 双分支可训练参数保持5458／1055125=0.517285%；10个可训练张量只含`id/dom_backbone.meta_adapter_time`，不含分类头、协方差、LDA或持久新头。

## Phase1数据与训练边界

- 只读取WiSig source receiver0～6、source days0～1；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- clean测试固定days2～3，与训练／选择物理样本不相交；训练中不读取Phase2 target support、target query、query真值／角色或target receiver字段。
- 任务池固定20个episode，权重为`Q_SAME_DOMAIN=40%`、`Q_RX_HOLDOUT=20%`、`Q_DAY_CHANNEL_HOLDOUT=15%`、`Q_CLEAN_TO_LEO=15%`、`Q_LEO_CROSS=10%`；K取1／2／5／10。
- 最终评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地验证与审查

- 配置：`configs/phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1.json`。相对scale16配置只允许`run_id`和`adapter.support_logit_scale`变化。
- 复用已通过269项邻近回归、11个生产入口编译和真实bundle回读的prototype-aligned实现；本候选只新增合法配置值8.0，不修改代码。
- 唯一一次独立P0/P1配置审查结论为P0无、P1无；审查确认相对scale16仅`run_id`和scale 16→8变化，真实dual模型仍为5458／1055125=0.517285%，实际bundle严格回读和Phase2 runner均得到`frozen_prototype_cosine_ce_v1`／8.0，无新增head、source或query路径。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1.out`
- 冻结命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1.json --meta_adapter_rank 8 --meta_adapter_sites time --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

- expected artifacts：正式bundle、冻结原型、`logs.jsonl`、`metrics.csv`、训练曲线、P0／final四场景评价、`run_summary.json`和config snapshot。
- 技术停止规则：只在协议越权、错误checkout／数据split、输出覆盖、launcher-wide故障、无训练进展或重复确定性异常时停止；不得因中间性能低停止。

## 后续门槛

Phase1仅在source-only选择规则允许且9个artifact完整时进入同row单seed Target5。Target5仍以`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp作为Target25门槛；失败则记录科学失败并继续下一少层候选。

## Release与真实checkpoint smoke

- 最终release提交：`c399fc047af05f07ae577c637d0ecda800f4902a`；远端分支OID独立回读一致。release归档本地／远端唯一一次SHA256均为`92cb1ffd757aceddc4cb68f8083d8791d519367859798e0dfb49e63f65097af6`，run专属checkout内8个相关入口远端编译通过。
- N607只读核对确认release、output root、stdout和同名Python进程原先均不存在；冻结checkpoint与WiSig数据存在，GPU0空闲。
- 真实`ADV3B02_CORE90_SOFT_E200`checkpoint无query smoke通过：只存在`id/dom_backbone.meta_adapter_time`，可训练参数5458／1055125=0.517285%，正式3步；bundle严格回读`frozen_prototype_cosine_ce_v1`／8.0，`query_read=false`、`target_read=false`。
- 2026-08-25 13:29:58 HKT由唯一owner按冻结命令启动，主PID=`2888480`；启动检查确认PPID=1、CWD／cmdline／run root均与预登记一致，GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`，进程显存488MiB，stdout已产生启动记录。

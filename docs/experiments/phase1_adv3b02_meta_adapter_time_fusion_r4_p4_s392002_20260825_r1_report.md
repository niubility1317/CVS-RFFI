# Time+Fusion Rank-4 Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1`
- 状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`971001b5b72a59adf59d7339fa6036a04d4fc539`；GitHub远端分支OID已独立回读一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`checkpoint。
- 科学锚点：fusion-only rank-4 Target5为0.0pp／0.0pp；fusion-only rank-8 Target5为-0.2778pp／0.0pp，且6个决策变化中5个在low-elev产生负迁移。

## 候选与唯一机制变量

- 容量分支已经由rank-4和rank-8完成可证伪；本候选回到rank-4锚点，只把可适配位置从`fusion`改为`time+fusion`，不加入freq位置、不增加更新步数、不改变P4元训练任务池与损失。
- 设计依据：low-elev负迁移更可能缺少时间／相位校准方向；用户设计报告和历史晚期块结果均提示time与fusion对floor更敏感，而freq晚期块基本无收益。
- 双分支实际可训练参数5780／1055449，占0.547634%，低于1%；共20个可训练张量，只含`id/dom_backbone.meta_adapter_time`与`id/dom_backbone.meta_adapter_fusion`，无分类头、协方差、LDA或持久新头。

## Phase1数据与训练边界

- 只读取WiSig source receiver0～6、source days0～1；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- clean测试固定days2～3，与训练／选择物理样本不相交；训练中不读取任何Phase2 target support、target query、query真值／角色或target receiver字段。
- 任务池固定20个episode，权重为`Q_SAME_DOMAIN=40%`、`Q_RX_HOLDOUT=20%`、`Q_DAY_CHANNEL_HOLDOUT=15%`、`Q_CLEAN_TO_LEO=15%`、`Q_LEO_CROSS=10%`；K取1／2／5／10。
- P4保持FOMAML+Meta-SGD、200个outer step、每步4个episode、正式3步inner update；最终必须评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地验证

- RED：未注册时，Phase1 config、单分支bundle和真实双分支严格回读3项新测试全部失败。
- GREEN：只注册`rank=4+sites=time,fusion`；`rank=8+sites=time,fusion`仍拒绝；历史tri／fusion profile保持不变。
- 配置：`configs/phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1.json`。相对已闭合fusion rank-4 r2只允许`run_id`和`adapter.sites`变化。
- 配置差异断言通过：顶层只变化`run_id`与`adapter`，且adapter内部只变化`sites`；259项Meta-Adapter Phase1／Phase2邻近回归通过，9个相关生产入口编译通过。
- 测试环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；CWD为本分支worktree根目录。
- 唯一一次独立P0/P1审查结论：P0无、P1无。审查确认精确profile、真实双分支预算、冻结CLI/config/run root、source-only数据边界和旧profile兼容均正确；不需要修复或复审。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1.out`
- 冻结命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1.json --meta_adapter_rank 4 --meta_adapter_sites time,fusion --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

- expected artifacts：正式bundle、冻结原型、`logs.jsonl`、`metrics.csv`、训练曲线、P0／final四场景评价、`run_summary.json`和config snapshot。
- 技术停止规则：只在协议越权、错误checkout／数据split、输出覆盖、launcher-wide故障、无训练进展或重复确定性异常时停止；不得因中间性能低停止。

## 后续门槛

Phase1只有在source-only选择规则允许且9个artifact完整时进入同row单seed Target5。Target5仍以`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp作为Target25门槛；失败则记录科学失败并继续下一少层候选。

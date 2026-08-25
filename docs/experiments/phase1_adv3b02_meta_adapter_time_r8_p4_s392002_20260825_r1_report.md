# Time-only Rank-8 Meta-Adapter P4 Phase1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1`
- 状态：`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`227889e09e2affce2af18bce15e39c86c7436b98`；GitHub远端分支OID已独立回读一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`checkpoint。
- 科学锚点：fusion rank8已产生6个Target5决策变化但在low-elev净负；time+fusion rank4产生最大0.0301分数变化但0个决策变化。二者均未晋级。

## 候选与唯一机制变量

- 本候选从已闭合fusion rank8配置出发，只把可适配位置从`fusion`替换为`time`；rank8、P4 FOMAML+Meta-SGD、任务池、损失、200个outer step和正式3步均不变。
- 机制目的：隔离时间／相位校准方向并保留rank8的可测更新容量，排除已在low-elev显示负迁移的fusion方向；不加入freq或额外层。
- 双分支实际可训练参数5458／1055125，占0.517285%，低于1%；共10个可训练张量，只含`id/dom_backbone.meta_adapter_time`，无fusion、freq、分类头、协方差、LDA或持久新头。

## Phase1数据与训练边界

- 只读取WiSig source receiver0～6、source days0～1；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- clean测试固定days2～3，与训练／选择物理样本不相交；训练中不读取任何Phase2 target support、target query、query真值／角色或target receiver字段。
- 任务池固定20个episode，权重为`Q_SAME_DOMAIN=40%`、`Q_RX_HOLDOUT=20%`、`Q_DAY_CHANNEL_HOLDOUT=15%`、`Q_CLEAN_TO_LEO=15%`、`Q_LEO_CROSS=10%`；K取1／2／5／10。
- 最终必须评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地验证

- RED：未注册时，Phase1 config、单分支bundle和真实双分支严格回读3项time rank8接受测试全部失败。
- GREEN：只注册`rank=8+sites=time`；`rank=4+sites=time`仍拒绝；历史tri、fusion、time+fusion profile保持不变。
- 配置：`configs/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1.json`。相对已闭合fusion rank8只允许`run_id`和`adapter.sites`变化；差异断言通过。
- 5项新接受／拒绝测试通过；262项Meta-Adapter Phase1／Phase2邻近回归通过，9个相关生产入口编译通过。测试环境为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，CWD为本分支worktree根目录。
- 唯一一次独立P0/P1审查结论：P0无、P1无。审查确认只新增`rank8+time`精确profile、time rank4继续拒绝、真实双分支预算和可训练集合正确、source-only config未漂移、冻结CLI必须显式保留rank／site／独立run root；不需要修复或复审。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1.out`
- 冻结命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1.json --meta_adapter_rank 8 --meta_adapter_sites time --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

- expected artifacts：正式bundle、冻结原型、`logs.jsonl`、`metrics.csv`、训练曲线、P0／final四场景评价、`run_summary.json`和config snapshot。
- 技术停止规则：只在协议越权、错误checkout／数据split、输出覆盖、launcher-wide故障、无训练进展或重复确定性异常时停止；不得因中间性能低停止。

## 后续门槛

Phase1只有在source-only选择规则允许且9个artifact完整时进入同row单seed Target5。Target5仍以`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp作为Target25门槛；失败则记录科学失败并继续下一少层候选。

## Release与真实checkpoint smoke

- release提交：`f4ab5e6684a2f5fea6a6a3336bc580e3de0ce1fb`；GitHub远端分支OID独立回读一致。
- 唯一release归档映射：本地`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1_release.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1_release.tar.gz`。本地／远端一次SHA256均为`682464a53f21d1435733617fd1ae4d9d81950d602ff776945972a60545920b43`。
- 新release、run root和正式stdout日志在落地前均不存在；未发现同run进程，GPU0～7均无计算进程。release已在独立checkout解包，9个生产入口远端编译通过。
- 真实`ADV3B02_CORE90_SOFT_E200`checkpoint无query smoke通过：严格bundle保存／回读成功；只发现id／dom两个time adapter模块、10个可训练张量、5458／1055125=0.517285%；rank=8、正式3步；`query_read=false`、`target_read=false`，无fusion、freq、分类头、LDA或协方差参数。

## 启动与一次健康检查

- 2026-08-25 11:34:33（N607）唯一启动成功，主PID为`2829308`；PID的CWD、完整cmdline、独立run root和GPU0映射均与预登记一致，未覆盖既有路径。
- ManySig初始化期间进程保持100% CPU并累计读取约2.39GB，先生成`config_snapshot.json`；没有把该阶段误报为训练进展。
- 11:45:56首次确认`logs.jsonl`真实增长至111个episode记录、outer step 27；记录均为正式`inner_steps=3`且loss有限。stdout无Traceback、RuntimeError、ValueError或CUDA OOM，故状态提升为`RUNNING`。

## Phase1最终证据

- 进程于2026-08-25 11:55:49自然完成，stdout状态为`ARTIFACTS_COMPLETE`，无Traceback、OOM、NaN或Inf；9／9项预期artifact均非空并已下载到`E:\type10-7\local_artifacts\meta_adapter_recovery\phase1_time_r8_r1_complete_20260825`独立回读。
- 训练精确覆盖outer step 0～199和800个episode，每步4个；任务分布为`Q_SAME_DOMAIN=320`、`Q_RX_HOLDOUT=160`、`Q_DAY_CHANNEL_HOLDOUT=120`、`Q_CLEAN_TO_LEO=120`、`Q_LEO_CROSS=80`；K分布为K1=280、K2=200、K5=160、K10=160。所有episode均为3步，全部loss有限。
- 正式bundle严格回读通过：5458／1055125=0.517285%，10个可训练张量只属于id／dom双分支的time adapter；无fusion、freq、分类头、LDA或协方差参数。
- source-only选择结论为`SOURCE_SELECTION_ELIGIBLE`：两个`V_select` holdout的A0→A3分别为1.0000→1.0000和0.8333→0.8333，最差A3变化为0.0pp；配置和run summary均无target键，`source_only=true`。

|场景|P0均值|最终均值|均值变化|P0 floor|最终floor|floor变化|
|---|---:|---:|---:|---:|---:|---:|
|clean|92.0464%|92.3619%|+0.3155pp|87.8286%|87.8786%|+0.0500pp|
|`leo_clear_weak`|79.2167%|79.1750%|-0.0417pp|52.2500%|52.0071%|-0.2429pp|
|`leo_low_elev_weak`|75.1821%|75.0845%|-0.0976pp|45.2786%|44.9429%|-0.3357pp|
|`leo_rain_weak`|74.9262%|74.7952%|-0.1310pp|43.8571%|43.3500%|-0.5071pp|

- 解释边界：time-only rank8在source clean上小幅正收益，但三类LEO weak均值和floor轻微下降；source选择规则仍允许进入Target5。这些source-side变化不等于目标域适配收益，必须由同rowtruth-last `DA1_REG0-DA0_REG0`裁决。

# CVS_META_ADAPTER_FUSION_R4_P4_V1最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1`
- 当前状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 固定代码与配置提交：`d2d06a25a17d488bd5d9ab9410264dab47764897`

## 候选与停止规则

本候选把可训练站点从time、freq和fusion三层收窄为仅fusion层，保留rank=4、P4 FOMAML+Meta-SGD、source-only元训练和正式3步适配。P1～P4三层候选已在Target5全部闭合为零收益，本轮不重复训练P1～P3；Phase1入口通过`active_candidate_id=P4`只运行P4，并同时产生同checkpoint的P0控制评价。

Phase1只读取receiver0～6、day0～1的source角色样本训练和选择；最终checkpoint必须评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。Phase1合法完成后才进入同row单seed Target5。仅当`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp时进入Target25；否则记录科学失败并推进下一机制。

## 本地验证与独立P0/P1审查

- RED确认旧校验拒绝fusion-only和通用schema；GREEN只登记`[fusion]`与既有`[time,freq,fusion]`两种profile，旧`tri_r4`schema仍只能使用三站点。
- 真实dual模型只创建`id_backbone.meta_adapter_fusion`和`dom_backbone.meta_adapter_fusion`；内循环参数2,890/1,052,557，占0.2746%，共10个非分类参数名，低于1%。
- Meta-Adapter Phase1/Phase2聚焦及邻近回归251项通过。
- 唯一一次独立P0/P1审查未发现P0，发现1个P1：直接训练命令若省略站点参数会回退旧三层默认。修复方式是不改变代码路径，而是在下方冻结命令中显式加入`--meta_adapter_sites fusion`。仅针对原问题的定点复审确认命令、config和入口值均为`fusion`，原P1已解除；未扩展审查范围。

## N607最小预登记

- 账户：普通`N607`用户`szu2070436088`，不使用管理员账号。
- 环境：现有`CVS-RFFI`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- GPU：0；启动前核对资源，不超过每GPU两个训练进程。
- 本地release归档：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1_release.tar.gz`
- 远端release归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1_release.tar.gz`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1.out`
- GPU：0。

冻结训练命令如下，必须原样包含`--meta_adapter_sites fusion`：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/train.py --dataset wisig --wisig_pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --init_checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --use_cvs_meta_adapter --meta_config configs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1.json --meta_adapter_rank 4 --meta_adapter_sites fusion --meta_inner_steps 3 --meta_inner_max_steps 5 --meta_output_root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_fusion_r4_p4_s392002_20260825_r1 --seed 392002 --wisig_equalized 1 --wisig_out_len 256 --wisig_domain rx_day --wisig_max_day123_per_combo 0 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_train_days 0,1 --wisig_test_days 2,3 --device cuda
```

预期artifact：`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`run_summary.json`、`config_snapshot.json`、`source_adaptation_curve.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`和`frozen_prototypes.npz`。

技术停止只允许协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、确定性重复异常或进程归属不清；不得因中间性能低而停止，也不得干预无关进程。

## N607发布与真实checkpoint smoke

- 启动前只读核对确认release目录、run root和stdout日志均不存在，run ID无匹配进程；GPU0～GPU7利用率均为0且无计算进程。冻结checkpoint为8,582,116字节，ManySig为2,359,341,461字节。
- release归档固定源提交为`a0a2db9f7bbe2c1f19850698cdde65581434ab59`，35,503,098字节、5,018个条目。唯一一次本地/远端SHA256均为`c6883a79156da8cfac6344751750271f52ab1d8bc2d1c4f1230ea67cc036351c`。
- 远端checkout内7个相关生产入口编译通过。
- 第一次外部smoke在真实checkpoint加载、站点和预算断言均通过后，因smoke脚本误把dual返回键写成`logits`而报`KeyError`；未创建run root、未读取target/query，也未修改release代码。只把非生产smoke断言改为真实接口`tx_logits`后再次运行。
- 第二次真实checkpoint smoke输出`REAL_FUSION_CHECKPOINT_NO_QUERY_SMOKE_PASS`：迁移路径为`rank0_legacy_shell`，站点仅为两个fusion adapter，内循环参数2,890/1,052,557，占0.2746%，正式3步，`query_read=false`、`target_read=false`，前向logits形状合法。

当前最高状态为`LANDED`。尚未证明训练启动、Phase1完成或目标域正向收益；后续证据按`RUNNING→ARTIFACTS_COMPLETE→ANALYZED`追加。

## N607启动健康检查

- 2026-08-25 08:04（Asia/Hong_Kong）完成唯一一次不可覆盖启动，训练PID=`2724363`。
- 启动脚本在打印PID后因本地管道附带的末尾CR使远端bash返回非零；没有据此重复启动。独立回读确认PID存在且唯一，CWD为预登记checkout，cmdline逐项绑定冻结config、run root、checkpoint、ManySig、GPU0和`--meta_adapter_sites fusion`。
- run root已创建且初始只含`config_snapshot.json`；stdout为576字节，包含启动时间、`Device: cuda`及三类LEO评价声明，未出现异常指纹。
- 56秒复核时进程状态为`Rl`，累计CPU时间56秒、CPU占用100%，GPU0显存594MiB；其余GPU无计算进程。日志暂未继续增长，结合持续CPU进展判定处于ManySig/源角色准备阶段，不构成非进展。

当前最高状态为`RUNNING`。这只证明唯一run启动健康，不代表训练或四场景评价完成，更不代表目标域正向收益。

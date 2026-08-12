# D92 TCRA safe-v2 Hard9+K1实验报告

## 状态

- run ID：`d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v2`
- 状态：`LOCAL_VERIFIED`
- 日期：2026-08-12
- 目标：仅在最难的9个未见performance outer上比较`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`与同outer的`E0_FULL_ONLY`，另保留1个K1协议活性行。
- 声明：development-only hard screen；完成前不作性能结论。

## 假设与严格裁决

候选只后处理E0 FULL的D42`coef2_qint8`，不增加fit、query MAC或持久状态。9个performance outer的均值必须同时满足：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy全部严格上升；forgetting、new→old、old→new全部严格下降。任一持平或反向即`REJECT_ROUTE`。

大胆提升目标依次为：`+1.0pp,+1.5pp,+1.0pp,+4.0pp,+0.5pp,-1.5pp,-0.5pp,-0.5pp`。资源硬门：registration wall P90≤150ms、相对E0配对wall中位数≤1.50、峰值增量≤512KiB、query/state exact、component-fit reduction≥0.80。

## 冻结矩阵

- schema：`p2_min_v1`；数据：既有`VALIDATED_ONCE`，不重验。
- 9个performance outer+1个K1 liveness outer；每个3个`leo_*_weak`场景；共10 jobs、30 scene-arm、8 shards。
- G0 outer`rx_7_7__seed_713106__k_10__new_5`已在独立truth-free G0使用，因此从本矩阵排除。
- selection SHA256：`4fc836fbe3960cf95bfdf9fdb9eba1d311fb47fa4cc2ff89b64acab7e88f8e61`。

## v1技术失败与唯一修复

v1在固定K5 smoke的`leo_low_elev_weak`已得到`active=true`、`fallback=false`、`safe_directional_pass=true`，但runner错误要求`aggregate_saturation_count==0`，把合法的1个已跳过饱和原子拒绝为`fit audit TCRA atomic receipt drift`；正式shard未启动，状态为`NO_PERFORMANCE_RESULT`。

v2只修runner收据闭包：`0<=aggregate_saturation_count<=rejected_atomic_ascent_count`。方法、阈值、矩阵、数据、smoke outer与裁决均不变。TDD覆盖合法值1及越界-1、3；定向测试26项通过；独立复核`P0=0,P1=0,APPROVE`。代码提交：`b66644db`。

## 交付与服务器路径

- runtime archive：`d92_tcra_safe_v2_hard9_runtime_b66644db.tar.gz`，5109182 bytes，SHA256`2586ead2ff1d9b3822e8772c26674b1a91ac3cd0e847c2aa76b42bd03c25ce0c`。
- config：`stage2_d92_tcra_safe_v2_hard10_v1.json`，7213 bytes，SHA256`9740ebd8f7368ea73bf8bdfb1ff57735e7407f89dab7b51a834988c4d6f9f13e`。
- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_safe_v2_hard9_source_b66644db_20260812_v2`。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v2`。
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v2`。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；每个shard独占一个GPU可见域并使用`cuda:0`。

## 健康、停止与产物

启动前必须核对Git/三件套hash、远端source/output/log不存在、8卡和同run进程。唯一detached launch后先执行真实checkpoint truth-free smoke，再启动8 shards。仅P0协议/安全违规、launcher确定性故障、错误hash/checkout、覆盖风险，或两个distinct outer在prediction前同指纹失败时停止；不得按性能停止，且同run不重试。

期望产物：10个job receipt、20组before/after prediction/COMMIT/fit/resource/execution、9个performance score、1个K1 score、8个shard summary。分析仅在全部immutable prediction闭合后由独立truth-side scorer完成，并与E0_FULL_ONLY同outer逐项配对。

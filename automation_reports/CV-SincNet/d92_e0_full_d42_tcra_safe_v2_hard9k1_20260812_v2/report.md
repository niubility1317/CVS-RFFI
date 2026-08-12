# D92 TCRA safe-v2 Hard9+K1实验报告

## 状态

- run ID：`d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v2`
- 状态：`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`
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

## Runner执行交接（2026-08-12）

本run按冻结交付物执行一次，smoke通过后启动8 shards；未改方法、代码、config、矩阵或阈值，未重试同run，未运行analyzer，runner未读取或解释性能。

### 状态与健康

- 唯一detached命令已执行一次；`fresh_run_retry=false`。
- smoke状态：`D92_TCRA_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；prediction/COMMIT/fit闭合，query fit/update/selection/truth/role/quota/global均false。
- prepare：`job_count=10`、`scene_arm_count=30`、manifest SHA256=`cb0832d90d30dff3804c403f31e2a1ededa0ac294cf32bc29640fcc701d2dc5c`。
- 8个shard日志均`status=PASS`、`failed_job_count=0`、stderr为空；未触发systemic stop。

### 最终计数

|类别|实际|
|---|---:|
|job receipts|10/10|
|prediction artifacts|20/20|
|COMMIT|20/20|
|fit/resource/execution|20/20各|
|score receipts|10/10|
|shard summaries|8/8|
|systemic stop|ABSENT|

### 远端/本地树证据

远端与本地完整取回根：`E:\type10-7\local_artifacts\d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v2`。tree SHA按`relative_path<TAB>size<TAB>file_sha256`计算，远端与本地逐项manifest diff均为0。

|树|count|bytes|tree SHA256|
|---|---:|---:|---|
|source|1338|70,936,865|`eba2ee0a79292e8196411bf48fc9d5e5740402d9ec59d4a14123f09e6bf76b04`|
|output|190|11,679,067|`b2ede3bb984101d8a528ba74482445744e31a6fb5c718c5c1ef41d8ec08a3a54`|
|logs|22|9,900|`3f18b955524fc5954d05d83bbcb16a3d9f2af0bfac5cb45e16ffb9a113cc567d`|
|truth sidecars|10|5,016,265|`d3b92a650c36a2d22c118dfef5a27738748013f468b7a5a69b8a45f3f597a627`|

10个manifest引用truth sidecar均完成远端/本地SHA核对，未打开truth内容。

### 清理

最终远端精确run进程=0、GPU compute=0；每次SSH/SCP后本地`ssh.exe`=0、TCP22 Established=0。共享代码仓库HEAD保持`db42a14c8cc9f6f715d096f472e94c0dd4b46fca`，runner未产生代码改动。

## 运行闭合

- 状态：`ANALYZED / REJECT_ROUTE`。
- 唯一detached launch执行1次；真实K5 smoke通过，8个shard均启动。
- 正式10/10 job完成；20/20 prediction、COMMIT、fit、resource、execution闭合；10/10 score；8/8 shard summary均`PASS`，failed=0；stderr为空，systemic stop不存在。
- 9个K>2 performance outer全部`active=true`、`fallback=false`；K1保持精确E0 alias；所有query fit/update/selection/truth/role/quota/global-reassignment字段均为false。
- 性能分析排除K1，只使用9个performance outer与完整125实验中同outer的`E0_FULL_ONLY`原始score逐行配对。

## 8指标总体结果

所有数值为9个performance outer的算术均值，单位为百分点；forgetting和双向混淆越低越好。

| 指标 | E0_FULL_ONLY | TCRA safe-v2 | 差值 | 严格方向通过 |
|---|---:|---:|---:|---|
| H_old_new | 72.216594 | 72.237968 | +0.021374 | 是 |
| old balanced accuracy | 73.641975 | 73.641975 | +0.000000 | 否 |
| c_old_acc | 73.641975 | 73.641975 | +0.000000 | 否 |
| old floor | 42.777778 | 42.777778 | +0.000000 | 否 |
| seen-new accuracy | 71.000000 | 71.037037 | +0.037037 | 是 |
| average forgetting | 13.271605 | 13.271605 | +0.000000 | 否 |
| new→old | 14.824074 | 14.787037 | -0.037037 | 是 |
| old→new | 15.987654 | 15.987654 | +0.000000 | 否 |

严格8项Pareto门仅3/8通过；大胆提升门0/8通过。路线直接`REJECT_ROUTE`，不得进入完整125。

## 同row差值

| performance outer | ΔH | Δold BA | Δc_old | Δfloor | Δseen-new | Δforgetting | Δnew→old | Δold→new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rx_3_19/713102/K10/new5 | +0.192368 | 0 | 0 | 0 | +0.333333 | 0 | -0.333333 | 0 |
| rx_7_14/713104/K10/new10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_7_7/713103/K10/new5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_7_7/713104/K10/new5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_7_7/713104/K5/new20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_7_7/713105/K10/new20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_8_8/713103/K10/new5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_8_8/713103/K5/new20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rx_8_8/713106/K5/new20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

唯一变化来自`rx_3_19/713102/K10/new5`的`leo_clear_weak`：after预测只改变1个query，seen-new提高1.0pp、new→old降低1.0pp；其余8个outer的before/after预测均与E0逐字节同标签。六个旧类在所有9个outer上均无任何accuracy变化，因此floor和forgetting不可能改善。

## 资源结果

每个outer先取3场景registration wall/peak的中位数，再与同outer E0配对。

| 资源门 | 实测 | 阈值 | 结果 |
|---|---:|---:|---|
| registration wall P90 | 336.968ms | ≤150ms | 失败 |
| 配对wall ratio中位数 | 2.184× | ≤1.50× | 失败 |
| 峰值增量最大差 | +372736B | ≤524288B | 通过 |
| query MAC/state | 与E0一致 | 精确一致 | 通过 |
| K>2 actual FULL fit | 1 | 1 | 通过 |

## 机制结论

TCRA在support上每场景实际选择16–58个D42原子，并保持旧类tail、新类tail与双向hinge的safe-directional门，但这些support代理变化几乎全部落在held-query决策边界之外。它证明了“旧类零损伤”可以做到，却没有形成用户要求的floor与遗忘改善；同时逐prefix真实score导致K5/new20和K10/new20资源显著放大。继续放宽support门或增加原子数量大概率只会增加耗时，不能由现有数据证明能改善held-query旧类floor。

最终决定：淘汰TCRA/TPCE式support-tail离散后处理轴；不做完整125、不做同轴参数扫描。下一轮必须直接优化与held-query旧类错误更一致、且不依赖逐prefix全support复算的统计量。


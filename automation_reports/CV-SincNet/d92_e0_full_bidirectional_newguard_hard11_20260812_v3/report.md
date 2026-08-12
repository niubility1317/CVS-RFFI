# D92 NewGuard v3量化感知回缩Hard11实验报告

## 实验身份

- run ID：`d92_e0_full_bidirectional_newguard_hard11_20260812_v3`
- 候选：`E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`
- candidate ID：`d92_e0_full_bidirectional_newguard_maxmin`
- 日期：2026-08-12
- 协议：`p2_min_v1`；复用`VALIDATED_ONCE`
- 声明：`DEVELOPMENT_ONLY_HARD_SCREEN`
- 当前状态：`LOCAL_VERIFIED / READY_TO_LAND / NO_PERFORMANCE_RESULT`
- 操作：Codex主代理完成方法集成；唯一N607 runner负责落地、运行、监控和取回

## 目标与比较

目标是以`E0_FULL_ONLY`为同排基线，在保持单FULL fit、单一全类query头和永久state不变的前提下，同时改善：

- `H_old_new`、`old_balanced_accuracy`、`c_old_acc`、`c_old_floor`、`seen_new_acc`；
- 降低`average_forgetting`、`new_to_old_rate`、`old_to_new_rate`。

v1真实checkpoint smoke因D42量化把三个场景的旧类tail最小变化分别推至约`-1.79e-4`、`-6.51e-4`、`-9.07e-4`而安全回退；v2诊断证实量化前tail为正、新类行byte-exact、fit为2/1。v1/v2均为`NO_PERFORMANCE_RESULT`，不参与性能比较。

## v3方法锁

- 288维联合特征、ground-spectrum Cauchy center、task-balanced covariance、F0固定。
- K>2仅一次FULL component fit；不使用BLOCK、OCF、LOO、Fisher、Pareto或多query头。
- 先在支持集上构造旧类内部零空间max-min候选，新类仿射行保持byte-exact。
- 固定部署量化感知尺度：
  `[128,64,32,16,8,4,2,1,0.5,...,2^-12]`。
- 每一级都用真实D42 two-level int8系数和FP16截距codec回环，逐项重检：
  新类行byte-exact、逐旧类tail不下降、新类margin不下降、旧类包络不增加。
- 保护容差固定为`1024*float32_eps=0.0001220703125`，不接受更负tail。
- 选择第一个即最大的安全尺度，并要求最终D42 decoded头不与E0全头byte-exact；否则exact E0 fallback。
- 最大支持侧信任域比例固定为`0.0128`；无receiver、scene、K、query或性能调参。
- K<=2保持D92 FULL exact alias。
- query fit/update/selection/truth/role/quota/global reassignment全部为零；query MAC和永久state与E0一致。
- codec尝试次数和解析MAC进入支持侧资源收据。

## 冻结矩阵

10个performance outer：

1. `rx_7_7__seed_713106__k_10__new_5`
2. `rx_7_7__seed_713104__k_5__new_20`
3. `rx_7_7__seed_713103__k_10__new_5`
4. `rx_8_8__seed_713103__k_5__new_20`
5. `rx_8_8__seed_713103__k_10__new_5`
6. `rx_8_8__seed_713106__k_5__new_20`
7. `rx_7_14__seed_713104__k_10__new_10`
8. `rx_3_19__seed_713102__k_10__new_5`
9. `rx_7_7__seed_713105__k_10__new_20`
10. `rx_7_7__seed_713104__k_10__new_5`

K1 liveness：`rx_20_1__seed_713106__k_1__new_20`。

每个outer固定`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，共11job、33scene-arm、8shard。K>2真实checkpoint smoke使用第1个performance outer；K1只检验alias/liveness，不进入性能均值。

## 基线与晋级门

E0完整125历史基线：

- `E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis\paired_rows.csv`
- SHA256：`6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a`
- per-old-class SHA256：`c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6`
- 11个对应raw score路径和SHA已冻结在method lock中；D92/E0不重跑。

任一八项均值方向不优于E0即`REJECT_ROUTE`；方向全对但幅度、稳定性或资源未过为`REVISE_ONCE`；全部过门才为`ADVANCE_TO_TARGET125_CANDIDATE`。幅度门：H+1.00pp、old BA+1.50pp、c_old+1.00pp、floor+4.00pp、seen-new+0.50pp、forgetting-1.50pp、两向混淆各-0.50pp。Hard11不会自动触发完整125。

## 本地实现与验证

- Git仓库：`E:\type10-7\code\snapshots\d92_125wt`
- 分支：`codex/d92-be-hard12-strict-pareto-20260811`
- 科学提交：`9f1a85df`；容差修复：`e8d6a451`
- 方法锁/runner提交：`7e69843a`
- TDD：量化tail翻转、最大安全尺度、raw失败继续回缩、20级全失败exact E0、真实D42 active、receipt漂移均先RED后GREEN。
- 集成验证：NewGuard核心、probe、slim、query、matrix、runner、analysis共`101 passed`。
- 静态验证：8个生产文件`py_compile`通过，config JSON通过，`bash -n`通过，`git diff --check`通过。
- 独立审查：`APPROVE`，`P0=0/P1=0`。

## 发布三件套

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `d92_newguard_runtime_closure_7e69843a.tar.gz` | 5,045,895 | `ceaec0bfc59da2859750bdbacfa579a0eea768dd2d8788af6e34173af911507e` |
| `stage2_d92_full_bidirectional_newguard_hard11_v1.json` | 6,980 | `2efccf8a96fa01c5515695d9c9d3a208fd0059e3824e70262457027c8b0548ca` |
| `launch.sh` | 4,058 | `78c09120e730a3338f49360bb5d749d9d001366b6af1ba078698c26dc5c8586c` |

runtime archive来自Git HEAD的完整tracked `code/`树，共1307个成员；已确认包含`code/cvsrffi/__init__.py`、NewGuard核心和Hard11 runner，且不存在`code/code`层级。

## N607冻结执行

- 普通账号SSH目标：`N607`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_newguard_source_snapshot_20260812_v3`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_bidirectional_newguard_hard11_20260812_v3`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_bidirectional_newguard_hard11_20260812_v3`
- GPU：0–7，每个shard通过`CUDA_VISIBLE_DEVICES=<shard>`绑定后使用`cuda:0`
- fresh retry：`false`

exact command：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_newguard_source_snapshot_20260812_v3 &&
nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

同步映射：

- archive→`source_root/d92_newguard_runtime_closure_7e69843a.tar.gz`
- method lock→`source_root/configs/stage2_d92_full_bidirectional_newguard_hard11_v1.json`
- launch→`source_root/launch.sh`

## 健康与停止规则

启动后先核对PID、CWD、cmdline、GPU映射和日志增长。K>2真实checkpoint smoke必须满足active=true、无fallback、部署尺度/保护数值闭合、2/1 fit、query零访问，才启动8shard。

只因query协议违规、wrong hash/checkout、覆盖风险、prediction closure缺失、launcher-wide确定性故障、至少2个distinct outer在prediction前出现同一异常指纹、OOM/NaN/无进展等系统故障停止。不得因任何性能指标停止。停止后保留并取回全部artifact，不重启、不覆盖。

预期正式artifact：11 job receipt、22 before/after prediction NPZ和COMMIT、22 fit/resource audit、11 score、8 shard summary，以及smoke闭包。完成后完整取回source/output/logs并核对树摘要。

## 正式分析与裁决（2026-08-12）

- 11份manifest指定truth sidecar已从原D92任务路径只读取回，11/11实际SHA与manifest一致；本地映射仍锁定`jobs/<outer>/offline/scorer/truth_sidecar.json`，receipt、score、manifest与实际文件四方SHA闭合。
- 分析artifact闭合：11个paired row、66个逐旧类row、33个scene row；10个performance outer与1个K1 liveness完整。
- analyzer输出：`E:\type10-7\local_artifacts\d92_e0_full_bidirectional_newguard_hard11_20260812_v3\analysis`。
- 本地分析适配只增加跨平台truth-root映射并修正资源汇总集合初始化；NewGuard matrix/runner/analyzer聚焦回归`37 passed`，`py_compile`与`git diff --check`通过。

| candidate | H | old BA | c_old_acc | old floor | seen-new | forgetting | new→old | old→new | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E0_FULL_ONLY | 73.3472% | 74.8611% | 74.8611% | 44.8333% | 72.0333% | 12.9167% | 15.0417% | 15.3333% | baseline |
| NewGuard v3 | 73.3472% | 74.8611% | 74.8611% | 44.8333% | 72.0333% | 12.9167% | 15.0417% | 15.3333% | `REJECT_ROUTE` |
| NewGuard−E0 | +0.0000pp | +0.0000pp | +0.0000pp | +0.0000pp | +0.0000pp | +0.0000pp | +0.0000pp | +0.0000pp | 八项严格方向均未通过 |

NewGuard在10/10 performance outer均`active=true`、无fallback，部署尺度为0.015625–0.25、尝试次数为10–14，但所有正式score指标均与E0完全相同。说明量化保护后的安全扰动虽然改变了部署头字节，却没有改变这些难例上的最终类别决策，不能带来floor、遗忘或新类收益。

资源方面，query MAC与永久state保持E0精确一致，peak p90只增加20KiB；但注册wall p90为175.999ms、相对E0为1.856×，超过150ms和1.5×冻结门。额外成本主要来自多轮真实D42 codec回检。

最终裁决：`REJECT_ROUTE`。不运行完整Target125，不继续微调NewGuard尺度或放宽保护容差。下一方法应直接在D42部署格点上做一次闭式/小规模联合margin求解，允许同时调整旧类与新类行，显式优化旧类CVaR/floor、遗忘、新类margin及双向混淆；必须保持单FULL fit、query/state不增，并避免多轮codec回缩。

下一轮完整研发目标已固化到`docs/D92_NEXT_STRICT_PARETO_TARGET_PROMPT_20260812.md`。

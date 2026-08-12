# D92 NewGuard Hard11最终实验与合规审计报告

## 实验身份

- run ID：`d92_e0_full_bidirectional_newguard_hard11_20260812_v3`
- 候选：`E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`
- candidate ID：`d92_e0_full_bidirectional_newguard_maxmin`
- 日期：2026-08-12
- 协议：`p2_min_v1`；复用`VALIDATED_ONCE`
- 声明：`DEVELOPMENT_ONLY_HARD_SCREEN`
- 最终状态：`ANALYZED / REJECT_ROUTE / DEVELOPMENT_ONLY / NOT_PROMOTABLE`
- 操作：Codex主代理完成方法集成；唯一N607 runner负责落地、运行、监控和取回

## 目标与比较

目标是以`E0_FULL_ONLY`为同排基线，在保持单FULL fit、单一全类query头和永久state不变的前提下，同时改善：

- `H_old_new`、`old_balanced_accuracy`、`c_old_acc`、`c_old_floor`、`seen_new_acc`；
- 降低`average_forgetting`、`new_to_old_rate`、`old_to_new_rate`。

v1真实checkpoint smoke因D42量化把三个场景的旧类tail最小变化分别推至约`-1.79e-4`、`-6.51e-4`、`-9.07e-4`而安全回退；v2诊断证实量化前tail为正、新类行byte-exact、fit为2/1。v1/v2均为`NO_PERFORMANCE_RESULT`，不参与性能比较。v3完成了开发矩阵，但其量化回缩变体后来被独立源码审计判定不符合最初冻结设计；因此v3数值只用于否证路线，不能作为正式或可推广性能声明。

## 冻结方法、完整公式与生命周期

令E0单FULL头的第`c`类参数为增广向量`theta_c=[w_c;b_c]`，support为`z_tilde=[z;1]`。全部注册新类support组成`X_new`，通过紧凑SVD行空间算子得到零空间作用`P_perp`，秩阈值固定为机器精度规则。每个旧类先在E0 support margin上冻结lower方法的bottom-20%尾部集合，再构造同式方向，并求一次确定性小型max-min：

```text
delta_internal_c = P_perp u_c
X_new delta_internal_c = 0
sum_c delta_internal_c = 0
delta_theta_c = delta_internal_c + [0;tau], tau <= 0
maximize t subject to:
  fixed old-tail margin gain >= t
  registered-new new-vs-old margin gain >= t
  ||delta_internal_c|| <= 1e-4 * support_score_rms/support_feature_rms
```

输入是E0一次FULL fit后的288维support、标签和仿射头；输出仍是一套D42 int8系数加FP16截距的单仿射头。新类参数行不得改变，额外统计量只在注册过程中短暂存在，query不保存零空间、尾集合或求解状态。生命周期按四态记录：本轮只观测`DA1_REG0`和`DA1_REG1`；`DA0_REG0/DA0_REG1`未运行。`REG0`下新类准确率和H为`N/A`。

终态方法锁只有一个候选、一个预注册强度`1.0`和一次正式D42回环。raw或deployed阶段的Xnew残差、旧类组零和、包络等式、逐旧类tail、新类margin、新类行byte-exact或`tau<=0`任一失败，都逐字节返回E0；不扫描第二强度。历史v3使用20档`128,...,2^-12`回缩，属于后来引入且已撤销的非合规实现，不能再发布。

## 历史v3运行锁（仅用于解释已取回artifact）

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

任一八项均值方向不优于E0即`REJECT_ROUTE`；八项方向全对、稳定性/资源/计算门也通过，但仅幅度不足时为`REVISE_ONCE`；稳定性、资源或计算门失败同样为`REJECT_ROUTE`；全部过门才为`ADVANCE_TO_TARGET125_CANDIDATE`。幅度门：H+1.00pp、old BA+1.50pp、c_old+1.00pp、floor+4.00pp、seen-new+0.50pp、forgetting-1.50pp、两向混淆各-0.50pp。Hard11不会自动触发完整125。

## 本地实现与验证

- Git仓库：`E:\type10-7\code\snapshots\d92_125wt`
- 分支：`codex/d92-be-hard12-strict-pareto-20260811`
- 科学提交：`9f1a85df`；容差修复：`e8d6a451`
- 方法锁/runner提交：`7e69843a`
- 历史v3 TDD覆盖量化回缩；终态纠偏另以“保护失败不得尝试第二强度”和“部署Xnew闭包失败不得激活”先RED后GREEN。
- 集成验证：NewGuard核心、probe、slim、query、matrix、runner、analysis共`101 passed`。
- 静态验证：8个生产文件`py_compile`通过，config JSON通过，`bash -n`通过，`git diff --check`通过。
- 历史发布前审查曾给出`P0=0/P1=0`，但终态独立源码审计发现2个P0：多强度回缩违反冻结设计、deployment pass遗漏三项部署闭包。报告以终态审计为准。

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

终态30个same-outer/same-scene资源配对显示：query MAC与永久state保持E0精确一致，但注册wall p90为179.172ms、配对中位比为1.74784×、peak最大增量为5,951,488字节，三项注册资源门均失败。额外成本主要来自多轮真实D42 codec回检。

最终裁决：`REJECT_ROUTE`。不运行完整Target125，不继续微调NewGuard尺度或放宽保护容差。下一方法应直接在D42部署格点上做一次闭式/小规模联合margin求解，允许同时调整旧类与新类行，显式优化旧类CVaR/floor、遗忘、新类margin及双向混淆；必须保持单FULL fit、query/state不增，并避免多轮codec回缩。

下一轮完整研发目标已固化到`docs/D92_NEXT_STRICT_PARETO_TARGET_PROMPT_20260812.md`。

## 终态严格复算与证据边界

终态analyzer修正了四处与原规格不一致的判定：资源展开为10个performance outer×3scene的30个同outer、同scene配对，wall ratio取30对中位数、候选wall取p90、peak差取最大值；稳定性把tie计作非退化但不计作seen-new/new→old“方向正确”，并要求至少2个receiver和2个scene出现严格H或seen-new收益；同时补入完整D92两状态fit库存`8*(K+1)`的逐row计算削减门。为保持v3 artifact不可变，复算时按历史v3 method-lock读取旧产物，不把终态单候选锁伪装成当时的运行锁。

- 复算目录：`E:\type10-7\local_artifacts\d92_e0_full_bidirectional_newguard_hard11_20260812_v3\analysis_strict_spec_v4`
- 复算只读取immutable v3与冻结E0 artifact；不改prediction、score或truth。
- `summary.json`：18,004字节，SHA256`bf8c963379089d0d5090ac06890386f33d4182e3bbd0547c4f99de205cc83552`。
- `gates.json`：11,551字节，SHA256`c6861b6792bc9b59d715757e2b19b204a45fb374456b056efe2dc04461d066e0`。
- `analysis.md`：8,062字节，SHA256`bc4980c05dfd29bdb96e0e0cd16a43d4ac7938e2d58e448be0b7523e20351aaa`。
- `scenario_rows.csv`：7,334字节，SHA256`42ba770f4a2ab4eb11f24810e1f1c5f484d5a48d24512f73fe353193808ab040`。
- 闭合：11个paired row、66个逐旧类row、33个scene row、10个performance outer+1个K1 liveness。
- 全部24份execution receipt均标记`formal_metric_claim_allowed=false`，声明范围是开发稳定性筛查；本节不把它升级为正式结论。

### 三方同排均值

|方法|H|old BA|c_old_acc|old floor|seen-new|forgetting|new→old|old→new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D92历史同排|74.1673%|76.1389%|76.1389%|48.5000%|72.3750%|11.6389%|15.1000%|14.6944%|
|E0_FULL_ONLY|73.3472%|74.8611%|74.8611%|44.8333%|72.0333%|12.9167%|15.0417%|15.3333%|
|NewGuard v3|73.3472%|74.8611%|74.8611%|44.8333%|72.0333%|12.9167%|15.0417%|15.3333%|
|NewGuard−E0|+0.0000pp|+0.0000pp|+0.0000pp|+0.0000pp|+0.0000pp|+0.0000pp|+0.0000pp|+0.0000pp|

D92八项由10份原始同排score重新计算；D92证据范围仍是`development_only_not_formal_confirmation`。NewGuard八项严格Pareto、幅度和稳定性门均失败。tie不是成功，也不是“方向正确”。

### 10个performance outer汇总

每行三场景聚合，NewGuard与E0的八项均逐row完全相同。

|outer|K|新类数|H|old BA/c_old|floor|seen-new|forgetting|new→old|old→new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|rx_7_7__seed_713106__k_10__new_5|10|5|83.5228%|85.8333%|63.3333%|81.3333%|9.7222%|17.0000%|9.4444%|
|rx_7_7__seed_713104__k_5__new_20|5|20|69.7850%|75.5556%|41.6667%|64.8333%|15.8333%|13.6667%|17.2222%|
|rx_7_7__seed_713103__k_10__new_5|10|5|82.6908%|81.1111%|53.3333%|84.3333%|12.2222%|13.6667%|12.2222%|
|rx_8_8__seed_713103__k_5__new_20|5|20|66.3842%|66.9444%|28.3333%|65.8333%|16.9444%|13.3333%|22.7778%|
|rx_8_8__seed_713103__k_10__new_5|10|5|77.4910%|76.6667%|53.3333%|78.3333%|8.6111%|17.6667%|10.8333%|
|rx_8_8__seed_713106__k_5__new_20|5|20|64.3795%|63.6111%|36.6667%|65.1667%|15.2778%|9.7500%|19.4444%|
|rx_7_14__seed_713104__k_10__new_10|10|10|73.1111%|73.0556%|33.3333%|73.1667%|15.0000%|12.5000%|17.2222%|
|rx_3_19__seed_713102__k_10__new_5|10|5|54.1577%|58.6111%|30.0000%|50.3333%|11.3889%|29.6667%|19.1667%|
|rx_7_7__seed_713105__k_10__new_20|10|20|76.9815%|79.4444%|50.0000%|74.6667%|15.5556%|7.8333%|16.6667%|
|rx_7_7__seed_713104__k_10__new_5|10|5|84.9684%|87.7778%|58.3333%|82.3333%|8.6111%|15.3333%|8.3333%|

### 10 outer×3 scene完整结果

|outer|scene|H|old acc|seen-new|new→old|old→new|ΔH|Δseen-new|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|rx_7_7__seed_713106__k_10__new_5|clear|87.82%|86.67%|89.00%|10.00%|8.33%|+0.00pp|+0.00pp|
|同上|low_elev|76.32%|82.50%|71.00%|27.00%|13.33%|+0.00pp|+0.00pp|
|同上|rain|86.11%|88.33%|84.00%|14.00%|6.67%|+0.00pp|+0.00pp|
|rx_7_7__seed_713104__k_5__new_20|clear|72.91%|76.67%|69.50%|12.75%|17.50%|+0.00pp|+0.00pp|
|同上|low_elev|67.32%|76.67%|60.00%|17.50%|13.33%|+0.00pp|+0.00pp|
|同上|rain|68.92%|73.33%|65.00%|10.75%|20.83%|+0.00pp|+0.00pp|
|rx_7_7__seed_713103__k_10__new_5|clear|85.41%|85.83%|85.00%|14.00%|9.17%|+0.00pp|+0.00pp|
|同上|low_elev|82.15%|83.33%|81.00%|15.00%|10.83%|+0.00pp|+0.00pp|
|同上|rain|80.07%|74.17%|87.00%|12.00%|16.67%|+0.00pp|+0.00pp|
|rx_8_8__seed_713103__k_5__new_20|clear|76.47%|75.00%|78.00%|10.50%|10.00%|+0.00pp|+0.00pp|
|同上|low_elev|61.51%|67.50%|56.50%|18.75%|25.83%|+0.00pp|+0.00pp|
|同上|rain|60.58%|58.33%|63.00%|10.75%|32.50%|+0.00pp|+0.00pp|
|rx_8_8__seed_713103__k_10__new_5|clear|86.99%|84.17%|90.00%|7.00%|5.00%|+0.00pp|+0.00pp|
|同上|low_elev|69.86%|80.00%|62.00%|32.00%|5.83%|+0.00pp|+0.00pp|
|同上|rain|73.43%|65.83%|83.00%|14.00%|21.67%|+0.00pp|+0.00pp|
|rx_8_8__seed_713106__k_5__new_20|clear|70.14%|66.67%|74.00%|6.25%|21.67%|+0.00pp|+0.00pp|
|同上|low_elev|61.88%|63.33%|60.50%|14.00%|21.67%|+0.00pp|+0.00pp|
|同上|rain|60.92%|60.83%|61.00%|9.00%|15.00%|+0.00pp|+0.00pp|
|rx_7_14__seed_713104__k_10__new_10|clear|76.83%|76.67%|77.00%|4.50%|18.33%|+0.00pp|+0.00pp|
|同上|low_elev|73.99%|76.67%|71.50%|18.00%|11.67%|+0.00pp|+0.00pp|
|同上|rain|68.32%|65.83%|71.00%|15.00%|21.67%|+0.00pp|+0.00pp|
|rx_3_19__seed_713102__k_10__new_5|clear|59.93%|65.83%|55.00%|32.00%|19.17%|+0.00pp|+0.00pp|
|同上|low_elev|49.60%|52.50%|47.00%|26.00%|20.00%|+0.00pp|+0.00pp|
|同上|rain|52.91%|57.50%|49.00%|31.00%|18.33%|+0.00pp|+0.00pp|
|rx_7_7__seed_713105__k_10__new_20|clear|85.05%|86.67%|83.50%|4.50%|13.33%|+0.00pp|+0.00pp|
|同上|low_elev|71.84%|75.83%|68.25%|10.25%|15.83%|+0.00pp|+0.00pp|
|同上|rain|74.00%|75.83%|72.25%|8.75%|20.83%|+0.00pp|+0.00pp|
|rx_7_7__seed_713104__k_10__new_5|clear|89.08%|89.17%|89.00%|10.00%|10.00%|+0.00pp|+0.00pp|
|同上|low_elev|83.96%|88.33%|80.00%|18.00%|9.17%|+0.00pp|+0.00pp|
|同上|rain|81.73%|85.83%|78.00%|18.00%|5.83%|+0.00pp|+0.00pp|

### receiver、scene、K/new_count和逐旧类分解

所有分组的NewGuard−E0均为0。下表给出NewGuard绝对值。

|分组|H|old acc|seen-new|new→old|old→new|
|---|---:|---:|---:|---:|---:|
|receiver 7-7|79.5129%|81.9444%|77.5000%|13.5000%|12.7778%|
|receiver 8-8|69.0860%|69.0741%|69.7778%|13.5833%|17.6852%|
|receiver 7-14|73.0484%|73.0556%|73.1667%|12.5000%|17.2222%|
|receiver 3-19|54.1466%|58.6111%|50.3333%|29.6667%|19.1667%|
|clear|79.0640%|79.3333%|79.0000%|11.1500%|13.2500%|
|low_elev|69.8436%|74.6667%|65.7750%|19.6500%|14.7500%|
|rain|70.6978%|70.5833%|71.3250%|14.3250%|18.0000%|
|K10/new5|76.3580%|78.0000%|75.3333%|18.6667%|12.0000%|
|K5/new20|66.7381%|68.7037%|65.2778%|12.2500%|19.8148%|
|K10/new10|73.0484%|73.0556%|73.1667%|12.5000%|17.2222%|
|K10/new20|76.9648%|79.4444%|74.6667%|7.8333%|16.6667%|

|旧类TX|NewGuard|E0|差值|
|---|---:|---:|---:|
|14-10|78.1667%|78.1667%|+0.0000pp|
|14-7|49.5000%|49.5000%|+0.0000pp|
|20-15|96.5000%|96.5000%|+0.0000pp|
|20-19|57.1667%|57.1667%|+0.0000pp|
|6-15|69.6667%|69.6667%|+0.0000pp|
|8-20|98.1667%|98.1667%|+0.0000pp|

### 资源、fallback和异常

|项目|结果|门|判定|
|---|---:|---:|---|
|query MAC|3168/4608/7488，逐row与E0相等|精确相等|PASS|
|永久state|8583/11888/18498字节，逐row与E0相等|精确相等|PASS|
|FULL actual fit|1|1|PASS|
|相对完整D92 fit reduction|最小95.8333%，均值97.1591%|≥80%|PASS|
|注册wall同场景配对中位比|1.74784×（30对）|≤1.50×|FAIL|
|注册wall p90|179.172ms（30场景值）|≤150ms|FAIL|
|peak同场景最大增量|5,951,488字节|≤512KiB|FAIL|
|v3 support优化MAC上界|0–47,808,352，K1为0|仅解析上界|仅报告，不冒充时延|
|v3 support临时内存上界|0–3,430,832字节|注册临时态|仅报告|

v3正式K>2的30个scene receipt均标记active、无fallback；K1三场景为exact alias。其部署候选需要10–15次codec尝试，这正是历史v3超时和设计不合规的直接来源。v1/v2均在真实smoke因`deployment_protection_failed`回退并停止，无性能结果。终态严格单候选真实D42 probe同样安全回退E0，因此不具备新的active smoke，且在路线已经因八项tie被否决后不再消耗N607重跑。

### changed files、验证与终态结论

终态纠偏修改NewGuard core、Slim、Query、Hard11 method-lock、runner、analyzer、配置及对应7个测试文件；并更新本追溯、计划和报告。核心变化是删除20档回缩、重命名single-candidate receipt、补全六项raw/deployed闭包复核、把资源门改为30个same-outer/same-scene配对、加入receiver/scene严格收益分散门，并补全完整D92 fit reduction门。

实现/配置/测试提交包含：`stage2_d92_bidirectional_newguard.py`、`stage2_d92_e0d_slim.py`、`stage2_d92_e0d_query_evaluation.py`、`stage2_d92_newguard_hard11.py`、`stage2_d92_newguard_hard11_analysis.py`、`run_d92_newguard_hard11.py`、NewGuard Hard11 JSON配置，以及7个对应core/probe/Slim/Query/matrix/runner/analyzer测试文件。证据提交包含本报告、NewGuard追溯、三轮回顾、下一轮目标prompt和实施计划终态记录。

- NewGuard聚焦：`146 passed`。
- 相邻E0OCF/FloorBoost回归：`51 passed`。首次命令使用了仓内不存在的旧E0OCF测试名，pytest在收集前退出；随后按`rg --files tests`返回的真实文件名串行重跑并全部通过，这不是项目测试失败。
- 生产文件`py_compile`、config JSON、runner/analyzer `--help`、`git diff --check`：PASS。
- 独立终审：`APPROVE / P0=0 / P1=0`。
- 机制与分析纠偏Git commit：`8ec37964`（`fix: enforce frozen NewGuard deployment gates`）。
- Target125：未运行。

唯一结论：`REJECT_ROUTE`。理由不是“提升太小”，而是八项严格均值全部tie，且历史v3资源wall门失败；按照冻结规则，tie禁止`REVISE_ONCE`。后续若继续研发，必须换方法家族并使用新candidate/run ID，不能继续扫描NewGuard强度或放宽保护容差。

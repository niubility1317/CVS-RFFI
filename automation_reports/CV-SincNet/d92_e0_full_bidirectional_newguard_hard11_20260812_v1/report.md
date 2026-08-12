# D92 E0 FULL BIDIRECTIONAL NEWGUARD MAXMIN Hard11实验报告

## 1. 实验身份

- 实验ID：`d92_e0_full_bidirectional_newguard_hard11_20260812_v1`
- 候选：`E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`
- candidate ID：`d92_e0_full_bidirectional_newguard_maxmin`
- 日期：2026-08-12
- 操作：Codex主代理集成；唯一N607 runner待本地冻结后接管
- 当前状态：`LOCAL_VERIFIED / READY_TO_LAND / NO_PERFORMANCE_RESULT`
- 声明范围：`DEVELOPMENT_ONLY_HARD_SCREEN`
- 协议：`p2_min_v1`
- 数据状态：复用`VALIDATED_ONCE`，不重复数据验证

## 2. 目标与假设

目标是在保持`E0_FULL_ONLY`单FULL fit、单query仿射头和永久state布局的同时，使以下Hard10同排均值全部严格改善：

- `H_old_new`、`old_balanced_accuracy`、`c_old_acc`、`c_old_floor`、`seen_new_acc`提高；
- `average_forgetting`、`new_to_old_error`、`old_to_new_error`降低。

假设：旧类内部残差被限制在注册新类support的增广零空间后，不会重现FloorBoost的new→old包络扩张；共享旧类截距只允许`tau<=0`，用旧类内部max-min获得的margin余量同时保护新类。

## 3. 已冻结负证据

Hard10同排均值：

| 方法 | H | old BA | old floor | seen-new | forgetting |
|---|---:|---:|---:|---:|---:|
| D92 | 74.1673% | 76.1389% | 48.5000% | 72.3750% | 11.6389% |
| E0_FULL_ONLY | 73.3472% | 74.8611% | 44.8333% | 72.0333% | 12.9167% |
| FloorBoost | 67.2838% | 77.8056% | 55.1667% | 59.8750% | 9.9722% |

FloorBoost相对E0虽然floor提高10.3333pp、forgetting降低2.9444pp，但seen-new下降12.1583pp、H下降6.0634pp、new→old增加17.4667pp，因此该路线已拒绝；本轮不得扫描其lambda、kappa、quantile或bias cap。

## 4. 方法锁

- 288维联合特征、ground-spectrum Cauchy center、task-balanced covariance和F0保持不变。
- K>2注册态只调用一次FULL component fit。
- 不调用BLOCK、OCF、LOO、Fisher、Pareto或多query头。
- 新类仿射行byte-exact。
- bottom tail固定为0.20，NumPy quantile `method="lower"`。
- 使用紧凑行空间/零空间算子，不显式构造289×289投影矩阵。
- 内部旧类残差满足新类support零响应及旧类组零和。
- 共享旧类截距`tau<=0`。
- FP32中间头和正式D42 int8/FP16部署头均须通过保护约束。
- 数值、可行性、量化或闭包失败时整头byte-exact回退E0。
- K<=2保持D92 FULL exact alias。
- 无参数、arm、query或checkpoint扫描。

## 5. 冻结矩阵

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

K1 liveness：

- `rx_20_1__seed_713106__k_1__new_20`

每行固定`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，共11job、33scene-arm、8shard。K>2真实smoke使用第1个performance outer；K1作为正常liveness job，不进入性能均值。

## 6. 历史基线

- `paired_rows.csv`：`E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis\paired_rows.csv`
- SHA256：`6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a`
- `per_old_class_rows.csv` SHA256：`c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6`
- 11个对应E0 raw score的逐文件SHA将由冻结config记录，用于真实计算old-balanced、双向混淆和逐旧类门。
- D92和E0均不重跑。

## 7. 本地Git与验证

- Git仓库：`E:\type10-7\code\snapshots\d92_125wt`
- 分支：`codex/d92-be-hard12-strict-pareto-20260811`
- 设计/追溯提交：`282072f6`
- 可行性缺口修订：`85d26143`
- 科学实现提交：`2df14f90`；部署回退修订：`35309a43`
- 机械实现提交：`474242c6`；P0/P1闭环：`ed1db427`、`ef9c7249`、`2d2ea755`
- 集成发布提交：`e5eadc04`
- 本地集成测试：`81/81`通过；Task2最终回归`52/52`通过；相关历史回归通过
- 独立P0/P1：科学复核`P0=0/P1=0`；机械复核`P0=0/P1=0`
- real-checkpoint smoke：已冻结为N607启动前的K>2 active-method强制门；只有`active=true`、无fallback、2/1fit和query零访问全部闭合才启动shard

## 8. 计划N607发布

- 普通账号目标：`N607`
- Python环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，发布前重新核验
- source root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_newguard_source_snapshot_20260812_v1`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_bidirectional_newguard_hard11_20260812_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_bidirectional_newguard_hard11_20260812_v1`
- GPU：0–7，每个shard通过`CUDA_VISIBLE_DEVICES=<shard>`绑定后使用`cuda:0`
- 精确启动模式：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_newguard_source_snapshot_20260812_v1 &&
nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

只有本地方法、测试、真实smoke、独立P0/P1、Git和三件套哈希冻结后，以上命令才成为获准的exact command。

发布三件套已冻结：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `d92_newguard_runtime_closure_2d2ea755.tar.gz` | 5,033,754 | `48074423354da375dde3b6488396bde31daadb490f2afc14cb93208469cee607` |
| `stage2_d92_full_bidirectional_newguard_hard11_v1.json` | 6,345 | `d41b116b2bb7fb8be1fb56512e9e47e7915e94b5fae57776ced9c875ceb5f523` |
| `launch.sh` | 4,058 | `38267ad139ed89a402cf663eaa668b4581d0589e7471e6dfca37f2e7fbcc2b6f` |

runtime archive共1,307个成员，包含`code/cvsrffi/__init__.py`、NewGuard核心、D92/E0D执行链、Hard11 runner/analyzer及其依赖，不存在`code/code`错误层级。同步映射固定为archive→`source_root/d92_newguard_runtime_closure_2d2ea755.tar.gz`、method lock→`source_root/configs/stage2_d92_full_bidirectional_newguard_hard11_v1.json`、launch→`source_root/launch.sh`。

## 9. 健康停止规则

只因以下原因停止：

- query truth/fit/update/selection/role/quota/global访问；
- wrong checkout/hash、覆盖风险或prediction closure缺失；
- launcher-wide确定性故障；
- 至少2个distinct outer在prediction前产生同一规范化异常指纹；
- OOM、NaN或无进展等预注册系统故障。

不得因H、accuracy、floor、forgetting或任何中间性能停止。

## 10. 晋级门

任一核心均值方向不优于E0，结论为`REJECT_ROUTE`。八项方向全部正确但未达到幅度，最多`REVISE_ONCE`。只有同时达到以下幅度、稳定性和资源门，才是`ADVANCE_TO_TARGET125_CANDIDATE`：

- H≥E0+1.00pp；
- old BA≥E0+1.50pp；
- c_old_acc≥E0+1.00pp；
- old floor≥E0+4.00pp；
- seen-new≥E0+0.50pp；
- forgetting≤E0-1.50pp；
- new→old≤E0-0.50pp；
- old→new≤E0-0.50pp；
- query MAC和永久state与E0精确相等；
- FULL主fit=1；
- median wall≤1.50×E0，p90≤150ms，peak≤E0+512KiB。

## 11. 结果表

当前没有性能结果。只有完成11/11预测、评分、artifact取回并由独立analyzer连接truth后，才填写同排表和唯一裁决。

| candidate | H | old BA | c_old_acc | old floor | seen-new | forgetting | new→old | old→new | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING | ANALYSIS_PENDING |

## 12. 证据边界

本矩阵是`DEVELOPMENT_ONLY_HARD_SCREEN`，不能代替完整125或正式推广声明。即使结论为`ADVANCE_TO_TARGET125_CANDIDATE`，本轮也不自动启动完整125。

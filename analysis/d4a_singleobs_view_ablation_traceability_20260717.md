# D4a单观测固定接收IQ后处理消融追踪

日期：2026-07-17  
范围：`receiver=20-1`、`seed=713101`、`K=10`、真实`new5`、三种互斥物理样本LEO_weak场景。  
声明边界：development-only，不是正式确认矩阵，也不具备formal launch authority。

## 1. 输入证据

- 合法row：`E:\type10-7\automation_reports\CV-SincNet\d4a_single_observation_smoke_20260717_010128\dev_k10_new5_r2`
- sealed Phase1 runtime SHA256：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`
- Phase1 checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- 消融artifact：`E:\type10-7\automation_reports\CV-SincNet\d4a_singleobs_view_ablation_20260717_013003`
- prediction COMMIT SHA256：`e24cdb11b6e185542574dc88f1a88d6fca2ed40ba4cb79bbd1321c009d228b53`
- score COMMIT SHA256：`747e71c03541e3665877add9e6d49f039782c37e09a2cc5666c89e00b0608779`

三场景support/query分别来自各自sealed包；同场景support/query物理token与received-IQ SHA不重叠，三场景全角色物理token与received-IQ SHA两两不重叠。预测阶段命令没有truth sidecar参数。

## 2. 预登记variant

独立脚本：

`code/scripts/explore_singleobs_view_ablation.py`

固定variant：

1. `base_mean`
2. `diag_mean_current3view`
3. `base_trimmed10`
4. `base_trimmed20`
5. `base_median`
6. `diag_trimmed10`
7. `diag_trimmed20`
8. `diag_median`
9. `base_2proto`
10. `diag_2proto`

所有方法最多2个prototype/class。`diag_*`仅拟合每场景support的288维对角scale；总参数为864。所有query保持一次backbone forward、一次FFT/RF描述符提取、单view逐样本分类。

`diag_mean_current3view`忠实对应当前D4a K10实现的预测数学：虽然登记base/plus/minus三个固定接收IQ后处理representation lineage，但K>=2时prototype与query分类只使用base view，plus/minus不改变当前K10 prediction。因此本轮未把plus/minus冒充额外物理样本或额外K，也没有把它们解释为性能来源。

## 3. Support-only选择

选择信息只来自注册前后、三场景support leave-one-physical-sample-out：

1. 最大化最差state×scenario逐类LOO准确率；
2. 最大化六个slice的平均逐类floor；
3. 最大化总体support LOO准确率；
4. 最大化最差LOO margin；
5. 同分时优先更少prototype和更少equalizer状态。

| rank | variant | 最差逐类LOO | 平均slice floor | 总体LOO | 最差margin |
|---:|---|---:|---:|---:|---:|
| 1 | `diag_2proto` | 0.2000 | 0.4333 | 0.7311 | -2.2040 |
| 2 | `base_2proto` | 0.2000 | 0.4167 | 0.7326 | -1.9150 |
| 3 | `base_median` | 0.0000 | 0.3833 | 0.7010 | -1.3830 |
| 10 | `diag_mean_current3view` | 0.0000 | 0.3000 | 0.7240 | -1.3866 |

锁定路线为`diag_2proto`。选择完成后才提取并提交全部variant的immutable query predictions；之后隔离scorer才打开truth sidecar。

## 4. 实际new5结果

### 4.1 Support-selected路线

| candidate | old before | old after | old floor after | seen-new | H | forgetting | 参数 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `diag_2proto` | 0.7667 | 0.6222 | 0.3333 | 0.6933 | 0.6559 | 14.44pp | 864 | 81,834B |

逐场景：

| 场景 | old | seen-new | H | old→new | new→old |
|---|---:|---:|---:|---:|---:|
| `leo_clear_weak` | 0.6417 | 0.7200 | 0.6786 | 0.2250 | 0.1300 |
| `leo_low_elev_weak` | 0.6167 | 0.6500 | 0.6329 | 0.2333 | 0.1400 |
| `leo_rain_weak` | 0.6083 | 0.7100 | 0.6552 | 0.2333 | 0.1000 |

逐类floor：

| TX | role | accuracy |
|---|---|---:|
| `20-19` | target_old | 0.3333 |
| `14-7` | target_old | 0.4167 |
| `14-10` | target_old | 0.5667 |
| `6-15` | target_old | 0.6667 |
| `20-15` | target_old | 0.8500 |
| `8-20` | target_old | 0.9000 |
| `1-18` | target_new | 0.3333 |
| `14-11` | target_new | 0.6833 |
| `18-10` | target_new | 0.7500 |
| `1-16` | target_new | 0.8167 |
| `8-3` | target_new | 0.8833 |

### 4.2 冻结后诊断对照

| variant | old after | old floor | seen-new | H | forgetting |
|---|---:|---:|---:|---:|---:|
| `base_mean` | 0.6306 | 0.4333 | 0.6933 | 0.6605 | 14.17pp |
| `base_2proto` | 0.6444 | 0.4167 | 0.6800 | 0.6617 | 16.11pp |
| `base_trimmed10` | 0.6222 | 0.4167 | 0.6933 | 0.6559 | 14.44pp |
| `diag_mean_current3view` | 0.6222 | 0.4167 | 0.6867 | 0.6529 | 14.17pp |
| `diag_2proto` | 0.6222 | 0.3333 | 0.6933 | 0.6559 | 14.44pp |

这些query结果是在全部prediction冻结后获得的，只能用于诊断下一轮机制，不能反向把`base_mean`或其他variant改写为本row的support-selected winner。

## 5. 结论与下一轮

1. 2-prototype提高了support LOO floor，但没有迁移为query旧类floor；`diag_2proto`出现明显support选择错配。
2. 当前对角equalizer没有改善总体new5 Pareto，且当前K10三view实现中的plus/minus并不参与prototype或query score，不能将“三view”本身声明为性能增强。
3. robust median/trimmed prototype没有解决`20-19`与`1-18`floor类；主要矛盾仍是类间重叠和注册后old→new混淆。
4. 下一轮不得在本row query上继续改选。应在新的development seed上预登记加入纯support稳定性门槛，例如2-prototype cluster occupancy、双prototype间距、leave-two-sample-out一致性和old registry保护项；若不稳定则自动回退单prototype。
5. 当前结果远低于K10 new5门槛：`old_acc>=0.92`、`min_old_class_acc>=0.88`、`seen_new_acc>=0.92`，不得进入125正式确认。

## 6. 验证

- `conda run -n ssr-gpu python -m py_compile code\scripts\explore_singleobs_view_ablation.py`
- predictor状态：`PREDICTIONS_COMMITTED_BEFORE_TRUTH_JOIN`
- scorer状态：`POST_PREDICTION_ISOLATED_SCORING_COMMITTED`
- predictor query truth access：`false`
- scorer query truth feedback：`false`
- trainable parameters：`864`
- adaptation epochs：`0`
- dense query graph：`0B`
- persistent state：`81,834B`


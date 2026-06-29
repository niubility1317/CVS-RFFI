# Phase2开放世界特征空间优化实验验证设计

生成时间：2026-06-29 18:52 CST
状态：实验设计完成；未启动新训练、未同步N607、未改远程文件。
相关改动：`31bab7c feat: add open-world feature geometry loss`。
验证目标：判断新增`open_world_feature_space_loss`是否改善地面训练`z_id`几何，并进一步改善Phase2旧类校准、新类注册和未知类拒识。

## 当前证据基线

本设计基于以下本地和只读远程证据：

| 证据 | 当前结论 | 对实验设计的影响 |
|---|---|---|
| `项目.md` | `z_id`用于TX身份、原型、注册和校准；`z_dom`不得进入TX原型距离；target receiver和unknown query不得进入地面训练或阈值拟合 | 新loss只能作用于identity feature，后续Stage2阈值和unknown评估必须隔离 |
| `code/train.py` | `--lambda_open_world_feat`已默认关闭接入，使用`select_generalization_feature`，默认`z_id` | 可直接做机制验证和Phase2 prototype导出 |
| `code/SSDG/train_ssdg.py` | 最新Phase1队列使用该入口；目前没有`--lambda_open_world_feat`参数 | 与最新Safe-SSDG主线做严格同口径比较前，需要先做默认关闭桥接；否则只能验证`code/train.py`路径 |
| `stage2_spaceborne_h06_phase1_cen51select_20260628_121456` | 8/8完成；只有`PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_SEED2_GPU7_A`通过final same-row floors：overall 88.6436、strict UDU 84.9433、receiver floor 83.4917、sat mean3 62.8526、sat floor3 60.8549；Safe-SSDG repair row未通过；`LOSS_ANOMALY`存在 | CEN51 refresh只能作为control floor，不能作为新方法成功；新loss必须同时看完整loss曲线和几何指标 |
| `stage2_spaceborne_h06_oldheadfar_repair_20260626_233310` | 48行完成负诊断；old max 0.1111，unknown_FAR可到0但old为0，old80_count=0，old80_far05_count=0 | 单纯拒识阈值或old-head修复不够，必须回到`z_id`几何和原型半径 |
| `stage2b_oldhead_sweep` | target-only上限显示target_old最高可到0.8333；但按old/unknown hmean筛选时FAR偏高，存在old recoverability和unknown rejection冲突 | 验证应先看old80，再看unknown FAR；不能只报old_acc或只报low FAR |
| `phase1_gpu0_jointsafe36_queue_20260629_0930` | 2026-06-29 18:52只读N607库存仍有12个该队列GPU compute进程；队列未完成 | 不应立刻启动新矩阵；先等当前队列完成并做完整日志/几何审计 |

只读远程检查记录：

- `tools\n607_ssh_preflight.ps1`：PASS，直接N607可达，项目根可见，8张RTX3090可见。
- `python tools\n607_training_inventory.py --direct-only --pretty`：2026-06-29 18:52 CST，`phase1_gpu0_jointsafe36_queue_20260629_0930`仍有12个GPU compute进程。
- SSH清理：本地`ssh.exe`进程0，N607/bridge TCP:22连接0。

## 核心假设

H1：在source-only地面训练中，`open_world_feature_space_loss`会降低同类`z_id`角半径、扩大TX类中心角间隔，并降低source-domain prototype overlap。

H2：若H1成立，Phase2-B old-class calibration会更容易达到`old_acc>=0.80`，且在相同support-only阈值规则下unknown FAR不应恶化。

H3：若H1和H2同时成立，Stage2-C seen-new enrollment的new prototype overlap应下降，`H_old_new`和unknown rejection才有继续优化价值。

失败解释优先级：

1. 若几何指标改善但strict UDU、receiver floor或satellite floors下降，说明loss权重或margin过强，不可进入Stage2。
2. 若source指标稳定但Phase2 old/unknown无改善，说明问题主要在target receiver shift或阈值层，转向support calibration/EVT而不是继续加训练loss。
3. 若`train_ow_feat_active_classes<2`长期出现，说明batch类别覆盖不足，优先修sampler或batch构成，不提高loss权重。

## 分阶段验证流程

### V0：当前队列完成审计

目的：把`JOINTSAFE36`作为最新source-only审计基线，而不是直接把新loss塞进正在运行的队列。

输入：

- N607 run：`phase1_gpu0_jointsafe36_queue_20260629_0930`
- 本地报告：`automation_reports/CV-SincNet/phase1_gpu0_jointsafe36_queue_20260629_0930/report.md`

完成条件：

| 检查 | 要求 |
|---|---|
| 调度状态 | 36/36完成或明确失败；不得只用startup/active状态 |
| 日志 | 全量stdout扫描，不能只读tail |
| 曲线 | 每个候选的`metrics_epoch.csv`/`metrics_epoch.jsonl`完整解析 |
| 结论 | 同一行绑定candidate、seed、strict UDU、receiver floor、sat mean/floor、pseudo precision、joint_safe_score |
| 边界 | 若仍有`LOSS_ANOMALY`或缺失loss telemetry，不得声明优化效果 |

输出：

- 最新地面训练候选排名。
- 至少3个后续对照锚点：CEN51 refresh control、JOINTSAFE36最佳source row、JOINTSAFE36最佳satellite-retention row。

### V1：入口一致性门

当前事实：新loss只在`code/train.py`可用，最新Safe-SSDG队列使用`code/SSDG/train_ssdg.py`。

选择：

| 选项 | 用途 | 是否可作为Safe-SSDG主线证据 |
|---|---|---|
| V1-A：直接用`code/train.py` | 快速机制验证`open_world_feature_space_loss`是否按预期改变`z_id`几何 | 否，只是机制验证 |
| V1-B：先把loss默认关闭桥接到`code/SSDG/train_ssdg.py` | 与`JOINTSAFE36`、CEN51/Safe-SSDG同入口比较 | 是，推荐主线 |

推荐：先做V1-B。桥接仍必须默认关闭；新增参数不得改变`train_ssdg.py`默认训练结果。桥接验证至少包括`py_compile`、`--help`/dry-run、synthetic loss测试和一个`train_ssdg.py --dry_run`。

### V2：地面训练8行小矩阵

运行时机：`JOINTSAFE36`完成并完成V0审计后，且N607 lane有安全容量。

矩阵名称建议：`phase1_owfeat8_source_geometry_20260630`。

统一协议：

- 训练阶段：Phase1 source-only weak/semi-supervised DG。
- 数据：ManySig source receivers，`labeled_ratio=0.10`，`unlabeled_ratio=0.70`，`source_val_ratio=0.20`。
- 禁止：target receiver样本、Stage2 support/query、unknown query、target threshold fitting。
- 特征：`z_id`，不使用`z_dom`做TX距离。
- checkpoint选择：沿用`joint_safe`保护；必须记录final和best上下文。

候选表：

| ID | 目的 | `lambda_open_world_feat` | radius deg | inter margin deg | sample margin deg | domain align | 判定用途 |
|---|---|---:|---:|---:|---:|---:|---|
| `OWFEAT8_C0_CONTROL` | 同入口同seed控制 | 0.0 | 12 | 55 | 5 | 0.00 | 排除入口/seed差异 |
| `OWFEAT8_C1_L005` | 低权重主候选 | 0.005 | 12 | 55 | 5 | 0.00 | 看几何是否温和改善 |
| `OWFEAT8_C2_L010` | 默认推荐权重 | 0.010 | 12 | 55 | 5 | 0.00 | 主候选 |
| `OWFEAT8_C3_L010_DOM03` | 加弱跨domain中心对齐 | 0.010 | 12 | 55 | 5 | 0.03 | 看receiver shift是否下降 |
| `OWFEAT8_C4_L010_DOM05` | 较强domain对齐 | 0.010 | 12 | 55 | 5 | 0.05 | 检查是否损伤TX identity |
| `OWFEAT8_C5_RADIUS10` | 更紧类内半径 | 0.010 | 10 | 55 | 5 | 0.03 | 检查radius收缩收益/风险 |
| `OWFEAT8_C6_INTER65` | 更大类间角间隔 | 0.010 | 12 | 65 | 5 | 0.03 | 检查old/new overlap |
| `OWFEAT8_C7_STRESS` | 高权重负控 | 0.030 | 10 | 70 | 5 | 0.05 | 若source指标崩，证明过强边界 |

训练期必须记录：

- `train_open_world_feat_loss`
- `train_ow_feat_compact`
- `train_ow_feat_inter`
- `train_ow_feat_sample_margin`
- `train_ow_feat_domain_align`
- `train_ow_feat_active_classes`
- `train_ow_feat_pos_angle_deg`
- `train_ow_feat_min_inter_deg`
- strict UDU、receiver floor、sat mean3/floor3、pseudo precision、loss非有限跳过次数。

地面阶段晋级规则：

| 指标 | 晋级要求 |
|---|---|
| source overall/strict UDU/receiver floor | 不低于同入口control超过1pp；若下降超过2pp直接停止进入Phase2 |
| sat mean3/sat floor3 | 不低于同入口control；sat floor下降超过1pp需人工复核 |
| `train_ow_feat_min_inter_deg` | 相对control提高或保持，且不是以source指标崩溃换取 |
| prototype radius p95/p99 | 相对control下降，至少一个候选在多数TX上下降 |
| domain shift | 若启用domain_align，`P_tx_dom` shift下降且receiver leakage不升高 |

### V3：离线原型和特征几何审计

对V2每个候选导出或重建Phase2 prototype package：

```powershell
python code\train.py --phase2_export_prototypes --phase2_export_feature_key z_id --phase2_export_split train
```

若使用`train_ssdg.py`桥接主线而不能直接调用`code/train.py`导出器，则必须先确认checkpoint兼容；不兼容时用`eval_feature_diagnosis.py`或专用离线feature exporter重建`z_id`特征，不得伪造`P_tx`。

几何审计指标：

| 维度 | 指标 |
|---|---|
| 类内 | 每个TX的mean/p95/p99/max angular radius |
| 类间 | min/median class-center angle，top-k nearest TX pairs |
| 跨domain | 同TX不同receiver/day domain center angle |
| overlap | old prototype radius overlap count、nearest non-old margin |
| 可部署性 | prototype包大小、推理延迟、缺失TX/domain计数 |

晋级规则：只选择最多2个候选进入V4，且必须同时满足source指标不退化和prototype radius/overlap改善。

### V4：Stage2-B old/unknown验证

目标：验证地面几何改善是否解决OLD80_FIRST的第一阶段，而不是只让unknown FAR下降。

协议：

- support：`R_t`中`Y_old`的K-shot，K取`5,10,20,50`。
- query：`R_t`中held-out target-old和`Y_unknown`，unknown query只评估。
- 阈值：只允许source统计或target-old support，不允许unknown query拟合。
- target receiver优先：`20-1`，再扩展`3-19`,`7-14`,`7-7`,`8-8`。
- channel view：主线`leo_clear_weak`，通过后再做`leo_low_elev_weak`,`leo_rain_weak`。

指标：

- `old_acc`
- `target_old_accepted_acc`
- coverage
- `unknown_FAR`
- FPR95/AUROC
- old_unknown_hmean
- rollback/defer rate

晋级规则：

| 条件 | 解释 |
|---|---|
| 必须 | `old_acc>=0.80` |
| 必须 | `unknown_FAR<=0.05`不能来自old coverage=0的退化行 |
| 必须 | 同一candidate同一K同一receiver行同时满足old和FAR |
| 禁止 | 使用unknown query调阈值后声明成功 |

若V4没有任何同一行满足old80+FAR05，但old80显著增加，下一步转阈值/EVT/energy层；若old80没有改善，回到V2/V3调整特征空间或support geometry。

### V5：Stage2-C seen-new/unknown验证

只在V4至少有一个非退化候选满足old80+FAR05后启动。

协议：

- support：`R_t`中`Y_old`K-shot和`Y_new`K-shot。
- query：target-old、seen-new、`Y_unknown`互斥。
- `Y_new`和`Y_unknown`必须来自ManyTx non-`Y_old`真实TX label，不使用synthetic/rank占位。

指标：

- old_acc
- seen_new_acc
- `H_old_new`
- unknown_FAR
- unknown->new confusion
- new->old confusion
- prototype overlap rejection count

晋级规则：

- `old_acc>=0.80`
- unknown_FAR<=0.05
- `H_old_new`优于同入口control
- new_acc_drop_pp<=2pp或明确标为exploratory

## 实验阻断条件

| 阻断 | 处理 |
|---|---|
| `JOINTSAFE36`仍有活跃训练进程 | 不启动V2；只做monitor和完成审计 |
| `train_ssdg.py`未桥接新loss却声称Safe-SSDG主线验证 | 阻断；改成机制验证或先补桥接 |
| 缺少完整loss/metrics曲线 | 只报告启动或不完整诊断，不做loss正常/优化收益声明 |
| source指标下降超过阈值 | 不进入Stage2 |
| old80只在coverage=0或accepted=0下达成 | 退化，不可晋级 |
| unknown query进入阈值拟合 | 协议违规，结果降级为无效 |

## 推荐执行顺序

1. 等`phase1_gpu0_jointsafe36_queue_20260629_0930`完成；做完整日志和metrics审计。
2. 选择V1-B：把`open_world_feature_space_loss`默认关闭桥接到`code/SSDG/train_ssdg.py`，本地验证后再生成V2矩阵。
3. 启动8行`OWFEAT8`小矩阵，不超过每GPU一个新增候选；若已有队列仍占用，保持monitor-only。
4. 对V2所有候选做V3离线prototype/geometry审计。
5. 只让最多2个候选进入V4 Stage2-B old/unknown。
6. 只有V4同一行满足old80+FAR05后，启动V5 Stage2-C。

## 最小本地验证命令

```powershell
conda activate ssr-gpu
python -m pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py code\tests\test_phase2_prototypes.py code\tests\test_open_world_head.py -q
python -m py_compile code\cvsrffi\losses.py code\cvsrffi\logging.py code\train.py
python code\train.py --help | Select-String -Pattern "open_world_feat|ow_feat|phase2_export"
```

若桥接`train_ssdg.py`：

```powershell
conda activate ssr-gpu
python -m py_compile code\SSDG\train_ssdg.py tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py
python code\SSDG\train_ssdg.py --dry_run --epochs 1 --lambda_open_world_feat 0.01 --ow_feat_domain_align_weight 0.03
```

## 当前结论

当前不应立即启动新实验。正确动作是先完成正在运行的`JOINTSAFE36`队列审计，再决定是否桥接`train_ssdg.py`并发起8行小矩阵。新增loss的验证不能只看`train_open_world_feat_loss`下降；必须同时证明source DG不退化、prototype半径/overlap改善，并在Stage2-B同一行达到old80+FAR05后，才进入Stage2-C。

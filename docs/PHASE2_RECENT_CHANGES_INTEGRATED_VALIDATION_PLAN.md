# Phase2近期改动联合实验验证计划

生成时间：2026-06-29 19:06 CST
状态：联合验证设计落地；未启动新训练；未同步N607；未修改远程文件。
目标：下一轮实验必须同时覆盖近期落地的多原型、Meta-SSL/源域episode、Phase2原型导出、开放世界多原型头和`z_id`特征空间优化，而不是只验证单个loss。

## 当前本地事实

| 能力 | 本地落点 | 当前可用状态 | 验证含义 |
|---|---|---|---|
| Meta-SSL/源域episode | `code/train.py`、`code/cvsrffi/meta_episodes.py` | `--use_meta_ssl_cvs`、`--use_meta_rxday_episodes`、`--lambda_meta_ssl`、`--lambda_ssl_tx`、`--lambda_ssl_proto`默认关闭可用 | 用源域`rx_day`episode和teacher/pseudo/prototype gate验证元学习式源域泛化；不是完整MAML |
| 训练期原型记忆 | `code/cvsrffi/losses.py::PrototypeMemoryBank`、`code/train.py` | `--use_proto_memory`、`--lambda_proto`、`--proto_domain_align_weight`默认关闭可用 | 作为地面训练`z_id`原型正则，不能替代Phase2离线导出 |
| 特征空间优化 | `code/cvsrffi/losses.py::open_world_feature_space_loss`、`code/train.py` | `--lambda_open_world_feat`和`--ow_feat_*`默认关闭可用 | 用角半径、类间角、样本margin和可选跨domain中心对齐优化`z_id`几何 |
| Phase2原型导出 | `code/cvsrffi/phase2_prototypes.py`、`code/train.py` | `--phase2_export_prototypes`默认关闭可用 | 训练结束后导出`P_tx`、`P_tx_dom`诊断、radius和geometry summary |
| 多原型开放世界头 | `code/cvsrffi/open_world_head.py::OpenWorldMultiPrototypeHead` | `from_phase2_export`、`add_target_prototypes`、`register_new_class(...,overlap_margin=...)`可用 | 用old source prototype、target-old support prototype和seen-new prototype共同决策old/new/unknown |
| 当前Safe-SSDG入口 | `code/SSDG/train_ssdg.py` | 只有`use_phase2_ground_prototypes`、`use_feature_masks`、`use_txrx_geometry_losses`audit接线；非零loss会触发`NotImplementedError` | 最新N607队列不能被解释为已验证主动多原型/特征空间loss |

`项目.md`仍是协议边界：`z_id`用于TX身份、原型、注册和校准；`z_dom`只能用于域诊断；地面训练不得使用target receiver、Stage2 support/query或unknown query做训练、模型选择、阈值拟合。

## 最新实验状态

2026-06-29 19:06 CST只读检查结果：

| 检查 | 结果 |
|---|---|
| `tools\n607_ssh_preflight.ps1` | PASS；N607直连可达，项目根可见，8张RTX3090可见 |
| `tools\n607_training_inventory.py --direct-only --pretty` | `phase1_gpu0_jointsafe36_queue_20260629_0930`仍有12个`code/SSDG/train_ssdg.py`GPU进程 |
| 当前队列机制 | 命令包含`--use_phase2_ground_prototypes true`、`--use_feature_masks true`、`--use_txrx_geometry_losses true`、`--phase1_distribution_audit_only true`，但`lambda_tx_proto/lambda_rx_proto/lambda_mask_aux/lambda_tx_supcon_masked/lambda_rx_supcon_masked/lambda_txrx_rect`均为0 |
| SSH清理 | 本地`ssh.exe`进程0，N607/bridge TCP:22连接0 |

结论：当前队列可作为“多原型/feature mask/txrx geometry audit基线”，不能作为主动loss收益证据。下一轮联合验证必须等该队列完成审计，或只在空闲容量下另启新run并明确不覆盖当前队列语义。

## 联合假设

H1：Meta-SSL源域episode和pseudo/prototype gate提高`z_id`对未见receiver/day的稳定性，表现为strict UDU、receiver floor和satellite floor不下降。

H2：训练期`PrototypeMemoryBank`和`open_world_feature_space_loss`共同收缩同TX角半径、扩大TX类中心角间隔，降低old/new/unknown原型重叠。

H3：Phase2离线`P_tx/P_tx_dom`导出和`OpenWorldMultiPrototypeHead`能把地面几何收益转化为同一Stage2-B行内`old_acc>=0.80`且`unknown_FAR<=0.05`，并在Stage2-C提升`H_old_new`。

H4：如果H1成立但H2/H3不成立，瓶颈在原型半径、target support校准或拒识门控；如果H2成立但source floor下降，几何loss过强；如果H3只在coverage接近0时成立，则是退化拒识，不可晋级。

## 验证总体结构

本计划采用“两入口、三层证据链”。

| 层级 | 主入口 | 使用近期改动 | 产出 |
|---|---|---|---|
| G层：地面表征训练 | `code/train.py` | Meta-SSL、PrototypeMemoryBank、open-world feature loss、Phase2导出CLI | source DG指标、loss曲线、`P_tx/P_tx_dom`和几何诊断 |
| S层：Safe-SSDG主线兼容 | `code/SSDG/train_ssdg.py` | 当前只做audit；主动loss需先桥接默认关闭 | 证明同主线入口可复现，不把`train.py`机制收益误报为SSDG收益 |
| D层：部署开放世界评估 | `code/cvsrffi/open_world_head.py`和Stage2工具 | Phase2原型包、多原型head、new-class overlap rejection | Stage2-B旧类/未知，Stage2-C旧类/seen-new/unknown同排结果 |

## V0：完成最新队列审计

目标：把`JOINTSAFE36`作为最新Safe-SSDG audit基线。

必须读取全量日志和metrics，不允许只看tail。结果表必须同一行绑定candidate、seed、label/pseudo epoch、sat schedule、strict UDU、receiver floor、sat mean/floor、joint_safe、loss异常和audit字段。

V0输出作为后续控制组之一：

| 控制组 | 用途 |
|---|---|
| CEN51 refresh control | 已完成强基线，但只是control floor |
| JOINTSAFE36最佳source row | 当前SSDG主线source-only audit锚点 |
| JOINTSAFE36最佳satellite-retention row | 星地视图保留能力锚点 |

## V1：入口一致性门

联合验证分两条路径：

| 路径 | 入口 | 何时用 | 证据等级 |
|---|---|---|---|
| V1-A机制联合验证 | `code/train.py` | 立即可设计，覆盖Meta-SSL、PrototypeMemoryBank、open-world feature loss和Phase2导出 | 机制证据，可说明模块组合是否值得进入SSDG主线 |
| V1-B主线桥接验证 | `code/SSDG/train_ssdg.py` | 需要先把主动loss以默认关闭方式桥接，且保留audit-only默认 | Safe-SSDG同入口证据，推荐作为正式主线 |

V1-B不得绕过`train_ssdg.py`当前保护：现有非零`lambda_tx_proto/lambda_rx_proto/lambda_mask_aux/lambda_tx_supcon_masked/lambda_rx_supcon_masked/lambda_txrx_rect`会报`NotImplementedError`，这是正确的安全边界。若要在SSDG入口主动使用多原型/特征空间loss，必须先做代码桥接、测试和Git提交。

## V2：近期改动联合地面训练矩阵

矩阵名建议：`phase1_recent_stack12_20260630`。

统一协议：

- 数据：ManySig source receivers；`labeled_ratio=0.10`、`unlabeled_ratio=0.70`、`source_val_ratio=0.20`。
- 特征：`--generalization_feature z_id`。
- 禁止：target receiver、Stage2 support/query、unknown query进入训练、阈值拟合或模型选择。
- 每行训练结束后显式开启`--phase2_export_prototypes --phase2_export_feature_key z_id`，导出`P_tx/P_tx_dom`。
- 每行保留完整loss telemetry，不能用最终epoch单点替代曲线。

候选表：

| ID | 类型 | Meta-SSL | 训练原型 | 特征空间优化 | Phase2导出 | 目的 |
|---|---|---|---|---|---|---|
| RS12_C0_CONTROL | 控制 | 关 | 关 | 关 | 开 | 同入口基线，排除导出本身影响 |
| RS12_C1_META | 单模块 | `--use_meta_ssl_cvs --use_meta_rxday_episodes --lambda_ssl_tx 0.03 --lambda_ssl_proto 0.01 --lambda_meta_ssl 0.005` | 关 | 关 | 开 | 验证源域episode/pseudo gate是否改善strict UDU和sat floor |
| RS12_C2_PROTO | 单模块 | 关 | `--use_proto_memory --lambda_proto 0.006 --proto_domain_align_weight 0.30` | 关 | 开 | 验证训练期原型正则是否缩小`P_tx`半径 |
| RS12_C3_OWFEAT | 单模块 | 关 | 关 | `--lambda_open_world_feat 0.005 --ow_feat_domain_align_weight 0.00` | 开 | 验证角度几何loss的独立效果 |
| RS12_C4_META_PROTO | 双模块 | 同C1 | 同C2 | 关 | 开 | 验证Meta-SSL伪标签和原型pull是否互补 |
| RS12_C5_PROTO_OWFEAT | 双模块 | 关 | `--lambda_proto 0.006` | `--lambda_open_world_feat 0.005 --ow_feat_domain_align_weight 0.02` | 开 | 验证训练原型和开集几何是否协同 |
| RS12_C6_META_OWFEAT | 双模块 | 同C1 | 关 | `--lambda_open_world_feat 0.005` | 开 | 验证episode稳定性和几何loss是否冲突 |
| RS12_C7_FULL_LOW | 全栈保守 | 同C1 | `--lambda_proto 0.004` | `--lambda_open_world_feat 0.004 --ow_feat_domain_align_weight 0.01` | 开 | 主候选，低权重防止source退化 |
| RS12_C8_FULL_MID | 全栈主候选 | `--lambda_ssl_tx 0.04 --lambda_ssl_proto 0.015 --lambda_meta_ssl 0.006` | `--lambda_proto 0.006` | `--lambda_open_world_feat 0.006 --ow_feat_domain_align_weight 0.02` | 开 | 主要晋级候选 |
| RS12_C9_FULL_DOM | 全栈域对齐 | 同C8 | `--proto_domain_align_weight 0.40` | `--ow_feat_domain_align_weight 0.04` | 开 | 验证跨receiver/day prototype shift下降 |
| RS12_C10_FULL_MARGIN | 全栈大间隔 | 同C8 | 同C8 | `--ow_feat_inter_margin_deg 65 --ow_feat_radius_deg 10` | 开 | 验证old/new overlap是否进一步下降 |
| RS12_C11_STRESS_NEG | 负控 | `--lambda_ssl_tx 0.08 --lambda_ssl_proto 0.03` | `--lambda_proto 0.012` | `--lambda_open_world_feat 0.020 --ow_feat_domain_align_weight 0.05` | 开 | 压力边界，预期可能损伤source floor |

晋级规则：

| 门槛 | 要求 |
|---|---|
| 默认行为 | 所有新增模块默认关闭；C0必须与普通入口一致 |
| source DG | overall、strict UDU、receiver floor任一相对C0下降超过2pp则不进Phase2 |
| satellite | `leo_clear_weak/leo_low_elev_weak/leo_rain_weak`mean和floor不劣于C0；若下降超过1pp需人工复核 |
| Meta-SSL | `meta_ssl_coverage`、`meta_ssl_proto_agree`、`meta_ssl_teacher_conf`必须非退化；伪标签覆盖不能集中到少数TX/receiver |
| 原型几何 | `radius_p95_mean_deg`下降、`min_interclass_angle_deg`上升或overlap count下降，至少满足两项 |
| 特征空间loss | `train_ow_feat_active_classes>=2`稳定出现；不能靠batch覆盖不足产生虚假低loss |
| 导出完整性 | 每行必须有`.pt`和`.json`原型包，包含`P_tx`、`P_tx_dom`统计和radii |

## V3：离线多原型几何审计

对V2每一行做统一审计：

| 指标族 | 指标 |
|---|---|
| 类内半径 | 每个TX的mean/p95/p99/max angular radius |
| 类间间隔 | min/median class-center angle、top-k nearest TX pairs |
| 跨域稳定性 | 同TX不同receiver/day center angle、`P_tx_dom`shift |
| 多原型可用性 | old source prototype、target-old support prototype、seen-new prototype的overlap clearance |
| 部署成本 | prototype count、包大小、决策延迟、缺失TX/domain数 |

只允许最多2个候选进入V4。选择标准不是单项最大值，而是同一候选行内source不退化、satellite不退化、几何改善和导出完整同时成立。

## V4：Stage2-B旧类校准和未知拒识

输入：V3晋级的最多2个`phase2_export`包。

使用`OpenWorldMultiPrototypeHead.from_phase2_export`加载old source prototype；使用target-old support通过`add_target_prototypes`或`shrink_target_prototypes`形成同类多原型；query只用于评估。

协议：

| 项 | 设置 |
|---|---|
| target receiver | 先`20-1`，通过后扩展`3-19`,`7-14`,`7-7`,`8-8` |
| K | `5,10,20,50` |
| channel view | 主线`leo_clear_weak`，通过后加`leo_low_elev_weak`,`leo_rain_weak` |
| 阈值 | 只用source统计和target-old support；unknown query禁止拟合 |
| 指标 | `old_acc`、accepted_acc、coverage、`unknown_FAR`、AUROC/FPR95、old_unknown_hmean、rollback/defer |

成功要求：同一candidate、同一K、同一receiver、同一view行内同时满足`old_acc>=0.80`、`unknown_FAR<=0.05`和非退化coverage。

## V5：Stage2-C seen-new注册、unknown拒识和遗忘控制

只在V4通过后启动。

使用`register_new_class(...,overlap_margin=...)`注册`Y_new`，拒绝与old source/target-old prototype半径重叠的新类support。`Y_new`和`Y_unknown`必须来自ManyTx non-`Y_old`真实TX label，并互斥。

指标：

| 指标 | 解释 |
|---|---|
| old_acc | 旧类保持，必须继续满足OLD80_FIRST |
| seen_new_acc | 新类注册后的识别准确率 |
| `H_old_new` | 旧类和seen-new的同排调和指标 |
| unknown_FAR | unknown误接收率，目标`<=0.05` |
| confusion | old->new、new->old、unknown->new |
| overlap rejection | 新类support被拒绝次数和最近old class |

若`seen_new_acc`提高但old_acc跌破0.80，不得写成Stage2-C成功。若unknown_FAR达标但coverage接近0，也不得晋级。

## 与当前`JOINTSAFE36`的关系

`JOINTSAFE36`已经用上近期“模块存在性”层面的能力：`phase2_prototypes`、`feature_masks`、`tx_rx_geometry`和`open_world_head`都通过`train_ssdg.py`audit import检查进入运行日志。但它没有主动使用这些loss，因为权重为0且`phase1_distribution_audit_only=true`。

因此下一步不是否定当前队列，而是把它作为V0 audit基线：

1. 若`JOINTSAFE36`完成后source/sat指标强，V2用其最佳row作为SSDG强基线。
2. 若`JOINTSAFE36`完成后loss或分布异常，V2仍用`code/train.py`做机制验证，但不得直接声称SSDG主线收益。
3. 若要让SSDG主线也“都用上”，必须先桥接主动loss并保留默认关闭。

## 最小本地验证命令

联合矩阵生成或桥接前至少执行：

```powershell
conda activate ssr-gpu
python -m py_compile code\train.py code\cvsrffi\losses.py code\cvsrffi\phase2_prototypes.py code\cvsrffi\open_world_head.py
python -m pytest code\tests\test_phase2_prototypes.py code\tests\test_open_world_head.py code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py -q
python code\train.py --help | Select-String -Pattern "use_meta_ssl_cvs|lambda_meta_ssl|use_proto_memory|lambda_proto|lambda_open_world_feat|phase2_export"
```

若新增SSDG桥接：

```powershell
conda activate ssr-gpu
python -m py_compile code\SSDG\train_ssdg.py
python code\SSDG\train_ssdg.py --help | Select-String -Pattern "phase2_ground_prototypes|feature_masks|txrx_geometry|open_world"
```

## 当前决策

立即动作不是启动新实验，而是：

1. 等`phase1_gpu0_jointsafe36_queue_20260629_0930`完成并做完整日志审计。
2. 用本计划的V2矩阵把Meta-SSL、PrototypeMemoryBank、open-world feature loss和Phase2导出放进同一个`code/train.py`机制验证。
3. 只让V2/V3同排通过的最多2个候选进入Stage2-B。
4. 若用户要求Safe-SSDG正式主线“都用上”，先做`code/SSDG/train_ssdg.py`默认关闭桥接，再生成SSDG版联合矩阵。

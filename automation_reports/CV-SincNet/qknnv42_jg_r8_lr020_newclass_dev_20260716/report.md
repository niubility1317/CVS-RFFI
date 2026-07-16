# JG_R8_LR020新类注册development实验报告

## 运行摘要

- 实验ID：`qknnv42_jg_r8_lr020_newclass_dev_20260716`
- 日期：2026-07-16
- 操作者：Codex主agent`root`＋子agent`jg020_registration_exp`
- 当前状态：`INITIAL_LAUNCHER_PARSE_FAILURE_REPAIRED_AWAITING_SAFE_RESUME`；Phase1 cache完成，Stage2-C尚未开始
- 目标：验证锁定的`JG_R8_LR020`轻量旧类适配器在合法Stage2-C新类注册后的同row旧类保持、新类准确率与资源表现。
- 声明边界：这是单receiver、单development seed的开发单元；即使指标达到门槛，也不能替代独立确认矩阵或宣称总目标完成。

## 锁定假设与比较对象

锁定候选来自清理后保留的真实主线：ADV3B02基础checkpoint、ground P4、target BPJG-LOPO `joint_gate` rank8、`lr=0.02`、`epochs=5`、最多50个optimizer step。假设是：仅凭已注册target support训练的6,400参数关键层更新，可以在不读取query、不使用old/new query角色或类别quota的前提下，把原先old-only适配收益扩展到真实seen-new enrollment。

同row比较对象：

1. strict direct ADV3B02旧类分类头：同checkpoint、同old query、1-view、无support/adapter/FFT/TTA；只报告old指标。
2. identity-only单qKNN：同support/query、同注册类别、同1-view；报告注册前/注册后与资源。
3. JG_R8_LR020：同support/query、同注册类别、同1-view；报告注册前/注册后与资源。

## Development cell

| 字段 | 锁定值 |
|---|---|
| target receiver | `20-1`（ManySig target index 7；ManyTx receiver index 10） |
| source receivers | `1-1,1-19,14-7,18-2,19-2,2-1,2-19` |
| old TX | `14-10,14-7,20-15,20-19,6-15,8-20` |
| seed | `713101` |
| K | `10`个互异物理sample ID/注册类 |
| new class scales | 真实嵌套`Y_new^5⊂Y_new^10⊂Y_new^20`：`1-16,1-18,18-10,14-11,8-3`；再追加`18-8,10-10,16-19,20-12,4-10`；再追加`13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6` |
| scenarios | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| query view | 默认单view；本轮不因query结果触发额外view |
| adapter | `joint_gate`, rank8, 6,400 trainable params（运行时重新审计） |
| optimizer budget | `lr=0.02`, 5epoch, <=50step；SGD无momentum优先，完整loss trace |

## 严格时序与当前算法边界

JG只在注册前6个旧类的K10 support上训练一次。训练输入为6类×10shot×3个LEO_weak场景，共180行；训练函数只收到连续class index 0–5，不包含target-new行，也没有old/new分支。随后冻结candidate runtime，对全部6+n类registered support前向并计算cosine prototype；5→10→20注册仅追加新类prototype，不重新训练adapter。receipt强制记录`adapt_fit_class_count=6`、`prototype_fit_class_count=6+n`、`new_support_gradient_used=false`和`adapter_retrained_at_registration=false`。

当前cell不是最终qKNNV42：查询严格固定1-view，分类头是temperature=18的单cosine prototype；没有FFT96、没有低置信度触发的自适应多View、没有多prototype或最终qKNN检索/融合。这里的`support_view_count=3`指同一物理support在3个正式LEO_weak场景中的匹配训练视图，不代表query端3-view TTA。

为保证每个5/10/20 row都能独立密封和审计，当前launcher会重复执行3次完全相同的old-only adapter fit。这是`experimental_repeated_compute=3`，不是算法部署必需。连续部署的`deployment_amortized_compute`只计1次6,400参数、5epoch、最多50step适配；后续5→10→20只增加registered support forward和prototype append。

## 本地真实机制证据

主线程已持久化并提交真实artifact parity：`automation_reports/CV-SincNet/qknnv42_k1_support_trust_adapt_20260716/cached_jg_real_parity.json`，SHA256=`d9cfcdab9d066e2f0888061ba814979200865f62811f6790441dd95ab65193b1`，来源commit=`89fc2ff`。在真实ADV3B02 checkpoint SHA256=`2699eed...59c98`和P4 SHA256=`95f9a8...de446`上严格加载成功，target trainable parameters=6,400；7行、batch size=3时仅3次full-backbone调用，完整`z_id`与缓存`feat_cls/feat_dac/feat_pa`后只重算`id_gate+joint_proj`的最大绝对误差为`5.066394805908203e-7<=1e-6`，PASS。该证据验证机制等价性，正式资源与结果仍以本次N607 enrollment receipt为准。

## Phase2硬合同

```text
phase2_sample_view_policy=leo_weak_only_no_clean_access
clean_sample_access=false
clean_derived_signal_access=false
phase2_clean_dataset_reachable=false
phase2_clean_cache_reachable=false
phase2_clean_control_flow_reachable=false
phase2_pretrained_artifact_policy=sealed_phase1_checkpoint_only
phase2_query_decision_policy=per_sample_all_registered_classes
phase2_query_role_oracle_access=false
phase2_query_true_batch_class_count_access=false
phase2_query_class_quota_access=false
phase2_query_batch_global_assignment=false
```

这些字段必须由密封package成员allowlist、sample-level LEO overlay provenance、pre-open SHA256审计、strict loader和实际打开文件集合支撑。仅有本报告中的声明不会放行实验。

## 权限隔离设计

```text
offline package builder (Phase2边界外)
  -> support-only sealed package + manifest
  -> query-only sealed package + truth-free manifest
  -> independent truth sidecar (predictor不可达)

support-only enrollment process
  input: support package + sealed ADV3B02/P4 artifacts + locked JG config
  output: adapter delta + registered prototypes + loss/resource trace

truth-free predictor
  input: query package + adapter/prototypes + all registered class labels
  output: immutable per-sample predictions + SHA256

isolated scorer
  input: immutable predictions + independent truth sidecar
  output: before/after/direct metrics, per-class/per-scenario tables
```

## 必须生成的同run结果

| candidate | new scale | scenario | old_acc_before_increment | old_acc_after_increment | min_old_class_acc | seen_new_acc | H_old_new | forgetting | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 待运行 | 5 | `leo_clear_weak` | - | - | - | - | - | - | pending |
| 待运行 | 10 | `leo_clear_weak` | - | - | - | - | - | - | pending |
| 待运行 | 20 | `leo_clear_weak` | - | - | - | - | - | - | pending |
| 待运行 | 5/10/20 | `leo_low_elev_weak` | - | - | - | - | - | - | pending |
| 待运行 | 5/10/20 | `leo_rain_weak` | - | - | - | - | - | - | pending |

完成时还必须附：逐old/new TX、逐scenario、receiver×TX sample count/correct/accuracy、old→new/new→old稀疏confusion、support/query零重叠、direct与identity-only同sample比较。

## 资源与门槛

| 项目 | 要求 |
|---|---|
| trainable params | <=50,000；锁定候选预期6,400，须实测 |
| epochs/steps | <=5epoch且<=50step |
| persistent adapter+head state | <=256KB |
| query graph | 禁止dense query graph |
| predictor decision | 逐样本、全部已注册类 |
| K10 old_acc | >=92% |
| K10 min_old_class_acc | >=88% |
| K10 seen_new_acc | 5类>=92%、10类>=90%、20类>=86% |
| Pareto | 同row报告MAC、适配/推理时延、峰值显存、持久状态与identity-only变化 |

## 本地与N607执行记录

| 项目 | 当前值 |
|---|---|
| 本地Git分支/起点 | `codex/cvs-rffi-release-20260626`；本实现提交前HEAD含主线程parity commit`89fc2ff` |
| 本地现有未提交修改 | 用户已有`mitigating_da_rootcause_20260710_104628/{progress.md,task_plan.md}`，不得触碰 |
| 新增实现 | `jg020_stage2c.py`、support-only enrollment、apply-only predictor、split package builder、顺序launcher、3个candidate lock、registered cache spec、3个边界descriptor、9项测试、traceability/report |
| 本地验证 | `ssr-gpu`中py_compile PASS；`pytest tests/test_jg020_stage2c_isolation.py -q`为9/9 PASS，主线程独立复验同为9/9；3个lock、cache spec校验PASS；launcher dry-run展开19阶段条目；`git diff --check` PASS |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| N607只读preflight | 2026-07-16 10:50:07 CST直连PASS；`dell-DSS8440`；项目根可见；8×RTX 3090均为10MiB/24576MiB、0%利用率 |
| N607训练inventory | 2026-07-16 10:53:48+0800；`active_training_processes=[]`、`gpu_compute=[]`、`unknown_training_active=false`、route=`direct` |
| SSH断开审计 | preflight与inventory后均为`N607_SSH_DISCONNECTED=PASS`，无残留`ssh.exe`或到N607/bridge的ESTABLISHED TCP22 |
| 远端环境 | 项目根`/home/szu2070436088/2510044040/CV-SincNet`；执行时使用`ssr-gpu`，启动前仍需校验目标文件/依赖哈希 |
| 首次启动 | 物理GPU0；PID=`2250148`；`CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n CVS-RFFI python -u paper_reproduction/scripts/launch_cvs_jg020_stage2c_dev_20260716.py --execute` |
| 首次外层日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/qknnv42_jg_r8_lr020_newclass_dev_20260716_launcher.out` |
| 首次运行状态 | launcher已退出；Phase1 cache成功，Stage2-C未开始；没有性能结果，不能报告candidate指标 |
| 预期输出 | sealed manifests、adapter/prototype、loss trace、immutable predictions、scorer tables、resource audit、完整日志 |

## 首次启动完整日志诊断与恢复设计

已完整读取首次`launcher.out`的97行、`phase1_offline_cache.log`的80行，并保存到本报告目录`remote_logs/initial_failure/`。Phase1为成功状态：`stage2_registered`三场景各1040行，物理sample根一致；`forbidden_members_checked_before_iq_read=true`、`clean_sample_access=false`。唯一warning是NumPy私有命名空间弃用提示；没有NaN、Inf、OOM、Killed或训练loss异常。

失败发生在任何Stage2-C row建立之前。cache builder输出的是带warning前缀的多行pretty JSON，launcher旧实现却执行`json.loads(lines[-1])`，因此把单独的`}`送入JSON解析器并触发`JSONDecodeError`。这是launcher控制流错误，不是数据、ADV3B02、JG适配或新类注册性能失败。

修复内容：

1. 从混合stdout末尾逆向识别最后一个完整JSON文档，支持warning＋pretty JSON与单行JSON。
2. 新增显式`--resume`；只允许复用已有`phase1_cache/cache_set.json`且尚无任何`new_5/new_10/new_20`row、尚无`execution_summary.json`的cache-only中断状态。
3. 不删除、不覆盖、不重建已有cache；若出现部分Stage2-C row则fail closed。
4. 本地`ssr-gpu`回归更新为10/10 PASS，并直接用首次完整80行cache log验证解析得到`stage2_registered/1040`。

## 启动后对话回顾与路线教训

2026-07-16刷新项目conversation index，共978条项目相关记录。重点命中线程`019f6882-849d-74c2-8c0b-534ae0257c49`、`019f6710-a7e4-7541-ba9f-fdb814a9f99c`与`019f6573-9453-7e90-b4b8-eabac68fe8e4`。回顾结论如下：

1. 任务不能退化为old-only域适应；必须在同一运行提供适应后注册前old指标和new5/10/20注册后的old/new/H/最低类/遗忘。本cell正为补齐该缺口。
2. 历史`88.8354%`只来自source-old、不同切分、单seed诊断，不能当作当前target Stage2-C最强性能；本次结果必须独立报告。
3. K1目标仍是适应后明显优于strict direct ADV3B02。历史JG/JP梯度在K1对identity为负，因此后续只能使用support-only信赖门或闭式轻量对齐，不得靠query选择补丁。
4. 多View被用户明确判断为高性能关键，但应采用默认1-view、低置信度再触发3/5-view。本cell故意先锁1-view cosine baseline，用于隔离JG和注册本身；它不是最终qKNNV42。
5. 禁止query角色Oracle、类别配额和global assignment是不可放宽边界；即使准确率下降也不能回到历史role/quota筛选。
6. 下一轮只有在本cell得到真实注册混淆后，才决定加入FFT96、对称qKNN/multi-prototype与自适应View；不能同时堆叠所有模块而失去归因。

## 当前风险与放行条件

1. enrollment package exact role set只含support、checkpoint、P4、direct mapping和lock；apply package exact role set只含query、3个runtime、head、receipt和lock。双向物理排除测试均通过，truth仅在独立scorer root。
2. `.cvspred`使用exclusive temp和atomic no-replace发布，首次可见即只读；重复目标路径测试按预期抛出`FileExistsError`。apply predictor不包含prototype fit或optimizer。
3. `20-1`下真实新TX逐类至少K10+Q20覆盖仍须由N607 offline cache build验证；配置预设每TX最多40行，但配置声明不是数据coverage证据。
4. 本cell只提供development证据；无FFT、无自适应多View、非最终qKNNV42，也不能替代5receiver×5seed确认矩阵。
5. N607直连preflight与活动训练inventory已通过，当前无活动训练；仍须完成Git提交、SCP映射、远端依赖哈希和目标路径不存在性审计后才允许启动。

# JG_R8_LR020新类注册development实验报告

## 运行摘要

- 实验ID：`qknnv42_jg_r8_lr020_newclass_dev_20260716`
- 日期：2026-07-16
- 操作者：Codex主agent`root`＋子agent`jg020_registration_exp`
- 当前状态：`EIGHTH_SCORER_HASH_BINDING_CALL_REPAIRED_AWAITING_RETRY7`；retry6的new5 enrollment与三个runtime parity已PASS，尚未生成prediction或正式指标
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
| 本地验证 | `ssr-gpu`中py_compile PASS；最新`pytest tests/test_jg020_stage2c_isolation.py -q`为14/14 PASS；3个lock、cache spec校验PASS；launcher dry-run展开19阶段条目；待提交前再次执行`git diff --check` |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| N607只读preflight | 2026-07-16 11:48:16 CST直连PASS；`dell-DSS8440`；项目根可见；8×RTX 3090均为10MiB/24576MiB、0%利用率 |
| N607训练inventory | 2026-07-16 11:48:33+0800；`active_training_processes=[]`、`gpu_compute=[]`、`unknown_training_active=false`、route=`direct` |
| SSH断开审计 | preflight与inventory后均为`N607_SSH_DISCONNECTED=PASS`，无残留`ssh.exe`或到N607/bridge的ESTABLISHED TCP22 |
| 远端环境 | 项目根`/home/szu2070436088/2510044040/CV-SincNet`；执行时使用`ssr-gpu`，启动前仍需校验目标文件/依赖哈希 |
| 首次启动 | 物理GPU0；PID=`2250148`；`CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n CVS-RFFI python -u paper_reproduction/scripts/launch_cvs_jg020_stage2c_dev_20260716.py --execute` |
| 首次外层日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/qknnv42_jg_r8_lr020_newclass_dev_20260716_launcher.out` |
| 首次运行状态 | launcher已退出；Phase1 cache成功，Stage2-C未开始；没有性能结果，不能报告candidate指标 |
| 预期输出 | sealed manifests、adapter/prototype、loss trace、immutable predictions、scorer tables、resource audit、完整日志 |

retry7同步映射（本地→N607同相对路径；其余retry6文件保持远端已验证版本）：

| 文件 | 本地SHA256 | 目的 |
|---|---|---|
| `paper_reproduction/scripts/build_cvs_jg020_split_packages.py` | `d3670d99dec84dbf0398ba57c8c6ca6a0211b48dabcc801b74919abdb2271e51` | 对source与新scoring manifest都显式绑定expected SHA256 |
| `paper_reproduction/scripts/launch_cvs_jg020_stage2c_dev_20260716.py` | `58c022318079f7d0b5e0bf9ef120c5314af4fc9ab29a632f1f64ebb5a2b247d3` | 允许不可覆盖的`retry7`运行根 |

retry4实际启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 nohup /opt/miniconda3/bin/conda run --no-capture-output -n CVS-RFFI python -u paper_reproduction/scripts/launch_cvs_jg020_stage2c_dev_20260716.py --execute --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg_r8_lr020_newclass_dev_20260716_retry4 --reuse-cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg_r8_lr020_newclass_dev_20260716/phase1_cache/cache_set.json > /home/szu2070436088/2510044040/CV-SincNet/logs/qknnv42_jg_r8_lr020_newclass_dev_20260716_retry4_launcher.out 2>&1 &
```

启动前远端复核：`retry4`根不存在；环境为Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、Torch`2.1.0+cu121`、NumPy`2.2.5`；4个同步文件SHA256与当时的retry4记录一致；远端py_compile、enrollment CLI import闭包和launcher dry-run均PASS。上表已更新为下一次retry5待同步哈希。

retry5已按相同命令启动，外层PID=`2287997`；retry6计划沿用相同命令，仅把run root和外层日志改为`..._retry6`。两者都只读复用原run的`phase1_cache/cache_set.json`，不删除或覆盖既有训练artifact。

## 首次启动完整日志诊断与恢复设计

已完整读取首次`launcher.out`的97行、`phase1_offline_cache.log`的80行，并保存到本报告目录`remote_logs/initial_failure/`。Phase1为成功状态：`stage2_registered`三场景各1040行，物理sample根一致；`forbidden_members_checked_before_iq_read=true`、`clean_sample_access=false`。唯一warning是NumPy私有命名空间弃用提示；没有NaN、Inf、OOM、Killed或训练loss异常。

失败发生在任何Stage2-C row建立之前。cache builder输出的是带warning前缀的多行pretty JSON，launcher旧实现却执行`json.loads(lines[-1])`，因此把单独的`}`送入JSON解析器并触发`JSONDecodeError`。这是launcher控制流错误，不是数据、ADV3B02、JG适配或新类注册性能失败。

修复内容：

1. 从混合stdout末尾逆向识别最后一个完整JSON文档，支持warning＋pretty JSON与单行JSON。
2. 新增显式`--resume`；只允许复用已有`phase1_cache/cache_set.json`且尚无任何`new_5/new_10/new_20`row、尚无`execution_summary.json`的cache-only中断状态。
3. 不删除、不覆盖、不重建已有cache；若出现部分Stage2-C row则fail closed。
4. 本地`ssr-gpu`回归更新为10/10 PASS，并直接用首次完整80行cache log验证解析得到`stage2_registered/1040`。

首次`--resume`使用PID=`2253380`。它成功复用Phase1 cache，并完成new5离线source bundle：11个注册类、support pool 110、query 220；随后在enrollment package manifest发布前fail closed。已完整读取并保存`remote_logs/resume1_failure/resume1.out`的22行日志。

第二个根因是NPZ表示层不一致：来源manifest按NumPy逻辑key记录`support_pool_leo_weak_iq`等8个成员，ZIP物理条目则自动带`.npy`后缀；旧JG validator直接比较二者，错误触发`JG020 NPZ member allowlist drift`。实测实际与descriptor逻辑key一一对应，没有额外或缺失成员。修复统一在逻辑key层执行精确有序allowlist，同时继续拒绝重复成员、目录成员、非`.npy`条目及额外/缺失key；materialisation也按`np.load(...).files`复验。

由于原run已含不可覆盖的new5中间目录，不删除、不移动、不覆盖。恢复路径锁定为新根`runs/qknnv42_jg_r8_lr020_newclass_dev_20260716_retry1`，只允许引用原run的`phase1_cache/cache_set.json`；retry1从空Stage2-C目录开始。修复后本地10/10 PASS、retry1 launcher dry-run PASS，并用真实new5 support NPZ验证`ACTUAL_SOURCE_NPZ_LOGICAL_ALLOWLIST_MATCH=True`。

retry1使用PID=`2256635`，成功通过new5 source bundle与enrollment package双重preflight；`query_member_reachable=false`、`truth_member_reachable=false`、`clean_member_reachable=false`均已在远端receipt中成立。随后enrollment CLI在导入阶段停止：脚本误从`paper_reproduction.scripts`导入实际位于`code/scripts`的TorchScript导出器。远端目标文件存在且SHA256与本地完全相同，因此不是漏同步文件，而是Python模块路径错误。完整17行外层日志与4行enrollment日志已保存到`remote_logs/retry1_import_failure/`；没有optimizer step、GPU训练或性能输出。

修复为显式把`code/scripts`加入CLI导入闭包，并按该目录的既有top-level模块名加载；新增真实子进程`--help`导入闭包测试，回归为11/11 PASS。retry1中间产物继续保留；下一次从空`runs/qknnv42_jg_r8_lr020_newclass_dev_20260716_retry2`开始并只读复用原密封cache。

retry2使用PID=`2259168`，再次通过source/enrollment隔离preflight，随后在模型构建前因严格时序重构遗留的旧变量名`support_rows`触发`NameError`。完整21行外层日志与8行enrollment日志已保存到`remote_logs/retry2_name_failure/`；仍无optimizer step。修复把3个模型的`input_len`统一绑定已构造的old-only`adapt_rows`，并增加静态回归断言：enrollment脚本中不得再出现`support_rows`，且3处模型输入长度均来自`adapt_rows`。本地11/11 PASS，下一根为不可覆盖的`..._retry3`。

retry3再次通过new5 source bundle与enrollment package双重preflight，远端receipt继续确认`query_member_reachable=false`、`truth_member_reachable=false`、`clean_member_reachable=false`；随后在首个cached JG tensor构造处触发`TypeError: expected np.ndarray (got numpy.ndarray)`。完整外层日志与enrollment日志已保存到`remote_logs/retry3_numpy_abi_failure/`。失败发生在optimizer建立和首个optimizer step之前，因此仍没有训练loss或性能结果。根因是N607当前`CVS-RFFI`环境的Torch 2.1与NumPy 2 C-ABI桥不兼容；这不是数据、协议、候选或GPU失败。

修复把JG020小型support、feature与logit在Torch/NumPy之间的边界改为经Python值的显式复制，避免`torch.from_numpy`和`.cpu().numpy()`依赖NumPy C-ABI；同时保留dtype、device和数值等价测试。新增回归断言确保JG020执行闭包不再出现这两类ABI桥调用，`ssr-gpu`中py_compile与12/12定向测试通过。前三个retry根继续只读保留；下一根锁定为不可覆盖的`runs/qknnv42_jg_r8_lr020_newclass_dev_20260716_retry4`，仍只读复用原密封Phase1 cache。

retry4已确认ABI修复有效并首次完整执行new5的support-only适配：5epoch、50个optimizer step，loss从`1.946215`降至`1.488509`，support train accuracy从`0.711111`升至`0.741667`，mean margin从`0.302293`升至`0.334063`；完整loss trace已回收到`remote_logs/loss_trace.json`和`loss_trace.csv`。训练无NaN、Inf、OOM或Killed，并已写出6,400参数FP16 delta、prototype head与direct runtime。此处仅是support训练诊断，不是query性能结果。

retry4随后在TorchScript parity阶段fail closed。ADV3B02底层把trace时batch维转换为Python整数，batch=2被固化；旧代码却用batch=8的probe调用已trace runtime，触发`shape '[4, 1, 256]' is invalid for input of size 4096`。修复把runtime合同显式锁为2-row microbatch：trace与parity使用相同2-row形状，receipt/apply predictor双向绑定该值，predictor逐2行向量化并要求query数可整除2；当前锁定5/10/20新类行的每类query=20，均满足整除。该microbatch只用于计算向量化，不读取role、truth、quota或全局类别分配。`ssr-gpu`中新增固定batch回归后py_compile与13/13测试通过；下一根为不可覆盖的`..._retry5`。

retry5确认2-row trace形状修复有效，再次完成new5的5epoch/50step support-only训练；随后`torch.jit.trace(check_trace=True)`在Torch`2.1.0+cu121`的内部重复trace比较中抛出`complex128 != float64`dtype comparison exception。该异常发生在runtime发布前，未进入truth-free predictor或scorer，仍无query性能结果。完整外层/enrollment日志与loss trace已回收到`remote_logs/retry5_*`。

JG020在保存并重新加载每个runtime后，本来就逐元素比较eager feature/logit与runtime feature/logit，容差为`1e-4`并在漂移时fail closed；因此重复的Torch内部trace checker不提供额外协议证据。retry6关闭`check_trace`，但保留加载后显式数值parity、runtime SHA256和receipt绑定。`ssr-gpu`中py_compile与13/13测试通过，测试静态锁定`check_trace=False`和显式parity调用同时存在。

retry6使用外层PID=`2291586`，首次完整完成new5 enrollment并生成PASS receipt：6,400可训练参数、5epoch/50step、适配壁钟`1.652494s`、峰值CUDA内存`31,924,224B`、持久状态`70,816B`。candidate/direct/identity三个runtime的eager-vs-loaded feature/logit最大绝对误差均为`0.0`，runtime SHA256已写入receipt。该结果证明训练与runtime导出链路健康，但仍不是query性能指标。

retry6随后在apply package的独立truth/scorer复制阶段因接口升级漏参而fail closed：`load_verified_scoring_sidecar()`现要求关键字`expected_scoring_manifest_sha256`，JG split builder的两处调用仍使用旧签名。修复对原source scoring manifest和新生成scoring manifest分别先计算SHA256再传入验证函数，并把新manifest SHA256写入builder结果；这加强truth-sidecar绑定，不改变预测输入或算法。`ssr-gpu`中py_compile与14/14测试通过；下一根为不可覆盖的`..._retry7`。

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

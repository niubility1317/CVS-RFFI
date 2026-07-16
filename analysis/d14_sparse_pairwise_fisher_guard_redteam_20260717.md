# D14稀疏pairwise Fisher双阶段门独立红队

## 结论

**增量终审判定：分层GO，整体仍NO-GO。**

- D14核心module与最新`development_select` runner的聚焦本地验证：**GO**。external expected seal、authority fail-closed、磁盘state重建与bitwise score round-trip、开发runner拒绝confirmation模式均已实质落地。
- 当前D14真实性能路线：**NO-GO**。真实support结果已经回退Z0，正候选未消除旧类遗忘；不得打开query或进入125确认矩阵。
- formal committed-state apply门：**代码级GO**。默认模式只接受FORMAL_SELECTED且promotion、support gate和formal authority全真的COMMIT；diagnostic加载必须显式设置`require_formal_promotion=False`。两种模式都核对state candidate、runtime、checkpoint和feature-code绑定。
- runner级zero回退：**部分关闭**。纯选择函数、zero state序列化和COMMIT加载反例已通过，但尚无最新runner完整`run()`的“三场景全部正arm失败→六个最终Z0 state→COMMIT”端到端测试或最新真实artifact。
- confirmation：开发runner误用风险已关闭，但`confirmation_apply_locked`尚未实现；K1/K5/K20独立exact-K闭环仍缺失。
- 本红队没有打开任何query、truth、prediction、score或scorer包，不产生D14性能成功声明。

## 2026-07-17增量终审

### 最新审计对象

|对象|SHA256|
|---|---|
|`code/cvsrffi/stage2_sparse_pairwise_fisher_guard.py`|`B39C1732124BF76C2ED5852E243D308A8BA3DDD2F2159FF5CE67B4E1E3032731`|
|`code/scripts/run_d14_support_only_pairwise_fisher_guard.py`|`28BC156CC4D89B54E6642823265D815DE103B34E0ACB1F72560A222816C78D4A`|
|`tests/test_stage2_sparse_pairwise_fisher_guard.py`|`04851D40A86002911146FB8096F89DED244013A5C83C3C1967C8BFCE35B8835F`|
|`tests/test_run_d14_support_only_pairwise_fisher_guard.py`|`0FF7734C29B5D127CBC5ED10299BA2A3DD08E7BCD3FFC80D8078A2A1117968C3`|
|`analysis/d14_sparse_pairwise_fisher_guard_traceability_20260717.md`|`ACA24AD2BF33EEC6DCFA4A357D9B64AC93D51DB5CBEFFFEDE8749602EEC36641`|

验证：

```text
py_compile: PASS
pytest: 17 passed
git diff --check: PASS
```

### 旧P0关闭矩阵

|旧项|最新状态|增量证据|剩余边界|
|---|---|---|---|
|P0-1 external expected seal|已关闭|`run()`强制接收before/after expected seal SHA，并原样传入verified bundle loader；旧self-hash调用已移除|expected SHA仍必须来自runner外部控制面，不能由调用包装器临时现算|
|P0-2 authority状态冲突|已关闭并fail closed|`promotion_ready=support_candidate_pass and formal_authority`；pre-open必须同时为formal authority、formal metric allowed和`CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS`；外部authority evidence还需外部expected SHA及package/seal绑定|当前现有bundle pre-open仍为`LOCAL_PROTOCOL_REPAIR_REQUIRED`，所以现实边界仍是diagnostic-only，不是formal promotion|
|P0-3磁盘state重建与formal apply|已关闭|NPZ+JSON经外部hash验证后重建immutable state；默认formal模式要求FORMAL_SELECTED、promotion/support gate/formal authority全真；diagnostic模式必须显式开启；两种模式均核对candidate/runtime/checkpoint/feature-code|真实formal apply仍需一个满足authority和性能门的新COMMIT；当前旧Z0 artifact本身不具备formal资格|
|P0-4 runner真实zero|部分关闭|新增纯`_select_candidate`失败回退、zero state写盘、COMMIT外部hash加载与empty-edge/gamma0断言|测试使用手工candidate rows和手工COMMIT，没有走完整`run()`三场景失败路径；最新真实D14目录也由旧runner SHA生成|
|P0-5开发/确认重选|开发侧已关闭|runner唯一合法mode为`development_select`；传`confirmation_apply_locked`立即fail closed|独立confirmation apply-only runner尚不存在，故仍不能进入5 receiver×确认seed矩阵|
|P0-6独立K1/K5/K20|未关闭|K1仅有数值闭合；runner仍只运行strict K10|必须使用独立sealed exact-K package，不能从K10切片|

### external seal和authority复核

最新入口要求：

```text
expected_before_seal_sha256
expected_after_seal_sha256
mode=development_select
```

长度不为64时在package读取前拒绝；bundle loader会核对实际seal哈希。源码不再包含：

```text
expected_seal_sha256=_sha256_file(seal)
```

authority门同时要求：

```text
before/after preopen formal_launch_authority=true
before/after preopen formal_metric_claim_allowed=true
before/after control_state=CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS
external authority evidence SHA=外部expected SHA
authority evidence绑定before/after package root和seal SHA
```

缺少authority evidence不会报虚假PASS，而是返回：

```text
DIAGNOSTIC_SUPPORT_ONLY_NO_AUTHORITY_EVIDENCE
formal_authority=false
promotion_ready=false
```

这关闭了原先“pre-open写LOCAL_PROTOCOL_REPAIR_REQUIRED而runner仍promotion”的冲突。

### committed-state loader与formal apply增量攻击

已通过：

- COMMIT文件必须匹配外部`expected_commit_sha256`；
- COMMIT必须绑定所请求state key；
- NPZ、metadata必须匹配COMMIT记录的SHA；
- loader重建完整state并触发state content/resource/binding校验；
- 写盘后随机feature score和prediction与内存state逐位一致。
- 默认`require_formal_promotion=True`时，COMMIT必须同时满足：
  - `status=SUPPORT_ONLY_D14_FORMAL_SELECTED_NO_QUERY_OPEN`；
  - `promotion_ready_for_single_query_candidate=true`；
  - `support_candidate_gate_pass_before_authority=true`；
  - `formal_launch_authority=true`；
  - `query_opened=false`。
- diagnostic state只能在调用者显式设置`require_formal_promotion=False`时加载，且该COMMIT必须包含`DIAGNOSTIC`状态、promotion=false和formal authority=false。
- formal与diagnostic两种模式均在state重建后核对：
  - `state.candidate_id`和state hyperparameter candidate等于COMMIT selected candidate；
  - sealed runtime SHA一致；
  - sealed Phase1 checkpoint SHA一致；
  - combined feature-code SHA一致。

红队实测：

```text
formal_matching PASS:d14_c1_balanced_light
selected_candidate_id REJECT:D14 committed state metadata binding drift
sealed_runtime_sha256 REJECT:D14 committed state metadata binding drift
sealed_phase1_checkpoint_sha256 REJECT:D14 committed state metadata binding drift
combined_feature_code_sha256 REJECT:D14 committed state metadata binding drift
default_reject_diagnostic REJECT:D14 COMMIT lacks formal promotion authority
```

因此原P0-3关闭。这里的“代码级GO”不等于当前真实D14已获得formal authority；它只证明未来满足性能与authority门的COMMIT具备fail-closed加载入口。

### runner级zero回退复核

新测试已经证明：

- 所有candidate row的`all_scenario_gate_pass=false`时，纯选择函数返回`d14_z0_true_zero_base`；
- zero state为empty old edges、empty new edges、`gamma_old=gamma_new=0`；
- zero state写盘后可通过COMMIT和loader重建。

但该测试没有执行完整`run()`，没有验证真实三场景evaluation、authority、最终六个state、audit、report与COMMIT共同形成一个Z0闭环。现有真实目录：

```text
E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d14_sparse_pairwise_fisher_v1_base
```

其COMMIT绑定旧runner SHA：

```text
CD49E5986709A6EF7FF770F653203B6959DBBD29E94CF3A35D5CCD5D3EDFFD8E
```

不是本次终审的最新runner SHA。因此原P0-4只能标记为部分关闭。补齐方式是使用合法diagnostic authority边界运行最新runner一次，或构造完整sealed小包集成测试，断言：

- 三场景全部正arm失败；
- selected candidate为Z0；
- before/after×3共六个最终state均为真实zero；
- `support_candidate_gate_pass_before_authority=false`；
- `promotion_ready=false`；
- COMMIT经formal loader语义检查时因未晋升而拒绝apply。

### development-only边界

增量反例：

```text
mode=confirmation_apply_locked
```

结果：

```text
D14RunnerError:
this runner is development_select only;
confirmation must apply one locked candidate
```

所以当前runner不会被误用于确认集重选arm。下一步必须单独实现confirmation apply-only流程；它只能读取开发COMMIT锁定的一个candidate，不能加载候选网格或重新执行support选择。

### 最新GO/NO-GO边界

|动作|判定|
|---|---|
|D14 module合成/单元验证|GO|
|最新development runner本地聚焦验证|GO|
|用当前协议状态执行diagnostic-only strict-K10 support复核|条件GO，必须提供外部expected seals，并明确authority false、NO_QUERY|
|把当前真实Z0结果声明为D14性能改进|NO-GO|
|用满足FORMAL_SELECTED、promotion/support/authority全真且绑定一致的新COMMIT加载state|代码级GO|
|用当前真实Z0/diagnostic COMMIT授权formal query|NO-GO|
|打开candidate-bound query|NO-GO|
|进入125确认矩阵|NO-GO|
|K1/K5/K20稳定性声明|NO-GO|

下文保留首次红队的完整攻击清单；其中P0状态以本增量终审矩阵为准。

## 审计对象与版本

|对象|SHA256|
|---|---|
|`code/cvsrffi/stage2_sparse_pairwise_fisher_guard.py`|`1286D75DDC41EDAF9B11B6162A9F17AFFD2D7A0AAA643081537938522C35120E`|
|`code/scripts/run_d14_support_only_pairwise_fisher_guard.py`|`CD49E5986709A6EF7FF770F653203B6959DBBD29E94CF3A35D5CCD5D3EDFFD8E`|
|`tests/test_stage2_sparse_pairwise_fisher_guard.py`|`04851D40A86002911146FB8096F89DED244013A5C83C3C1967C8BFCE35B8835F`|
|`tests/test_run_d14_support_only_pairwise_fisher_guard.py`|`F9C60A00C755172C47A7BBED0C6BD5ADEAD56156035B1E059793CEEE397F8A99`|
|`analysis/d14_sparse_pairwise_fisher_guard_traceability_20260717.md`|`5BEC730FC38AD7F63EA1B6892367C96A58AD8F4DC7D94FB0BA882EF848FD03C8`|

验证命令：

```powershell
$env:PYTHONPATH='E:\type10-7\github_publish\CVS-RFFI-repo\code'
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile `
  code/cvsrffi/stage2_sparse_pairwise_fisher_guard.py `
  code/scripts/run_d14_support_only_pairwise_fisher_guard.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q `
  tests/test_stage2_sparse_pairwise_fisher_guard.py `
  tests/test_run_d14_support_only_pairwise_fisher_guard.py
```

结果：`14 passed`。

## 已确认通过的机制

|检查项|结论|证据|
|---|---|---|
|单一received-IQ视图|PASS|runner只登记`operator_id="base"`，每次特征抽取强制physical batch=1；未调用LEO信道模拟器|
|strict K10|PASS|每类必须恰好10行，rank必须为`0..9`，token、post-channel IQ SHA必须唯一|
|跨三场景物理行复用|PASS但见后述包层限制|runner拒绝跨场景重复sample token或重复post-channel IQ SHA|
|旧pair稀疏性|PASS|最多3条，确定性最大权匹配，端点互斥；不是dense class/query图|
|pair选择held泄漏|PASS|旧pair只使用outer fold train K8中的内部LOO统计|
|独立选择带宽|PASS|`select_band_old`与预测触发`band_old`已分离，并进入state hash和统一candidate lock|
|new rival选择|PASS|旧样本正确性由train-only内部LOO D14 Before判定，包含锁定old edge修正；new样本使用内部LOO new prototype|
|held2不可回流|PASS|除既有测试外，红队同时极端修改fold0 held old和held new；train selection SHA、selection tensor SHA、old pair和new rival全部不变，held指标允许变化|
|After旧score锁|PASS|new修正读取immutable Before，要求stored rival等于Before old argmax，只修改对应new score；旧列逐位相同|
|无query role/quota入口|PASS|formal API只接收state和恰好一个runtime-authorized artifact；无truth、role、quota、true batch count或global assignment参数|
|K1数值闭合|PASS|显式全零K1反例中prototype和edge数组均finite，old/new edges全部关闭|
|稀疏部署数组|PASS|持久化state不保存variance、`w`或midpoint，只保存prototype、edge索引、`direction+bias`和rival|
|真实zero数学状态|PASS|module级zero为base、空old edge、空new edge、`gamma_old=gamma_new=0`，随机feature上与alpha0 score/prediction逐位相同|

红队额外反例结果：

```text
held_old_new_train_state_invariant=True
held_metrics_changed=True
zero_k1_finite_closed=True
```

## P0阻断项

### P0-1：detached seal的“期望哈希”由待验证seal自身现算，缺少外部信任根

runner在`code/scripts/run_d14_support_only_pairwise_fisher_guard.py:179`调用：

```text
expected_seal_sha256=_sha256_file(seal)
```

这会把`expected_seal_sha256`从外部控制面信任根退化为“文件等于自身哈希”的恒真检查。最小反例是同时替换一个自洽package和对应seal；runner重新计算新seal哈希后仍会进入内部一致性验证，外部方法锁无法发现替换。

必须门：

1. CLI必须显式接收`--before-seal-sha256`和`--after-seal-sha256`，或从已验证、不可由本runner改写的上游COMMIT/control state读取。
2. 上游信任根必须同时绑定receiver、seed、registration state、K、package root和seal SHA。
3. runner输出必须记录“用户/控制面提供的expected SHA”和“实际SHA”，不得仅记录现算值。

### P0-2：pre-open证据明确是`LOCAL_PROTOCOL_REPAIR_REQUIRED`，runner却可能标记可晋升

被调用的`somph_predictor_bundle`审计当前明确输出：

```text
formal_launch_authority=False
formal_metric_claim_allowed=False
control_state=LOCAL_PROTOCOL_REPAIR_REQUIRED
```

位置：`code/cvsrffi/somph_predictor_bundle.py:1100-1102`。

D14 runner把该审计原样放入`preopen_audit`，但`promotion_ready`仅由support指标决定，见`code/scripts/run_d14_support_only_pairwise_fisher_guard.py:714`。因此一个输出可同时声称：

```text
preopen_audit.control_state=LOCAL_PROTOCOL_REPAIR_REQUIRED
promotion_ready_for_single_query_candidate=True
```

这是直接的控制状态冲突。

必须门：

1. 只要before或after pre-open audit不是当前协议下的完整PASS，`promotion_ready`必须强制为`False`。
2. 如果materialization后的真实IQ重算和provenance交叉检查足以关闭旧的structural-only状态，必须由bundle loader生成新的明确状态，例如`CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS`，不能由D14 runner自行推断。
3. D14 audit和COMMIT必须显式声明完整Phase2字段：`phase2_clean_dataset_reachable=false`、`phase2_clean_cache_reachable=false`、`phase2_clean_control_flow_reachable=false`、`phase2_pretrained_artifact_policy=sealed_phase1_checkpoint_only`以及全部source/query边界。当前仅写若干缩写布尔值不足以成为launchable row。

### P0-3：所谓state round-trip没有重建可预测state，尚不是可部署artifact

`_write_state`会写NPZ和JSON，并比较NPZ数组与内存数组，但没有：

- 从NPZ+JSON重建`SparsePairwiseFisherState`；
- 重新执行完整state validation和content SHA计算；
- 使用重建state做score/prediction逐位等价测试；
- 提供formal apply阶段可调用的sealed state loader。

当前测试`test_actual_state_serialization_roundtrip_and_resource_audit`只证明“刚写出的数组可读回”，不证明“部署端可安全加载并预测”。

必须门：

1. 新增只接受NPZ、JSON、COMMIT和外部expected COMMIT SHA的state loader。
2. loader必须验证COMMIT→NPZ/JSON哈希、JSON→state content SHA、数组dtype/shape/finite/read-only、class顺序、old/new边、hyperparameter lock和runtime/checkpoint/code绑定。
3. 从磁盘重建state后，在随机feature和合法单样本artifact上与内存state的score、prediction逐位相同。
4. 任何metadata字段、数组字节、class顺序、edge顺序、runtime SHA或candidate lock变异都必须fail closed。

### P0-4：真实zero回退只有module级证明，没有runner级“全部正arm失败”闭环测试

inline选择逻辑看起来会在全部正候选失败时选`d14_z0_true_zero_base`，但当前runner测试没有构造“三场景所有正arm失败”的最小反例，也没有核验最终保存的六个state文件确实为zero。

必须门：

1. 把候选排序/选择抽成可独立测试的纯函数。
2. 输入三个场景均失败的candidate rows，断言选择zero、`promotion_ready=False`。
3. 对最终before/after NPZ+JSON逐个断言base、空edge、gamma0，并用重建state验证alpha0逐位等价。
4. 断言失败正arm没有被写成selected state。

### P0-5：当前runner每次都会重选arm，不能直接用于独立确认seed/receiver矩阵

项目目标要求先在开发seed锁定统一K-shot工作点和超参数，再在5个receiver×至少5个确认seed×3个场景上固定应用。当前`run()`每次都遍历4个candidate并重新选择。若在每个确认receiver/seed上分别运行，可能分别选出`c1/c2/c3`，形成确认集调参和场景外的隐式Oracle。

最小反例：

```text
development receiver选择c1
confirmation receiver A重新选择c2
confirmation receiver B重新选择c3
```

即使各次都“三场景统一”，整体确认矩阵仍没有共享同一个开发锁。

必须门：

1. 区分`development_select`与`confirmation_apply_locked`两种不可混用模式。
2. confirmation模式只能读取开发COMMIT中的单一candidate和lock SHA，禁止访问其他arm结果。
3. 每个receiver、seed、场景、K和新类规模的state/audit必须引用同一development lock SHA。

### P0-6：K1/K5/K20独立exact-K压力闭环尚未实现

当前runner只接受strict K10并在报告中明确defer K1/K5/K20。module的K1测试只证明数值不崩溃，不是独立K1性能实验；也没有K5相对K10不下降3pp的证据。

必须门：

1. K1、K5、K10、K20分别使用独立sealed exact-K package。
2. 不得从K10切片，也不得让K1/K5 runner看到K10余量。
3. 使用开发阶段锁定的candidate rebuild各K state；确认阶段不得重新选arm。
4. K5需要逐receiver、逐场景、逐类与K10比较，并执行3pp硬门。

## P1修复项

### P1-1：support provenance清单缺少class和rank

`_feature_provenance`目前只记录physical sample ID、父IQ SHA、operator、view seed和feature SHA，没有class handle、support rank和registration state。用户要求合法TX/receiver/support-query清单；当前audit难以直接证明“每个类恰好K个独立物理样本”。

建议每行追加：

```text
receiver
seed
scenario
registration_state
class_handle
support_rank_within_class
physical_sample_id
parent_received_iq_sha256
feature_sha256
```

### P1-2：跨场景检查无法识别“同一clean来源换token后叠加不同LEO状态”

当前runner能拒绝重复token和重复post-channel IQ SHA，但如果同一clean物理IQ被上游改成三个不同token，并分别生成三个不同LEO_weak观测，Phase2端仅看post-channel SHA无法识别。

该检查不能通过让Phase2读取clean ID解决。必须由Phase2边界外的package builder产生不可用于模型决策的“物理源独立性审计attestation”，seal只暴露PASS和审计哈希；D14 runner验证attestation信任根，不读取clean样本、clean hash或clean-derived特征。

### P1-3：Pareto延迟/显存口径尚不完整

当前延迟比较为：

- D14：每次包含state validation、query normalization和head score；
- qKNN：预先normalization后只做矩阵乘；
- 显存：只报告整次D14 run的CUDA峰值，没有同输入下D14与identity qKNN的分别峰值。

因此当前结果只能标记为support-row head microbenchmark，不能作为最终星上Pareto结论。正式报告需采用同一输入、同一warm-up、同一同步、同一argmax/top-k口径，并分别测量D14和identity qKNN的CPU RSS、CUDA allocated/reserved、延迟分布和state总序列化字节。

### P1-4：报告表展示raw array state而非实际序列化总state

audit中已有`serialized_state_total_bytes`，但`report.md`主表展示的是`persistent_array_state_bytes`。最终星上资源表应以NPZ+JSON+必要loader metadata的总字节为主，raw array只作为分解项。

### P1-5：输出目录不是原子提交

runner先创建最终output目录，再逐个写只读文件；中途失败会留下无COMMIT的半成品。虽然不能形成合法成功声明，但容易被后续工具误扫。

建议写入同级临时目录，完成全部readback和COMMIT后原子rename；消费者必须要求外部expected COMMIT SHA且拒绝无COMMIT目录。

## 双向new修正的必须性能门

After旧score逐位冻结并不自动等于旧类不遗忘，因为new score仍可能越过旧score。

最小旧类侵入反例：

```text
Before old rival=0.60,new base=0.59
Fisher evidence=+1,gamma_new=0.05
After new=0.64，旧样本从old翻为new
```

最小新类压制反例：

```text
Before old rival=0.59,new base=0.60
Fisher evidence=-1,gamma_new=0.05
After new=0.55，新样本从new翻为old
```

因此任何正arm必须在同一candidate、同一三场景锁下同时满足：

1. Before每个旧类不低于alpha0；
2. After每个旧类不低于D14 Before和alpha0；
3. After每个新类不低于alpha0；
4. old forgetting≤0；
5. old floor、new floor、`H_old_new`和joint均不退化；
6. 任一场景失败则整arm失败；
7. 全部正arm失败则真实zero；
8. support选择完成并提交COMMIT前保持`NO_QUERY_OPEN`。

当前runner已实现大部分指标布尔门，但在P0-1至P0-4关闭前，这些指标不得提升控制状态。

## 最终门禁清单

|阶段|判定|
|---|---|
|module单元测试和纯合成红队|GO|
|当前runner读取真实sealed enrollment包|NO-GO，先修复外部seal信任根和pre-open控制状态冲突|
|把当前输出标为support candidate promotion ready|NO-GO|
|打开任何candidate-bound query|NO-GO|
|进入125确认矩阵|NO-GO|
|宣称可部署state|NO-GO，先完成真实反序列化和预测round-trip|
|宣称K1/K5/K20稳定性|NO-GO，等待独立exact-K package|

修复优先级：

1. 外部expected seal/COMMIT信任根；
2. 当前协议pre-open/runtime access audit PASS与promotion硬绑定；
3. sealed state loader和磁盘预测round-trip；
4. runner级真实zero失败闭环；
5. development lock与confirmation apply-only分离；
6. 独立K1/K5/K20 package压力；
7. 完整support清单和统一Pareto口径。

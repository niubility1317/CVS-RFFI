# D9 floor-focused稀疏operator融合追踪

日期：2026-07-17

范围：基于D7a的三个固定received-IQ operator，以support-only物理样本删除验证选择每类至多两个operator的稀疏凸权重，替代每类单operator硬选择。本文不读取或使用任何既有query、clean或source artifact，也不声明真实性能。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D9-01|`项目.md`§7.1、§7.1.1|只处理已经密封的单一LEO_weak received IQ；不读取clean/source，不生成额外LEO状态，计算view不增加K|`code/cvsrffi/stage2_floor_sparse_operator_fusion.py`|verified|接口与resource audit单测通过|只接收received IQ/operator features、support标签、物理ID和父IQ哈希|
|D9-02|任务要求|固定候选为单operator及两operator稀疏凸融合，每类最多2个operator|同上|verified|候选集合、权重和active component单测通过|候选权重只含1、0.75/0.25、0.5/0.5及反向0.25/0.75|
|D9-03|任务要求|只用support leave-two稳健选择，优先floor类`20-19/1-18`|同上|verified|selection trace单测通过|K10执行leave-two|
|D9-04|任务要求|设置总体与最差类非退化门|同上|verified|总体、floor和逐类baseline/final单测通过|组合失败时全局回退到每类base单view|
|D9-05|Stage2-C注册边界|After旧类权重、prototype、calibration bitwise锁定，只追加新类；防止新类侵入旧类floor|同上|verified|prefix tensor/calibration与旧类逐类门单测通过|注册输入保留全部旧类support及新类support|
|D9-06|`项目.md`§7.2|Query逐样本面对全部注册类，无label/role/quota/global assignment/query fit|同上|verified|公开API签名、batch局部性和samplewise seal单测通过|向量化batch不改变单样本结果|
|D9-07|任务资源约束|0参数、0epoch、最多3个去重backbone前向、无dense query图、状态不超过256KB|同上|verified|resource audit与forward计数单测通过|按实际NumPy dtype计状态|
|D9-08|证据边界|代码、测试、追踪完成；不得用已评分query声称D9性能|本文、测试|verified|反向审计完成|未读取或声明既有query性能|
|D9-09|独立审计阻断|每个operator feature逐样本绑定`parent_received_iq_sha256/operator_id/view_seed`|代码、测试|verified|缺失/错位provenance fail-closed单测通过|逐row验证|
|D9-10|独立审计阻断|Query extractor必须携带不可变samplewise sealed契约；拒绝普通或batch-dependent callback|代码、测试|verified|plain及batch-dependent callback反例通过|seal创建时比较batch与逐样本输出|
|D9-11|K-shot统一工作点|K10锁定operator/weight后，K1/K5只能用嵌套support重建prototype，不得重选候选或calibration|代码、测试|verified|K1/K5前缀lineage及bitwise锁单测通过|K1/K5不执行删除选择|
|D9-12|独立审计阻断|资源必须从真实base状态/MAC累加，合计超限fail closed|代码、测试|verified|base+head精确计数与超限反例通过|合计状态超过256KB即阻断|
|D9-13|独立审计阻断|联合门要求总体和每个类均非退化；After保存并验证完整旧support lineage|代码、测试|verified|逐类门与old lineage错位反例通过|最差类门不替代每类门|
|D9-14|真实support-only集成|只打开20-1/713101/K10/new5 before/after enrollment-only包并生成不可变state/audit/COMMIT|`code/scripts/run_d9_support_only_enrollment.py`、真实artifact|verified|seal、成员allowlist、COMMIT哈希与只读权限复核通过|未打开query/truth/prediction/score/scorer|
|D9-15|D8b复用|runner的package root/seal/output/device均为参数，不硬编码当前row|同上、`tests/test_run_d9_support_only_enrollment.py`|verified|CLI参数与禁止参数测试通过|可直接替换为D8b enrollment package|
|D9-16|D8b strict K10门|payload中每类实际可达support必须恰为10，不能只依赖manifest `k_shot=10`后截断20-shot池|runner、runner测试、D8b artifact|verified|20-shot反例fail closed；D8b新包before 6×10、after 11×10逐scenario预检通过|旧包封存为`PROTOCOL_INVALID_KSHOT_REACHABILITY`|
|D9-17|D8b support lock|在D8b新strict K10包上只做before/after锁定与K1/K5 prototype重建证明|D8b D9 artifact|verified|COMMIT/audit/report哈希、39文件只读、12个nested证明通过|无query/prediction/scorer|
|D9-18|正式feature authority|真实runner使用的operator feature必须由不可伪造的正式authority绑定，而不能仅依赖普通mapping与自声明provenance|runner、support COMMIT|blocked|现有runner直接构造feature mapping；COMMIT未绑定feature-authority seal|机制单测不能替代正式绑定|
|D9-19|正式samplewise extractor|真实backbone callback必须证明逐样本独立，batch扩展不能改变feature|runner、support COMMIT|blocked|模块有合成`SamplewiseSealedFeatureExtractor`反例，但support runner实际调用`forward_zid160(...,batch_size=64)`且未封存callback seal|禁止candidate-bound query|
|D9-20|旧feature fingerprint|After复用旧support时必须绑定旧received IQ到正式feature fingerprint，不能只把内存中的before feature数组复制到after|runner、state audit|blocked|当前仅核对旧IQ/token/hash并复用旧feature数组，未保存可反向验证的正式feature fingerprint root|旧状态tensor锁不能替代feature authority|
|D9-21|代码身份闭包|support COMMIT必须绑定实现与runner代码SHA|support COMMIT|blocked|D9两个COMMIT均无代码SHA字段|artifact不能授权后续query|

## 预登记候选

固定operator：

- `base`
- `dc_rms`
- `dc_rms_spec15`

每类候选：

- 三个单operator：`(1.0,0.0)`
- 每个无序operator对的`(0.75,0.25)`、`(0.5,0.5)`、`(0.25,0.75)`

每类最多两个非零operator。候选集合、排序和tie-break在query打开前固定。

## 选择与回退预登记

Before：

1. 对物理support执行每类leave-two-out；若每类少于4个但至少2个，则leave-one-out。
2. 每fold只用保留support重算三个operator下的prototype和calibration。
3. 每类候选在held-out support上与全base基线比较。
4. 候选必须保证目标类准确率不低于基线；按目标类准确率、目标类最差fold准确率、总体准确率、最差真实margin、稀疏度和固定候选顺序选择。
5. 合并各类选择后要求总体准确率、最差类准确率和每个类准确率均不低于全base基线，否则整体回退全base。
6. `20-19/1-18`若存在，必须在trace中标记`floor_priority=true`；它们与其他类适用相同非退化硬门，不获得query信息或特殊类别配额。

After：

1. 旧类operator index、权重、prototype、calibration逐bit锁定。
2. 仅为新增类执行support删除选择并追加状态。
3. 新类选择同时使用旧support做intrusion guard；每个旧类support准确率不得低于全base新增类基线。
4. 合并新增类后重新核查new总体、新类floor、old总体与每个旧类support准确率；失败时只把新类回退为base，旧类状态不变。

## K10选择锁

- 正式选择入口要求各类统一K并记录完整有序`class/physical_sample_id/parent_received_iq_sha256`lineage。
- K10状态生成不可变selection lock。
- K1/K5入口必须逐类复用K10有序support的前1/5个物理ID及父IQ哈希，只重建已锁operator下的prototype。
- `operator_indices`、`weights`、`calibrations`和selection lock必须bitwise/值相等；K1/K5不得运行候选比较、门限选择或回退决策。

## Samplewise query extractor

Query API只接受`SamplewiseSealedFeatureExtractor`，其中必须包含固定extractor ID、64位契约SHA、`batch_independent=true`和`query_updates=0`。普通callback即使输出形状正确也拒绝。该契约允许一次向量化backbone前向处理一个operator batch，但禁止callback根据同batch其他query、顺序、类别数量或统计量改变单样本feature。

## 资源累加

D9状态必须保存调用方提供并验证的真实base persistent bytes和base head MAC。正式resource audit报告：

- `combined_persistent_state_bytes=base_persistent_state_bytes+d9_incremental_state_bytes`
- `combined_head_macs_per_query=base_head_macs_per_query+d9_incremental_head_macs_per_query`

合计状态超过256KB时构建或重建立即fail closed。

## 与D8隔离

D9不修改`analysis/d8_second_physical_block_traceability_20260717.md`及D8数据构建路径。D8未来可作为未评分development物理块调用D9，但本实现阶段不打开D8 query或既有已评分query。

## 验证结果

```text
python -m pytest -q tests/test_stage2_floor_sparse_operator_fusion.py tests/test_run_d9_support_only_enrollment.py
.............                                                            [100%]
13 passed

python -m py_compile code/cvsrffi/stage2_floor_sparse_operator_fusion.py code/scripts/run_d9_support_only_enrollment.py
PASS
```

反向审计：17项verified，0项deferred，0项rejected，4项blocked。D9稀疏operator选择、逐类非退化门、K10锁与K1/K5 prototype-only机制仍为已验证的support侧证据；正式feature authority、真实batch64 callback的samplewise seal、旧feature fingerprint和代码SHA闭包均未完成。因此现有D9 artifact只能保留为support diagnostic，不能授权candidate-bound query或性能声明。

附带联合回归观察：共享工作树中的D7a实现已被另一并行修复改为新签名，但旧D7a测试仍调用旧`physical_sample_ids`接口，产生5项`TypeError`。D9自身测试不受影响，本任务未修改D7a。

## 真实sealed enrollment support-only结果

成功artifact：

`E:\type10-7\automation_reports\CV-SincNet\d4a_single_observation_smoke_20260717_010128\dev_k10_new5_r2\d9_support_only_v2`

COMMIT SHA256：

`dcc2337e358c42860fb287aee159938df08b6dc0413814cfe94182b2c9cbfeac`

状态：

`SUPPORT_ONLY_D9_LOCKED_NO_QUERY_OPEN`

|场景|状态|类数|support overall baseline→final|support floor baseline→final|状态|去重operator|
|---|---|---:|---:|---:|---:|---|
|`leo_clear_weak`|before|6|0.8500→0.8500|0.6000→0.6000|26,316B|base|
|`leo_clear_weak`|after|11|0.6800→0.7000|0.2000→0.3000|48,226B|3种|
|`leo_low_elev_weak`|before|6|0.7833→0.7833|0.6000→0.6000|26,316B|base|
|`leo_low_elev_weak`|after|11|0.6400→0.7000|0.1000→0.4000|48,226B|3种|
|`leo_rain_weak`|before|6|0.8500→0.8833|0.7000→0.7000|26,316B|3种|
|`leo_rain_weak`|after|11|0.5200→0.5800|0.2000→0.2000|48,226B|3种|

这些数值仅是注册support删除验证和floor门，不是query性能。三个场景before/after各保存一个K10状态，并保存K1/K5共12个prototype-only重建状态；全部证明operator index、weight、calibration和selection lock未重选。

逐类最终组件共53行：

- base单view：33行
- spec15单view：9行
- base 0.75 + spec15 0.25：4行
- base 0.25 + DC/RMS 0.75：3行
- DC/RMS单view：2行

包内只有opaque class handle，没有真实TX标签。runner为保持只读sealed enrollment边界，没有访问外部cache反查`20-19/1-18`，因此只输出全部opaque类的候选/floor证据，未作重点TX定向声明。

第一次执行目录`d9_support_only_v1`在After K1重建的append-only registry顺序检查中被阻断；该目录没有`COMMIT.json`、没有`support_audit.json`，明确不是可用artifact。修复后使用全新`v2`目录成功，未覆盖或删除失败证据。

## D8b strict K10-only复用结果

旧D8b包虽然manifest写`K=10`，但before/after每类实际可达20条，已被数据构建方封存为：

`d8b_k10_new5_enrollment_PROTOCOL_INVALID_KSHOT_REACHABILITY`

D9 runner已增加硬门：读取support class/rank元数据后，要求每类物理行数恰为10且rank集合严格为`0..9`；不再允许从20-shot可达池内部截取前10条。

新严格包预检：

- before：每scenario 60=`6×10`
- after：每scenario 110=`11×10`
- 三scenario每类均恰10，rank仅`0..9`
- detached seal均存在

成功D9 artifact：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d9_support_only_strict_k10_v1`

COMMIT SHA256：

`c2103429117d3a45574b26edad461f046f66e5a94e35c41df8a0341ecd5e127d`

|场景|状态|support overall baseline→final|support floor baseline→final|状态|operator|
|---|---|---:|---:|---:|---|
|clear|before|0.7167→0.7167|0.1000→0.1000|26,316B|base|
|clear|after|0.4600→0.5000|0.2000→0.3000|48,226B|3种|
|low|before|0.7333→0.7667|0.3000→0.3000|26,316B|3种|
|low|after|0.5000→0.5400|0.1000→0.1000|48,226B|3种|
|rain|before|0.7333→0.7333|0.3000→0.3000|26,316B|base|
|rain|after|0.5800→0.5800|0.4000→0.4000|48,226B|3种|

这些仍然只是support删除验证。floor为0.1–0.4，D9在D8b support侧同样不具备晋升依据。输出共39个只读文件；COMMIT引用的audit/report哈希重算一致；12个K1/K5证明全部保持operator、weight、calibration和K10 lineage lock。`query_package_opened/query_truth_opened/query_prediction_opened/query_score_opened/scorer_opened`全部为false。

## 正式绑定NO-GO

D9两个真实support artifact均已新增只读：

`SUPPORT_PROTOCOL_BINDING_INCOMPLETE_NOT_SELECTED.json`

位置：

- `E:\type10-7\automation_reports\CV-SincNet\d4a_single_observation_smoke_20260717_010128\dev_k10_new5_r2\d9_support_only_v2`
- `E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d9_support_only_strict_k10_v1`

marker明确`candidate_bound_query_generation_authorized=false`、`candidate_bound_query_package_created=false`，且所有query/truth/prediction/score/scorer标志为false。既有D8b D9结构query包不得反向解释为D9已获选择授权。

# qKNNv4.2正式Stage2-B/C极轻路线追踪

状态：`ACTIVE_ROUTE_EXPLORATION_LOCAL_PROTOCOL_REPAIR_REQUIRED`。

本工作严格使用`ADV3B02_CORE90_SOFT_E200`作为基座。Stage2-B旧类域适应和Stage2-C真实seen-new注册同等重要；正式结果必须同时给出注册前和注册后。Phase2的support、query及所有适配/注册/评估信号必须在Phase2边界前已叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`，clean样本和任何clean派生信号必须物理不可达。

## 正式实现追踪账本

本表按`项目.md`第7.1、7.2、9.2、9.3、10.3节及`项目实验.md`第13–16节执行。只有`verified`项可作为正式已落地证据；其余项不得支撑launch或性能声明。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|T01|`项目.md`7.1|Phase2包根与NPZ在IQ materialization前执行exact allowlist、解压边界、CRC和无额外成员检查|`code/cvsrffi/stage2_predictor_bundle.py`;`tests/test_stage2_predictor_bundle.py`|verified|关联测试43项PASS|通用sealed package P0已加固；不等于SOMP-H专用包已完成|
|T02|`项目.md`7.1、AGENTS.md|实际OS账本拒绝额外execve及未登记sealed package成员，并检查必需成员确实打开|`code/cvsrffi/phase2_isolated_runner.py`;`tests/test_phase2_isolated_runner.py`|verified|runner+sealed integration共12项PASS|exact member检查为opt-in；SOMP-H调用方必须强制传入|
|T03|`项目.md`7.1|sample-level LEO overlay provenance、scenario、satellite seed和post-channel IQ SHA在打开IQ payload前进入独立trust root|`somph_leo_weak_lineage_seal.py`;`somph_lineage_authority.py`;离线签名CLI|authority_chain_implemented_official_cache_built|固定Ed25519公钥验证预签authority lock；逐字段绑定cache-set、dataset、exporter、build spec、channel closure、physical/IQ/overlay root；bundle/COMMIT强制提交builder receipt与official cache-spec manifest；扩大回归296项PASS|官方locked spec生成的首个N607 maxK20 cache已完成且exact40 PASS；真实lock/签名/bundle仍待生成，`formal_launch_authority=false`|
|T04|`项目.md`7.1、7.2|support-only enrollment与query-only apply物理隔离，两个进程均不挂载truth/scorer；apply不挂载support|`somph_predictor_entry.py`;`somph_isolated_runner.py`;profile-specific closure与固定CLI|implemented|外部pre-run binding/runtime SHA、固定system-root allowlist、非注入式生产subprocess、exact output/resource receipt、额外execve与apply打开support负测试PASS；相关扩大回归237项PASS|仅Windows伪strace；必须在Linux/N607真实bwrap+strace后才能升为`verified`|
|T05|`项目.md`7.2、9.3|注册前/后采用物理独立统一class registry包；predictor内部无old/new边界，scorer侧才配对角色与physical ID|`somph_offline_target_package.py`;`somph_predictor_bundle.py`;`somph_metric_scorer.py`|implemented_real_input_pending|formal builder固定消费外部expected authority COMMIT；同一max-new20 cache派生new5/10/20嵌套registry；before/after旧support/query物理摘要匹配；query按无role/class信息的HMAC全局置换|真实head/apply package、prediction和scorer artifact尚未从N607真实cache产生|
|T06|`项目.md`7.1、`项目实验.md`13|钉死正式ADV3B02 checkpoint SHA并强制`adv3b02_z_id160_fp32`输出|`somph_predictor_runtime.py`;`somph_predictor_bundle.py`;`somph_predictor_entry.py`;profile-specific closure|implemented|固定entry只调用`load_torchscript_backbone_same_fd`读取sealed checkpoint descriptor；32维和FP16输出拒绝；method lock canonical SHA；enrollment/apply closure不含generic入口或对侧SOMP-H脚本|真实ADV3B02 checkpoint尚未在N607隔离进程执行，公开内核API不能单独作为正式证据|
|T07|`项目.md`7.2|每个query独立面对当前统一registry，禁止truth/role/Q20/quota/order/global assignment/dense graph|SOMP-H runtime、single-stream artifact、专用bundle/request schema|implemented|apply batch固定1并拒绝非singleton；query exact schema、registry replay、shuffle/singleton、cross-scenario token set与support token→class/rank映射检查PASS|真实OS closure与逐样本open/execve trace尚缺|
|T08|`项目.md`7.2|先冻结不可变单stream prediction artifact，再由隔离scorer打开truth并输出前后指标|`somph_prediction_artifact.py`;`somph_metric_scorer.py`;`somph_head_artifact.py`;专用runner|implemented|prediction artifact v2移除`new_class_count`和post-run resource SHA；head/prediction均0444、atomic noreplace；scorer侧才持有new-count/truth|真实scorer OS ledger与evidence重算待补；receipt固定`LOCAL_PROTOCOL_REPAIR_REQUIRED`|
|T09|`项目.md`10.3、`项目实验.md`13|从实际head/runtime重算状态、MAC、enrollment/query时延、峰值显存/内存和backbone forward|专用runtime receipt与resource auditor|implemented|26类实际capsule重算state=76320B、active=25440B、head MAC/query=13142；每query forward count=1|合成CPU时延不可作为正式Pareto；真实隔离进程峰值显存/内存/时延仍待测|
|T10|`项目.md`10.3、`项目实验.md`14–16|开发K10锁定后完成5receiver×至少5确认seed×3场景×new5/10/20及K1/5/10/20、逐类/逐receiver/日志/Pareto|`somph_formal_matrix.py`;`somph_cache_build_matrix.py`;matrix、runner、自动化报告|schema_and_cache_plan_verified_data_blocked|结构模板固定120个Stage2-B行、360个Stage2-C pair、确认seed713102–713106；另生成30个5receiver×seed713101–713106的maxK20+Q20 cache spec；既有`20-1/713101`development cache已在N607通过三场景exact40 gate|真实data-bound pair数仍为0；其余29个receiver/seed cache cell未构建，数据补齐前不启动Phase2正式矩阵|
|T11|`项目.md`7.1、AGENTS.md|从真实cache/build-spec/exporter/channel/dataset bytes构造不可覆盖、可离线签名的authority lock，不接受独立TX/dataset声明或query/truth/quota参数|`somph_authority_lock_builder.py`;固定CLI；authority consumer；builder测试|official_cache_ready_unsigned_lock_pending|生产wrapper钉死30-cell manifest SHA并按cell解析spec；builder与registered bundle consumer均独立重建固定channel config、role seed公式、exact40和physical count；bundle强制提交并复核builder receipt与official manifest/cell/spec；扩大296项PASS|正式`rx_20_1_seed_713101`cache已由locked spec生成并通过exact40；下一步直接构建unsigned lock、离线签名与bundle；私钥不得同步|

## 当前证据边界

- JG_R8_LR020的25行K10 Stage2-B矩阵只有旧类，old accuracy均值78.8222%，不能代替Stage2-C。
- JG_R8_LR020单development Stage2-C在new5/10/20下注册后old为57.78%/52.78%/50.83%，seen-new为61.00%/37.33%/20.67%，遗忘19.44–26.39pp，属于合法负证据。
- CSIL、MoPC-HR、Orthogonal Incremental三种对照的完整严格矩阵均未达到目标；matched Stage2-C MRIOR仍缺失。
- 当前JG锁硬编码单receiver、单seed和K10；OS级访问账本、资源capsule及正式Pareto闭环尚未完成。

## 追踪表

|ID|要求|状态|证据/下一步|
|---|---|---|---|
|R01|Phase2`LEO_weak-only`且clean/clean-derived物理不可达|signed_authority_implemented_real_recompute_required|lineage链已加入固定公钥签名authority、真实dataset/build-spec/exporter/channel闭包和逐样本root复核；缓存计划固定只输出预叠加三种`leo_*_weak`的Phase1离线cache；真实N607 authority bundle与Linux OS ledger尚未接入|
|R02|逐样本全部注册类、无角色/真实批次数/quota/global assignment|runtime_entry_implemented_real_os_pending|apply request无batch/new-count/query-count字段，entry固定singleton backbone＋singleton head并面对统一registry；正式证据仍须真实N607 trace|
|R03|不可变prediction与independent scorer隔离|artifact_v2_predictor_runner_implemented_scorer_os_pending|`.cvspred`无future `new_class_count`或post-run resource SHA；head/prediction只读atomic noreplace；entry resource receipt和输出exact allowlist已后验绑定，scorer独立OS ledger尚缺|
|R04|Stage2-B注册前与Stage2-C注册后matched pair|authority_bound_package_producer_real_run_pending|offline producer已生成物理独立before/after enrollment package和apply staging；formal finalizer重新验证authority COMMIT并绑定本row enrollment root/seal、lineage、cache、registry；真实N607 payload尚未产生|
|R05|真实嵌套5/10/20 seen-new TX覆盖|one_official_max20_cache_built_29_cells_pending|new20合法顺序固定，new5/new10为同一authority/cache的嵌套前缀；首个官方`20-1/713101`maxK20+Q20 cache已由locked spec构建并通过exact40，其余29个cell待建|
|R06|K10开发锁定、K5独立matched确认|implemented_not_integrated|method lock采用canonical JSON唯一SHA；package K绑定独立于开发K10锁；正式项目另要求K1/K20遗忘压力|
|R07|多receiver、多seed、多场景确认|cache_specs_locked_data_pending|exact matrix template固定development=`20-1/713101/K10`、confirmation=`713102–713106`；30个receiver/seed cache spec、90个全局唯一卫星seed和固定N607离线根已生成；全部840状态行仍是UNBOUND模板|
|R08|K10 old>=92%、旧类floor>=88%|pending|真实LEO矩阵未运行|
|R09|K10 seen-new 5/10/20>=92/90/86%|pending|真实LEO矩阵未运行|
|R10|K5较K10下降<=3pp|schema_implemented_data_pending|`validate_k_family`已强制K1/K5/K10/K20共享K20 pool、query、runtime和场景；真实密封包与正式评分尚缺|
|R11|注册后旧类遗忘控制|pending|SOMP-H专门针对prototype拥挤，真实性能待验证|
|R12|adapter<=50k、<=20epoch、无dense query图|runtime_resource_recompute_local_only|纯SOMP-H为0参数/0epoch/0step；26类三场景从实际capsule重算state=76320B、active-scenario state=25440B、head MAC/query=13142；真实隔离进程时延/显存/内存receipt仍缺|
|R13|identity-only及三种方法Pareto|pending|baseline改为独立artifact；ProtoNet 0参数/0step只能做零维不劣＋性能/MAC/状态/时延/显存Pareto|
|R14|完整日志或闭式求解诊断|structural_logging_real_run_pending|闭式support-only单元审计可重算head张量、state与MAC；不可变head artifact、canonical execution receipt、mean/p95/max singleton latency字段已实现；正式时延、峰值内存、包成员和OS访问日志待真实隔离运行|
|R15|合法TX/receiver/support-query清单|one_official_registered_cell_verified_29_cells_pending|旧6TX、新20TX嵌套顺序和5receiver已固定；官方locked spec生成的`20-1/713101`cache在N607通过三场景exact40 gate：每场景1040行、26个单元、每单元严格40条，physical sample root跨场景一致；其余29个receiver/seed cell尚未构建|
|R16|自动化报告和Git提交|current_authority_commit_pending|根目录报告、locked formal matrix、30-cell cache spec和首个官方cache结果已更新；authority、signer、bundle consumer、offline producer及相关扩大回归296项PASS；既有提交=`5d5e0ed`、`07e8ddb`、`0952371`，本轮authority修复待提交|
|R17|每3个turn回顾目标和对话|implemented|已完成本轮三轮回顾：拒绝clean cache、query侧Q20/ordering、结构JSON冒充真实证据及注册后切片模拟注册前|
|R18|外部authority不可由同一Phase2调用者自签|implemented_real_signing_pending|生产验签使用函数体literal issuer/key-id/public key；离线私钥仅存根报告offline controller且ACL限制，签名CLI固定OpenSSL路径/SHA并返回receipt SHA；真实cache authority lock尚未签名|

## 本轮authority与真实缓存计划里程碑

- 正式矩阵artifact固定为`somph_formal_matrix_locked.json`，文件SHA256=`06f86484b2443e7198c67aac13083cf90efef4b043eee2e15b85e7fe55625f4e`，内部matrix SHA256=`8ccddb7f5b78624ecd2c6d4a62dd86b008e9e11f2d47c6f701c7c58062ff08e8`。它仍是`UNBOUND_REQUIREMENT_TEMPLATE`，不是性能证据。
- 30-cell cache spec manifest固定5receiver×seed`713101–713106`，每cell为旧6+新20、每类maxK20 support pool+Q20 query、三种`leo_*_weak`，manifest文件SHA256=`0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5`。
- N607缓存根固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/somph_stage2bc_leo_weak_cache_20260716`；post-build gate未通过前状态保持`LOCAL_PROTOCOL_REPAIR_REQUIRED`。
- N607只读preflight于2026-07-16通过：8张RTX 3090可见、无活动训练进程、项目根和ManySig/ManyTx可见、约7.6TB可用空间。该检查没有启动或修改远端作业。
- 既有`20-1/713101`development cache已通过N607只读exact40 coverage gate：三种`leo_*_weak`场景各1040行、26个role/TX/receiver单元、每单元严格40条，physical sample root=`cc9ccdecf256b7d6ece705d193723ef8d6c54dabce800732feae00e178ca143a`。这只覆盖30-cell计划中的1个cell，且审计明确`formal_launch_authority=false`。
- 本轮验证集合共279项PASS；formal matrix已去除对完整predictor bundle的非必要导入，Phase1离线coverage gate只依赖轻量协议常量。该结果只证明本地代码、schema、fail-closed边界和合成fixture闭合，不代表target-old/seen-new门槛达标，不授权Phase2正式launch。

## SOMP-H首条路线

文件：`paper_reproduction/cvs_aligned/support_only_multiprototype_head.py`。

机制：全注册类support-only对角类内残差白化、每类最多2个压缩原型、centroid混合和support几何hubness惩罚。所有query使用同一score函数并独立计算；API不接收query标签、角色、配额或query集合图。持久状态和per-query MAC直接由实际tensor shape重算。部署状态可封装为无pickle FP16 capsule，使用精确成员集合和schema；任何额外query真值成员都会被加载器拒绝。

验证：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' -m py_compile paper_reproduction/cvs_aligned/support_only_multiprototype_head.py tests/test_support_only_multiprototype_head.py
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' -m pytest -p no:cacheprovider -q tests/test_support_only_multiprototype_head.py tests/test_jg020_stage2c_isolation.py tests/test_adv3b02_ci_heads.py
git diff --check -- paper_reproduction/cvs_aligned/support_only_multiprototype_head.py tests/test_support_only_multiprototype_head.py
```

结果：核心与协议相关测试合计41项PASS；只有既有TorchScript弃用/trace警告。`screen_support_only_multiprototype_head.py`会在feature tensor加载前拒绝任何clean共存cache；现有legacy 20-new feature NPZ已被该边界排除。SOMP-H已采用独立before/after单stream capsule、纯密封ADV3B02 z_id160、K20 prefix、support feature/head payload、opaque token摘要、注册pair与K-family绑定及FP16 flight-state推理。该结果仅证明机制、序列化和fail-closed接口可用，不是Stage2-B/C性能成功，也不授予N607正式启动权限。

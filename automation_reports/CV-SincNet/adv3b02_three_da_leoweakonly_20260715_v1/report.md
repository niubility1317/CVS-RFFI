# ADV3B02三种域适应方法LEO_weak-only重跑报告

## 一、实验登记

|字段|内容|
|---|---|
|实验ID|adv3b02_three_da_leoweakonly_20260715_v1|
|登记时间|2026-07-15（Asia/Hong_Kong）|
|执行者|Codex|
|阶段|Stage2-B target-old few-shot domain adaptation|
|目标|按最新项目协议重跑ProtoNet CDA、MRIOR-SDA、DADDA-SDA；每方法5个target receiver×5个seed×5个K，共125次，合计375次正式方法运行|
|历史比较|替代2026-07-14批次中Phase2直接读取raw/clean IQ并在runner内部生成LEO视图的375行历史artifact|
|当前状态|ARTIFACT_COMPLETE；375/375正式行完成，失败0，全量协议与artifact审计PASS|

## 二、假设与声明边界

假设：将source和target-old样本在Phase1/offline边界预先叠加`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`并封存为带逐样本provenance的post-channel IQ缓存后，三种方法仍能在不接触clean样本、clean派生feature/logit/prototype和query Oracle的条件下完成同协议Stage2-B比较。

声明边界：

- 仅回答六个旧类的target-domain LEO_weak-only适应，不报告target-new、seen-new或unknown/open-set性能。
- 三种方法均为共享ADV3B02的CVS extension，不是原论文架构的严格复现。
- ProtoNet CDA冻结ADV3B02；MRIOR-SDA和DADDA-SDA更新ADV3B02 identity backbone，属于非轻型full-backbone比较，不得据此声明星上轻量部署成功。
- 历史clean-access结果统一保留为`PROTOCOL_INVALID_FOR_PHASE2`，不得与本次结果合并排名。

## 三、实验矩阵与数据协议

|项目|设置|
|---|---|
|方法|ProtoNet CDA、MRIOR-SDA、DADDA-SDA|
|target receiver|`20-1`、`3-19`、`7-14`、`7-7`、`8-8`|
|seed|`713101`、`713102`、`713103`、`713104`、`713105`|
|K|`1,2,5,10,20`，每个receiver×seed共用一个target cache；support为前K个物理sample，query固定为预留20-shot support池之后的20个样本/类|
|旧类|`14-10`、`14-7`、`20-15`、`20-19`、`6-15`、`8-20`|
|source receiver|`1-1`、`1-19`、`14-7`、`18-2`、`19-2`、`2-1`、`2-19`|
|场景|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|Phase1/offline准备|26个LEO_weak cache任务+25个无真值predictor bundle任务+1个运行时密封任务，共52个；只有前26个任务可读取ManySig|
|Phase2正式运行|每方法125行，共375行；Phase2配置不包含任何pkl/dataset路径|
|query决策|逐样本面对全部六个注册旧类；无角色Oracle、无类别数量、无类别配额、无全局批量分配|
|预测/评分隔离|predictor仅接收opaque query ID、LEO_weak query IQ、注册support和类别表；预测artifact固化并校验SHA256后，独立scorer才连接truth sidecar|
|OS隔离|N607内核6.8、Landlock ABI 4；predictor使用`no_new_privs`+Landlock文件白名单，另由`strace`生成实际文件访问账本|

每个launchable row必须记录：

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

## 四、方法设置

|方法|输入与目标|参数更新|资源/声明|
|---|---|---|---|
|ProtoNet CDA|target-old LEO support的ADV3B02 `z_id160`类均值原型；欧氏最近原型分类|0步梯度更新|冻结backbone对比|
|MRIOR-SDA|同场景sealed source LEO+target-old LEO support；source CE+target-support CE+DV-KL；估计器每步7次上升|200步/场景|更新ADV3B02 identity backbone，非轻型比较|
|DADDA-SDA|同场景sealed source LEO+target-old LEO support；source CE+target-support CE+MMD+LMMD+动态alpha|200步/场景|更新ADV3B02 identity backbone，非轻型比较|

## 五、本地变更与验证

### 5.1代码/config/script

|文件|用途|
|---|---|
|`code/scripts/build_cvs_leo_weak_iq_cache.py`|新增`stage2_target_old`封存缓存scope|
|`paper_reproduction/cvs_aligned/adv3b02_supervised_da_runner.py`|移除pkl、raw IQ和运行时信道叠加入口；source仅加载LEO cache，target仅加载v2 truth-free predictor package内的注册support与query|
|`code/cvsrffi/stage2_predictor_bundle.py`|同一文件描述符完成hash、NPZ成员白名单审计与IQ读取；禁止query真值/角色/配额字段|
|`code/scripts/build_cvs_stage2_predictor_bundle.py`|Phase2外生成opaque query ID、无真值predictor bundle和独立truth sidecar|
|`code/scripts/build_phase2_runtime_seal.py`|生成密封包root digest、runtime code digest、成员白名单和pre-run隔离证据|
|`code/scripts/run_phase2_landlock_isolated.py`|使用Landlock ABI和`no_new_privs`限制predictor文件访问|
|`paper_reproduction/scripts/score_adv3b02_three_da_predictions.py`|预测进程退出后独立连接truth并生成old_acc、逐类和逐receiver统计|
|`paper_reproduction/cvs_aligned/supervised_da.py`|fail-closed校验clean可达性与query Oracle guard|
|`paper_reproduction/scripts/build_adv3b02_three_da_leo_weak_plan.py`|生成26个offline缓存任务和375个Phase2任务|
|`paper_reproduction/scripts/run_adv3b02_three_da_cache_plan.py`|仅执行并验证offline缓存准备，明确`phase2_started=false`|
|`paper_reproduction/scripts/run_cvs_publication_matrix.py`|支持显式GPU设备并在artifact审计中检查新协议字段|
|`paper_reproduction/scripts/summarize_adv3b02_three_da.py`|只接受v2 sealed-cache结果并检查完整协议|
|`paper_reproduction/configs/adv3b02_stage2b_three_da_leo_weak_only_20260715_n607.json`|不含dataset路径的Phase2基准配置|

### 5.2本地验证结果

|验证|结果|
|---|---|
|`python -m py_compile ...`|PASS|
|相关pytest 57项|`57 passed`；包含v2 predictor package的相对路径、detached seal、同fd审计、篡改和symlink拒绝测试|
|计划生成|PASS；26个cache build+25个predictor bundle build+1个runtime seal，375个formal method rows|
|矩阵dry-run|PASS；首行为`protonet_cda_stage2b_rx20-1_k1_seed713101`，末行为`dadda_sda_stage2b_rx8-8_k20_seed713105`|
|N607隔离能力只读探测|内核`6.8.0-117-generic`、Landlock ABI 4；`bwrap`因user namespace权限失败、Docker无daemon权限，因此采用Landlock等价隔离|

本地生成计划：

`E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\adv3b02_three_da_leoweakonly_20260715_v1_plan\plan_manifest.json`

## 六、N607同步与启动记录

直接SSH预检已PASS，8张RTX 3090当时均空闲。第三次offline准备已完成26/26个LEO_weak cache，随后因预测器打包器升级后的CLI/layout与旧plan不一致而在第1个bundle前停止；未生成predictor package、runtime seal或正式Phase2行。已在本地commit `98ea663`完成三目录物理隔离接口迁移并通过33项相关测试，待重新同步后复用26个已验证cache继续准备。

|字段|内容|
|---|---|
|本地Conda环境|`ssr-gpu`|
|远端工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端Python/Conda环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1`|
|远端log root|`.../stage2_logs/`|
|GPU/PID|正式分片GPU0–7分别为PID `1630806`、`1630865`、`1630924`、`1631084`、`1631232`、`1631381`、`1631553`、`1631676`|
|代码版本|正式启动时Git HEAD `d3fa2d4`；最终runtime代码闭包commit `51be8e2`，本地编译、相关pytest与375行matrix dry-run均PASS|
|offline准备命令|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_adv3b02_three_da_cache_plan.py --plan-manifest runs/adv3b02_three_da_leoweakonly_20260715_v1/plan/plan_manifest.json --execute`|
|正式worker命令|由`plan_manifest.json.commands.phase2_workers`给出8个shard；52/52离线准备及三方法smoke通过后已于2026-07-15 17:24 CST启动|
|期望输出|每行`metrics.json`、`split_manifest.json`、`resolved_config.json`、`score_table.csv`、详细分组统计和完整loss trace|

下一次同步映射：commit `98ea663`中的`code/scripts/{build_cvs_stage2_predictor_bundle,build_phase2_runtime_seal}.py`、ADV3B02 runner、独立scorer、plan/cache-plan脚本和配置同步至N607同相对路径；本地`local_artifacts/adv3b02_three_da_leoweakonly_20260715_v1_plan/{phase2_config.json,plan_manifest.json,cache_specs/,package_artifacts/}`同步至远端run root的`plan/`。既有26个cache不覆盖、不删除。

计划SCP命令均使用`scp -F E:\type10-7\tools\n607_ssh_config`：按`code/scripts/`、`paper_reproduction/cvs_aligned/`、`paper_reproduction/scripts/`、`paper_reproduction/configs/`四个目标目录分别同步上述文件，再将`phase2_config.json`、`plan_manifest.json`、`cache_specs/`和`package_artifacts/`同步至`N607:/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/plan/`。同步前只读复核为相关进程0、8张GPU空闲、cache manifest 26、predictor manifest 0、磁盘余量7.6TB。

启动诊断记录：初次后台命令因远端根目录旧`cvsrffi`包遮蔽`code/cvsrffi`而退出；commit `e325503`已修复`sys.path`优先级。第二次因offline spec误写`/CV-SincNet/ManySig.pkl`而退出；服务器实际路径为`/CV-SincNet/Dataset_WigSig/ManySig.pkl`，已修正。第三次PID `1604043`完成26/26缓存后，旧plan在第1个bundle调用已升级打包器时因CLI不兼容退出。三次均未启动正式Phase2行；第三次产生的26个LEO_weak cache保留并将在续跑时逐项hash复核后跳过。

第四次offline准备PID `1615166`在首个bundle完成密封后，由末尾字符串泄漏扫描误把固定Phase1 checkpoint内部合法的旧类标签映射识别为query truth泄漏而fail-closed退出；首个predictor manifest、scoring manifest和detached seal各1个已落盘，正式行仍为0。修复只豁免已预注册且受hash密封的`checkpoint.bin`、`adapter.bin`和`head.bin`无结构字节扫描；新生成support/query/package manifest仍执行成员白名单、同fd hash审计和泄漏扫描。修复后相关pytest仍为`33 passed`，首个已密封bundle不删除，由续跑时严格verifier决定复用或阻断。

第五次offline准备PID `1616795`已完成并正常退出：`expected=52`、`verified=52`、`cache_verified=26`、`predictor_bundle_verified=25`、`runtime_seal_verified=true`、`phase2_started=false`。其中27项严格校验后跳过复用，25项新建；全局inventory root SHA256为`8c33405898992adaa812975b2fd42c371302c88a60734cc335d5653c90a4c5c2`，runtime code SHA256为`199a16fe9bd09110e4d1793402aefbac2d5cc9a81c535d90a407eabb304c3fe2`。

下一步smoke固定`receiver=20-1`、`seed=713101`、`K=5`，三方法各1行，输出到`runs/adv3b02_three_da_leoweakonly_20260715_v1/smoke_runs/`，使用与正式worker相同的Landlock launcher、artifact allowlist、runtime evidence和独立scorer；smoke成功后才允许启动8个正式shard。

第一次smoke PID `1617869`在首行、模型加载前退出，`completed=0`、`failed=1`。行日志为`ModuleNotFoundError: No module named 'paper_reproduction'`；访问trace证明Landlock允许具体runtime文件但缺少仓库根目录遍历权限。修复仅将两个`runtime_code_root`及其共同项目父目录加入`runtime_code_list_dirs`，权限为`LIST_DIR`而非任意文件读取；本地重新编译和33项pytest通过。失败smoke artifact保留，复测使用`smoke_runs_v2/`，并在复测前重建runtime seal/allowlist/evidence。

第二次smoke PID `1619013`已能导入`paper_reproduction`，但随后在读取`baselines/__init__.py`时被Landlock阻断，仍为`completed=0`、`failed=1`且未加载模型。`model_dual_cvsincnet`和复现辅助模块确实依赖`baselines`代码；因此将`baselines/`作为第三个`runtime_code_root`纳入逐文件SHA256、目录遍历白名单和runtime code digest，不开放任何数据目录或未登记文件。plan生成测试通过并重新生成52项/375行plan；下一次复测使用`smoke_runs_v3/`。

第三次smoke PID `1620292`已通过所有导入、包审计并进入support张量构造，但远端NumPy 2.2.5与PyTorch 2.1.0的`torch.from_numpy`桥接最小复现同样报`TypeError: expected np.ndarray (got numpy.ndarray)`。已将runner与独立scorer共6处转换改为`torch.frombuffer(memoryview(...)).reshape(...).clone()`的显式ABI安全拷贝；远端float32/int64最小复现通过，本地编译与34项相关pytest通过。数据值、样本、损失、优化步数和决策规则均未改变；同步后必须重建runtime code digest，复测使用`smoke_runs_v4/`。

第四次smoke PID `1622232`的predictor已返回0，但post-run访问审计发现整棵`code/`白名单包含历史`phase2_frozen_manytx_unknown_diagnostic.py`等未使用文件，Landlock构建规则时的`O_PATH`访问触发禁止路径，故指标未评分。已进一步移除runner对`train_ssdg`、`eval_feature_diagnosis`、`class_incremental`和`wisig_runtime`的宽导入链，以runner内最小等价函数重建严格ADV3B02并记录loss；runtime seal从3棵目录改为22个显式Python文件白名单，不包含dataset loader、ManySig/ManyTx/clean诊断或独立scorer。重生成plan后仍为52项离线准备和375行，编译与34项pytest通过；复测使用`smoke_runs_v5/`。

第五次smoke PID `1626556`在最小代码闭包下到达checkpoint反序列化，但checkpoint `args`包含`baseline_origin_sat_view.SatViewStage`，因此`torch.load(weights_only=False)`需要该类定义；`weights_only=True`最小测试确认因该类不在PyTorch 2.1内置安全集合而失败。`baseline_origin_sat_view.py`仅定义dataclass/数学/torch辅助，不导入dataset或信道数据入口，故把该单文件加入显式代码白名单；仍不允许任何dataset loader或clean/ManySig/ManyTx路径。复测使用`smoke_runs_v6/`。

第六次smoke PID `1628089`已`completed=3`、`failed=0`。最终最小runtime seal的global inventory SHA256为`463ad98e75bf81720844add010b153a1bd53096d59df52fef48d921c4a194f31`，runtime code SHA256为`8ef15765fa639647b4ed6f0cbdd23a0408acdcc0e177a789bccb6b5919de0182`；allowlist禁止路径命中为0。三行均为receiver `20-1`、seed `713101`、K=5，checkpoint严格加载均为0 missing/0 unexpected/0 shape mismatch，访问审计PASS、forbidden hit 0、prediction/scoring进程隔离PASS、每行360个评分样本。

正式运行于2026-07-15 17:24 CST启动8个GPU分片，shard 0–7分别固定GPU0–7。17:25首次健康检查显示8张GPU均有独立计算进程，已生成14/375个`metrics.json`，8个顶层分片日志均无`Traceback`、`FAILED`或`ERROR`标记。正式结果根目录为`runs/adv3b02_three_da_leoweakonly_20260715_v1/stage2_runs/`，分片事件与摘要位于同run root的`stage2_logs/`。

|smoke方法|before old_acc|after old_acc|delta|loss rows|loss有限|访问审计|判定|
|---|---:|---:|---:|---:|---|---|---|
|ProtoNet CDA|61.11%|60.00%|-1.11pp|3|是|PASS|允许正式启动|
|MRIOR-SDA|61.11%|83.06%|+21.94pp|33|是|PASS|允许正式启动|
|DADDA-SDA|61.11%|71.39%|+10.28pp|33|是|PASS|允许正式启动|

## 七、成功条件与风险

成功条件：

- 26/26缓存、25/25 predictor bundle和1/1运行时密封通过审计；
- 375/375行均有Landlock执行证明、实际文件访问账本、密封prediction artifact和独立scoring audit；
- 375/375方法artifact完整，每方法125行，每K档25行；
- checkpoint严格加载为0 missing/0 unexpected/0 shape mismatch；
- support/query零重叠，三个场景均为预叠加LEO_weak缓存；
- 所有query Oracle/类别数量/类别配额/全局分配guard均为false；
- MRIOR/DADDA的loss trace全部有限；
- 结果按同一run row报告before/after/delta，不拼接边际最大值。

主要风险：

- MRIOR/DADDA更新完整identity backbone，计算与持久状态明显高于星上极轻型路线，只能作为资源较重的比较方法。
- source输入从clean改为同场景LEO_weak anchor后，优化分布与历史375行不同；性能变化必须归因于协议修复后的新输入边界，而不是代码回归。
- 不能使用query选择早停或针对困难receiver修改步数；固定200步可能继续产生负迁移。
- 任一缓存manifest缺字段、hash不一致或包含禁止成员时必须停止，不能降级运行。

## 八、正式结果表

### 8.1完成与审计结论

正式运行于2026-07-15 17:59 CST全部结束。8个worker完成数合计375、失败0、跳过0；三方法各125行，每个方法×K档各25行。严格汇总器返回`artifact_complete=true`、`errors=[]`。375行共包含135,000条逐样本评分记录和8,625条loss trace；所有行均使用同一ADV3B02 checkpoint SHA256 `2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。

每行均通过以下硬审计：checkpoint严格加载0 missing/0 unexpected/0 shape mismatch；support/query零重叠；Phase2仅可达预叠加LEO_weak缓存；无clean样本及clean派生信号；无query角色、类别数量、类别配额和全局分配；predictor与scorer进程隔离；Landlock实际文件访问审计PASS；truth仅在不可变prediction artifact落盘并校验SHA256后由独立scorer连接。MRIOR-SDA与DADDA-SDA全部loss有限。

### 8.2总体结果

表中95%CI按125个receiver×seed×K任务行计算；胜/负表示同一run的after old_acc相对before old_acc提高/下降，未出现平局。

|方法|before old_acc|after old_acc|95%CI|平均增益|胜/负任务|平均适配时延|
|---|---:|---:|---:|---:|---:|---:|
|MRIOR-SDA|73.60%|**82.58%**|81.06%–84.09%|**+8.98pp**|105/20|17.90s|
|DADDA-SDA|73.60%|78.35%|76.79%–79.91%|+4.75pp|99/26|14.62s|
|ProtoNet CDA|73.60%|66.85%|64.49%–69.20%|−6.75pp|10/115|0.05s|

结论：在新版Phase2协议下，MRIOR-SDA是三种方法中唯一同时达到最高平均old_acc、最大平均增益和最多正迁移任务的方法；DADDA-SDA次之。ProtoNet CDA计算最轻，但原型估计在LEO_weak域偏移下普遍负迁移，不能因时延低而表述为有效适应路线。

### 8.3逐K主表

每行含25个任务，即5个receiver×5个seed；before old_acc均为同一固定query集合上的未适应ADV3B02结果，因此各K档均为73.60%。

|方法|K|before old_acc|after old_acc|95%CI|平均增益|胜/负|平均时延|
|---|---:|---:|---:|---:|---:|---:|---:|
|MRIOR-SDA|1|73.60%|77.22%|73.26%–81.19%|+3.62pp|18/7|17.78s|
|MRIOR-SDA|2|73.60%|79.51%|76.07%–82.96%|+5.91pp|19/6|17.65s|
|MRIOR-SDA|5|73.60%|82.59%|79.80%–85.38%|+8.99pp|22/3|17.71s|
|MRIOR-SDA|10|73.60%|85.82%|83.31%–88.33%|+12.22pp|22/3|18.35s|
|MRIOR-SDA|20|73.60%|**87.74%**|85.35%–90.14%|**+14.14pp**|24/1|18.00s|
|DADDA-SDA|1|73.60%|74.94%|71.02%–78.87%|+1.34pp|15/10|14.34s|
|DADDA-SDA|2|73.60%|76.14%|72.49%–79.80%|+2.54pp|17/8|14.58s|
|DADDA-SDA|5|73.60%|78.19%|74.79%–81.59%|+4.59pp|21/4|14.66s|
|DADDA-SDA|10|73.60%|80.31%|77.27%–83.36%|+6.71pp|22/3|14.86s|
|DADDA-SDA|20|73.60%|82.16%|79.40%–84.91%|+8.56pp|24/1|14.65s|
|ProtoNet CDA|1|73.60%|58.67%|53.05%–64.28%|−14.93pp|0/25|0.04s|
|ProtoNet CDA|2|73.60%|64.70%|59.25%–70.15%|−8.90pp|1/24|0.06s|
|ProtoNet CDA|5|73.60%|68.98%|64.05%–73.91%|−4.62pp|2/23|0.06s|
|ProtoNet CDA|10|73.60%|70.42%|65.83%–75.01%|−3.18pp|3/22|0.06s|
|ProtoNet CDA|20|73.60%|71.48%|67.12%–75.84%|−2.12pp|4/21|0.02s|

K增加时三种方法均改善，但改善含义不同：MRIOR-SDA与DADDA-SDA从K=1起总体正迁移，且随support增加持续扩大收益；ProtoNet CDA只是逐步减轻负迁移，直到K=20仍未回到未适应基线。

### 8.4逐receiver结果

每行含25个K×seed任务；增益仍为同一run配对差值。

|方法|receiver|after old_acc|95%CI|平均增益|
|---|---|---:|---:|---:|
|MRIOR-SDA|20-1|83.43%|81.33%–85.54%|+18.82pp|
|MRIOR-SDA|3-19|69.06%|66.44%–71.67%|+8.72pp|
|MRIOR-SDA|7-14|89.93%|89.14%–90.73%|−0.12pp|
|MRIOR-SDA|7-7|86.78%|85.14%–88.42%|+6.56pp|
|MRIOR-SDA|8-8|83.69%|81.74%–85.64%|+10.91pp|
|DADDA-SDA|20-1|75.91%|74.00%–77.82%|+11.30pp|
|DADDA-SDA|3-19|65.31%|63.70%–66.92%|+4.98pp|
|DADDA-SDA|7-14|89.93%|89.04%–90.83%|−0.12pp|
|DADDA-SDA|7-7|82.49%|81.58%–83.40%|+2.27pp|
|DADDA-SDA|8-8|78.10%|76.51%–79.69%|+5.32pp|
|ProtoNet CDA|20-1|60.98%|58.55%–63.41%|−3.63pp|
|ProtoNet CDA|3-19|48.50%|45.29%–51.71%|−11.83pp|
|ProtoNet CDA|7-14|83.53%|81.69%–85.38%|−6.52pp|
|ProtoNet CDA|7-7|74.81%|73.03%–76.60%|−5.41pp|
|ProtoNet CDA|8-8|66.42%|63.91%–68.94%|−6.36pp|

MRIOR-SDA与DADDA-SDA的主要收益来自20-1、3-19、7-7和8-8；在未适应性能已高的7-14上二者均为−0.12pp，说明固定200步仍存在轻微饱和域负迁移。ProtoNet CDA在5个receiver上全部平均负迁移，其中3-19最严重。

### 8.5逐LEO_weak场景结果

|方法|场景|after old_acc|95%CI|平均增益|
|---|---|---:|---:|---:|
|MRIOR-SDA|leo_clear_weak|85.99%|84.56%–87.41%|+10.09pp|
|MRIOR-SDA|leo_low_elev_weak|81.29%|79.60%–82.99%|+8.76pp|
|MRIOR-SDA|leo_rain_weak|80.45%|78.81%–82.10%|+8.09pp|
|DADDA-SDA|leo_clear_weak|81.15%|79.61%–82.69%|+5.25pp|
|DADDA-SDA|leo_low_elev_weak|77.39%|75.63%–79.16%|+4.86pp|
|DADDA-SDA|leo_rain_weak|76.50%|74.84%–78.16%|+4.13pp|
|ProtoNet CDA|leo_clear_weak|70.08%|67.53%–72.63%|−5.82pp|
|ProtoNet CDA|leo_low_elev_weak|65.99%|63.52%–68.45%|−6.55pp|
|ProtoNet CDA|leo_rain_weak|64.48%|62.02%–66.94%|−7.89pp|

三种方法均在`leo_rain_weak`最困难。MRIOR-SDA和DADDA-SDA在三个弱信道场景均保持正平均增益；ProtoNet CDA在三个场景均为负。

### 8.6声明边界与正式判定

- 本次375行属于Stage2-B target-old域适应，只含6个注册旧类；不存在新类注册，因此不报告`new_acc`、H或unknown指标。
- MRIOR-SDA与DADDA-SDA更新完整ADV3B02 identity backbone，虽性能优于未适应基线，但计算和可写状态均不属于星上极轻型方案。
- ProtoNet CDA近似无训练时延，但准确率结论为负，不能用其轻量性替代有效性判定。
- 2026-07-14历史clean-access三方法结果继续标记为`PROTOCOL_INVALID_FOR_PHASE2`，不纳入本次排名或统计。
- 正式推荐次序为MRIOR-SDA>DADDA-SDA>ProtoNet CDA；若优先性能，MRIOR-SDA为当前三方法主比较；若限制计算资源，需另行比较不更新backbone的合规轻量路线，而不能直接采用本次ProtoNet CDA。

汇总artifact：`summary/audit.json`、`summary/method_overall_summary.csv`、`summary/method_k_summary.csv`、`summary/receiver_summary.csv`、`summary/scenario_summary.csv`、`summary/class_summary.csv`、`summary/loss_summary.csv`、`summary/per_run_results.csv`、`summary/per_scenario_results.csv`。

### 8.7逐candidate/run联合明细

下表每行指标来自同一个candidate/run，不拼接跨run边际值。ProtoNet CDA的ADV3B02梯度更新数为0；MRIOR-SDA和DADDA-SDA均为三个场景各200步，共600次ADV3B02梯度更新。所有行缓存审计与最终判定均为PASS。

|candidate/run|方法|receiver|K|seed|before old_acc|after old_acc|delta|适配时延|ADV3B02更新|审计/判定|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|mrior_sda_stage2b_rx20-1_k1_seed713101|MRIOR-SDA|20-1|1|713101|61.11%|70.56%|+9.44pp|18.03s|600|PASS|
|mrior_sda_stage2b_rx20-1_k1_seed713102|MRIOR-SDA|20-1|1|713102|69.72%|81.94%|+12.22pp|17.51s|600|PASS|
|mrior_sda_stage2b_rx20-1_k1_seed713103|MRIOR-SDA|20-1|1|713103|63.61%|75.56%|+11.94pp|18.85s|600|PASS|
|mrior_sda_stage2b_rx20-1_k1_seed713104|MRIOR-SDA|20-1|1|713104|64.72%|74.44%|+9.72pp|18.33s|600|PASS|
|mrior_sda_stage2b_rx20-1_k1_seed713105|MRIOR-SDA|20-1|1|713105|63.89%|78.06%|+14.17pp|18.71s|600|PASS|
|mrior_sda_stage2b_rx20-1_k2_seed713101|MRIOR-SDA|20-1|2|713101|61.11%|80.28%|+19.17pp|18.41s|600|PASS|
|mrior_sda_stage2b_rx20-1_k2_seed713102|MRIOR-SDA|20-1|2|713102|69.72%|84.44%|+14.72pp|17.32s|600|PASS|
|mrior_sda_stage2b_rx20-1_k2_seed713103|MRIOR-SDA|20-1|2|713103|63.61%|79.17%|+15.56pp|16.60s|600|PASS|
|mrior_sda_stage2b_rx20-1_k2_seed713104|MRIOR-SDA|20-1|2|713104|64.72%|80.28%|+15.56pp|17.04s|600|PASS|
|mrior_sda_stage2b_rx20-1_k2_seed713105|MRIOR-SDA|20-1|2|713105|63.89%|79.17%|+15.28pp|17.75s|600|PASS|
|mrior_sda_stage2b_rx20-1_k5_seed713101|MRIOR-SDA|20-1|5|713101|61.11%|83.61%|+22.50pp|18.35s|600|PASS|
|mrior_sda_stage2b_rx20-1_k5_seed713102|MRIOR-SDA|20-1|5|713102|69.72%|87.22%|+17.50pp|16.11s|600|PASS|
|mrior_sda_stage2b_rx20-1_k5_seed713103|MRIOR-SDA|20-1|5|713103|63.61%|85.56%|+21.94pp|17.93s|600|PASS|
|mrior_sda_stage2b_rx20-1_k5_seed713104|MRIOR-SDA|20-1|5|713104|64.72%|81.11%|+16.39pp|16.51s|600|PASS|
|mrior_sda_stage2b_rx20-1_k5_seed713105|MRIOR-SDA|20-1|5|713105|63.89%|83.06%|+19.17pp|18.50s|600|PASS|
|mrior_sda_stage2b_rx20-1_k10_seed713101|MRIOR-SDA|20-1|10|713101|61.11%|86.94%|+25.83pp|17.64s|600|PASS|
|mrior_sda_stage2b_rx20-1_k10_seed713102|MRIOR-SDA|20-1|10|713102|69.72%|90.28%|+20.56pp|18.79s|600|PASS|
|mrior_sda_stage2b_rx20-1_k10_seed713103|MRIOR-SDA|20-1|10|713103|63.61%|85.00%|+21.39pp|18.06s|600|PASS|
|mrior_sda_stage2b_rx20-1_k10_seed713104|MRIOR-SDA|20-1|10|713104|64.72%|84.44%|+19.72pp|17.82s|600|PASS|
|mrior_sda_stage2b_rx20-1_k10_seed713105|MRIOR-SDA|20-1|10|713105|63.89%|87.22%|+23.33pp|17.02s|600|PASS|
|mrior_sda_stage2b_rx20-1_k20_seed713101|MRIOR-SDA|20-1|20|713101|61.11%|89.72%|+28.61pp|16.36s|600|PASS|
|mrior_sda_stage2b_rx20-1_k20_seed713102|MRIOR-SDA|20-1|20|713102|69.72%|91.67%|+21.94pp|18.05s|600|PASS|
|mrior_sda_stage2b_rx20-1_k20_seed713103|MRIOR-SDA|20-1|20|713103|63.61%|87.22%|+23.61pp|18.12s|600|PASS|
|mrior_sda_stage2b_rx20-1_k20_seed713104|MRIOR-SDA|20-1|20|713104|64.72%|89.44%|+24.72pp|18.85s|600|PASS|
|mrior_sda_stage2b_rx20-1_k20_seed713105|MRIOR-SDA|20-1|20|713105|63.89%|89.44%|+25.56pp|19.08s|600|PASS|
|mrior_sda_stage2b_rx3-19_k1_seed713101|MRIOR-SDA|3-19|1|713101|66.67%|65.83%|-0.83pp|18.64s|600|PASS|
|mrior_sda_stage2b_rx3-19_k1_seed713102|MRIOR-SDA|3-19|1|713102|60.83%|56.39%|-4.44pp|16.83s|600|PASS|
|mrior_sda_stage2b_rx3-19_k1_seed713103|MRIOR-SDA|3-19|1|713103|57.50%|59.17%|+1.67pp|17.65s|600|PASS|
|mrior_sda_stage2b_rx3-19_k1_seed713104|MRIOR-SDA|3-19|1|713104|55.28%|59.72%|+4.44pp|17.24s|600|PASS|
|mrior_sda_stage2b_rx3-19_k1_seed713105|MRIOR-SDA|3-19|1|713105|61.39%|61.94%|+0.56pp|17.60s|600|PASS|
|mrior_sda_stage2b_rx3-19_k2_seed713101|MRIOR-SDA|3-19|2|713101|66.67%|68.06%|+1.39pp|17.56s|600|PASS|
|mrior_sda_stage2b_rx3-19_k2_seed713102|MRIOR-SDA|3-19|2|713102|60.83%|61.11%|+0.28pp|18.52s|600|PASS|
|mrior_sda_stage2b_rx3-19_k2_seed713103|MRIOR-SDA|3-19|2|713103|57.50%|59.72%|+2.22pp|17.63s|600|PASS|
|mrior_sda_stage2b_rx3-19_k2_seed713104|MRIOR-SDA|3-19|2|713104|55.28%|63.89%|+8.61pp|17.64s|600|PASS|
|mrior_sda_stage2b_rx3-19_k2_seed713105|MRIOR-SDA|3-19|2|713105|61.39%|66.67%|+5.28pp|17.24s|600|PASS|
|mrior_sda_stage2b_rx3-19_k5_seed713101|MRIOR-SDA|3-19|5|713101|66.67%|73.33%|+6.67pp|18.79s|600|PASS|
|mrior_sda_stage2b_rx3-19_k5_seed713102|MRIOR-SDA|3-19|5|713102|60.83%|70.56%|+9.72pp|18.29s|600|PASS|
|mrior_sda_stage2b_rx3-19_k5_seed713103|MRIOR-SDA|3-19|5|713103|57.50%|66.94%|+9.44pp|16.93s|600|PASS|
|mrior_sda_stage2b_rx3-19_k5_seed713104|MRIOR-SDA|3-19|5|713104|55.28%|71.67%|+16.39pp|15.47s|600|PASS|
|mrior_sda_stage2b_rx3-19_k5_seed713105|MRIOR-SDA|3-19|5|713105|61.39%|67.78%|+6.39pp|19.11s|600|PASS|
|mrior_sda_stage2b_rx3-19_k10_seed713101|MRIOR-SDA|3-19|10|713101|66.67%|76.94%|+10.28pp|19.02s|600|PASS|
|mrior_sda_stage2b_rx3-19_k10_seed713102|MRIOR-SDA|3-19|10|713102|60.83%|75.00%|+14.17pp|19.20s|600|PASS|
|mrior_sda_stage2b_rx3-19_k10_seed713103|MRIOR-SDA|3-19|10|713103|57.50%|71.94%|+14.44pp|17.81s|600|PASS|
|mrior_sda_stage2b_rx3-19_k10_seed713104|MRIOR-SDA|3-19|10|713104|55.28%|75.00%|+19.72pp|18.54s|600|PASS|
|mrior_sda_stage2b_rx3-19_k10_seed713105|MRIOR-SDA|3-19|10|713105|61.39%|72.50%|+11.11pp|18.36s|600|PASS|
|mrior_sda_stage2b_rx3-19_k20_seed713101|MRIOR-SDA|3-19|20|713101|66.67%|80.00%|+13.33pp|18.89s|600|PASS|
|mrior_sda_stage2b_rx3-19_k20_seed713102|MRIOR-SDA|3-19|20|713102|60.83%|76.67%|+15.83pp|18.47s|600|PASS|
|mrior_sda_stage2b_rx3-19_k20_seed713103|MRIOR-SDA|3-19|20|713103|57.50%|75.83%|+18.33pp|18.29s|600|PASS|
|mrior_sda_stage2b_rx3-19_k20_seed713104|MRIOR-SDA|3-19|20|713104|55.28%|75.56%|+20.28pp|18.77s|600|PASS|
|mrior_sda_stage2b_rx3-19_k20_seed713105|MRIOR-SDA|3-19|20|713105|61.39%|74.17%|+12.78pp|16.38s|600|PASS|
|mrior_sda_stage2b_rx7-14_k1_seed713101|MRIOR-SDA|7-14|1|713101|90.56%|87.78%|-2.78pp|18.76s|600|PASS|
|mrior_sda_stage2b_rx7-14_k1_seed713102|MRIOR-SDA|7-14|1|713102|93.61%|90.56%|-3.06pp|18.24s|600|PASS|
|mrior_sda_stage2b_rx7-14_k1_seed713103|MRIOR-SDA|7-14|1|713103|88.61%|87.50%|-1.11pp|18.53s|600|PASS|
|mrior_sda_stage2b_rx7-14_k1_seed713104|MRIOR-SDA|7-14|1|713104|88.06%|89.72%|+1.67pp|18.27s|600|PASS|
|mrior_sda_stage2b_rx7-14_k1_seed713105|MRIOR-SDA|7-14|1|713105|89.44%|89.72%|+0.28pp|17.48s|600|PASS|
|mrior_sda_stage2b_rx7-14_k2_seed713101|MRIOR-SDA|7-14|2|713101|90.56%|88.61%|-1.94pp|18.63s|600|PASS|
|mrior_sda_stage2b_rx7-14_k2_seed713102|MRIOR-SDA|7-14|2|713102|93.61%|91.94%|-1.67pp|16.98s|600|PASS|
|mrior_sda_stage2b_rx7-14_k2_seed713103|MRIOR-SDA|7-14|2|713103|88.61%|88.06%|-0.56pp|19.13s|600|PASS|
|mrior_sda_stage2b_rx7-14_k2_seed713104|MRIOR-SDA|7-14|2|713104|88.06%|87.50%|-0.56pp|17.92s|600|PASS|
|mrior_sda_stage2b_rx7-14_k2_seed713105|MRIOR-SDA|7-14|2|713105|89.44%|87.50%|-1.94pp|18.47s|600|PASS|
|mrior_sda_stage2b_rx7-14_k5_seed713101|MRIOR-SDA|7-14|5|713101|90.56%|87.50%|-3.06pp|16.79s|600|PASS|
|mrior_sda_stage2b_rx7-14_k5_seed713102|MRIOR-SDA|7-14|5|713102|93.61%|92.78%|-0.83pp|16.83s|600|PASS|
|mrior_sda_stage2b_rx7-14_k5_seed713103|MRIOR-SDA|7-14|5|713103|88.61%|89.17%|+0.56pp|19.35s|600|PASS|
|mrior_sda_stage2b_rx7-14_k5_seed713104|MRIOR-SDA|7-14|5|713104|88.06%|90.00%|+1.94pp|18.63s|600|PASS|
|mrior_sda_stage2b_rx7-14_k5_seed713105|MRIOR-SDA|7-14|5|713105|89.44%|87.50%|-1.94pp|18.11s|600|PASS|
|mrior_sda_stage2b_rx7-14_k10_seed713101|MRIOR-SDA|7-14|10|713101|90.56%|89.44%|-1.11pp|18.35s|600|PASS|
|mrior_sda_stage2b_rx7-14_k10_seed713102|MRIOR-SDA|7-14|10|713102|93.61%|92.22%|-1.39pp|18.11s|600|PASS|
|mrior_sda_stage2b_rx7-14_k10_seed713103|MRIOR-SDA|7-14|10|713103|88.61%|91.39%|+2.78pp|18.67s|600|PASS|
|mrior_sda_stage2b_rx7-14_k10_seed713104|MRIOR-SDA|7-14|10|713104|88.06%|91.11%|+3.06pp|19.04s|600|PASS|
|mrior_sda_stage2b_rx7-14_k10_seed713105|MRIOR-SDA|7-14|10|713105|89.44%|88.61%|-0.83pp|19.02s|600|PASS|
|mrior_sda_stage2b_rx7-14_k20_seed713101|MRIOR-SDA|7-14|20|713101|90.56%|88.89%|-1.67pp|17.84s|600|PASS|
|mrior_sda_stage2b_rx7-14_k20_seed713102|MRIOR-SDA|7-14|20|713102|93.61%|94.17%|+0.56pp|17.98s|600|PASS|
|mrior_sda_stage2b_rx7-14_k20_seed713103|MRIOR-SDA|7-14|20|713103|88.61%|91.67%|+3.06pp|18.52s|600|PASS|
|mrior_sda_stage2b_rx7-14_k20_seed713104|MRIOR-SDA|7-14|20|713104|88.06%|92.78%|+4.72pp|16.71s|600|PASS|
|mrior_sda_stage2b_rx7-14_k20_seed713105|MRIOR-SDA|7-14|20|713105|89.44%|92.22%|+2.78pp|17.60s|600|PASS|
|mrior_sda_stage2b_rx7-7_k1_seed713101|MRIOR-SDA|7-7|1|713101|78.89%|82.22%|+3.33pp|17.13s|600|PASS|
|mrior_sda_stage2b_rx7-7_k1_seed713102|MRIOR-SDA|7-7|1|713102|81.11%|86.39%|+5.28pp|16.94s|600|PASS|
|mrior_sda_stage2b_rx7-7_k1_seed713103|MRIOR-SDA|7-7|1|713103|76.94%|81.67%|+4.72pp|18.66s|600|PASS|
|mrior_sda_stage2b_rx7-7_k1_seed713104|MRIOR-SDA|7-7|1|713104|82.78%|82.78%|-0.00pp|17.07s|600|PASS|
|mrior_sda_stage2b_rx7-7_k1_seed713105|MRIOR-SDA|7-7|1|713105|81.39%|79.72%|-1.67pp|17.07s|600|PASS|
|mrior_sda_stage2b_rx7-7_k2_seed713101|MRIOR-SDA|7-7|2|713101|78.89%|81.39%|+2.50pp|17.39s|600|PASS|
|mrior_sda_stage2b_rx7-7_k2_seed713102|MRIOR-SDA|7-7|2|713102|81.11%|85.28%|+4.17pp|17.55s|600|PASS|
|mrior_sda_stage2b_rx7-7_k2_seed713103|MRIOR-SDA|7-7|2|713103|76.94%|83.33%|+6.39pp|16.66s|600|PASS|
|mrior_sda_stage2b_rx7-7_k2_seed713104|MRIOR-SDA|7-7|2|713104|82.78%|82.50%|-0.28pp|17.49s|600|PASS|
|mrior_sda_stage2b_rx7-7_k2_seed713105|MRIOR-SDA|7-7|2|713105|81.39%|83.89%|+2.50pp|18.09s|600|PASS|
|mrior_sda_stage2b_rx7-7_k5_seed713101|MRIOR-SDA|7-7|5|713101|78.89%|86.39%|+7.50pp|14.60s|600|PASS|
|mrior_sda_stage2b_rx7-7_k5_seed713102|MRIOR-SDA|7-7|5|713102|81.11%|89.44%|+8.33pp|16.24s|600|PASS|
|mrior_sda_stage2b_rx7-7_k5_seed713103|MRIOR-SDA|7-7|5|713103|76.94%|83.61%|+6.67pp|17.38s|600|PASS|
|mrior_sda_stage2b_rx7-7_k5_seed713104|MRIOR-SDA|7-7|5|713104|82.78%|85.56%|+2.78pp|16.98s|600|PASS|
|mrior_sda_stage2b_rx7-7_k5_seed713105|MRIOR-SDA|7-7|5|713105|81.39%|87.78%|+6.39pp|18.21s|600|PASS|
|mrior_sda_stage2b_rx7-7_k10_seed713101|MRIOR-SDA|7-7|10|713101|78.89%|89.72%|+10.83pp|18.18s|600|PASS|
|mrior_sda_stage2b_rx7-7_k10_seed713102|MRIOR-SDA|7-7|10|713102|81.11%|94.44%|+13.33pp|18.14s|600|PASS|
|mrior_sda_stage2b_rx7-7_k10_seed713103|MRIOR-SDA|7-7|10|713103|76.94%|86.11%|+9.17pp|18.06s|600|PASS|
|mrior_sda_stage2b_rx7-7_k10_seed713104|MRIOR-SDA|7-7|10|713104|82.78%|88.89%|+6.11pp|18.83s|600|PASS|
|mrior_sda_stage2b_rx7-7_k10_seed713105|MRIOR-SDA|7-7|10|713105|81.39%|90.56%|+9.17pp|18.24s|600|PASS|
|mrior_sda_stage2b_rx7-7_k20_seed713101|MRIOR-SDA|7-7|20|713101|78.89%|90.28%|+11.39pp|18.78s|600|PASS|
|mrior_sda_stage2b_rx7-7_k20_seed713102|MRIOR-SDA|7-7|20|713102|81.11%|94.17%|+13.06pp|18.49s|600|PASS|
|mrior_sda_stage2b_rx7-7_k20_seed713103|MRIOR-SDA|7-7|20|713103|76.94%|89.17%|+12.22pp|17.21s|600|PASS|
|mrior_sda_stage2b_rx7-7_k20_seed713104|MRIOR-SDA|7-7|20|713104|82.78%|91.11%|+8.33pp|19.37s|600|PASS|
|mrior_sda_stage2b_rx7-7_k20_seed713105|MRIOR-SDA|7-7|20|713105|81.39%|93.06%|+11.67pp|18.67s|600|PASS|
|mrior_sda_stage2b_rx8-8_k1_seed713101|MRIOR-SDA|8-8|1|713101|73.06%|79.44%|+6.39pp|17.17s|600|PASS|
|mrior_sda_stage2b_rx8-8_k1_seed713102|MRIOR-SDA|8-8|1|713102|72.22%|83.33%|+11.11pp|16.99s|600|PASS|
|mrior_sda_stage2b_rx8-8_k1_seed713103|MRIOR-SDA|8-8|1|713103|70.56%|71.94%|+1.39pp|18.10s|600|PASS|
|mrior_sda_stage2b_rx8-8_k1_seed713104|MRIOR-SDA|8-8|1|713104|73.61%|75.56%|+1.94pp|18.45s|600|PASS|
|mrior_sda_stage2b_rx8-8_k1_seed713105|MRIOR-SDA|8-8|1|713105|74.44%|78.61%|+4.17pp|16.26s|600|PASS|
|mrior_sda_stage2b_rx8-8_k2_seed713101|MRIOR-SDA|8-8|2|713101|73.06%|79.44%|+6.39pp|17.01s|600|PASS|
|mrior_sda_stage2b_rx8-8_k2_seed713102|MRIOR-SDA|8-8|2|713102|72.22%|85.00%|+12.78pp|17.33s|600|PASS|
|mrior_sda_stage2b_rx8-8_k2_seed713103|MRIOR-SDA|8-8|2|713103|70.56%|79.17%|+8.61pp|17.60s|600|PASS|
|mrior_sda_stage2b_rx8-8_k2_seed713104|MRIOR-SDA|8-8|2|713104|73.61%|78.89%|+5.28pp|18.46s|600|PASS|
|mrior_sda_stage2b_rx8-8_k2_seed713105|MRIOR-SDA|8-8|2|713105|74.44%|82.50%|+8.06pp|16.87s|600|PASS|
|mrior_sda_stage2b_rx8-8_k5_seed713101|MRIOR-SDA|8-8|5|713101|73.06%|83.61%|+10.56pp|18.62s|600|PASS|
|mrior_sda_stage2b_rx8-8_k5_seed713102|MRIOR-SDA|8-8|5|713102|72.22%|85.28%|+13.06pp|19.13s|600|PASS|
|mrior_sda_stage2b_rx8-8_k5_seed713103|MRIOR-SDA|8-8|5|713103|70.56%|82.22%|+11.67pp|18.56s|600|PASS|
|mrior_sda_stage2b_rx8-8_k5_seed713104|MRIOR-SDA|8-8|5|713104|73.61%|83.61%|+10.00pp|19.94s|600|PASS|
|mrior_sda_stage2b_rx8-8_k5_seed713105|MRIOR-SDA|8-8|5|713105|74.44%|79.44%|+5.00pp|17.50s|600|PASS|
|mrior_sda_stage2b_rx8-8_k10_seed713101|MRIOR-SDA|8-8|10|713101|73.06%|85.56%|+12.50pp|18.81s|600|PASS|
|mrior_sda_stage2b_rx8-8_k10_seed713102|MRIOR-SDA|8-8|10|713102|72.22%|89.72%|+17.50pp|18.32s|600|PASS|
|mrior_sda_stage2b_rx8-8_k10_seed713103|MRIOR-SDA|8-8|10|713103|70.56%|90.00%|+19.44pp|16.59s|600|PASS|
|mrior_sda_stage2b_rx8-8_k10_seed713104|MRIOR-SDA|8-8|10|713104|73.61%|86.94%|+13.33pp|18.80s|600|PASS|
|mrior_sda_stage2b_rx8-8_k10_seed713105|MRIOR-SDA|8-8|10|713105|74.44%|85.56%|+11.11pp|19.21s|600|PASS|
|mrior_sda_stage2b_rx8-8_k20_seed713101|MRIOR-SDA|8-8|20|713101|73.06%|86.94%|+13.89pp|16.98s|600|PASS|
|mrior_sda_stage2b_rx8-8_k20_seed713102|MRIOR-SDA|8-8|20|713102|72.22%|90.83%|+18.61pp|18.78s|600|PASS|
|mrior_sda_stage2b_rx8-8_k20_seed713103|MRIOR-SDA|8-8|20|713103|70.56%|89.44%|+18.89pp|17.54s|600|PASS|
|mrior_sda_stage2b_rx8-8_k20_seed713104|MRIOR-SDA|8-8|20|713104|73.61%|89.44%|+15.83pp|19.67s|600|PASS|
|mrior_sda_stage2b_rx8-8_k20_seed713105|MRIOR-SDA|8-8|20|713105|74.44%|89.72%|+15.28pp|14.57s|600|PASS|
|dadda_sda_stage2b_rx20-1_k1_seed713101|DADDA-SDA|20-1|1|713101|61.11%|65.00%|+3.89pp|14.89s|600|PASS|
|dadda_sda_stage2b_rx20-1_k1_seed713102|DADDA-SDA|20-1|1|713102|69.72%|74.72%|+5.00pp|13.68s|600|PASS|
|dadda_sda_stage2b_rx20-1_k1_seed713103|DADDA-SDA|20-1|1|713103|63.61%|70.00%|+6.39pp|14.48s|600|PASS|
|dadda_sda_stage2b_rx20-1_k1_seed713104|DADDA-SDA|20-1|1|713104|64.72%|70.56%|+5.83pp|13.61s|600|PASS|
|dadda_sda_stage2b_rx20-1_k1_seed713105|DADDA-SDA|20-1|1|713105|63.89%|72.50%|+8.61pp|13.05s|600|PASS|
|dadda_sda_stage2b_rx20-1_k2_seed713101|DADDA-SDA|20-1|2|713101|61.11%|70.56%|+9.44pp|13.16s|600|PASS|
|dadda_sda_stage2b_rx20-1_k2_seed713102|DADDA-SDA|20-1|2|713102|69.72%|75.56%|+5.83pp|14.33s|600|PASS|
|dadda_sda_stage2b_rx20-1_k2_seed713103|DADDA-SDA|20-1|2|713103|63.61%|70.00%|+6.39pp|15.29s|600|PASS|
|dadda_sda_stage2b_rx20-1_k2_seed713104|DADDA-SDA|20-1|2|713104|64.72%|71.94%|+7.22pp|15.06s|600|PASS|
|dadda_sda_stage2b_rx20-1_k2_seed713105|DADDA-SDA|20-1|2|713105|63.89%|74.17%|+10.28pp|13.00s|600|PASS|
|dadda_sda_stage2b_rx20-1_k5_seed713101|DADDA-SDA|20-1|5|713101|61.11%|71.39%|+10.28pp|15.67s|600|PASS|
|dadda_sda_stage2b_rx20-1_k5_seed713102|DADDA-SDA|20-1|5|713102|69.72%|80.56%|+10.83pp|11.97s|600|PASS|
|dadda_sda_stage2b_rx20-1_k5_seed713103|DADDA-SDA|20-1|5|713103|63.61%|74.72%|+11.11pp|15.46s|600|PASS|
|dadda_sda_stage2b_rx20-1_k5_seed713104|DADDA-SDA|20-1|5|713104|64.72%|76.67%|+11.94pp|15.68s|600|PASS|
|dadda_sda_stage2b_rx20-1_k5_seed713105|DADDA-SDA|20-1|5|713105|63.89%|77.78%|+13.89pp|15.96s|600|PASS|
|dadda_sda_stage2b_rx20-1_k10_seed713101|DADDA-SDA|20-1|10|713101|61.11%|77.50%|+16.39pp|14.57s|600|PASS|
|dadda_sda_stage2b_rx20-1_k10_seed713102|DADDA-SDA|20-1|10|713102|69.72%|82.22%|+12.50pp|15.60s|600|PASS|
|dadda_sda_stage2b_rx20-1_k10_seed713103|DADDA-SDA|20-1|10|713103|63.61%|77.50%|+13.89pp|15.02s|600|PASS|
|dadda_sda_stage2b_rx20-1_k10_seed713104|DADDA-SDA|20-1|10|713104|64.72%|76.67%|+11.94pp|15.16s|600|PASS|
|dadda_sda_stage2b_rx20-1_k10_seed713105|DADDA-SDA|20-1|10|713105|63.89%|78.61%|+14.72pp|15.71s|600|PASS|
|dadda_sda_stage2b_rx20-1_k20_seed713101|DADDA-SDA|20-1|20|713101|61.11%|80.56%|+19.44pp|15.11s|600|PASS|
|dadda_sda_stage2b_rx20-1_k20_seed713102|DADDA-SDA|20-1|20|713102|69.72%|85.00%|+15.28pp|15.78s|600|PASS|
|dadda_sda_stage2b_rx20-1_k20_seed713103|DADDA-SDA|20-1|20|713103|63.61%|79.72%|+16.11pp|12.78s|600|PASS|
|dadda_sda_stage2b_rx20-1_k20_seed713104|DADDA-SDA|20-1|20|713104|64.72%|81.67%|+16.94pp|15.46s|600|PASS|
|dadda_sda_stage2b_rx20-1_k20_seed713105|DADDA-SDA|20-1|20|713105|63.89%|82.22%|+18.33pp|15.49s|600|PASS|
|dadda_sda_stage2b_rx3-19_k1_seed713101|DADDA-SDA|3-19|1|713101|66.67%|62.50%|-4.17pp|15.41s|600|PASS|
|dadda_sda_stage2b_rx3-19_k1_seed713102|DADDA-SDA|3-19|1|713102|60.83%|61.67%|+0.83pp|14.47s|600|PASS|
|dadda_sda_stage2b_rx3-19_k1_seed713103|DADDA-SDA|3-19|1|713103|57.50%|57.78%|+0.28pp|15.53s|600|PASS|
|dadda_sda_stage2b_rx3-19_k1_seed713104|DADDA-SDA|3-19|1|713104|55.28%|58.06%|+2.78pp|14.03s|600|PASS|
|dadda_sda_stage2b_rx3-19_k1_seed713105|DADDA-SDA|3-19|1|713105|61.39%|65.00%|+3.61pp|15.21s|600|PASS|
|dadda_sda_stage2b_rx3-19_k2_seed713101|DADDA-SDA|3-19|2|713101|66.67%|65.00%|-1.67pp|14.61s|600|PASS|
|dadda_sda_stage2b_rx3-19_k2_seed713102|DADDA-SDA|3-19|2|713102|60.83%|63.06%|+2.22pp|14.58s|600|PASS|
|dadda_sda_stage2b_rx3-19_k2_seed713103|DADDA-SDA|3-19|2|713103|57.50%|63.61%|+6.11pp|13.92s|600|PASS|
|dadda_sda_stage2b_rx3-19_k2_seed713104|DADDA-SDA|3-19|2|713104|55.28%|59.17%|+3.89pp|14.60s|600|PASS|
|dadda_sda_stage2b_rx3-19_k2_seed713105|DADDA-SDA|3-19|2|713105|61.39%|63.61%|+2.22pp|14.48s|600|PASS|
|dadda_sda_stage2b_rx3-19_k5_seed713101|DADDA-SDA|3-19|5|713101|66.67%|68.89%|+2.22pp|14.03s|600|PASS|
|dadda_sda_stage2b_rx3-19_k5_seed713102|DADDA-SDA|3-19|5|713102|60.83%|63.89%|+3.06pp|15.63s|600|PASS|
|dadda_sda_stage2b_rx3-19_k5_seed713103|DADDA-SDA|3-19|5|713103|57.50%|60.83%|+3.33pp|14.09s|600|PASS|
|dadda_sda_stage2b_rx3-19_k5_seed713104|DADDA-SDA|3-19|5|713104|55.28%|65.83%|+10.56pp|15.19s|600|PASS|
|dadda_sda_stage2b_rx3-19_k5_seed713105|DADDA-SDA|3-19|5|713105|61.39%|63.61%|+2.22pp|13.92s|600|PASS|
|dadda_sda_stage2b_rx3-19_k10_seed713101|DADDA-SDA|3-19|10|713101|66.67%|71.67%|+5.00pp|15.44s|600|PASS|
|dadda_sda_stage2b_rx3-19_k10_seed713102|DADDA-SDA|3-19|10|713102|60.83%|65.56%|+4.72pp|15.63s|600|PASS|
|dadda_sda_stage2b_rx3-19_k10_seed713103|DADDA-SDA|3-19|10|713103|57.50%|66.94%|+9.44pp|14.78s|600|PASS|
|dadda_sda_stage2b_rx3-19_k10_seed713104|DADDA-SDA|3-19|10|713104|55.28%|69.17%|+13.89pp|15.31s|600|PASS|
|dadda_sda_stage2b_rx3-19_k10_seed713105|DADDA-SDA|3-19|10|713105|61.39%|66.11%|+4.72pp|14.37s|600|PASS|
|dadda_sda_stage2b_rx3-19_k20_seed713101|DADDA-SDA|3-19|20|713101|66.67%|72.78%|+6.11pp|15.14s|600|PASS|
|dadda_sda_stage2b_rx3-19_k20_seed713102|DADDA-SDA|3-19|20|713102|60.83%|70.00%|+9.17pp|14.31s|600|PASS|
|dadda_sda_stage2b_rx3-19_k20_seed713103|DADDA-SDA|3-19|20|713103|57.50%|70.28%|+12.78pp|16.13s|600|PASS|
|dadda_sda_stage2b_rx3-19_k20_seed713104|DADDA-SDA|3-19|20|713104|55.28%|68.89%|+13.61pp|16.18s|600|PASS|
|dadda_sda_stage2b_rx3-19_k20_seed713105|DADDA-SDA|3-19|20|713105|61.39%|68.89%|+7.50pp|15.01s|600|PASS|
|dadda_sda_stage2b_rx7-14_k1_seed713101|DADDA-SDA|7-14|1|713101|90.56%|90.00%|-0.56pp|15.45s|600|PASS|
|dadda_sda_stage2b_rx7-14_k1_seed713102|DADDA-SDA|7-14|1|713102|93.61%|93.61%|-0.00pp|15.32s|600|PASS|
|dadda_sda_stage2b_rx7-14_k1_seed713103|DADDA-SDA|7-14|1|713103|88.61%|87.50%|-1.11pp|12.53s|600|PASS|
|dadda_sda_stage2b_rx7-14_k1_seed713104|DADDA-SDA|7-14|1|713104|88.06%|86.94%|-1.11pp|14.25s|600|PASS|
|dadda_sda_stage2b_rx7-14_k1_seed713105|DADDA-SDA|7-14|1|713105|89.44%|88.06%|-1.39pp|14.29s|600|PASS|
|dadda_sda_stage2b_rx7-14_k2_seed713101|DADDA-SDA|7-14|2|713101|90.56%|89.44%|-1.11pp|15.44s|600|PASS|
|dadda_sda_stage2b_rx7-14_k2_seed713102|DADDA-SDA|7-14|2|713102|93.61%|93.33%|-0.28pp|15.46s|600|PASS|
|dadda_sda_stage2b_rx7-14_k2_seed713103|DADDA-SDA|7-14|2|713103|88.61%|88.33%|-0.28pp|15.95s|600|PASS|
|dadda_sda_stage2b_rx7-14_k2_seed713104|DADDA-SDA|7-14|2|713104|88.06%|88.61%|+0.56pp|14.97s|600|PASS|
|dadda_sda_stage2b_rx7-14_k2_seed713105|DADDA-SDA|7-14|2|713105|89.44%|88.61%|-0.83pp|15.53s|600|PASS|
|dadda_sda_stage2b_rx7-14_k5_seed713101|DADDA-SDA|7-14|5|713101|90.56%|90.56%|+0.00pp|15.14s|600|PASS|
|dadda_sda_stage2b_rx7-14_k5_seed713102|DADDA-SDA|7-14|5|713102|93.61%|94.17%|+0.56pp|15.27s|600|PASS|
|dadda_sda_stage2b_rx7-14_k5_seed713103|DADDA-SDA|7-14|5|713103|88.61%|88.33%|-0.28pp|16.09s|600|PASS|
|dadda_sda_stage2b_rx7-14_k5_seed713104|DADDA-SDA|7-14|5|713104|88.06%|87.78%|-0.28pp|14.79s|600|PASS|
|dadda_sda_stage2b_rx7-14_k5_seed713105|DADDA-SDA|7-14|5|713105|89.44%|87.78%|-1.67pp|15.04s|600|PASS|
|dadda_sda_stage2b_rx7-14_k10_seed713101|DADDA-SDA|7-14|10|713101|90.56%|89.72%|-0.83pp|13.18s|600|PASS|
|dadda_sda_stage2b_rx7-14_k10_seed713102|DADDA-SDA|7-14|10|713102|93.61%|94.72%|+1.11pp|15.35s|600|PASS|
|dadda_sda_stage2b_rx7-14_k10_seed713103|DADDA-SDA|7-14|10|713103|88.61%|88.33%|-0.28pp|15.40s|600|PASS|
|dadda_sda_stage2b_rx7-14_k10_seed713104|DADDA-SDA|7-14|10|713104|88.06%|89.17%|+1.11pp|14.33s|600|PASS|
|dadda_sda_stage2b_rx7-14_k10_seed713105|DADDA-SDA|7-14|10|713105|89.44%|89.17%|-0.28pp|16.13s|600|PASS|
|dadda_sda_stage2b_rx7-14_k20_seed713101|DADDA-SDA|7-14|20|713101|90.56%|89.72%|-0.83pp|14.20s|600|PASS|
|dadda_sda_stage2b_rx7-14_k20_seed713102|DADDA-SDA|7-14|20|713102|93.61%|94.17%|+0.56pp|14.59s|600|PASS|
|dadda_sda_stage2b_rx7-14_k20_seed713103|DADDA-SDA|7-14|20|713103|88.61%|89.17%|+0.56pp|14.78s|600|PASS|
|dadda_sda_stage2b_rx7-14_k20_seed713104|DADDA-SDA|7-14|20|713104|88.06%|90.56%|+2.50pp|12.05s|600|PASS|
|dadda_sda_stage2b_rx7-14_k20_seed713105|DADDA-SDA|7-14|20|713105|89.44%|90.56%|+1.11pp|15.14s|600|PASS|
|dadda_sda_stage2b_rx7-7_k1_seed713101|DADDA-SDA|7-7|1|713101|78.89%|81.67%|+2.78pp|15.08s|600|PASS|
|dadda_sda_stage2b_rx7-7_k1_seed713102|DADDA-SDA|7-7|1|713102|81.11%|81.94%|+0.83pp|14.62s|600|PASS|
|dadda_sda_stage2b_rx7-7_k1_seed713103|DADDA-SDA|7-7|1|713103|76.94%|77.78%|+0.83pp|14.01s|600|PASS|
|dadda_sda_stage2b_rx7-7_k1_seed713104|DADDA-SDA|7-7|1|713104|82.78%|80.28%|-2.50pp|13.21s|600|PASS|
|dadda_sda_stage2b_rx7-7_k1_seed713105|DADDA-SDA|7-7|1|713105|81.39%|80.28%|-1.11pp|15.89s|600|PASS|
|dadda_sda_stage2b_rx7-7_k2_seed713101|DADDA-SDA|7-7|2|713101|78.89%|83.33%|+4.44pp|14.80s|600|PASS|
|dadda_sda_stage2b_rx7-7_k2_seed713102|DADDA-SDA|7-7|2|713102|81.11%|80.83%|-0.28pp|13.92s|600|PASS|
|dadda_sda_stage2b_rx7-7_k2_seed713103|DADDA-SDA|7-7|2|713103|76.94%|79.44%|+2.50pp|13.21s|600|PASS|
|dadda_sda_stage2b_rx7-7_k2_seed713104|DADDA-SDA|7-7|2|713104|82.78%|80.28%|-2.50pp|14.48s|600|PASS|
|dadda_sda_stage2b_rx7-7_k2_seed713105|DADDA-SDA|7-7|2|713105|81.39%|81.67%|+0.28pp|14.83s|600|PASS|
|dadda_sda_stage2b_rx7-7_k5_seed713101|DADDA-SDA|7-7|5|713101|78.89%|84.17%|+5.28pp|14.15s|600|PASS|
|dadda_sda_stage2b_rx7-7_k5_seed713102|DADDA-SDA|7-7|5|713102|81.11%|81.39%|+0.28pp|15.31s|600|PASS|
|dadda_sda_stage2b_rx7-7_k5_seed713103|DADDA-SDA|7-7|5|713103|76.94%|80.56%|+3.61pp|14.12s|600|PASS|
|dadda_sda_stage2b_rx7-7_k5_seed713104|DADDA-SDA|7-7|5|713104|82.78%|81.94%|-0.83pp|14.73s|600|PASS|
|dadda_sda_stage2b_rx7-7_k5_seed713105|DADDA-SDA|7-7|5|713105|81.39%|82.22%|+0.83pp|13.89s|600|PASS|
|dadda_sda_stage2b_rx7-7_k10_seed713101|DADDA-SDA|7-7|10|713101|78.89%|85.83%|+6.94pp|16.43s|600|PASS|
|dadda_sda_stage2b_rx7-7_k10_seed713102|DADDA-SDA|7-7|10|713102|81.11%|85.00%|+3.89pp|12.80s|600|PASS|
|dadda_sda_stage2b_rx7-7_k10_seed713103|DADDA-SDA|7-7|10|713103|76.94%|79.72%|+2.78pp|14.37s|600|PASS|
|dadda_sda_stage2b_rx7-7_k10_seed713104|DADDA-SDA|7-7|10|713104|82.78%|83.89%|+1.11pp|14.02s|600|PASS|
|dadda_sda_stage2b_rx7-7_k10_seed713105|DADDA-SDA|7-7|10|713105|81.39%|85.28%|+3.89pp|14.96s|600|PASS|
|dadda_sda_stage2b_rx7-7_k20_seed713101|DADDA-SDA|7-7|20|713101|78.89%|84.72%|+5.83pp|14.94s|600|PASS|
|dadda_sda_stage2b_rx7-7_k20_seed713102|DADDA-SDA|7-7|20|713102|81.11%|86.11%|+5.00pp|14.19s|600|PASS|
|dadda_sda_stage2b_rx7-7_k20_seed713103|DADDA-SDA|7-7|20|713103|76.94%|83.61%|+6.67pp|14.40s|600|PASS|
|dadda_sda_stage2b_rx7-7_k20_seed713104|DADDA-SDA|7-7|20|713104|82.78%|84.44%|+1.67pp|15.64s|600|PASS|
|dadda_sda_stage2b_rx7-7_k20_seed713105|DADDA-SDA|7-7|20|713105|81.39%|85.83%|+4.44pp|14.20s|600|PASS|
|dadda_sda_stage2b_rx8-8_k1_seed713101|DADDA-SDA|8-8|1|713101|73.06%|75.28%|+2.22pp|15.30s|600|PASS|
|dadda_sda_stage2b_rx8-8_k1_seed713102|DADDA-SDA|8-8|1|713102|72.22%|76.11%|+3.89pp|13.06s|600|PASS|
|dadda_sda_stage2b_rx8-8_k1_seed713103|DADDA-SDA|8-8|1|713103|70.56%|68.61%|-1.94pp|13.30s|600|PASS|
|dadda_sda_stage2b_rx8-8_k1_seed713104|DADDA-SDA|8-8|1|713104|73.61%|73.06%|-0.56pp|14.26s|600|PASS|
|dadda_sda_stage2b_rx8-8_k1_seed713105|DADDA-SDA|8-8|1|713105|74.44%|74.72%|+0.28pp|13.64s|600|PASS|
|dadda_sda_stage2b_rx8-8_k2_seed713101|DADDA-SDA|8-8|2|713101|73.06%|76.94%|+3.89pp|14.17s|600|PASS|
|dadda_sda_stage2b_rx8-8_k2_seed713102|DADDA-SDA|8-8|2|713102|72.22%|73.89%|+1.67pp|13.42s|600|PASS|
|dadda_sda_stage2b_rx8-8_k2_seed713103|DADDA-SDA|8-8|2|713103|70.56%|72.50%|+1.94pp|15.15s|600|PASS|
|dadda_sda_stage2b_rx8-8_k2_seed713104|DADDA-SDA|8-8|2|713104|73.61%|72.78%|-0.83pp|15.22s|600|PASS|
|dadda_sda_stage2b_rx8-8_k2_seed713105|DADDA-SDA|8-8|2|713105|74.44%|76.94%|+2.50pp|14.83s|600|PASS|
|dadda_sda_stage2b_rx8-8_k5_seed713101|DADDA-SDA|8-8|5|713101|73.06%|78.89%|+5.83pp|15.16s|600|PASS|
|dadda_sda_stage2b_rx8-8_k5_seed713102|DADDA-SDA|8-8|5|713102|72.22%|78.33%|+6.11pp|13.61s|600|PASS|
|dadda_sda_stage2b_rx8-8_k5_seed713103|DADDA-SDA|8-8|5|713103|70.56%|76.11%|+5.56pp|14.09s|600|PASS|
|dadda_sda_stage2b_rx8-8_k5_seed713104|DADDA-SDA|8-8|5|713104|73.61%|78.33%|+4.72pp|14.04s|600|PASS|
|dadda_sda_stage2b_rx8-8_k5_seed713105|DADDA-SDA|8-8|5|713105|74.44%|80.00%|+5.56pp|13.55s|600|PASS|
|dadda_sda_stage2b_rx8-8_k10_seed713101|DADDA-SDA|8-8|10|713101|73.06%|82.22%|+9.17pp|14.28s|600|PASS|
|dadda_sda_stage2b_rx8-8_k10_seed713102|DADDA-SDA|8-8|10|713102|72.22%|81.94%|+9.72pp|13.98s|600|PASS|
|dadda_sda_stage2b_rx8-8_k10_seed713103|DADDA-SDA|8-8|10|713103|70.56%|80.00%|+9.44pp|14.31s|600|PASS|
|dadda_sda_stage2b_rx8-8_k10_seed713104|DADDA-SDA|8-8|10|713104|73.61%|79.72%|+6.11pp|15.80s|600|PASS|
|dadda_sda_stage2b_rx8-8_k10_seed713105|DADDA-SDA|8-8|10|713105|74.44%|81.11%|+6.67pp|14.44s|600|PASS|
|dadda_sda_stage2b_rx8-8_k20_seed713101|DADDA-SDA|8-8|20|713101|73.06%|82.50%|+9.44pp|13.58s|600|PASS|
|dadda_sda_stage2b_rx8-8_k20_seed713102|DADDA-SDA|8-8|20|713102|72.22%|84.17%|+11.94pp|15.29s|600|PASS|
|dadda_sda_stage2b_rx8-8_k20_seed713103|DADDA-SDA|8-8|20|713103|70.56%|81.94%|+11.39pp|14.21s|600|PASS|
|dadda_sda_stage2b_rx8-8_k20_seed713104|DADDA-SDA|8-8|20|713104|73.61%|83.33%|+9.72pp|15.25s|600|PASS|
|dadda_sda_stage2b_rx8-8_k20_seed713105|DADDA-SDA|8-8|20|713105|74.44%|83.06%|+8.61pp|12.51s|600|PASS|
|protonet_cda_stage2b_rx20-1_k1_seed713101|ProtoNet CDA|20-1|1|713101|61.11%|50.83%|-10.28pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx20-1_k1_seed713102|ProtoNet CDA|20-1|1|713102|69.72%|55.83%|-13.89pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx20-1_k1_seed713103|ProtoNet CDA|20-1|1|713103|63.61%|47.50%|-16.11pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx20-1_k1_seed713104|ProtoNet CDA|20-1|1|713104|64.72%|55.00%|-9.72pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k1_seed713105|ProtoNet CDA|20-1|1|713105|63.89%|46.11%|-17.78pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k2_seed713101|ProtoNet CDA|20-1|2|713101|61.11%|63.89%|+2.78pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k2_seed713102|ProtoNet CDA|20-1|2|713102|69.72%|59.44%|-10.28pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k2_seed713103|ProtoNet CDA|20-1|2|713103|63.61%|59.17%|-4.44pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k2_seed713104|ProtoNet CDA|20-1|2|713104|64.72%|61.94%|-2.78pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k2_seed713105|ProtoNet CDA|20-1|2|713105|63.89%|60.00%|-3.89pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k5_seed713101|ProtoNet CDA|20-1|5|713101|61.11%|60.00%|-1.11pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k5_seed713102|ProtoNet CDA|20-1|5|713102|69.72%|68.61%|-1.11pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k5_seed713103|ProtoNet CDA|20-1|5|713103|63.61%|60.00%|-3.61pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k5_seed713104|ProtoNet CDA|20-1|5|713104|64.72%|63.89%|-0.83pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k5_seed713105|ProtoNet CDA|20-1|5|713105|63.89%|66.94%|+3.06pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k10_seed713101|ProtoNet CDA|20-1|10|713101|61.11%|63.61%|+2.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx20-1_k10_seed713102|ProtoNet CDA|20-1|10|713102|69.72%|70.28%|+0.56pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k10_seed713103|ProtoNet CDA|20-1|10|713103|63.61%|58.61%|-5.00pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k10_seed713104|ProtoNet CDA|20-1|10|713104|64.72%|61.39%|-3.33pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k10_seed713105|ProtoNet CDA|20-1|10|713105|63.89%|65.83%|+1.94pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx20-1_k20_seed713101|ProtoNet CDA|20-1|20|713101|61.11%|61.67%|+0.56pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx20-1_k20_seed713102|ProtoNet CDA|20-1|20|713102|69.72%|68.61%|-1.11pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx20-1_k20_seed713103|ProtoNet CDA|20-1|20|713103|63.61%|64.44%|+0.83pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx20-1_k20_seed713104|ProtoNet CDA|20-1|20|713104|64.72%|64.17%|-0.56pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx20-1_k20_seed713105|ProtoNet CDA|20-1|20|713105|63.89%|66.67%|+2.78pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx3-19_k1_seed713101|ProtoNet CDA|3-19|1|713101|66.67%|47.78%|-18.89pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx3-19_k1_seed713102|ProtoNet CDA|3-19|1|713102|60.83%|41.94%|-18.89pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx3-19_k1_seed713103|ProtoNet CDA|3-19|1|713103|57.50%|27.78%|-29.72pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx3-19_k1_seed713104|ProtoNet CDA|3-19|1|713104|55.28%|36.11%|-19.17pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx3-19_k1_seed713105|ProtoNet CDA|3-19|1|713105|61.39%|42.50%|-18.89pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx3-19_k2_seed713101|ProtoNet CDA|3-19|2|713101|66.67%|48.89%|-17.78pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k2_seed713102|ProtoNet CDA|3-19|2|713102|60.83%|44.44%|-16.39pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k2_seed713103|ProtoNet CDA|3-19|2|713103|57.50%|37.22%|-20.28pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx3-19_k2_seed713104|ProtoNet CDA|3-19|2|713104|55.28%|38.89%|-16.39pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k2_seed713105|ProtoNet CDA|3-19|2|713105|61.39%|47.22%|-14.17pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k5_seed713101|ProtoNet CDA|3-19|5|713101|66.67%|57.78%|-8.89pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx3-19_k5_seed713102|ProtoNet CDA|3-19|5|713102|60.83%|53.06%|-7.78pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k5_seed713103|ProtoNet CDA|3-19|5|713103|57.50%|45.83%|-11.67pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx3-19_k5_seed713104|ProtoNet CDA|3-19|5|713104|55.28%|41.39%|-13.89pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k5_seed713105|ProtoNet CDA|3-19|5|713105|61.39%|53.89%|-7.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k10_seed713101|ProtoNet CDA|3-19|10|713101|66.67%|59.17%|-7.50pp|0.07s|0|PASS|
|protonet_cda_stage2b_rx3-19_k10_seed713102|ProtoNet CDA|3-19|10|713102|60.83%|53.61%|-7.22pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx3-19_k10_seed713103|ProtoNet CDA|3-19|10|713103|57.50%|51.39%|-6.11pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k10_seed713104|ProtoNet CDA|3-19|10|713104|55.28%|49.72%|-5.56pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k10_seed713105|ProtoNet CDA|3-19|10|713105|61.39%|55.28%|-6.11pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx3-19_k20_seed713101|ProtoNet CDA|3-19|20|713101|66.67%|60.00%|-6.67pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx3-19_k20_seed713102|ProtoNet CDA|3-19|20|713102|60.83%|53.89%|-6.94pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx3-19_k20_seed713103|ProtoNet CDA|3-19|20|713103|57.50%|52.22%|-5.28pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx3-19_k20_seed713104|ProtoNet CDA|3-19|20|713104|55.28%|52.78%|-2.50pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx3-19_k20_seed713105|ProtoNet CDA|3-19|20|713105|61.39%|59.72%|-1.67pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-14_k1_seed713101|ProtoNet CDA|7-14|1|713101|90.56%|81.94%|-8.61pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-14_k1_seed713102|ProtoNet CDA|7-14|1|713102|93.61%|73.61%|-20.00pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-14_k1_seed713103|ProtoNet CDA|7-14|1|713103|88.61%|76.39%|-12.22pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-14_k1_seed713104|ProtoNet CDA|7-14|1|713104|88.06%|77.50%|-10.56pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-14_k1_seed713105|ProtoNet CDA|7-14|1|713105|89.44%|74.44%|-15.00pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-14_k2_seed713101|ProtoNet CDA|7-14|2|713101|90.56%|86.94%|-3.61pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k2_seed713102|ProtoNet CDA|7-14|2|713102|93.61%|81.11%|-12.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k2_seed713103|ProtoNet CDA|7-14|2|713103|88.61%|80.56%|-8.06pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k2_seed713104|ProtoNet CDA|7-14|2|713104|88.06%|82.78%|-5.28pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k2_seed713105|ProtoNet CDA|7-14|2|713105|89.44%|81.94%|-7.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k5_seed713101|ProtoNet CDA|7-14|5|713101|90.56%|85.83%|-4.72pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-14_k5_seed713102|ProtoNet CDA|7-14|5|713102|93.61%|88.33%|-5.28pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k5_seed713103|ProtoNet CDA|7-14|5|713103|88.61%|81.11%|-7.50pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-14_k5_seed713104|ProtoNet CDA|7-14|5|713104|88.06%|83.89%|-4.17pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k5_seed713105|ProtoNet CDA|7-14|5|713105|89.44%|86.11%|-3.33pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k10_seed713101|ProtoNet CDA|7-14|10|713101|90.56%|86.94%|-3.61pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k10_seed713102|ProtoNet CDA|7-14|10|713102|93.61%|91.11%|-2.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k10_seed713103|ProtoNet CDA|7-14|10|713103|88.61%|82.22%|-6.39pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-14_k10_seed713104|ProtoNet CDA|7-14|10|713104|88.06%|83.33%|-4.72pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k10_seed713105|ProtoNet CDA|7-14|10|713105|89.44%|87.78%|-1.67pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-14_k20_seed713101|ProtoNet CDA|7-14|20|713101|90.56%|85.83%|-4.72pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-14_k20_seed713102|ProtoNet CDA|7-14|20|713102|93.61%|92.50%|-1.11pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-14_k20_seed713103|ProtoNet CDA|7-14|20|713103|88.61%|83.89%|-4.72pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-14_k20_seed713104|ProtoNet CDA|7-14|20|713104|88.06%|84.72%|-3.33pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-14_k20_seed713105|ProtoNet CDA|7-14|20|713105|89.44%|87.50%|-1.94pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-7_k1_seed713101|ProtoNet CDA|7-7|1|713101|78.89%|75.00%|-3.89pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-7_k1_seed713102|ProtoNet CDA|7-7|1|713102|81.11%|67.22%|-13.89pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-7_k1_seed713103|ProtoNet CDA|7-7|1|713103|76.94%|63.89%|-13.06pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-7_k1_seed713104|ProtoNet CDA|7-7|1|713104|82.78%|62.22%|-20.56pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-7_k1_seed713105|ProtoNet CDA|7-7|1|713105|81.39%|75.00%|-6.39pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx7-7_k2_seed713101|ProtoNet CDA|7-7|2|713101|78.89%|76.39%|-2.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k2_seed713102|ProtoNet CDA|7-7|2|713102|81.11%|70.00%|-11.11pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k2_seed713103|ProtoNet CDA|7-7|2|713103|76.94%|70.83%|-6.11pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k2_seed713104|ProtoNet CDA|7-7|2|713104|82.78%|75.28%|-7.50pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-7_k2_seed713105|ProtoNet CDA|7-7|2|713105|81.39%|76.39%|-5.00pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k5_seed713101|ProtoNet CDA|7-7|5|713101|78.89%|78.06%|-0.83pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k5_seed713102|ProtoNet CDA|7-7|5|713102|81.11%|75.56%|-5.56pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-7_k5_seed713103|ProtoNet CDA|7-7|5|713103|76.94%|76.67%|-0.28pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k5_seed713104|ProtoNet CDA|7-7|5|713104|82.78%|78.33%|-4.44pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k5_seed713105|ProtoNet CDA|7-7|5|713105|81.39%|77.22%|-4.17pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-7_k10_seed713101|ProtoNet CDA|7-7|10|713101|78.89%|77.78%|-1.11pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k10_seed713102|ProtoNet CDA|7-7|10|713102|81.11%|78.33%|-2.78pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-7_k10_seed713103|ProtoNet CDA|7-7|10|713103|76.94%|74.72%|-2.22pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx7-7_k10_seed713104|ProtoNet CDA|7-7|10|713104|82.78%|78.61%|-4.17pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k10_seed713105|ProtoNet CDA|7-7|10|713105|81.39%|77.78%|-3.61pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx7-7_k20_seed713101|ProtoNet CDA|7-7|20|713101|78.89%|77.50%|-1.39pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-7_k20_seed713102|ProtoNet CDA|7-7|20|713102|81.11%|78.06%|-3.06pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-7_k20_seed713103|ProtoNet CDA|7-7|20|713103|76.94%|73.61%|-3.33pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-7_k20_seed713104|ProtoNet CDA|7-7|20|713104|82.78%|78.89%|-3.89pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx7-7_k20_seed713105|ProtoNet CDA|7-7|20|713105|81.39%|76.94%|-4.44pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx8-8_k1_seed713101|ProtoNet CDA|8-8|1|713101|73.06%|51.94%|-21.11pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx8-8_k1_seed713102|ProtoNet CDA|8-8|1|713102|72.22%|59.44%|-12.78pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx8-8_k1_seed713103|ProtoNet CDA|8-8|1|713103|70.56%|52.50%|-18.06pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx8-8_k1_seed713104|ProtoNet CDA|8-8|1|713104|73.61%|61.39%|-12.22pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx8-8_k1_seed713105|ProtoNet CDA|8-8|1|713105|74.44%|62.78%|-11.67pp|0.04s|0|PASS|
|protonet_cda_stage2b_rx8-8_k2_seed713101|ProtoNet CDA|8-8|2|713101|73.06%|58.61%|-14.44pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k2_seed713102|ProtoNet CDA|8-8|2|713102|72.22%|65.28%|-6.94pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx8-8_k2_seed713103|ProtoNet CDA|8-8|2|713103|70.56%|63.33%|-7.22pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k2_seed713104|ProtoNet CDA|8-8|2|713104|73.61%|59.17%|-14.44pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k2_seed713105|ProtoNet CDA|8-8|2|713105|74.44%|67.78%|-6.67pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k5_seed713101|ProtoNet CDA|8-8|5|713101|73.06%|67.50%|-5.56pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k5_seed713102|ProtoNet CDA|8-8|5|713102|72.22%|63.89%|-8.33pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k5_seed713103|ProtoNet CDA|8-8|5|713103|70.56%|71.67%|+1.11pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx8-8_k5_seed713104|ProtoNet CDA|8-8|5|713104|73.61%|68.33%|-5.28pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx8-8_k5_seed713105|ProtoNet CDA|8-8|5|713105|74.44%|70.56%|-3.89pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k10_seed713101|ProtoNet CDA|8-8|10|713101|73.06%|69.17%|-3.89pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k10_seed713102|ProtoNet CDA|8-8|10|713102|72.22%|71.94%|-0.28pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k10_seed713103|ProtoNet CDA|8-8|10|713103|70.56%|69.17%|-1.39pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx8-8_k10_seed713104|ProtoNet CDA|8-8|10|713104|73.61%|70.83%|-2.78pp|0.05s|0|PASS|
|protonet_cda_stage2b_rx8-8_k10_seed713105|ProtoNet CDA|8-8|10|713105|74.44%|71.94%|-2.50pp|0.06s|0|PASS|
|protonet_cda_stage2b_rx8-8_k20_seed713101|ProtoNet CDA|8-8|20|713101|73.06%|72.78%|-0.28pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx8-8_k20_seed713102|ProtoNet CDA|8-8|20|713102|72.22%|71.67%|-0.56pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx8-8_k20_seed713103|ProtoNet CDA|8-8|20|713103|70.56%|69.17%|-1.39pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx8-8_k20_seed713104|ProtoNet CDA|8-8|20|713104|73.61%|75.83%|+2.22pp|0.02s|0|PASS|
|protonet_cda_stage2b_rx8-8_k20_seed713105|ProtoNet CDA|8-8|20|713105|74.44%|73.89%|-0.56pp|0.02s|0|PASS|

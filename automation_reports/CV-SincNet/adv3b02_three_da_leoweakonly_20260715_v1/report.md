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
|当前状态|N607_OFFLINE_PREPARATION_RUNNING；PID 1604043；正式375行尚未启动|

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
|GPU/PID|offline准备PID `1604043`已退出；26个cache已完成，25个bundle和1个runtime seal待续跑；正式worker未启动|
|代码版本|待同步Git commit `98ea663`；本地`py_compile` PASS、相关pytest `33 passed`、375行matrix dry-run PASS|
|offline准备命令|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_adv3b02_three_da_cache_plan.py --plan-manifest runs/adv3b02_three_da_leoweakonly_20260715_v1/plan/plan_manifest.json --execute`|
|正式worker命令|由`plan_manifest.json.commands.phase2_workers`给出8个shard；必须在52/52离线准备和三方法smoke通过后才允许启动|
|期望输出|每行`metrics.json`、`split_manifest.json`、`resolved_config.json`、`score_table.csv`、详细分组统计和完整loss trace|

下一次同步映射：commit `98ea663`中的`code/scripts/{build_cvs_stage2_predictor_bundle,build_phase2_runtime_seal}.py`、ADV3B02 runner、独立scorer、plan/cache-plan脚本和配置同步至N607同相对路径；本地`local_artifacts/adv3b02_three_da_leoweakonly_20260715_v1_plan/{phase2_config.json,plan_manifest.json,cache_specs/,package_artifacts/}`同步至远端run root的`plan/`。既有26个cache不覆盖、不删除。

计划SCP命令均使用`scp -F E:\type10-7\tools\n607_ssh_config`：按`code/scripts/`、`paper_reproduction/cvs_aligned/`、`paper_reproduction/scripts/`、`paper_reproduction/configs/`四个目标目录分别同步上述文件，再将`phase2_config.json`、`plan_manifest.json`、`cache_specs/`和`package_artifacts/`同步至`N607:/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/plan/`。同步前只读复核为相关进程0、8张GPU空闲、cache manifest 26、predictor manifest 0、磁盘余量7.6TB。

启动诊断记录：初次后台命令因远端根目录旧`cvsrffi`包遮蔽`code/cvsrffi`而退出；commit `e325503`已修复`sys.path`优先级。第二次因offline spec误写`/CV-SincNet/ManySig.pkl`而退出；服务器实际路径为`/CV-SincNet/Dataset_WigSig/ManySig.pkl`，已修正。第三次PID `1604043`完成26/26缓存后，旧plan在第1个bundle调用已升级打包器时因CLI不兼容退出。三次均未启动正式Phase2行；第三次产生的26个LEO_weak cache保留并将在续跑时逐项hash复核后跳过。

第四次offline准备PID `1615166`在首个bundle完成密封后，由末尾字符串泄漏扫描误把固定Phase1 checkpoint内部合法的旧类标签映射识别为query truth泄漏而fail-closed退出；首个predictor manifest、scoring manifest和detached seal各1个已落盘，正式行仍为0。修复只豁免已预注册且受hash密封的`checkpoint.bin`、`adapter.bin`和`head.bin`无结构字节扫描；新生成support/query/package manifest仍执行成员白名单、同fd hash审计和泄漏扫描。修复后相关pytest仍为`33 passed`，首个已密封bundle不删除，由续跑时严格verifier决定复用或阻断。

第五次offline准备PID `1616795`已完成并正常退出：`expected=52`、`verified=52`、`cache_verified=26`、`predictor_bundle_verified=25`、`runtime_seal_verified=true`、`phase2_started=false`。其中27项严格校验后跳过复用，25项新建；全局inventory root SHA256为`8c33405898992adaa812975b2fd42c371302c88a60734cc335d5653c90a4c5c2`，runtime code SHA256为`199a16fe9bd09110e4d1793402aefbac2d5cc9a81c535d90a407eabb304c3fe2`。

下一步smoke固定`receiver=20-1`、`seed=713101`、`K=5`，三方法各1行，输出到`runs/adv3b02_three_da_leoweakonly_20260715_v1/smoke_runs/`，使用与正式worker相同的Landlock launcher、artifact allowlist、runtime evidence和独立scorer；smoke成功后才允许启动8个正式shard。

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

运行完成后按同一candidate/run行补全。

|candidate/run|方法|receiver|K|seed|场景|before old_acc|after old_acc|delta|loss/adapter摘要|缓存审计|最终判定|
|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
|待运行|—|—|—|—|—|—|—|—|—|—|PENDING|

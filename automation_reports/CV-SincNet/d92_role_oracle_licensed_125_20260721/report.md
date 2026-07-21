# D92 Role-Oracle特许125上限实验报告

## 预注册

- 实验ID：`d92_role_oracle_licensed_125_20260721`
- 日期：2026-07-21
- 操作者：Codex主agent；N607唯一发布owner将在本地冻结后单独登记
- 当前状态：`LOCAL_VERIFIED_PREREGISTERED_AWAITING_GIT_COMMIT`
- 结果标记：`LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE`
- 目标：测量D92在仅额外获知每个query的`old/new`角色、但不知道具体TX时的逐样本角色内分类上限，并与同版本、同row、同seed、同support状态、同INT8 score matrix的无Oracle结果严格配对。
- 声明边界：本实验复用`p2_min_v1`固定数据capsule及其余数据约束，但经用户特许偏离“query不得提供old/new角色”的正式决策规则。结果不属于协议合法性能，不得正式晋级、进入正式leaderboard、替代合法baseline或反馈选择后续方法参数。

## 当前最强版本判定

本次冻结D92而不是D81。D92是当前最新协议下联合H、旧类floor和遗忘Pareto最强的可运行开发版本，但原结论仍是`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不是正式晋级方法。

- D92最终证据提交：`f65f8934e9021fa4ef9acc8d2924bf6b968f5966`
- D92 retry2实际方法提交：`87012f4138c1cd308468ef74e238131af949c651`
- D92 retry2矩阵manifest SHA256：`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`
- D92 retry2 summary SHA256：`71bba2c9c8ae8fb3731c508438ce6db01d95d2b3fd5a00208ce8ca8ec54f5de9`
- D92 retry2 gates SHA256：`c9afc828398fb318628e4286f03fe45e38ca061efb8ca5527651dff5ef423924`
- D92相对D81的K10/new20同row变化：注册后old`+2.622pp`、min-old`+4.600pp`、H`+0.964pp`、forgetting`-2.622pp`，代价为seen-new`-0.653pp`。
- 排除历史qKNNV42：其高分依赖dense transductive query图、整批query状态或旧协议信息面，不满足当前逐query独立的`p2_min_v1`要求。

## 冻结假设与双臂机制

D92的稳健中心、旧/新support协方差、固定`0.5Σ_old+0.5Σ_new`、等先验LDA、K1回退、INT8/FP16状态及全部fit过程保持不变。Role信息不得进入support拟合、中心、协方差、温度、回退、量化或候选选择。

每个row、scenario只执行一次D92 fit、一次query feature前向和一次INT8 score计算，然后由同一份score matrix派生：

|arm|候选集合|状态|
|---|---|---|
|`D92_NO_ORACLE_MATCHED`|全部实际注册类`Y_old∪Y_new`|正式D92逐样本argmax；不得读取role capsule|
|`D92_ROLE_ORACLE_LICENSED`|old query在全部`Y_old`内argmax；new query在全部`Y_new`内argmax|只读取当前query的role bit；不得读取具体TX|

注册前只有旧类，两臂必须逐元素一致。注册后Oracle只消除跨角色竞争，不能改变角色内score、类排序或状态。

## Role capsule最小权限

允许字段仅为不可推断具体TX的opaque query token及`old/new`角色。禁止包含或利用：

- 具体TX/class handle、类别索引、真值标签、physical ID；
- unknown角色、真实batch类别集合、每类数量、角色数量或quota；
- query排序、标签分块、Hungarian、OT或global reassignment；
- 任何会重新拟合D92、更新状态或选择候选的信息。

无Oracle prediction与共享score必须先写入不可变artifact和`COMMIT.json`；随后licensed wrapper才可打开role capsule。specific-TX truth只能由两臂prediction均提交后的独立scorer读取。

## 完整125矩阵

- 接收机：`20-1`,`3-19`,`7-14`,`7-7`,`8-8`
- seed：`713102`,`713103`,`713104`,`713105`,`713106`
- slice：`K10/new5`,`K10/new10`,`K10/new20`,`K5/new20`,`K1/new20`
- scenario：`leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak`
- 总量：125个job、375个scene单元；每个job内部同时产生两臂，不拆成250个独立job。

125仅为特许上限稳定性screen，不进入合法1200确认矩阵，也不得反向选择D92或后续DA/head参数。

## 必报指标与分层

两臂及Oracle−NoOracle配对差值必须在同一row、同一scene内报告：

- `old_acc_before_increment`
- `old_acc_after_increment`
- `seen_new_acc`
- `H_old_new`
- `balanced_accuracy_all_registered`
- `floor_all_registered`
- `min_old_class_acc`
- `min_new_class_acc`
- `average_forgetting=old_before-old_after`

必须落盘125行row表、375行scene表、全部逐类准确率、old→new/new→old/old→old/new→new混淆，以及receiver×slice、scene×slice、receiver×scene×slice、seed×slice和slice总体汇总。统计区间以每个slice的25个`receiver×seed`簇为单位，三个scenario作为簇内观测，不把375个scene误当作独立样本。

## 不变量和失败门

下列条件全部通过才能把结果称为有效的特许上限：

1. 125/125 job及375/375 scene严格成对完成；
2. 两臂共享相同Git版本、row、seed、support状态、query token、INT8 state和score SHA；
3. fresh无Oracle结果与D92 retry2对应artifact逐项bit-exact；
4. 注册前两臂prediction完全相同；
5. 无Oracle已落在真实角色内时，Oracle prediction不变；
6. Oracle的old→new和new→old均为0；
7. Oracle的old/new/H/BA/floor/min-old/min-new不低于无Oracle，forgetting不高于无Oracle；
8. role capsule exact schema通过，predictor未读取具体TX、quota或跨query统计；
9. prediction在specific-TX truth连接前完成不可变提交；
10. 代码、矩阵、输入和输出SHA无漂移。

任一单调性或同score不变量失败均优先判定为实现/配对错误，不解释为统计波动。历史无Oracle汇总不得冒充本次fresh配对臂。

## 资源假设

Oracle decoder不新增fit、backbone前向、score MAC、optimizer step或持久模型状态；只增加逐样本角色子集argmax。role capsule及配对证据属于评估wire artifact，不计入可部署模型状态，但必须单独报告serialized bytes和额外决策时延。

## 本地版本、修改和验证计划

- Git工作树：`E:\type10-7\code\snapshots\d92_125wt`
- 分支：`codex/d92-role-oracle-125`
- 起始提交：`f65f8934e9021fa4ef9acc8d2924bf6b968f5966`
- 计划新增：纯role-only decoder、D92 paired evaluator、专用125 launcher、paired summarizer及对应测试。
- 计划最小修改：通用row pipeline仅增加显式licensed candidate路由和role-only sidecar投影；原D92路径必须回归保持不变。
- 本地环境：`conda activate ssr-gpu`
- 本地冻结门：focused tests、D92/D81邻接回归、`py_compile`、diff review、独立协议/数学/authority review、Git clean commit和文件SHA表全部完成。

## N607发布预登记

本地冻结后由唯一实验子agent独占run ID，主agent不并发启动同一实验。子agent负责直接N607预检、精确文件同步、远端SHA/compile、不可覆盖输出检查、GPU/进程现场、detached launch、短连接监控、完整日志读取、artifact回收和结构化交接。

- 预定run ID：`d92_role_oracle_licensed_125_20260721`
- 预定远端源码快照：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_role_oracle_source_20260721`
- 预定远端输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_role_oracle_licensed_125_20260721`
- 预定远端日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_role_oracle_licensed_125_20260721`
- Conda/Python：沿用经远端预检确认的CVS-RFFI GPU环境；精确路径、GPU、PID和命令在启动前补录。
- retry默认不授权；若发生技术错误，子agent只回收证据并交回主agent复审，不自行修改方法或重启。

### 冻结验证证据

- 2026-07-21最终`py_compile`：通过。
- 2026-07-21冻结回归：`41 passed`，命令为`conda run -n ssr-gpu python -m pytest -q tests/test_stage2_d92_licensed_role_oracle.py tests/test_stage2_d92_role_oracle_query_evaluation.py tests/test_run_d92_role_oracle_125.py tests/test_summarize_d92_role_oracle_125.py tests/test_run_cvs_somph_diag_row_pipeline.py`。
- pytest退出码为0；结束时仅出现Windows临时目录`pytest-current`清理权限噪声，不属于测试失败。
- `git diff --check`：通过。

|冻结文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d92_licensed_role_oracle.py`|`42869bc0f3f6283f80c4a917326af037d993e819b9c220f614e60d5655567f9f`|
|`code/cvsrffi/stage2_d92_role_oracle_query_evaluation.py`|`8e34a2da0b6eb56e27bb69f4e7183d2c4db1693764ae30ab8d9639963bd73e66`|
|`code/cvsrffi/stage2_d92_role_oracle_records.py`|`7c075e1a0c1bb778d6cbb6244ae77d4ac3f98562b6d06780ae7ea7aa8ff24ad4`|
|`code/scripts/run_cvs_somph_diag_row_pipeline.py`|`60923fb9ec90552814403effa28efcfc6afeff78df04e3a30e7cde7b4817821f`|
|`code/scripts/run_d92_role_oracle_125.py`|`64114d34a533bae38679414c5fdaf4764d8467a376a4fbafbfce749bf39697f7`|
|`code/scripts/build_d92_role_oracle_125_summary_manifest.py`|`3c7bf602c12574902b16ea60508f9e6321af69e1dbc7506ef5aef7255cb5e2c6`|
|`code/scripts/summarize_d92_role_oracle_125.py`|`086345559662b4c5177469c6004793cd04f52107b0b4abba8518b7b6dea19605`|

### 冻结运行输入与命令模板

- D92配对参考根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`
- cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`
- authority：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`
- Phase1 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- sealed runtime：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`
- method lock：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json`
- ground component：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`
- ground manifest SHA256：`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`
- 每个shard命令固定调用`run_d92_role_oracle_125.py`，参数为上述输入、不可覆盖输出根、`--cpu-threads 2 --shard-index <0..7> --shard-count 8 --device cuda:0`；实际`CUDA_VISIBLE_DEVICES`、PID和GPU占用由唯一N607发布子agent完成预检后登记。

## 完成后结果表

待完整125 artifact回收后填写。无论性能多高，最终判定保持`LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE`。

## 首次发布技术失败与retry1预注册

首次run已在N607真实启动8个shard（GPU0–7，PID`1828208`–`1828215`），但125/125 row均在方法执行前因发布快照漏同步既有`code/scripts/probe_d92_registration_balanced_covariance.py`而失败：`125 JOB_START / 0 JOB_COMPLETE / 125 JOB_FAILED`。进程均自然退出，未kill、未覆盖、未修改方法；该run标记为`TECHNICAL_RELEASE_FAILURE / NO_PERFORMANCE_RESULT`，不得解释为方法性能。

现预注册不可覆盖技术retry1：

- retry run ID：`d92_role_oracle_licensed_125_retry1_20260721`
- Git方法提交仍为`0184952cf7283632b0330727bd1bfe6b3026e44e`；方法、参数、矩阵、数据和结果标签不变。
- 唯一修复：把该提交中完整已跟踪`code/cvsrffi`与`code/scripts`按原结构同步至新源码快照，禁止使用未提交`sitecustomize.py`或远端代码修改。
- 新源码快照：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_role_oracle_source_retry1_20260721`
- 新输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_role_oracle_licensed_125_retry1_20260721`
- 新日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_role_oracle_licensed_125_retry1_20260721`
- 启动前新增门：远端显式导入`from scripts import probe_d92_registration_balanced_covariance`，并执行完整launcher dry import；失败则不得launch。
- 仍由同一个N607发布agent独占run；主agent不并发启动。retry1只授权这一项打包修复，不授权方法修改、调参或失败后的再次自动retry。

## retry1技术失败与retry2预注册

retry1完成895/895文件SHA、七个冻结文件SHA、显式probe import、`py_compile`、launcher help和`bash -n`后，在GPU0–7以PID`1837491`–`1837498`启动；但125/125 row在子row pipeline入口因漏同步Git跟踪的代码根级`training_controls.py`统一失败，仍为`0 COMPLETE`。该run同样只标记`TECHNICAL_RELEASE_FAILURE / NO_PERFORMANCE_RESULT`，不是方法性能失败。

现预注册不可覆盖技术retry2：

- run ID：`d92_role_oracle_licensed_125_retry2_20260721`
- 方法提交仍为`0184952cf7283632b0330727bd1bfe6b3026e44e`，只扩大冻结发布包覆盖面，不改代码、参数、矩阵、数据或声明边界。
- 同步该提交的完整Git跟踪`code/`树：1244文件、工作树原始字节约60.85MiB；逐文件SHA必须1244/1244通过。
- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_role_oracle_source_retry2_20260721`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_role_oracle_licensed_125_retry2_20260721`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_role_oracle_licensed_125_retry2_20260721`
- 新增无数据深层入口门：`PYTHONPATH=<source>/code <python> -c "import scripts.run_cvs_somph_diag_row_pipeline"`及row pipeline`--help`必须通过，确保完整子进程import链已封闭；不得用真实target row做smoke或选择。
- 仍由同一唯一N607发布owner执行。retry2失败后不授权自动retry。

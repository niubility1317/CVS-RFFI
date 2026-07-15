# qKNNv42严格Phase2预测-评分隔离管线报告

## 一、实验登记

|字段|内容|
|---|---|
|实验ID|qknnv42_phase2_strict_pipeline_20260715|
|日期|2026-07-15（Asia/Hong_Kong）|
|执行者|Codex|
|目标|按最新`项目.md`为qKNNv42/ADV3B02建立truth-free预测、sealed/tamper-evident prediction artifact、独立评分和可验证运行时隔离的Stage2诊断基础设施|
|当前状态|`LOCAL_DIAGNOSTIC_PIPELINE_PASS`且`formal_launch_authority=false`；严格qKNN Stage2-C尚未同步或启动N607|
|声明边界|本报告证明本地合同、artifact链和模拟执行控制流；不证明Linux OS隔离已实际执行，不包含新的正式性能结果|

## 二、最新协议与目标

本轮先后读取根`AGENTS.md`、更新后的根`项目.md`、当前control manifest、versioned prompt、workflow contract和optimizer state，并以更新后的`项目.md`为科学/data protocol唯一解释来源。

正式Stage2-C目标：

|指标|门槛|
|---|---:|
|old_acc|≥92%|
|最低旧类准确率|≥88%|
|5个seen-new准确率|≥92%|
|10个seen-new准确率|≥90%|
|20个seen-new准确率|≥86%|
|K=5相对K=10下降|≤3pp|
|K=1适应收益|非负，且相对strict direct ADV3B02≥+2pp，paired CI下界>0|

资源约束：

|档位|训练参数|适应轮数|持久状态|推理视图|
|---|---:|---:|---:|---:|
|优选|≤50k|≤20epoch|≤256KB|默认1-view，低置信度自适应触发3/5-view|
|放宽|≤100k|≤40epoch|≤512KB|最多5次forward|

禁止项包括role Oracle、query真实批次类别数、类别quota、query拟合、dense query graph和scorer反馈预测器。

## 三、文档一致性与恢复记录

在将根`项目.md`镜像到Git承载面时，曾因动态patch输入截断导致根文件短暂损坏。本轮立即停止后续修改，使用已登记快照和20个历史patch事件按原UTC顺序确定性恢复，并重新核验完整协议。

|检查项|结果|
|---|---|
|根`项目.md`行数|694|
|根`项目.md`SHA256|`179873019b46dc63cb421e61e09063589df2060e86f77f82929a55da7285b148`|
|Git镜像`项目.md`SHA256|`179873019b46dc63cb421e61e09063589df2060e86f77f82929a55da7285b148`|
|deprecated字段|`phase2_query_class_count_access`在当前协议正文中不存在|
|当前字段|`phase2_query_true_batch_class_count_access=false`存在|
|K=1 direct gain与预测/评分隔离条款|存在|
|本地preflight|`PENDING_LOCAL_ARTIFACTS`，无协议blocker|

## 四、实现结果

### 4.1输入与输出

输入：

- Phase1/offline生成并密封的LEO_weak predictor package；
- 外部candidate/plan锁定的detached seal SHA256；
- 7文件最小predictor runtime closure；
- 每个formal cell绑定3个固定LEO_weak场景的support/query opaque artifact；
- 9字段pre-run runtime evidence bundle；
- 独立scorer根中的truth sidecar。

输出：

- 单一、不可覆盖、只读的`.cvspred`容器；
- payload→manifest→seal哈希链；
- 5路预测流：candidate after/before、identity after/before、strict direct；
- 每样本实际view count；
- 后验filesystem access audit、predictor stdout receipt和13字段post-run evidence；
- 独立scorer产生的3个正式场景行、逐样本预测、old/new/H、最低类、forgetting、adaptation和K=1 direct delta。

### 4.2关键方法

|模块|方法|效果|
|---|---|---|
|truth-free bundle|双物理根、HMAC opaque class/query/support/overlay token、exact member allowlist|predictor不可读取truth/role/query真实类别计数|
|strict request|先校验3基础+4clean+5query共12字段和exact schema|在打开Phase2 payload前fail closed|
|minimal runtime closure|仅7个生产文件，AST exact import closure，拒绝dataset/training/legacy/dynamic import|切断旧loader和clean控制流|
|pre-run evidence bundle|外部seal锚、closure、package同fd预审、controller hash、实际bwrap/strace/python、固定system root allowlist和scorer/truth物理分根交叉绑定|拒绝孤立SHA、任意runtime目录和调用方自定义data root|
|OS策略|`/runtime/code`和package只读、仅`/output`可写、无网络、drop all caps、clearenv；strace写入sandbox外父进程持有的继承FD|为Linux物理隔离提供诊断执行合同，尚未取得真实Linux证据|
|sealed prediction|`O_EXCL`临时文件、fsync、atomic no-replace、只读权限、exact 8 NPZ字段|提供sealed/tamper-evident和防API覆盖属性，不宣称宿主同UID下绝对不可变|
|independent scorer|先验证外部artifact SHA和seal SHA，再按`(scenario,query_token)`精确连接truth|scorer无法反馈adapter、门限、回滚或候选选择|
|双重复验|isolated runner在预测前后复验同一bundle、closure、package、controller和实际参数|检测持续漂移；不能证明同UID“替换→执行→恢复”的瞬时TOCTOU已消除|

## 五、本地变更

核心文件：

- `code/cvsrffi/phase2_runtime_contract.py`
- `code/cvsrffi/stage2_predictor_bundle.py`
- `code/cvsrffi/stage2_predictor_runtime.py`
- `code/cvsrffi/stage2_predictor_entry.py`
- `code/cvsrffi/stage2_prediction_artifact.py`
- `code/cvsrffi/stage2_metric_scorer.py`
- `code/cvsrffi/phase2_runtime_closure.py`
- `code/cvsrffi/phase2_bwrap_policy.py`
- `code/cvsrffi/phase2_pre_run_evidence.py`
- `code/cvsrffi/phase2_isolated_runner.py`

CLI：

- `code/scripts/build_cvs_stage2_predictor_bundle.py`
- `code/scripts/build_cvs_stage2_predictor_request.py`
- `code/scripts/build_cvs_stage2_runtime_closure.py`
- `code/scripts/build_cvs_stage2_pre_run_evidence.py`
- `code/scripts/run_cvs_stage2_predictor.py`
- `code/scripts/run_cvs_stage2_bwrap_isolated.py`
- `code/scripts/score_cvs_stage2_sealed_prediction.py`

追踪面：

- `analysis/phase2_runtime_isolation_traceability_20260715.md`
- `项目.md`

## 六、本地验证

执行环境：`ssr-gpu`。

|验证|结果|
|---|---|
|严格相关pytest合集|117项全部通过|
|端到端链|真实bundle、真实closure、真实pre-run bundle、request、fake subprocess控制流、`.cvspred`、独立scorer和post-run evidence全部贯通；结果仅为`LOCAL_DIAGNOSTIC_PASS`|
|负向验证|deprecated字段、truth/role污染、quota、路径逃逸、symlink、hash篡改、NPZ成员漂移、artifact覆盖、receipt/attestation篡改、closure/executable漂移、任意system data root、空scorer root、缺场景、场景token集合不一致和view count 257绕回均被拒绝|
|警告|仅TorchScript弃用提示；当前不影响合同正确性|
|Linux bwrap真实执行|未执行；Windows集成使用fake subprocess验证控制流，不声明OS隔离PASS|

验证命令：

```text
python -m pytest code/tests/test_phase2_runtime_contract.py tests/test_stage2_predictor_bundle.py tests/test_build_cvs_stage2_predictor_bundle.py tests/test_build_cvs_stage2_predictor_request.py tests/test_stage2_predictor_runtime.py tests/test_stage2_predictor_entry.py tests/test_stage2_prediction_artifact.py tests/test_run_cvs_stage2_predictor.py tests/test_stage2_metric_scorer.py tests/test_stage2_sealed_pipeline_integration.py tests/test_phase2_bwrap_policy.py tests/test_phase2_runtime_closure.py tests/test_phase2_isolated_runner.py tests/test_phase2_pre_run_evidence.py -q
```

## 七、矩阵边界与服务器状态

按更新后的`项目.md`，一个formal cell同时绑定3个场景：

|阶段|prediction cell|scorer场景行|
|---|---:|---:|
|Stage2-C|300|900|
|Stage2-B|100|300|

本严格qKNN管线没有执行SSH、SCP或N607启动。仓库中并发出现的`adv3b02_three_da_leoweakonly_20260715_v1`记录的是另一条Landlock Stage2-B三方法比较路线；其运行事实不能替代本报告的strict qKNN runtime闭环，也不能计入本报告性能达标。

已知N607上的`bwrap`因user namespace权限不可用；因此正式qKNN下一步必须二选一并经过相同bundle复验与后验open ledger：

1. 将当前strict runner迁移到经验证的Landlock等价隔离；
2. 由管理员提供可用的unprivileged user namespace/bwrap执行条件。

在该问题解决前，严格qKNN Stage2-C保持`launch_authority=false`。

提交前独立代码审查还确认两个不能由当前Windows本地测试消除的Critical blocker：adapter/head/TTA生成provenance尚未绑定到外部candidate/plan trust root；普通宿主目录只读bind无法抵御同UID瞬时替换后恢复。代码已把二者写入`formal_launch_blockers`并将isolated runner总体状态降为`LOCAL_DIAGNOSTIC_PASS`，因此任何post-run字段都不得被解释为`PROTOCOL_VALID`或正式启动授权。

## 八、结果与下一步

### 8.1当前结果表

|candidate/run|机制|receiver/TX split|K|seed|old_acc|min old|seen-new|H|forgetting|adapter/资源|最终判定|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|qknnv42_phase2_strict_pipeline_20260715|协议与运行时诊断基础设施|未运行正式cell|—|—|—|—|—|—|—|7文件closure；adapter资源不可由prediction artifact证明|`LOCAL_DIAGNOSTIC_PASS_ONLY`；formal=false|

### 8.2下一步顺序

1. 将adapter/head/TTA生成provenance绑定外部candidate/plan trust root，并采用固定inode、不同UID只读所有权或等价不可变snapshot关闭输入TOCTOU；
2. 将现有Landlock等价隔离改造成消费同一pre-run bundle的strict executor，并在N607做1个非正式smoke cell；
3. 校验真实open ledger、prediction seal和独立scorer后，修复effective8 v14 plan/candidate lock；
4. 先运行Stage2-C K=1小矩阵，验证相对strict direct ADV3B02的正收益和paired CI；
5. 再扩展K=5/10/20及5/10/20新类，按共同candidate/run行报告accuracy、最低类和forgetting；
6. 达到协议、资源和性能门槛后才允许300-cell正式矩阵。

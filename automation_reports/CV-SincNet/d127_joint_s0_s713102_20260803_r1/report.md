# D127 S0固定格式预注册报告骨架

> 本文件是固定格式预注册骨架。所有尚未由冻结实现、运行交接或独立评分产生的字段必须保持`PENDING`，不得在实验前用推测值替换。

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`d127_joint_s0_s713102_20260803_r1`|
|时间戳|`2026-08-03`|
|阶段/矩阵|D127 joint，S0|
|当前状态|`LOCAL_IMPLEMENTING/NO_NEW_PERFORMANCE_RESULT`|
|协议模式|`p2_min_v1`|
|报告类型|固定格式preregistration骨架|

### 操作员分工

|角色|职责边界|
|---|---|
|Primary|Sol High：集成、协议解释、分析与最终晋级判断|
|Implementation/runner|Terra Max：冻结实现；唯一N607 runner；不得调参、改方法或重复启动|
|Mechanical skeleton|Luna：仅机械生成和维护本报告骨架；不得作科学判断、改代码或启动实验|

## 2.目标与假设

### Objective

在已冻结的D127 joint候选、Phase1部署bundle和一次验证的Phase2固定接收IQ上执行S0矩阵，验证冻结目标文档规定的三条方向性H/正确计数条件，并形成同一row的before/after预测、独立评分和证据闭合。当前仅完成预注册骨架，尚无性能结果。

### Hypothesis/comparison

- 假设：D127 joint候选在相同输入、相同S0矩阵和同一row配对规则下，按目标文档规定的三个方向性条件相对于预注册比较对象达到要求；精确方向、公式、阈值和判定标签必须从活动目标文档原文复制，当前均为`PENDING`。
- 比较对象：冻结目标文档指定的同协议、同矩阵比较对象；候选名称、实现commit、文件hash和比较规则为`PENDING`，不得以开发期最佳值或Oracle/clean-access结果替代。
- 结果边界：`LOCAL_IMPLEMENTING/NO_NEW_PERFORMANCE_RESULT`；实现落地、进程完成或诊断阴性证据均不自动构成可晋级性能结果。

## 3.冻结S0矩阵

|字段|冻结值|
|---|---|
|seed|`713102`|
|receiver|`{20-1,3-19,7-14}`|
|K-shot|`{1,5}`|
|new_count|`20`|
|正式场景|`{leo_clear_weak,leo_low_elev_weak,leo_rain_weak}`|
|receiver/TX split完整映射|`PENDING`（仅可从冻结目标文档填入）|
|完整row枚举及数量|`PENDING`|
|Phase1 receiver-held折叠|7个receiver-held折；每cell前5个为support、后9个为query；K1为K5前缀|
|class对称性|仅循环标签置换；不按具体class ID设置分支、权重或阈值|
|protocol_schema|`p2_min_v1`|
|phase2_data_status|`PENDING`（仅核对既有句柄，不重新验证）|

S0只运行上述冻结矩阵。不得新增、删减或重排receiver、TX、K、scene、seed；不得从局部row或有利结果外推完整矩阵。

## 4.输入资产与谱系

|资产|远端路径/标识|SHA256或receipt|
|---|---|---|
|当前Git基线|`d655775b`；补充commit`ade9e987`、`49d281dd`、`c496b5ee`；早期谱系`45485b18`、`fec8c14b`、`3d07db6e`|补充commit文件hash：`PENDING`|
|Phase1 checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|固定source/received IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|文件SHA256：`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`；receipt SHA256：`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|D92注册根目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|manifest SHA256：`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|D127 method lock|`configs/d127_joint_s0_method_lock_20260803.json`|`7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650`|
|qKNN真实绑定LODO|路径/manifest：`PENDING`|LODO SHA256：`7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08`；quantization receipt SHA256：`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|L_s label join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|capsule/split句柄|沿用已验证资产；精确`capsule_id`、`split_id`、allowlist和访问账本|`PENDING`（仅句柄核对，不触发数据重验）|

Phase2运行时只读取不可变deployment bundle、已验证固定received IQ、当前row合法support标签/注册表和算法配置。禁止clean/source样本、source feature/cache/replay、query真值、真实old/new/unknown角色、真实batch类别计数、class quota、Hungarian/optimal transport/global reassignment或任何跨query全局重排进入预测。

## 5. Before/after闭合要求

|闭合面|预注册要求|状态|
|---|---|---|
|Before registration|同一row、同一旧类query集合和同一逐样本规则生成不可变old-class预测；记录`old_acc_before`、覆盖/计数闭合及对应artifact|`PENDING`|
|After registration|冻结support更新/注册后，在同一row、同一旧类query及新类query口径下生成不可变预测；记录`old_acc_after`、`seen_new_acc`、`H_old_new`、floor、forgetting及对应artifact|`PENDING`|
|Prediction closure|先封存预测，再由独立scorer读取opaque query ID和真值；scorer不得回流任何适配、注册、阈值、选择或重跑决策|`PENDING`|
|Row closure|每个receiver/TX、scene、K、seed组合必须有成对before/after预测和独立评分，缺失即标记未闭合|`PENDING`|

## 6.唯一S0门槛

S0门槛仅包括活动目标文档规定的以下三条方向性H/正确计数条件；不添加其它性能、资源或有利子集门槛。精确文字必须在冻结目标文档确认后填入。

|Gate|条件（必须原文复制）|证据列|状态|
|---|---|---|---|
|S0-G1|方向性H条件1：`PENDING`|同一row的before/after独立评分|`PENDING`|
|S0-G2|方向性H条件2：`PENDING`|同一row的before/after独立评分|`PENDING`|
|S0-G3|方向性正确计数条件：`PENDING`|逐row预测/计数闭合|`PENDING`|

## 7.运行交接字段（均待填）

|字段|值|
|---|---|
|exact new Git commit|`PENDING`|
|changed-file hashes|`PENDING`|
|local-to-remote sync mapping|`PENDING`|
|N607 exact command|`PENDING`|
|Conda/Python environment|`PENDING`（默认环境名为`ssr-gpu`，以交接实值为准）|
|N607 CWD|`PENDING`|
|GPU allocation/occupancy snapshot|`PENDING`|
|remote log path|`PENDING`|
|main PID/child PID binding|`PENDING`|
|output/prediction/score paths|`PENDING`|
|expected artifacts and manifests|`PENDING`|
|health-check schedule and receipts|`PENDING`|
|focused protocol-negative tests|`PENDING`|
|real-checkpoint no-query smoke|`PENDING`|
|independent P0/P1 review|`PENDING`|

本骨架创建不运行项目测试、不使用SSH/SCP、不启动或停止N607实验；上述交接字段只能由实现验证和唯一runner在交接时填写。

## 8.停止规则与明确排除项

- 仅在出现P0协议/安全违规，或至少两个不同row在产生预测前出现同一确定性异常指纹时，停止派发并按run-owned PID绑定规则处理；必须保留已有日志、退出码和部分artifact。
- 绝不因低准确率、低H、floor、forgetting或其它性能值停止实验；低性能不是健康停止条件。
- 本S0不运行`588`、`fresh63`或`repeated125`，不新增或重跑其它矩阵。
- 不做Phase2数据重验证；只有固定received-IQ字节、物理ID、receiver/TX集合、scene分配、K、support/query划分或协议schema改变时才按控制规则重验。候选、adapter、超参、epoch、prototype/update rule、method lock、checkpoint推理状态或报告格式变化不触发数据重验。
- Oracle、clean/source-access、query-fit、role/quota/global-assignment或跨run最优值均不得进入正式S0结论。

## 9.同一row结果表（待运行后填写）

每一行必须保持候选、机制、矩阵、before/after、旧类/新类/unknown指标、覆盖与安全字段和最终判定的联合上下文；禁止把不同row的独立极值拼成一行。

|candidate ID|机制/category|receiver/TX split|scene|K-shot|seed|new_count|old_acc_before|old_acc_after|seen_new_acc|unknown_acc|H_old_new|min_old_acc|min_new_acc|forgetting|coverage|rollback|defer|loss/adapter summary|final verdict|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|

## 10.完成后回填区

|项目|状态/内容|
|---|---|
|最终运行状态（`LANDED`/`RUNNING`/`ARTIFACTS_COMPLETE`/`ANALYZED`或技术停止）|`PENDING`|
|启动、首波和完整日志证据|`PENDING`|
|prediction/score/coverage/archive闭合|`PENDING`|
|best epoch/checkpoint reference|`PENDING`|
|逐candidate/逐experiment同一row结果表|`PENDING`|
|异常、偏差与协议审计|`PENDING`|
|解释与晋级结论|`PENDING`|
|下一实验建议|`PENDING`|

**预注册结论：**当前仅登记冻结输入、S0矩阵、合法性边界和三条S0门槛，状态保持`LOCAL_IMPLEMENTING/NO_NEW_PERFORMANCE_RESULT`。任何性能结论须等完整同一row预测、独立评分和门槛证据闭合后再写入。

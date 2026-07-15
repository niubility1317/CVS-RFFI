# qKNNv42/ADV3B02极轻型Stage2-B/C优化报告

## 一、任务登记

|字段|内容|
|---|---|
|任务ID|`qknnv42_extreme_light_optimization_20260715`|
|日期|2026-07-15（Asia/Hong_Kong）|
|执行者|Codex|
|基底模型|`ADV3B02_CORE90_SOFT_E200`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|当前状态|`LOCAL_PROTOCOL_REPAIR_REQUIRED_FORMAL_LAUNCH_AUTHORITY_FALSE`|
|远端动作|无SSH、无SCP、无N607启动|

本任务使用同一份严格加载的ADV3B02 checkpoint构建candidate、identity-only和strict direct三条配对预测流。CEN51、JREF、OPGAC和OA-MSE不再拥有当前基底或默认路线权限，仅保留为历史对照。

## 二、目标与验收门槛

正式输入必须是5个目标receiver上的密封`leo_*_weak`support/query artifact；适配、校准、reference、prototype、门限、回滚和TTA决策同样不得读取clean或clean派生信号。输出必须先是truth-free sealed prediction，再由独立scorer连接truth生成指标。

|维度|正式要求|
|---|---|
|开发K值|仅K=10允许选adapter、head、epoch、超参数和TTA门限|
|确认K值|K=1/5/10/20；K=1/5/20不得回流选参|
|确认矩阵|5receiver×至少5个独立confirm seed×4个K×3个新类规模；每个prediction cell一次输出3个LEO场景，即300cell/900场景行|
|K10旧类|`old_acc≥92%`，`min_old_class_acc≥88%`|
|K10新类|5/10/20个seen-new分别≥92%/90%/86%|
|K5|相对matched K10各关键指标下降≤3pp|
|K1|相对identity-only适应收益≥0；相对strict direct ADV3B02总体及逐receiver≥+2pp，paired 95% CI下界>0|
|遗忘|K=1/5/10/20均不劣于matched identity-only|
|资源|训练参数≤50,000、适配≤20epoch、持久状态≤256KB、无dense query graph|
|推理|逐样本默认1-view；仅低置信度时自适应触发3-view或5-view|

禁止role Oracle、query真实批次类别数、类别quota、query标签拟合、Hungarian/OT/global assignment、dense query-query graph以及scorer反馈预测器。

## 三、方法与底层假设

当前主候选不是重新训练ADV3B02 backbone，而是在其冻结特征上接入≤50k参数的key-layer/late-feature低秩残差、support-only分类头和自适应多视图策略。其目标是把历史60epoch固定5-view路线拆成三个可审计部分：

1. 用小参数残差吸收target receiver与LEO_weak通道偏移，同时冻结大部分身份判别backbone，降低旧类灾难性遗忘。
2. 用K10 support-only监督、identity保持、类间margin和cross-view一致性联合损失训练；所有损失只使用support，不读取query标签或query图。
3. 第1个view先做逐样本全注册类预测；仅当margin低、entropy高或view disagreement超过由source validation/support确定的门限时，再追加至3/5-view，并记录每样本实际forward数。

K1的核心不是扩大adapter，而是收紧identity保持：adapter以identity初始化，使用旧类anchor蒸馏/feature cosine约束、残差范数约束和support augmentation一致性，避免1-shot噪声把ADV3B02原有边界拉坏。K1验收只能来自锁定candidate的独立确认结果。

## 四、输入与输出

### 4.1预测器输入

- 外部trust root绑定的ADV3B02 checkpoint、adapter/head和TTA policy SHA256；
- 5个target receiver、真实嵌套`Y_new^5⊂Y_new^10⊂Y_new^20`的sealed target package；
- 物理嵌套`K1⊂K5⊂K10⊂K20`support token及固定query token；
- 每个cell的3个密封LEO_weak场景artifact；
- 3基础+4clean不可达+5query决策共12字段运行时合同。

预测器不得收到TX truth、old/new role、`query_per_tx`、raw PKL路径、cache build spec或任何scorer参数。

### 4.2预测器输出

- 不可覆盖的`.cvspred`密封容器及payload→manifest→seal哈希链；
- candidate after/before、identity after/before、strict direct五路逐样本预测；
- 每样本实际view count、adapter训练日志、参数/状态/MAC/延迟/显存记录；
- 运行时open ledger和禁止路径命中统计。

独立scorer验证artifact SHA后才连接truth sidecar，输出old/new/H、逐类floor、K值遗忘、K1配对增益及95% CI。

## 五、历史证据重新分级

|资产|同row关键结果|资源/机制|当前协议判定|用途|
|---|---|---|---|---|
|历史legacy 92.28%H|old94.52%、new90.14%、H92.28%|60epoch`id_norm_late_feature`、固定5-view、FFT96、场景筛选及角色/类别配额约束|不同切分、20新类、单seed的legacy diagnostic；含当前禁止机制|只说明多视图与特征适配可能有效，不是正式基线|
|`full_nonoracle125`|125行历史诊断|E60 clean-derived特征、固定TTA5、FFT96、dense transductive query graph|`PROTOCOL_INVALID_FOR_PHASE2`|不得用于当前选参或性能声明|
|`nondense_adapter_epoch_sweep`|去除了dense/Oracle|manifest仍有raw clean repair、clean loss和ManyTx proxy unknown train|`PROTOCOL_INVALID_FOR_PHASE2`|不得用于当前选参|
|`idnorm_tta5_1000` E30|old73.133%、new63.793%、H66.762%；K1 H比baseline低2.284pp|289,685参数、579,370B FP16、固定5-view、仅2个新类|超资源且K1负收益；历史诊断|量化固定5-view/重adapter成本|
|support-only taskadapt E2|old71.347%、new58.207%、H63.140%，相对baseline H下降0.028pp|154参数、2epoch|很轻但无正收益、仅2个新类|负面对照|
|adaptive V11单cell|fixed5：old73.889%、min old41.667%、new75.500%、H74.686%；adaptive H73.547%|约31,200参数、自适应平均2.296 forward|缺严格runtime与完整矩阵，`UNVERIFIED_UNDER_CURRENT_PROTOCOL`|自适应TTA机制诊断|
|effective8 v14 source holdout|source base86.678%/floor70.498%；adaptive87.341%/floor71.169%；平均1.124 forward|44,048参数、12epoch、约88,096B adapter|资源外形PASS；target matrix未运行，strict接入缺失|当前唯一主候选骨架|
|ADV3B02 MRIOR旧Stage2-B|375行整体old82.58%；K1 77.22%、K10 85.82%、K20 87.74%；receiver3-19为69.06%|全backbone更新600次/row|旧schema、无target-new且暴露`query_per_tx`，非当前正式模板|重DA计算量/性能对照|

历史结果表明性能差距大的根因不是单一随机波动，而是切分、new-class数量、seed、clean-derived适配、固定5-view、FFT、场景筛选以及角色/类别配额共同改变了任务难度和可用信息。删除这些信息后，合法路线的当前可验证结果明显低于legacy 92.28%H，不能横向直接比较。

## 六、本轮控制修复

|文件|修改|验证|
|---|---|---|
|`analysis/qknnv42_extreme_light_stage2_traceability_20260715.md`|建立需求→实现→验证映射|Git commit `7a43be9`|
|active `stage2_prompt.md`|新增2026-07-15 ADV3B02/qKNNv42 override，撤销旧OPGAC/JREF默认权限|根/Git镜像待最终SHA复核|
|`tools/optimizer_workflow_contract.md`|固化K10开发、K1/K5/K20确认、资源/性能/隔离合同|根/Git镜像待最终SHA复核|
|`stage2_optimizer_state.json`|更新基底、12字段、门槛、容量和fail-closed状态|根/Git镜像字节一致；核心断言PASS|
|`tools/update_qknnv42_stage2_control_20260715.py`|幂等定点更新mutable state|`py_compile`与二次运行PASS|

控制面回归：`python -m pytest -q tests/test_monitor_optimizer_closed_loop_prompt.py code/tests/test_optimizer_workflow_tools.py`返回71/71通过。唯一告警为根目录`.pytest_cache`无写权限，不影响测试内容。根/Git镜像复核如下：

|artifact|SHA256|镜像|
|---|---|---|
|active `stage2_prompt.md`|`cc0f69c0ed98c6594ed8d6a0a558fac289ad2491135cc1427c498ffb32a9aed3`|字节一致|
|`optimizer_workflow_contract.md`|`723421cb9d2ee69afb254a241157bbbbc1804d31149abfbe0ca3641b020e3ff7`|字节一致|
|`stage2_optimizer_state.json`|`61848e2ea38337b9ba33282d844a915f0d7e3fa61b033607e498c8ff79fe43da`|字节一致|

修改前快照位于`E:\type10-7\code\snapshots\qknnv42_extreme_light_control_repair_20260715\`。根目录不是Git仓库，所有意图变更均镜像到`E:\type10-7\github_publish\CVS-RFFI-repo`承载。

## 七、当前阻塞与下一步

当前launchable candidate数量仍为0，不能直接去N607启动300cell矩阵。必须依次关闭：

1. 将effective8的44,048参数adapter、head和TTA policy绑定外部candidate/plan trust root并接入唯一strict request builder。
2. 在Phase2外构建25份真实ManyTx target package，完成逐TX样本覆盖、固定query ID、`K1⊂K5⊂K10⊂K20`和`Y_new^5⊂Y_new^10⊂Y_new^20`密封证据。
3. 从Phase2 config/request/runtime删除`query_per_tx`、truth、role和build spec；一次prediction cell同时输出3个场景。
4. 关闭immutable snapshot/TOCTOU缺口，在N607通过真实Landlock等价strict smoke并生成post-run open ledger。
5. 只运行K10开发证据并锁定candidate；随后执行300cell/900场景行独立确认矩阵。

本报告当前是设计、审计和控制修复记录，不是性能达标声明，也不是N607部署成功证据。

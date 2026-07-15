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
|当前状态|`SOURCE_FFT_ABLATION_COMPLETE_WEIGHT2_ONLY_PASS_TARGET_MATRIX_UNRUN`|
|远端动作|N607 source-only FFT消融4/4完成；进程与GPU已释放；未启动target矩阵|

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

## 七、ADV3B02/effective8严格接入实现

本轮把主候选明确为“地面effective8 LoRA＋星上零梯度support校准＋自适应1→3→5-view”。ADV3B02原checkpoint不在目标侧更新；地面只更新8个直接影响`z_id/feat_joint`的Linear层：`t_proj`、`f_proj`、`pa_proj.0`、`fuse.0`以及classifier中的`id_proj.0`、`pa_proj.0`、`id_gate.0`、`joint_proj.0`。rank16 LoRA共16个A/B张量、44,048个训练参数，训练12epoch。

### 7.1输入、处理与输出

|环节|输入|方法|输出|
|---|---|---|---|
|地面导出|固定SHA的ADV3B02 checkpoint、source LEO_weak训练得到的FP16 effective8 LoRA|严格重建两份同源模型；一份保持base，另一份注入LoRA后合并到8个Linear；对injected、merged和TorchScript逐张量比对|`base_runtime.ts`、`candidate_runtime.ts`、parity receipt|
|candidate capsule|candidate lock、两份TorchScript、LoRA state、training manifest、source feature stats、head lock、TTA policy|重新核对candidate lock自哈希、不可变artifact/cache/code哈希、8层/16张量、参数/epoch/字节以及禁止权限|包外可指定SHA256的candidate capsule|
|星上support登记|K-shot注册support的3个LEO_weak场景base view|不做target backbone反向传播；使用统一全局对角对齐、稳健原型和Gram score transform构建闭式head|每个注册类的FP16 prototype、`C×C`score transform及可选alignment状态|
|逐query推理|单个truth-free LEO_weak query、全部注册类、source锁定TTA门限|先执行1-view；仅当top-2 margin或3-view disagreement未过门限时追加至3/5-view|candidate/identity/direct预测及每样本实际view count|

### 7.2资源口径

评测包总大小、评测峰值资源和星上持久增量是三个不同指标。双TorchScript与密封support/query使评测包必然大于256KiB；256KiB约束只用于“预装ADV3B02、由delta重建candidate、且不额外长期保存merged完整副本”的星上增量状态。

|资源项|5个新类`C=11`|10个新类`C=16`|20个新类`C=26`|
|---|---:|---:|---:|
|LoRA FP16张量payload|88,096B|88,096B|88,096B|
|闭式head FP16状态|6,920B|9,760B|15,740B|
|6个TTA float32门限|24B|24B|24B|
|理论张量payload合计|95,040B|97,880B|103,860B|

candidate capsule同时记录`adapter_serialized_file_bytes`。正式资源判定使用“真实序列化delta文件＋head＋24B门限”，不能用88,096B理论tensor payload替代实际部署文件大小。真实v14 adapter尚未拉取到本地，因此当前只能确认理论payload小于256KiB；真实序列化增量仍是待验证项。

### 7.3本地验证结果

|检查|结果|含义|
|---|---|---|
|`py_compile`|3个新增实现文件通过|语法与导入路径可用|
|candidate capsule测试|6/6通过|外部trust root、层集合、字节重算、candidate权限和merged副本错误均能fail closed|
|TorchScript trace测试|1/1通过|测试模型的`z_id/logits`在eager与trace间逐值一致|
|strict runtime测试|6/6通过|nested K、五路预测、FFT96确定性以及50k/20epoch/256KiB越界拒绝生效|
|联合测试|13/13通过|仅证明本地实现合同；不代表真实ADV3B02/effective8 artifact parity或性能达标|

TorchScript测试出现PyTorch对`torch.jit.trace`的弃用提示，但当前严格runtime使用`torch.jit.load`，该提示不改变测试通过结论。后续可以迁移到`torch.export`，迁移前必须保持相同的文件描述符加载、哈希和数值parity合同。

### 7.4本轮qKNN性能机制增量

本轮的工程重点已转到qKNN性能，不再继续扩展非必要协议握手。所有新机制均只使用source validation或注册support，不读取query批次统计、真实类别数、old/new角色或类别配额。

|机制|底层问题|实现|额外星上状态/计算|预期作用|证据状态|
|---|---|---|---|---|---|
|`consensus67`稳健原型|K=1时单个物理support的某个接收View可能成为离群点|每类保留平均两两相似度最高的`ceil(2V/3)`个观测后求球面中心|不增加状态；support登记阶段增加小规模`V×V`相似度|降低单View异常对K1原型方向的牵引|本地机制测试通过；真实性能待测|
|有界partial Gram|原始`(G+λI)^{-1}`在原型高度相关时会放大弱特征方向|Gram逆的谱增益限幅至`[0.5,2.0]`，再用`mix∈{0.25,0.5,1.0}`与identity插值|head仍为`C×C`；每query至多958 MAC|改善相邻类边界，同时避免K1数值放大导致最低类崩塌|本地机制测试通过；真实性能待测|
|support不确定性偏置|不同类的注册View稳定性不同|用类内View到原型的余弦离散度生成小偏置；`β=0`始终保留|新增`C`个FP16标量|降低不稳定support对query的过度吸引|本地机制测试通过；源域三重identity保护已启用|
|源域三重identity保护|只最大化平均准确率可能牺牲最差episode或最低类|候选必须同时满足平均准确率、最差episode准确率、平均最低类准确率均不低于identity|仅地面锁定开销|防止K1平均收益以旧类floor退化为代价|本地选择逻辑测试通过|
|部署匹配multi-view worst-K损失|训练先平均两个View，部署却把多View作为相关support观测，目标不一致|地面LoRA训练直接输入`[V,N,D]`；support跨View/物理shot建原型，query逐View交叉熵，K间用log-sum-exp最坏风险|参数仍44,048、epoch仍12；仅地面每步约增加至原来1.5—2倍的轻量head运算|优先改善K1，同时约束K=5/10/20遗忘|36项定向测试通过；需要重训实测|
|性能优先自适应View|仅靠margin可能让“高margin但全类绝对相似度很低”的样本错误停在1-view；简单均值会受振荡View影响|1-view同时检查margin和top1 score；3/5-view使用`mean(score)-β·std(score)`稳定性下置信界；源域选择先最大化准确率，再最小化forward|TTA状态由12B增至24B；默认仍1-view|让额外View集中到低置信度或跨View不稳定样本，避免固定5-view|85项联合回归通过；实际forward/准确率待测|
|FFT能量重平衡|历史`[z_id,2·z_fft]`归一化后FFT能量占80%，`z_id`和LoRA收益被稀释|增加`fft_weight={0.5,0.7,1.0,2.0}`源域消融；对应FFT能量约20%、32.9%、50%、80%|不增加参数、状态或backbone forward|先恢复ADV3B02身份特征主导，再检验FFT是否提供互补增益|CLI与能量审计测试通过；N607消融待运行|

这里的优先假设是：K=1正收益首先取决于“保住ADV3B02边界并让support原型更稳”，而不是继续扩大adapter。FFT权重和head规则均必须在source receiver holdout上锁定；target K1/K5/K20结果只能用于独立确认，不能反向选参。

本轮完整定向回归为85/85通过，覆盖candidate capsule、TorchScript、strict runtime、FFT能量映射、K1 head、自适应TTA、source锁定、candidate lock、benchmark及地面遗忘损失。该结果仍只证明实现一致性，不等同于准确率达标。

## 八、性能优先实验队列与最低合规边界

### 8.1立即执行的性能实验

1. 复用同一ADV3B02 checkpoint和同一v14 effective8 adapter，先在source receiver holdout运行`fft_weight={0.5,0.7,1.0,2.0}`。联合排序以K1平均准确率为第一目标，同时硬约束最差episode和最低类不低于identity；并记录K=5/10/20同头结果与平均forward。
2. 用胜出的FFT权重对比三种训练目标：identity adapter、原`z_rep`均值worst-K、部署匹配multi-view worst-K。训练参数固定44,048，epoch优先采用8/12/16，不超过20。
3. 在source锁定FFT、epoch、head和TTA门限后，只先跑5receiver的K10开发矩阵；达不到旧类floor时，优先调整identity/relation权重和Gram mix，不增加adapter层数。
4. K10锁定后再一次性运行K=1/5/20确认。K1若不能稳定超过strict direct ADV3B02，则该候选直接判负，不以其他K值补偿。

### 8.2最低必要合规工作

当前launchable candidate数量仍为0，不能直接去N607启动300cell矩阵。必须依次关闭：

1. source-only FFT/训练损失消融属于性能诊断，可以先运行；结果不得写成target部署成功声明。
2. 正式target K10运行前只关闭会影响输入、输出、资源和禁止Oracle结论的必要缺口；不再扩展与qKNN性能无关的握手机制。
3. K10锁定后执行K=1/5/20独立确认；完整300cell/900场景行仍由独立scorer给出最终结论。

本报告当前是设计、审计和控制修复记录，不是性能达标声明，也不是N607部署成功证据。

## 九、N607 source-only FFT权重消融启动登记

|字段|内容|
|---|---|
|实验ID|`qknn_ground_effective8_fft_source_ablation_20260715_v15`|
|目标|在不重训adapter、不增加参数和forward的条件下，检验FFT96权重从2.0降至0.5/0.7/1.0能否改善K1及旧类floor|
|比较对象|同一ADV3B02 checkpoint、同一v14 effective8 adapter、同一source receiver holdout；仅`fft_weight`变化|
|运行性质|source-only性能诊断；不得直接promotion或写成target结论|
|本地版本|Git commit `3d64252`＋自适应约束修复commit `44baec3`|
|本地验证|主回归85/85通过；审查后TTA约束定向回归39/39通过|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|Conda环境|`CVS-RFFI`|
|GPU计划|weight0.5/0.7/1.0/2.0分别使用GPU0/1/2/3；每GPU1个验证进程|
|输出根|`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/qknn_ground_effective8_fft_source_ablation_20260715_v15/`|
|日志根|`logs/qknn_ground_effective8_fft_source_ablation_20260715_v15/`|
|预期输出|每个权重独立的`validation_result.json`、source feature stats、head/TTA lock与launch manifest|

### 9.1启动前证据

- 2026-07-15 20:07+08:00直接SSH preflight通过；N607项目根和8张RTX3090可见。
- live inventory显示`active_training_processes=[]`，8张GPU均约10MiB占用、0%利用率。
- v14 adapter存在，文件大小94,054B；training manifest与source-validation cache set均存在。
- 独立审查指出：新的multi-view worst-K训练仍需按physical ID排除同源场景副本，因此本次不启动重训，只复用旧v14 adapter做FFT source诊断。
- 本轮禁用source C=6绝对top1阈值，只消融margin＋std-LCB；TTA候选在排序前必须满足平均forward≤3.0且额外View率≥5%。

### 9.2同步与启动命令

待同步文件：

|本地文件|N607目标|
|---|---|
|`paper_reproduction/cvs_aligned/extreme_light_adapter.py`|同相对路径|
|`paper_reproduction/cvs_aligned/k1_symmetric_head.py`|同相对路径|
|`paper_reproduction/cvs_aligned/adaptive_rxlight_tta.py`|同相对路径|
|`paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py`|同相对路径|
|`paper_reproduction/scripts/launch_cvs_ground_lora_fft_ablation_v15.sh`|同相对路径|

服务器启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash paper_reproduction/scripts/launch_cvs_ground_lora_fft_ablation_v15.sh
```

成功启动必须记录4个独立PID、GPU、日志和输出目录；启动后仅作短连接只读监控，不保留SSH会话。

### 9.3实际启动结果

2026-07-15 20:13:07+08:00启动成功。launcher与GPU worker是父子进程，因此PID分别记录：

|FFT权重|GPU|launcher PID|GPU worker PID|启动显存|日志|
|---:|---:|---:|---:|---:|---|
|0.5|0|1773278|1773312|582MiB|`logs/qknn_ground_effective8_fft_source_ablation_20260715_v15/w0p5.log`|
|0.7|1|1773279|1773313|582MiB|`logs/qknn_ground_effective8_fft_source_ablation_20260715_v15/w0p7.log`|
|1.0|2|1773280|1773314|582MiB|`logs/qknn_ground_effective8_fft_source_ablation_20260715_v15/w1p0.log`|
|2.0|3|1773281|1773315|582MiB|`logs/qknn_ground_effective8_fft_source_ablation_20260715_v15/w2p0.log`|

四份日志均已创建。启动初期只有`numpy.core._internal`弃用告警，没有Traceback、OOM或artifact缺失。启动复核后本地到N607及bridge的`ESTABLISHED`连接数为0。

## 十、启动后历史对话复盘

对话索引已于2026-07-15重新构建，共收录977条`E:\type10-7`相关记录。本轮重点复核了qKNNv42报告、V42 K×新类矩阵、V92 support anchor和Oracle硬化历史，并与当前对话逐条对照。

### 10.1用户引导与当前落实状态

|用户持续要求|历史含义|当前落实|仍需完成|
|---|---|---|---|
|以汇报形式讲清方法、输入、输出和效果|qKNN是ADV3B02之上的Stage2-C注册头，不是新backbone|本报告已分开方法、I/O、资源、历史结果和证据边界|实验完成后补同row结果表，不以单项最大值代替候选|
|解释历史94.52/90.14/92.28与后续125运行差距|历史行叠加60epoch adapter、固定5-view、FFT96、场景/角色/配额信息，且切分、类数和seed不同|已降级为legacy diagnostic，禁止作为当前目标基线|新实验必须与strict direct/identity在同support/query上配对|
|固定ADV3B02作为基底|不能切回CEN51、JREF、OPGAC等旧路线|checkpoint和SHA已固定；本次消融复用同一v14 adapter|真实性能表继续记录同一checkpoint SHA|
|压缩60epoch adapt，只更新关键层|允许地面训练小模块，星上不做重backbone训练|effective8 rank16仅44,048参数、12epoch；目标层集中在`z_id/feat_joint`路径|修复physical-ID episode后重训8/12/16epoch对比|
|多View是性能关键，改为低置信度自适应1→3→5|不能简单删除5-view，也不能固定5次forward|已实现margin＋std-LCB，forward≤3作为候选可行域；本轮禁用C=6绝对score门限|加入K1与最低类非退化约束后再锁TTA|
|重点提升1-shot旧类适应，K=1明显优于direct ADV3B02|其他DA在1-shot常为负收益，qKNN必须给出正贡献|加入稳健原型、bounded Gram、identity三重保护；K1仍是第一排序目标|当前source旧证据仅约+0.404pp，离+2pp目标仍远；需FFT实测和无泄漏重训|
|同时优化不同K的遗忘率|不能只报K10或平均H|source锁定同一head并输出K=1/5/10/20 identity对照|target确认矩阵尚未运行；需同rowforgetting ledger|
|禁止角色Oracle与类别配额|逐样本面对全部注册类|runtime、报告和历史结果分级均已禁止|不得为提升性能重新引入old/new bias或query类别数|
|理论分析ADV3B02哪些层最适合、损失如何设计|应从模型结构和旧类边界解释，不只扫epoch|effective8选择8个直接影响`z_id/feat_joint`的Linear；loss含identity/cosine/margin/relation/Gram/worst-K/view consistency|multi-view worst-K当前只是multi-rx surrogate，未完成三场景physical配组|
|把重点放在qKNN性能，而非协议握手|满足项目底线即可|已先启动零新增参数的FFT性能消融；formal仍保持false|后续开发时间优先用于K1/floor/TTA/loss，不扩展非必要合同|

### 10.2复盘发现的未闭合技术项

1. 新增multi-view worst-K loss尚未接收physical ID。同一物理样本的不同场景副本在全局shuffle后可能分落support/query，形成虚假近邻；因此本次没有用新loss重训。下一版必须按physical ID分组并排除同源query，且报告措辞改为`multi-rx-view surrogate`，直到真实三场景配组完成。
2. `consensus67`在3-view leave-one-out时每折只剩2-view，与mean等价，现有source锁参无法真正选择它。应改为完整3-view建头、episode外独立source physical query评分。
3. TTA虽已把forward上限纳入可行域，但尚未把K1准确率、K1最低类以及其余K非退化作为候选硬约束。仅拼接四个K最大化总体准确率仍可能牺牲K1。
4. FFT权重目前只贯通source validator。若0.5/0.7/1.0胜出，还需把权重封存进candidate lock，并由正式benchmark从lock读取，不能直接改一个全局常量后声称promotion。
5. 6个TTA float32门限应统一按24B统计；candidate capsule与本报告已修正，但旧training/benchmark资源字段仍有12B残留，需要在下一性能提交顺手统一。

这些项都直接影响qKNN性能结论或K1可信度，优先级高于新增数据协议握手。

## 十一、FFT权重消融完成结果

四个验证进程均于2026-07-15 20:13:40+08:00前结束。完整日志各523—526行，均已逐文件扫描；无Traceback、OOM或Killed。weight0.5/0.7/1.0的`conda run`返回非零是验证器对`locked_head_each_k_not_worse_than_identity`判负后的预期退出，不是运行崩溃。weight2.0正常返回且全部gate通过。

本地拉取证据位于`E:\type10-7\automation_reports\CV-SincNet\qknnv42_extreme_light_optimization_20260715\remote_artifacts_fft_v15\`，包含4份完整日志、4份`source_validation.json`、4份`promotion_manifest.json`、source feature stats及launch manifest。

### 11.1同权重总体结果

下表所有指标来自同一权重行；`base1`是未合并LoRA的ADV3B02 frozen feature head，`LoRA1`是同权重单View，`LoRA adaptive`使用对应source锁定TTA。这里只是6个source旧类receiver holdout，不是target 5/10/20新类结论。

|FFT权重|FFT能量占比|source gate|base1 acc/floor|LoRA1 acc/floor|LoRA adaptive acc/floor|平均forward|source选中head|判定|
|---:|---:|---|---:|---:|---:|---:|---|---|
|0.5|20.0%|FAIL|86.563%/70.498%|86.678%/69.349%|87.024%/69.828%|1.293|alignment＋uncertainty0.25|K1及多K稳定性不足|
|0.7|32.9%|FAIL|86.621%/70.498%|86.621%/69.349%|86.995%/70.881%|1.222|uncertainty0.25|K5/K20轻微低于identity|
|1.0|50.0%|FAIL|86.563%/70.498%|86.621%/69.349%|87.140%/71.456%|1.263|ridge0.03＋mix0.25＋uncertainty0.25|K20 fixed1低于identity0.058pp|
|2.0|80.0%|PASS|86.678%/70.498%|86.794%/69.349%|87.630%/71.552%|1.339|identity mean head|当前source winner；未证明target|

### 11.2同row nested-K结果

`Δfixed1`和`Δadaptive`均相对同权重、同K的identity mean fixed1；floor差异同样保持在该行内。由于本实验只有source旧类，表中的差值是source nested-K非退化诊断，不等同于正式target遗忘率。

|FFT权重|K|locked fixed1 acc/floor|identity fixed1 acc/floor|Δfixed1 acc/floor|adaptive acc/floor|Δadaptive acc|平均forward|判定|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0.5|1|86.390%/68.582%|86.448%/69.732%|-0.058/-1.149pp|86.275%/69.732%|-0.173pp|1.300|FAIL|
|0.5|5|86.505%/69.349%|86.678%/70.498%|-0.173/-1.149pp|87.024%/70.498%|+0.346pp|1.291|fixed1 FAIL|
|0.5|10|86.563%/69.349%|86.678%/70.115%|-0.115/-0.766pp|87.313%/70.115%|+0.634pp|1.288|fixed1 FAIL|
|0.5|20|87.082%/69.349%|86.967%/69.732%|+0.115/-0.383pp|87.486%/68.966%|+0.519pp|1.293|floor FAIL|
|0.7|1|86.505%/70.115%|86.448%/69.732%|+0.058/+0.383pp|86.448%/70.068%|+0.000pp|1.213|K1无明确增益|
|0.7|5|86.678%/70.498%|86.736%/70.881%|-0.058/-0.383pp|87.082%/71.264%|+0.346pp|1.216|fixed1 FAIL|
|0.7|10|86.621%/70.115%|86.621%/70.115%|+0.000/+0.000pp|87.024%/71.264%|+0.404pp|1.221|PASS row|
|0.7|20|86.851%/69.732%|86.967%/69.349%|-0.115/+0.383pp|87.428%/70.115%|+0.461pp|1.236|fixed1 FAIL|
|1.0|1|86.505%/69.732%|86.448%/69.732%|+0.058/+0.000pp|86.505%/70.408%|+0.058pp|1.257|正增益过小|
|1.0|5|86.736%/70.881%|86.736%/70.881%|+0.000/+0.000pp|87.255%/72.031%|+0.519pp|1.260|PASS row|
|1.0|10|86.736%/70.498%|86.736%/70.498%|+0.000/+0.000pp|87.255%/72.031%|+0.519pp|1.258|PASS row|
|1.0|20|87.082%/70.115%|87.140%/69.349%|-0.058/+0.766pp|87.543%/70.498%|+0.404pp|1.278|fixed1 FAIL|
|2.0|1|87.197%/69.732%|87.197%/69.732%|+0.000/+0.000pp|87.255%/72.031%|+0.058pp|1.317|PASS，但远低于+2pp|
|2.0|5|87.428%/71.264%|87.428%/71.264%|+0.000/+0.000pp|87.716%/72.414%|+0.288pp|1.325|PASS|
|2.0|10|87.140%/70.498%|87.140%/70.498%|+0.000/+0.000pp|87.716%/72.031%|+0.577pp|1.330|PASS|
|2.0|20|87.313%/69.732%|87.313%/69.732%|+0.000/+0.000pp|87.832%/69.732%|+0.519pp|1.384|PASS|

### 11.3结论与下一轮决策

1. “FFT占80%稀释LoRA，所以降低FFT权重会直接提升K1”的假设在当前source holdout上被否定。weight2.0同时给出最高总体adaptive accuracy、最高总体floor和唯一完整source PASS；不能把0.5/0.7/1.0推进candidate lock。
2. weight2.0的K1 adaptive为87.255%、floor72.031%、平均1.317次forward。它相对同权重identity fixed1只增加0.058pp；相对source base1增加0.577pp，仍明显低于用户要求的+2pp及paired CI下界>0。
3. source选择最终退回identity mean head，说明当前`consensus67/partial Gram/uncertainty`组合没有形成稳健正收益。低权重分支选中的复杂head反而在某些K破坏identity非退化。
4. 自适应View本身有持续小幅正收益，且平均forward仅1.21—1.38，证明“默认1-view＋低置信度追加View”方向成立；但当前TTA排序仍需显式加入K1 accuracy/floor约束。
5. 下一轮不再继续扩大FFT权重网格。优先修复：完整3-view建头＋episode外query选择`consensus67`、按physical ID构建无泄漏multi-scenario训练episode、K1/floor受约束TTA。FFT权重暂锁2.0作为source基线，待新的head/loss独立消融后再判断target适用性。

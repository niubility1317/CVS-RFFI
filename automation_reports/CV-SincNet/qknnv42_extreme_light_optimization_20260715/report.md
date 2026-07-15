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
|当前状态|`SOURCE_ADAPT_LAYER_ABLATION_COMPLETE_P4_SELECTED_TARGET_BPJG_UNRUN`|
|远端动作|N607关键层/损失消融8/8完成；进程与GPU已释放；未启动target BP-JG矩阵|

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


## 十二、当前优先级切换：ADV3B02 adapt关键层/损失消融

用户已明确要求先把重点放在adapt，因此暂停继续扩大qKNN head、Gram和TTA门限搜索。现有证据表明，target support侧20epoch四层feature-head LoRA能显著改善单cell，但旧类floor仍仅约51.67%；扩大到late＋head、hard-class DRO和cross-view CE均无联合收益。下一轮先在source receiver holdout上回答“哪些层最值得更新、8epoch能否替代12/60epoch、K1保护损失是否有效”，再决定是否进入target support快速适配。

### 12.1实现变化

- formal ground LoRA新增三种可比较层组：
  - `projection_feature`：`t_proj/f_proj/pa_proj.0/fuse.0`，rank16为18,448参数；
  - `feat_joint`：`id_proj.0/pa_proj.0/id_gate.0/joint_proj.0`，rank16为25,600参数；
  - `effective_feature`：上述8层，rank8为22,024参数、rank16为44,048参数。
- 所有配置仍冻结ADV3B02原始参数与CosFace；LoRA恒等初始化，训练后可合并，星上持续query不增加MAC。
- `nested_k_worst_prototype_risk`现按`sample_id`选择唯一物理support，并从query排除该物理样本的所有场景副本，消除同物理样本泄漏；当前仍准确表述为`multi_rx_view_surrogate`，不冒充完整三场景registered pairing。
- 训练统一压到8epoch；相对历史60epoch减少86.7%，相对v14 12epoch再减少33.3%。

本地`ssr-gpu`验证：两个脚本`py_compile`通过，support LoRA与ground forgetting定向测试34/34通过，`bash -n`及`git diff --check`通过。

### 12.2预注册8路矩阵

|候选|层组|rank|参数|epoch|损失profile|GPU|
|---|---|---:|---:|---:|---|---:|
|`p4_r16_e8_std`|前4投影层|16|18,448|8|v14保守权重|0|
|`h4_r16_e8_std`|后4 feature-head层|16|25,600|8|v14保守权重|1|
|`e8_r8_e8_std`|全部8层|8|22,024|8|v14保守权重|2|
|`e8_r16_e8_std`|全部8层|16|44,048|8|v14保守权重|3|
|`p4_r16_e8_k1`|前4投影层|16|18,448|8|K1边界保护增强|4|
|`h4_r16_e8_k1`|后4 feature-head层|16|25,600|8|K1边界保护增强|5|
|`e8_r8_e8_k1`|全部8层|8|22,024|8|K1边界保护增强|6|
|`e8_r16_e8_k1`|全部8层|16|44,048|8|K1边界保护增强|7|

保守profile沿用v14的relation/prototype-Gram/worst-K/multiview权重`0.5/0.25/0.5/0.25`。K1增强profile为`1.0/0.5/1.0/0.5`，同LEO teacher anchor仍保持22，reference margin由7.5提高到10，避免以放松旧边界换取表面K1收益。两种profile均固定FFT96权重2.0，不读target、query、clean、old/new角色或类别配额。

实验ID：`qknn_ground_adapt_layer_loss_ablation_20260715_v16`。远端输出根为`runs/qknn_ground_adapt_layer_loss_ablation_20260715_v16/`，日志根为`logs/qknn_ground_adapt_layer_loss_ablation_20260715_v16/`。启动脚本为`paper_reproduction/scripts/launch_cvs_ground_lora_adapt_ablation_v16.sh`。每路依次执行source-only训练与同一receiver holdout验证；主排序先看K1 fixed1/adaptive准确率和最低类，再检查K=5/10/20不退化与平均forward。该矩阵是source诊断，不是target达标声明。

### 12.3下一target adapt候选

若source层组消融证明final feature-head仍是有效位置，下一唯一target快速适配候选为`BP-JG-LoRA`：仅对合并后的ground model之`id_gate.0`和`joint_proj.0`注入rank8 LoRA，共6,400参数、FP16 delta 12,800B；统一5epoch，SGD无momentum，K1自然5步、K10自然50步。损失使用跨View prototype CE、全注册类margin不下降、feature anchor、prototype Gram保持/去混淆和轻量View一致性；所有项对全部注册类对称，不使用query、role或quota。该机制当前仅为待实现/待验证假设，不写成性能结论。


### 12.4启动前证据与同步

- Git版本：`11ea241`；本地分支`codex/cvs-rffi-release-20260626`。
- 本地`ssr-gpu`：34/34定向测试通过，两个训练脚本`py_compile`、launcher `bash -n`和`git diff --check`均通过。
- 2026-07-15 20:35+08:00 direct N607 preflight通过；项目根、时间和8张RTX3090可见。
- live inventory为`active_training_processes=[]`、`gpu_compute=[]`；8张GPU均10MiB、0%利用率。
- local-first同步后SHA256逐文件一致：
  - `train_apply_phase1_iq_preadapter_20260703.py`：`193f20474cea8b4347f399337a4d2881ce1bd9f52526e6e9f88c5f4dc3ae8fe5`；
  - `train_export_cvs_support_lora_adapter.py`：`563b90126df602d86ff7cce0831214f06e8eedf8cc58c2a0323da6171d89c7d8`；
  - `launch_cvs_ground_lora_adapt_ablation_v16.sh`：`7d91bd967327834641e4d23820bc6942857dc2f93cd463827b911929ea3ebbb0`。
- 远端`CVS-RFFI`环境下`py_compile`和`bash -n`通过；run/log目标根均不存在，满足不覆盖约束。
- 唯一启动命令：
  ```bash
  cd /home/szu2070436088/2510044040/CV-SincNet
  bash paper_reproduction/scripts/launch_cvs_ground_lora_adapt_ablation_v16.sh
  ```

## 十三、v16层组消融完成与高效域适应设计锁定

### 13.1实验完成状态与日志审计

实验`qknn_ground_adapt_layer_loss_ablation_20260715_v16`已完成8/8。8份完整日志共7,696行，其中4份`effective_feature`日志各1,002行，2份`feat_joint`和2份`projection_feature`日志各922行；逐文件扫描`Traceback/RuntimeError/OOM/Killed/Segmentation fault/NaN/Inf`均为0命中。8个source validation均`PASS`，远端训练进程和GPU已释放。

本地完整artifact位于`E:\type10-7\automation_reports\CV-SincNet\qknnv42_extreme_light_optimization_20260715\remote_artifacts_adapt_v16\`。该实验只使用source `leo_weak` receiver holdout，是关键层选择证据，不是target receiver、target-new或MRIOR胜出证据。

### 13.2八路同row结果

总体表中的`base1`、`LoRA1`和`adaptive`来自同一candidate的source holdout；floor是同row最低类准确率。所有candidate固定FFT96权重2.0和8epoch。

|candidate|层组/损失|参数|base1 acc/floor|LoRA1 acc/floor|adaptive acc/floor|adaptive相对base1|平均forward|source gate|结论|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|`p4_r16_e8_k1`|`projection_feature`/K1保护|18,448|86.678%/70.498%|87.140%/69.732%|**87.803%/71.839%**|**+1.125pp**|1.364|PASS|综合首选；最少参数且最高adaptive均值|
|`p4_r16_e8_std`|`projection_feature`/保守|18,448|86.678%/70.498%|86.563%/69.732%|87.486%/71.839%|+0.808pp|1.332|PASS|计算最稳，但单View负迁移|
|`h4_r16_e8_k1`|`feat_joint`/K1保护|25,600|86.678%/70.498%|86.736%/70.115%|87.644%/71.743%|+0.966pp|2.850|PASS|额外View过多，K20 floor退化|
|`h4_r16_e8_std`|`feat_joint`/保守|25,600|86.678%/70.498%|86.678%/70.498%|87.529%/72.222%|+0.851pp|1.339|PASS|稳定但不优于P4|
|`e8_r8_e8_k1`|`effective_feature`/K1保护|22,024|86.678%/70.498%|86.736%/69.349%|87.673%/71.264%|+0.995pp|1.352|PASS|K1 nested adaptive出现-0.058pp|
|`e8_r8_e8_std`|`effective_feature`/保守|22,024|86.678%/70.498%|86.736%/70.498%|87.529%/72.222%|+0.851pp|1.361|PASS|floor较好但均值无优势|
|`e8_r16_e8_k1`|`effective_feature`/K1保护|44,048|86.678%/70.498%|86.794%/69.349%|87.529%/72.031%|+0.851pp|1.356|PASS|参数增加2.39倍，无性能回报|
|`e8_r16_e8_std`|`effective_feature`/保守|44,048|86.678%/70.498%|86.794%/69.349%|87.572%/71.743%|+0.894pp|1.338|PASS|参数增加2.39倍，无性能回报|

嵌套K表中的单元格为`adaptive acc/floor/ΔA`；`ΔA`只表示adaptive View相对同一适应特征上的identity mean head，不是相对strict direct ADV3B02的增益。

|candidate|K1|K5|K10|K20|关键异常|
|---|---:|---:|---:|---:|---|
|`p4_r16_e8_k1`|87.543%/72.797%/+0.173pp|87.947%/72.414%/+0.461pp|87.889%/72.414%/+0.231pp|87.832%/69.732%/+0.000pp|K20无adaptive增益，但未出现均值负增益|
|`p4_r16_e8_std`|87.082%/71.429%/+0.000pp|87.601%/72.797%/+0.519pp|87.659%/72.031%/+0.634pp|87.601%/70.115%/+0.231pp|K1无增益|
|`h4_r16_e8_k1`|87.659%/72.797%/+0.346pp|87.716%/72.414%/+0.519pp|87.543%/72.797%/+0.288pp|87.659%/68.966%/+0.231pp|平均forward从K1的2.721升至K20的3.031|
|`h4_r16_e8_std`|87.140%/69.728%/+0.115pp|87.601%/72.797%/+0.750pp|87.716%/72.797%/+0.692pp|87.659%/69.732%/+0.231pp|K1 floor低于identity|
|`e8_r8_e8_k1`|87.370%/71.648%/**-0.058pp**|87.716%/72.031%/+0.346pp|87.832%/72.031%/+0.404pp|87.774%/69.349%/+0.173pp|K1 adaptive负增益|
|`e8_r8_e8_std`|87.082%/71.429%/+0.058pp|87.486%/72.797%/+0.461pp|87.659%/72.797%/+0.461pp|87.889%/70.498%/+0.519pp|均值稳定但不优于P4|
|`e8_r16_e8_k1`|87.024%/70.748%/+0.173pp|87.601%/72.414%/+0.461pp|87.716%/73.180%/+0.346pp|87.774%/70.115%/+0.231pp|参数最大|
|`e8_r16_e8_std`|87.197%/71.769%/+0.115pp|87.543%/72.414%/+0.519pp|87.716%/72.031%/+0.634pp|87.832%/70.115%/+0.461pp|参数最大|

最终epoch8的`p4_r16_e8_k1`训练损失为1.930，其中CE为1.182、worst-K为1.376、nested K1/K5/K10/K20分别为1.448/1.148/1.170/1.196，所有项有限。winner adapter含18,448个FP16元素，tensor payload 36,896B，真实序列化文件40,124B，SHA256为`95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446`。

### 13.3从底层决定“更新哪些层”

|位置|是否更新|理由|资源/风险判断|
|---|---|---|---|
|Sinc滤波器、time/frequency/PA卷积块|冻结|负责提取发射机硬件纹理；K1 support不足以重新估计时域/频域滤波器，更新会同时移动所有旧类边界|反向需保存长序列中间激活，是显存和时延主要来源；灾难性遗忘风险最高|
|GroupNorm/全局归一化统计|冻结|少shot与三View高度相关，统计量方差大；直接改norm容易把View差异误当成域统计|参数虽少，但K1不稳定，历史`id_norm_late_feature`仍有明显遗忘|
|`t_proj(96→160)`、`f_proj(32→160)`、局部`pa_proj.0(64→160)`、`fuse.0(321→160)`|地面更新，主选|都位于分支池化之后、最终身份头之前；能修正时域/频域/PA分支尺度和融合方向，同时不重写底层指纹滤波器|rank16共18,448参数；v16实测Pareto最优；合并后query零额外LoRA MAC|
|`id_proj.0`、classifier `pa_proj.0`|target侧冻结|直接重塑身份核和PA缺陷特征；少shot下自由度过大，容易用新类support拉坏旧类几何|v16后4层整体更新没有优于P4，故不进入星上首选|
|`id_gate.0(160→160)`、`joint_proj.0(320→160)`|target侧仅低秩更新|前者只调节PA缺陷特征对身份核的门控幅度，后者只做最终joint embedding的小旋转；它们最靠近qKNN使用的`feat_joint`，可用极少参数完成receiver-specific对齐|rank8分别2,560/3,840参数，共6,400；合并后零额外LoRA MAC|
|CosFace旧类head|冻结且不用于新类决策|checkpoint的CosFace只含source旧类权重，更新会天然偏向旧类且无法公平容纳target-new|新旧类统一交给support原型qKNN，避免扩头和旧/新角色分支|
|domain backbone/`z_dom`|冻结|正式qKNN读取身份`feat_joint`，更新domain branch不直接改善分类空间|纯额外计算和状态，不进入候选|

结论是“先投影、后门控”，不是“越靠后越好”或“更新层越多越好”：地面用P4修正通道/receiver共性偏移；星上只用JG做目标receiver的最后一小步几何校准。

### 13.4提出的算法：P4-BPJG-qKNN

算法分成两个不同时训练的阶段。

1. **地面P4-LoRA**：在固定ADV3B02上，只对4个池化后投影层注入rank16 LoRA，使用source `leo_weak`多View训练8epoch；训练完成后合并进Linear权重并作为预装candidate。其作用是学习跨receiver、跨LEO弱信道的共性修正。
2. **星上BP-JG-LoRA**：收到目标receiver的带标签注册support后，只对`id_gate.0`和`joint_proj.0`注入rank8 LoRA。每个物理shot对应3个`leo_*_weak`View，但仍只计1-shot；按shot index构造episode，每个episode同时包含全部注册类并对所有身份使用同一规则。统一训练5epoch；K1共5个optimizer step，K10共50个step。默认SGD、无momentum、`lr=5e-3`、`weight_decay=1e-4`、`grad_clip=1.0`、`LoRA alpha=8`、prototype temperature 18。
3. **qKNN注册与推理**：适应完成后合并JG LoRA，使用全部old＋new support建立同一球面原型头。query默认1-view；只有当前样本margin低、top1绝对相似度低或跨View不稳定时追加3/5-view，不使用query批次统计、角色或类别配额。

星上输入是固定SHA的ADV3B02＋已合并P4 candidate、目标receiver的old/new注册support标签、物理sample ID和3个弱信道View。训练阶段不接收query。输出是JG delta、全部注册类原型、对称qKNN head及自适应View门限；独立query推理输出所有注册类上的逐样本预测和实际View数。

对归一化support特征`z_i`、留一View类原型`p_c^{-v}`和适应前特征/原型`z_i^0,p_c^0`，目标函数固定为：

`L=L_xproto+2L_boundary+0.5L_anchor+0.5L_gram+0.25L_sep+0.1L_view`。

- `L_xproto`：跨View留一prototype CE。当前View只由其他View形成的prototype监督，直接优化新旧类统一的qKNN决策。
- `L_boundary`：若`m_i=cos(z_i,p_y^{-v})-max_{j≠y}cos(z_i,p_j^{-v})`，则惩罚`[m_i^0+0.02-m_i]_+`，要求每个注册类的support边界相对base不下降。
- `L_anchor=1-cos(z_i,z_i^0)`：限制K1特征漂移。
- `L_gram`：保持`C×C`support prototype Gram关系，防止旧类整体几何被新类support扭曲。
- `L_sep`：对全部注册类对称地压低过高的prototype互相似度，不使用hard-class quota或DRO类配额。
- `L_view`：约束同一物理shot的三View特征一致；不把三View伪装成3-shot。

该损失同时面向旧类保持和新类可分性。它不使用target CosFace logit蒸馏，因为CosFace没有新类权重；改用base feature/boundary/Gram作为角色对称教师。

### 13.5资源预算与MRIOR比较口径

|资源项|P4地面阶段|BP-JG星上阶段|部署合计/门槛|
|---|---:|---:|---:|
|可训练参数|18,448|6,400|两个阶段不同时训练；星上仅6,400|
|FP16 tensor payload|36,896B|12,800B|加26类head 15,688B和TTA 24B后理论65,408B|
|真实序列化文件|P4实测40,124B|待实现后实测|最终必须≤256KB|
|optimizer step|离线8epoch|K1 5步；K10 50步|MRIOR为600步/row，分别少120倍和12倍|
|持续query额外LoRA MAC|合并后0|合并后0|只保留qKNN小头和自适应View|
|多View计算|source winner平均1.364次forward|目标侧待测|相对固定5-view已实测减少72.7%|
|适配时延/峰值显存|离线记录|目标侧待测|时延目标≤MRIOR的25%，显存≤50%|

历史合法MRIOR-SDA证据只有Stage2-B旧类：125个receiver×seed×K任务总体old_acc 82.58%，K1/K5/K10/K20分别77.22%/82.59%/85.82%/87.74%，平均适配时延17.90s；每row对三个场景各更新200步，共600次full-backbone梯度更新。它没有target-new输出，因此不能据此声称“加入新类后优于MRIOR”。

正式比较分两条：

- Stage2-B使用MRIOR-SDA native旧类结果；
- Stage2-C构建披露为CVS extension的`MRIOR-SDA＋同一对称qKNN enrollment`：MRIOR和P4-BPJG使用相同old/new support、相同query、相同原型头和相同自适应View，唯一区别是full-backbone MRIOR适应与6,400参数JG适应。这样才能比较加入5/10/20个新类后的`old_acc/seen_new_acc/H_old_new`。

“显著高于且更轻”锁定为硬门槛：每个K的Stage2-B `old_acc`以及Stage2-C `old_acc/seen_new_acc/H_old_new`配对均值至少高于matched MRIOR 2pp且paired 95%CI下界>0，逐receiver不得低于MRIOR；参数≤MRIOR的5%、step≤10%、平均时延≤25%、峰值显存≤50%、持久状态≤256KB。当前v16只证明P4层选择，尚未满足target、新类或MRIOR显著胜出门槛。

### 13.6下一轮最小target开发矩阵

固定P4 ground candidate、K10开发数据和全部损失/优化器，只比较target可写层容量：

|arm|target更新层|rank|target参数|epoch|用途|
|---|---|---:|---:|---:|---|
|`P4_IDENTITY`|无梯度|0|0|0|判断target反向本身是否必要|
|`P4_JP_R8`|`joint_proj.0`|8|3,840|5|最小最终嵌入旋转|
|`P4_JG_R8`|`id_gate.0＋joint_proj.0`|8|6,400|5|主候选|
|`P4_JG_R16`|`id_gate.0＋joint_proj.0`|16|12,800|5|用户允许的性能放宽对照|

只用K10开发row选定一个arm；排序先检查`old_acc/min_old_class_acc/seen_new_acc/H_old_new`和相对MRIOR配对差，再检查时延、显存、状态与平均forward。锁定后K1/K5/K20只做确认，不在其query上继续调层、rank、损失权重或View门限。


## 十四、P4-BPJG-qKNN本地实现与v17启动前锁定

### 14.1实现结果

P4-BPJG-qKNN已从设计状态进入可运行实现。主链固定为：严格装载ADV3B02 checkpoint→按SHA256装载并合并P4 ground adapter→仅对target侧`joint_proj.0`或`id_gate.0＋joint_proj.0`注入LoRA→使用target receiver注册support的3个匹配LEO弱信道View训练→将target FP16 artifact落盘并立即回读→合并LoRA→导出适应特征→使用对全部注册类对称的prototype-only qKNN评测。

关键实现锁如下：

- BP-JG主候选只更新`id_gate.0＋joint_proj.0`，rank8为6,400参数；放宽对照rank16为12,800参数；JP最小对照rank8为3,840参数。
- 训练固定5epoch、SGD无momentum、`lr=0.005`、`weight_decay=0.0001`、`grad_clip=1`、temperature18、最多50步。
- 实际步数已由trainer级回归锁定：K1/K5/K10/K20分别5/25/50/50步；K20每个episode对每类取2个shot，共10个episode/epoch。
- 所有K统一`support_pool_max_k=20`。回归证明K1⊂K5⊂K10⊂K20、四个K使用完全相同query ID、support/query零交集。
- 每个View的row-level physical ID必须逐项等于第一个View；仅标签相同但physical ID次序漂移会fail closed。
- P4 artifact必须在`torch.load`前匹配SHA256；不同target scope/rank或P4 SHA具有不同run ID，不会互相覆盖。
- target LoRA性能状态以持久化FP16 artifact为准：训练后先保存、再回读、最后合并和导出，避免用内存FP32 patch评测却交付FP16 patch。
- LoRA合并后所有参数重新冻结；合并等价检查同时拒绝非有限值和大于`1e-5`的差异。
- qKNN固定`head_mode=qknn`、`support_representation=prototype_only`、`feature_adapter=none`、`labelprop=disabled`、`decision=per_sample_argmax`、`old_anchor_bias=0`；保留legacy K10/K5字段、旧类bias或额外head训练都会被拒绝。
- target优化器不读取query、old/new角色、类别配额或dense graph；类别ID置换前后BP-JG各损失项严格不变。

### 14.2真实ADV3B02集成验证

本地使用正式基座文件`best_joint_safe_ssdg.pth`，SHA256为`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；P4 winner为`adapter_fp16.pt`，SHA256为`95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446`。

|验证项|结果|
|---|---:|
|ADV3B02 strict checkpoint load|PASS|
|部署特征键|`feat_joint`|
|P4更新参数|18,448|
|P4合并最大绝对误差|4.17e-7|
|JG-r8更新参数|6,400|
|JG-r8 FP16 tensor payload|12,800B|
|JG恒等注入前后特征最大误差|0|
|FP16 roundtrip后合并最大误差|0|
|合并后特征最大误差|0|
|合并后可训练参数|0|

本地`ssr-gpu`最终回归为104/104通过，覆盖support LoRA、micro-IQ、adaptive View、Stage2 runner、candidate lock和class-incremental相邻路径；`py_compile`、v17 launcher `bash -n`、专用config验证和`git diff --check`均通过。

### 14.3v17最小开发矩阵

实验ID为`qknnv42_p4_bpjg_dev20_k10_20260715_v17`。本轮只使用receiver `8-8`、seed713101、K10和20个注册新类，先回答target层组/容量是否产生正收益；不在本轮调K1/K5/K20。

|arm|target更新层|rank|参数|epoch/step|GPU|
|---|---|---:|---:|---:|---:|
|`P4_IDENTITY`|无target梯度|0|0|0/0|0|
|`P4_JP_R8`|`joint_proj.0`|8|3,840|5/50|1|
|`P4_JG_R8`|`id_gate.0＋joint_proj.0`|8|6,400|5/50|2|
|`P4_JG_R16`|`id_gate.0＋joint_proj.0`|16|12,800|5/50|3|

每路独立输出training manifest、完整5epoch loss trace、FP16 target state、合并后3场景cache、Stage2-C `metrics.json/detailed_metrics.csv`及`resource_audit.json`。资源审计不再把训练器的FP16理论估算冒充真实部署state；实验完成后从qKNN runner读取实际float64 `persistent_state_bytes`，再加P4/target真实序列化文件和24B自适应View门限，硬判定是否≤256KB。

专用config为`paper_reproduction/configs/cvs_qknnv42_p4_bpjg_dev20_k10_20260715_n607.json`，launcher为`paper_reproduction/scripts/launch_cvs_p4_bpjg_dev20_k10_v17.sh`。远端输出根计划为`runs/qknnv42_p4_bpjg_dev20_k10_20260715_v17/`，日志根为`logs/qknnv42_p4_bpjg_dev20_k10_20260715_v17/`。唯一启动命令计划为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash paper_reproduction/scripts/launch_cvs_p4_bpjg_dev20_k10_v17.sh
```

### 14.4版本与声明边界

本轮本地文件SHA256：

|文件|SHA256|
|---|---|
|`train_export_cvs_micro_iq_adapter.py`|`aef73233505b23664e148a46e427b84199860f3b9f5568339f50b966adc9758b`|
|`train_export_cvs_support_lora_adapter.py`|`e0f9417fdb7c57ff8b35a597da5860824fd1ffb41a77eab9417f767e9acc3009`|
|`cvs_qknnv42_p4_bpjg_dev20_k10_20260715_n607.json`|`07a331cb6aab64b833711c464ab167c4625c8cdf7c1ff18e0631a5c0e6268641`|
|`launch_cvs_p4_bpjg_dev20_k10_v17.sh`|`ed7e64711d15052013f7c326b5b262444c85ad27e75e4f3328435f28cf6a5bf2`|
|`test_support_lora_adapter.py`|`698202c5fa53e4ea7bddd471ae1e99bcdbcdf67b7fac2a6922898b2bb8345649`|

`E:\type10-7`根目录仍不是Git仓库；实现、config、launcher、测试和本报告均落在Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。本轮开发config显式`resource_diagnostic_only=true`且`formal_claim_authority=false`：其输入是历史post-channel raw-IQ诊断cache，因此可用于层/rank性能筛选，但不能替代`项目.md`要求的正式sealed LEO-only Stage2-C结果，也不能据此声称显著优于MRIOR。正式结论仍需5receiver×至少5seed、5/10/20新类、K1/5/10/20和matched MRIOR配对置信区间。


### 14.5N607同步与启动授权证据

- Git提交：`3c4f178 feat: add lightweight P4 BPJG target adaptation`；分支`codex/cvs-rffi-release-20260626`。
- 2026-07-15 21:34+08:00 direct N607 preflight通过；项目根、服务器时间和8张RTX3090可见。
- live inventory为`active_training_processes=[]`、`gpu_compute=[]`，不存在centralized/federated/unknown训练占用。
- local-first同步映射：
  - `E:\type10-7\github_publish\CVS-RFFI-repo\paper_reproduction\scripts\train_export_cvs_micro_iq_adapter.py`→`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/scripts/train_export_cvs_micro_iq_adapter.py`；
  - `E:\type10-7\github_publish\CVS-RFFI-repo\paper_reproduction\scripts\train_export_cvs_support_lora_adapter.py`→`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py`；
  - `E:\type10-7\github_publish\CVS-RFFI-repo\paper_reproduction\configs\cvs_qknnv42_p4_bpjg_dev20_k10_20260715_n607.json`→`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/configs/cvs_qknnv42_p4_bpjg_dev20_k10_20260715_n607.json`；
  - `E:\type10-7\github_publish\CVS-RFFI-repo\paper_reproduction\scripts\launch_cvs_p4_bpjg_dev20_k10_v17.sh`→`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/scripts/launch_cvs_p4_bpjg_dev20_k10_v17.sh`。
- 4个同步文件远端SHA256与14.4本地值逐项一致；ADV3B02 checkpoint和P4 artifact的远端SHA也分别匹配`2699eedc...`与`95f9a8ba...`。
- 远端CVS-RFFI Python `py_compile`、launcher `bash -n`、专用config类对称qKNN锁和3个input cache存在性均通过；run/log目标根均不存在。
- 每个GPU只安排1个训练进程，低于项目允许的每卡2个上限。launcher完成有界启动后退出，不保留SSH或monitor会话。


## 十五、v17完成结果：容量不是瓶颈，层位置与同shot捷径才是瓶颈

### 15.1完成状态与声明边界

实验`qknnv42_p4_bpjg_dev20_k10_20260715_v17`已在N607完成，全部训练、评测、FP16 artifact、完整日志和资源审计已回收到本地`remote_artifacts_bpjg_v17`。原始`P4_IDENTITY`训练/导出成功，但评测器拒绝自定义resource tier；修复评测兼容后在独立目录`v17_identity_repair1`重跑identity评测，没有删除或覆盖原结果。以下仍是receiver `8-8`、seed713101、K10、20新类的单行development diagnostic，不能写成正式Stage2-C或显著优于MRIOR结论。

### 15.2同一候选行联合结果

|candidate|层/rank|old_acc|new_acc|H|遗忘率|min old|min new|相对identity old/new/H|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|`P4_IDENTITY`|无target更新|51.111%|26.333%|34.422%|29.722%|13.333%|0%|基线|加入20类后旧类严重塌缩|
|`P4_JP_R8`|`joint_proj`/8|51.944%|26.500%|34.734%|30.000%|16.667%|0%|+0.833/+0.167/+0.312pp|小幅改善floor，但遗忘略增|
|`P4_JG_R8`|`id_gate＋joint_proj`/8|52.778%|26.583%|34.991%|29.167%|18.333%|0%|+1.667/+0.250/+0.570pp|v17最佳，但远未达到目标|
|`P4_JG_R16`|`id_gate＋joint_proj`/16|52.778%|26.583%|34.991%|29.167%|18.333%|0%|+1.667/+0.250/+0.570pp|预测与rank8完全相同，无容量收益|

|candidate/scenario|old_acc|new_acc|H|old-before|遗忘率|
|---|---:|---:|---:|---:|---:|
|`P4_IDENTITY/clear`|53.333%|31.750%|39.804%|82.500%|29.167%|
|`P4_IDENTITY/low`|45.833%|27.000%|33.982%|78.333%|32.500%|
|`P4_IDENTITY/rain`|54.167%|20.250%|29.479%|81.667%|27.500%|
|`P4_JP_R8/clear`|54.167%|32.250%|40.429%|84.167%|30.000%|
|`P4_JP_R8/low`|45.833%|26.750%|33.783%|79.167%|33.333%|
|`P4_JP_R8/rain`|55.833%|20.500%|29.989%|82.500%|26.667%|
|`P4_JG_R8/clear`|55.833%|32.250%|40.885%|84.167%|28.333%|
|`P4_JG_R8/low`|45.833%|27.000%|33.982%|79.167%|33.333%|
|`P4_JG_R8/rain`|56.667%|20.500%|30.108%|82.500%|25.833%|
|`P4_JG_R16/clear`|55.833%|32.250%|40.885%|84.167%|28.333%|
|`P4_JG_R16/low`|45.833%|27.000%|33.982%|79.167%|33.333%|
|`P4_JG_R16/rain`|56.667%|20.500%|30.108%|82.500%|25.833%|

### 15.3训练与资源结果

|candidate|target参数|step|训练时延|峰值CUDA|support前向等价|最终loss|最终margin/base margin|真实总状态|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|`P4_IDENTITY`|0|0|0s|0|0|0|—|77,033B|
|`P4_JP_R8`|3,840|50|4.401s|166,224,384B|4,680|2.9334|-0.06633/-0.07158|85,431B|
|`P4_JG_R8`|6,400|50|4.136s|166,234,624B|4,680|2.9399|-0.06664/-0.07158|91,141B|
|`P4_JG_R16`|12,800|50|4.495s|166,260,224B|4,680|2.9368|-0.06646/-0.07158|103,941B|

真实总状态按qKNN runner `persistent_state_bytes=35,977B`＋P4文件40,124B＋target文件＋24B门限计算，四臂均显著低于256KB。相对历史MRIOR平均17.90s，三条target适配仅约4.14–4.50s，已达到时延≤25%的轻量级目标附近；50步也仅为MRIOR每row 600次full-backbone更新的1/12。但性能门槛完全未达到，不能因资源PASS而promotion。

### 15.4底层诊断

1. **主故障是26类竞争，不是单纯域偏移。**P4 identity的old-before三场景均值约80.83%，加入20类后old_acc降至51.11%，说明旧类特征本身尚可，但统一26类原型空间发生严重重叠。
2. **rank不是瓶颈。**JG-r8与JG-r16逐项预测完全相同；继续增大rank只增加状态，不增加有效自由度。
3. **`id_gate`几乎没有真正更新。**JG-r8中`id_gate`合并delta Frobenius范数仅为base的0.00483%，而`joint_proj`为0.52971%。ADV3B02中门控实际为`feat_id*(1+0.35*sigmoid(g))`，受0.35系数和sigmoid共同限制，因此少步LoRA很难通过它重塑类间几何。
4. **原跨View CE存在同shot捷径。**每个shot index episode每类只有一个物理shot，prototype虽然排除了当前View，却仍包含同一物理shot的另外两个View。网络可用接收View一致性完成匹配，却没有被迫学会“同类不同物理shot”之间的可迁移边界。
5. **loss仍未把support边界推为正值。**50步后margin仍约-0.0666，support训练正确率不足，表明当前监督结构比rank更值得优先修改。

因此v18不增加rank，不更新卷积backbone，而是同时改变两件事：把可写层换到更有直接几何作用的`id_proj/fuse`，并用leave-one-physical-shot-out消除同shot捷径。


## 十六、v18高效域适应设计：P4-BPJG-LOPO

### 16.1梯度更新层选择

|target层组|数据流位置|作用|rank8参数|判断|
|---|---|---|---:|---|
|`joint_gate`|`id_gate＋joint_proj`|保留v17锚点，验证单改loss的收益|6,400|必要对照|
|`identity_joint`|`id_proj＋joint_proj`|`id_proj`直接旋转由`fuse`输出得到的160维身份核，再由`joint_proj`校正最终qKNN空间|6,400|主候选；参数不变但梯度作用更直接|
|`fusion_joint`|`fuse＋joint_proj`|`fuse`在池化后融合time/frequency/rho，能先纠正receiver/LEO造成的分支尺度与方向偏移，再修正最终类几何|7,688|稍上游但仍不保存长卷积反向图，资源可控|

Sinc、time/frequency/PA卷积、归一化、CosFace head和domain branch继续冻结。这样既避免K10用少量support重写硬件指纹滤波器，也把星上可训练参数限制在7,688以内。

### 16.2LOPO目标

对一个episode中物理support样本`i`、View集合`V`和类别`c`，构造排除该物理样本全部View的原型：

`p_c^(-i)=Norm(sum_{j:y_j=c,j!=i} sum_{v in V} z_jv / count)`。

再用每个`z_iv`对`p_c^(-i)`计算prototype CE和boundary保持。这样同一物理shot的其他View不能进入其教师原型，模型必须依赖同类的另一个物理shot，直接针对跨shot泛化与26类原型去混淆。prototype bank实现为`[physical,C,D]`向量化张量，不在Python中逐样本/逐类循环，也不产生GPU同步点。

K10采用环形相邻shot配对：episode `i`包含每类shot `i`与`(i+1) mod K`，10个episode/epoch；每个物理shot每epoch出现2次。K20仍将20个shot分成10组，每组2个shot，每个shot每epoch只出现1次。K1没有第二个物理shot，自动回退到原leave-one-view目标，保证K1路径可运行；后续K1专用增益仍需独立设计和确认。

损失权重保持`CE＋2boundary＋0.5anchor＋0.5Gram＋0.25sep＋0.1view`不变，仅改变prototype排除规则，避免同时改层、loss权重和优化器而无法归因。SGD无momentum、5epoch、50步、`weight_decay=1e-4`、temperature18、clip1保持不变。

### 16.3计算与存储预算

K10、26类、3View时，v17为base预计算780次样本前向＋50×78次episode前向，共4,680；LOPO配对后为780＋50×156，共8,580，增加83.3%，位于用户允许的50%–100%性能优先放宽区间。相对MRIOR的full-backbone 600步，LOPO仍只有50个optimizer step，且只对≤7,688参数反向。

|层组|参数|FP16 payload|预计真实总状态|部署额外LoRA MAC|
|---|---:|---:|---:|---:|
|`joint_gate-r8`|6,400|12,800B|约91KB|合并后0|
|`identity_joint-r8`|6,400|12,800B|约91KB|合并后0|
|`fusion_joint-r8`|7,688|15,376B|约94KB|合并后0|

训练时仍只使用3个已注册弱信道View；query默认1-view，自适应多View属于后续阶段，本轮不混入adapt层/损失筛选。

### 16.4v18四臂开发矩阵

实验ID锁定为`qknnv42_p4_bpjg_lopo_dev20_k10_20260715_v18`，仍只使用receiver `8-8`、seed713101、K10、20新类development row。

|arm|target层|lr|rank|参数|用途|
|---|---|---:|---:|---:|---|
|`JG_R8_LOPO_LR005`|`id_gate＋joint_proj`|0.005|8|6,400|仅改变LOPO目标，与v17同层同lr比较|
|`JG_R8_LOPO_LR020`|`id_gate＋joint_proj`|0.020|8|6,400|验证原更新幅度过小假设|
|`IJ_R8_LOPO_LR010`|`id_proj＋joint_proj`|0.010|8|6,400|主层组候选|
|`FJ_R8_LOPO_LR010`|`fuse＋joint_proj`|0.010|8|7,688|分支融合层候选|

选择顺序仍是old_acc→min old→new_acc→H→遗忘率，且必须先显著改善identity；资源只作硬约束，不用来掩盖性能不足。本轮config继续锁定`resource_diagnostic_only=true`、`formal_claim_authority=false`，不使用query训练、old/new角色、类别配额、dense graph或额外head训练。


### 16.5本地实现与验证

- `train_export_cvs_support_lora_adapter.py`已增加`bp_jg_lopo`目标、`identity_joint/fusion_joint`层组、K10环形双shot episode、K20无遗漏单覆盖、K1安全回退、向量化LOPO prototype bank，以及包含objective/scope/rank/lr/P4 SHA的防碰撞run ID。
- 真实ADV3B02＋P4集成验证通过：`joint_gate/identity_joint/fusion_joint`分别严格为6,400/6,400/7,688个可训练参数；三者LoRA恒等注入的最大特征误差均为0；P4合并最大绝对误差为4.17e-7。
- 专用support测试52/52通过；support、micro-IQ、adaptive View、candidate lock、class-incremental和Stage2 runner相邻回归共140/140通过；Python `py_compile`、v18 launcher `bash -n`、四臂CLI部署锁、config类对称qKNN锁和`git diff --check`均通过。
- Git承载面本轮意图文件SHA256：
  - `train_export_cvs_support_lora_adapter.py`：`f985f5e5f718f1c60ab75e6b41684bf4962edce454c1612a7d2f7c0e14406f7e`；
  - `cvs_qknnv42_p4_bpjg_lopo_dev20_k10_20260715_n607.json`：`9dd8867174cd7896a8c5d57783015917b3497d369c2ccf4136744f6267c4c96f`；
  - `launch_cvs_p4_bpjg_lopo_dev20_k10_v18.sh`：`93afd895e0bbbb165bc9a8e6bec5c56ff066f1518f2ea291a512c444a2016fb5`；
  - `test_support_lora_adapter.py`：`1c414f36153c22965d12172d4500e20d27304b06197c0ae2f0ffe9a5f590ba07`。


### 16.6独立审查、协议边界与N607决策

提交前独立代码审查确认LOPO向量化数学正确：每个held physical sample只从其真实类prototype中减去该样本的全部View，其他类保持完整support；K1 fallback和v17旧run ID兼容也正确。审查要求补强的算法证据已完成：

- 新增逐physical、逐class慢速reference，与向量化LOPO的prototype、CE及输入梯度逐项对齐；显式断言held sample全部View不会进入其真实类教师原型。
- 新增K1/K5/K10/K20真实LOPO trainer参数化回归，分别锁定5/25/50/50步、1/5/10/10个episode、每shot出现次数、exclusion mode与36/330/660/720个toy前向等价。
- 新增环境artifact驱动的真实ADV3B02＋P4三层组集成回归，覆盖严格checkpoint load、精确层名/参数、非目标参数完全冻结、非零FP16 patch roundtrip、合并parity、最终160维有限特征和合并后0可训练参数。
- 独立增量复核结论为`Ready to merge: Yes`，原Critical与三项Important均已关闭；唯一非阻塞提示是持续在CI/N607验证记录中注明真实artifact环境变量与SHA，避免将缺少artifact时的skip误读为真实集成通过。

审查同时指出v18原config引用历史未密封post-channel raw-IQ cache。根据当前`项目.md`第7.1节，该输入不能因为`diagnostic-only`而继续生成新的可运行调参候选。因此本轮采取最小fail-closed处理，而不把工作重心转移到协议工程：

- config明确`phase2_runtime_isolation_status=LOCAL_PROTOCOL_REPAIR_REQUIRED`、`launch_authority=false`并记录3个实际blocker；
- 补齐逐样本统一决策及四个query访问禁令字段；
- launcher在检查或打开任何feature/raw-IQ cache之前硬拒绝启动，只有绑定`PROTOCOL_VALID`的密封LEO package后才允许复用该adapt矩阵；
- 2026-07-15 22:11本地direct N607 preflight通过，8张RTX3090均空闲，live inventory无训练进程；但没有SCP、没有创建远端v18输出目录、没有启动实验。该决定是协议阻断，不是性能失败。

最终本地回归为专用support测试52/52、相关路径140/140通过；唯一告警为既有`torch.cuda.amp.autocast`弃用提示。最终文件SHA256：

|文件|SHA256|
|---|---|
|`train_export_cvs_support_lora_adapter.py`|`f985f5e5f718f1c60ab75e6b41684bf4962edce454c1612a7d2f7c0e14406f7e`|
|`cvs_qknnv42_p4_bpjg_lopo_dev20_k10_20260715_n607.json`|`9dd8867174cd7896a8c5d57783015917b3497d369c2ccf4136744f6267c4c96f`|
|`launch_cvs_p4_bpjg_lopo_dev20_k10_v18.sh`|`93afd895e0bbbb165bc9a8e6bec5c56ff066f1518f2ea291a512c444a2016fb5`|
|`test_support_lora_adapter.py`|`1c414f36153c22965d12172d4500e20d27304b06197c0ae2f0ffe9a5f590ba07`|
+

## 十七、v19合法source LEO_weak层组筛选计划

### 17.1目的与证据边界

实验ID为`qknnv42_p4_bpjg_lopo_source_k10_20260715_v19`。v18的P4-BPJG-LOPO数学与资源合同已经完成，但当前target开发config仍引用历史未密封cache，不能打开。为继续把精力放在adapt性能，本轮只使用Phase1离线生成并严格验证的`source_validation` LEO_weak cache set，在receiver `2-19`上比较四个关键层组/学习率；不读取任何Phase2 target artifact，不生成target性能或matched MRIOR胜出声明。

假设为：v17中`id_gate`实际相对变化仅0.00483%，说明门控层梯度几乎不起作用；在相同P4初始化、K10、5epoch、50步和LOPO目标下，直接更新`id_proj＋joint_proj`或`fuse＋joint_proj`应比`id_gate＋joint_proj`产生更大的source receiver holdout正收益，同时不降低最低类准确率。

### 17.2输入、处理与输出

|环节|输入|处理|输出|
|---|---|---|---|
|输入审计|ADV3B02固定checkpoint、P4 adapter固定SHA、sealed `source_validation/cache_set.json`|调用`load_verified_leo_weak_cache_set`，要求`source_validation`、role仅为source、3个正式LEO_weak场景、sample-level overlay和physical ID逐行一致|cache/receiver/checkpoint/P4审计|
|物理划分|source validation receiver `2-19`的6类物理样本|每类按固定seed排列；前20个构成嵌套support池，K10取其前10个；query固定为K20池外全部物理样本|K1⊂K5⊂K10⊂K20且四个K共享query|
|support适配|每类K10×6类×3个预注册LEO场景View|P4先合并；目标LoRA rank8；K10环形双shot episode；LOPO排除held physical sample的全部View；SGD、5epoch、50步|FP16 target delta、完整loss trace、时延/显存/forward审计|
|source评估|与support pool物理不相交的固定source query|同一全注册类cosine prototype qKNN逐样本argmax；分别评估P4 identity和adapted；strict direct ADV3B02只走原分类头|3场景逐row及aggregate accuracy、最低类、逐类准确率和delta|

四臂固定为：

|candidate|目标层|lr|参数|epoch/step|选择用途|
|---|---|---:|---:|---:|---|
|`JG_R8_LR005`|`id_gate＋joint_proj`|0.005|6,400|5/50|与v17同层同lr的LOPO对照|
|`JG_R8_LR020`|`id_gate＋joint_proj`|0.020|6,400|5/50|检验原更新幅度过小|
|`IJ_R8_LR010`|`id_proj＋joint_proj`|0.010|6,400|5/50|身份映射主候选|
|`FJ_R8_LR010`|`fuse＋joint_proj`|0.010|7,688|5/50|融合边界主候选|

source screen仅在`adapted accuracy>P4 identity accuracy`且`adapted min-class accuracy≥P4 identity floor`时标为PASS。若多臂PASS，按accuracy delta→floor delta→适配时间排序；该排序只决定哪一臂值得接入正式sealed Stage2-C，不替代target K10开发或matched MRIOR门槛。

### 17.3资源与服务器计划

- 训练参数6,400–7,688，仅为50k上限的12.8%–15.4%；adapter合并后每query新增MAC为0。
- optimizer最多50步、5epoch，不保存optimizer state；持久状态按P4真实文件＋target FP16文件＋6×160×FP16 prototype＋24B门限估算并要求≤256KB。
- support训练只前向3个已密封LEO场景View；query不参与梯度、门限或候选内部更新。
- N607计划GPU0–3各运行一臂，每GPU一个进程；预计输出`result.json`、`loss_trace.json/csv`、`adapter_state_fp16.pt`、`process_status.tsv`。launcher是异步提交器，manifest显式记录`launch_only=true`；每个后台子进程通过`EXIT trap`无条件写独立状态回执，区分PASS、source screen判负和基础设施失败。
- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- 远端输出根：`runs/qknnv42_p4_bpjg_lopo_source_k10_20260715_v19/`。
- 远端日志根：`logs/qknnv42_p4_bpjg_lopo_source_k10_20260715_v19/`。
- Conda环境：`CVS-RFFI`。
- 唯一启动命令：`bash paper_reproduction/scripts/launch_cvs_p4_bpjg_lopo_source_v19.sh`。
- 启动前必须重新执行direct SSH preflight和live GPU/process inventory；若有活动任务，按每GPU最多2个训练进程规则重新分配或仅监控。

### 17.4本地实现和验证

新增文件及SHA256：

|文件|SHA256|用途|
|---|---|---|
|`screen_cvs_p4_bpjg_lopo_source.py`|`6f93e71469477c26bd46e6ce306524f91ba2856be9da3b830411a54b8bbe8031`|sealed source cache加载、nested physical split、P4/LOPO训练和同row评估|
|`launch_cvs_p4_bpjg_lopo_source_v19.sh`|`3814bda5ba90a65fa8e3e86defb56552491b2bc02b0f39743a4b531b28dd5f45`|四GPU四臂启动与独立状态回执|
|`test_source_bpjg_lopo_screen.py`|`3168e588346ee5a3fa44871202ff48d2f33e86e3be58633f2b260e48954c4dc3`|K1/5/10/20完整嵌套/固定query、View-major physical对齐和漂移拒绝|

`ssr-gpu`本地验证：新增3项测试与原support 52项合计55/55通过，真实ADV3B02＋P4三层组artifact测试实际执行而非skip；Python `py_compile`和launcher `bash -n`通过。实现同时锁定checkpoint SHA并在加载前后复验，strict direct不执行support enrollment，适配峰值显存只在单个ADV3B02实例驻留时测量。唯一告警仍是既有`torch.cuda.amp.autocast`弃用提示。

独立增量复核结论为`Ready，可启动source-only v19筛选`，无剩余Critical或Important；该Ready只覆盖source-only筛选，不改变正式Stage2-C的fail-closed状态。

### 17.5正式target接入边界

只读审计确认现成已提交且在N607完成375/375行的严格基底是`sealed package→Landlock allowlist＋strace→immutable prediction→isolated scorer`，但当前正式predictor仍只支持Stage2-B，v18 trainer也仍让legacy query cache在进程文件边界可达。因此v19 source筛选可独立运行；正式P4-BPJG-LOPO Stage2-C必须另设support-only enrollment进程，使Landlock allowlist不包含任何query文件，再密封target delta并由第二个truth-free predictor消费。当前工作树中的memfd/symmetric-head增量属于未提交他人改动，本轮不依赖、不覆盖。

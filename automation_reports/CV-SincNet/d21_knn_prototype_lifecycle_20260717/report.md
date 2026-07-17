# D21 KNN原型生命周期与Phase1压缩几何

- experiment ID：`d21_knn_prototype_lifecycle_20260717`
- timestamp：2026-07-17
- operator：Codex
- objective：在正式Stage2-B/C `LEO_weak-only`协议下，以极轻量KNN原型生命周期同时推进目标域适应和新类注册，利用与ADV3B02共同封存的Phase1压缩中心、域偏移和P90半径改善旧类floor，并抑制5/10/20个新类下的旧类遗忘与新旧混杂。
- current status：Phase1 exporter提交`dc8f8bda`、D21 lifecycle提交`735232d5`、联合bundle提交`ff6def83`、联合bundle运行器提交`0e724ebd`已落地。用户已锁定`L/U/V=0.07/0.63/0.30`、`rho_label=0.10`与epoch200 final权重，协议提交为`fe8421b3`。当前停止扩展协议握手，使用已保存单一LEO_weak Phase2资产研发逐样本域适应、新类注册和floor保护。

## 假设与比较对象

核心假设：旧类遗忘应拆为“旧状态漂移”和“新类竞争侵入”。Phase1锚、Stage2-B旧snapshot与Stage2-C追加registry三层隔离后，旧状态漂移可以被严格消除；P90半径和每新类至多一条稀疏碰撞边界只需处理竞争侵入。

计划比较：

|candidate|机制|训练参数/epoch|用途|
|---|---|---:|---|
|L0|target-only spherical centroid，radius/boundary off|0/0|identity-only prototype基线|
|L1|鲁棒snapshot+Phase1 global radius prior+LOO/LTO收缩|0/0|域适应与置信尺度|
|L2|L1+每新类至多1条support-only稀疏碰撞边界|0/0|抑制old→new/new→old混杂|
|DALI|domain20 max-old-preserving旧类内部重排|0/0|后续与L2正交集成|

## 协议边界

- target support/query均必须已经叠加且只叠加一个`leo_*_weak`信道；Phase2不访问clean/source样本或其未授权衍生信号。
- Phase1导出只读原ADV3B02有标签source train split，发生在任何target可达前；bundle不含原始IQ、单样本feature/distance、count、路径、full-precision prototype或可独立替换sidecar。
- Stage2-B只使用registered old support；Stage2-C显式验证全部当前旧support和新support。旧prototype/radius/score列逐位锁定，只追加新状态。
- query逐样本面对全部注册类；无query fit、角色Oracle、真实batch类别数、quota、global assignment或dense query图。

## Phase1压缩几何

|项目|结果|
|---|---:|
|中心域|domain20，global max-min|
|残差rank|3|
|方向数值payload|4,278B|
|P90 radius payload|96B|
|当前registry/schema|658B|
|总逻辑状态|5,032B|
|历史v1稠密数值状态|25,428B|
|压缩倍数|5.81×|
|真实历史v1复压缩mean/min cosine|0.9998432/0.9993398|
|最大角误差|2.0821°|
|13个非中心域一次重建|37,440MAC|
|每query重建MAC|0|

P90使用4096-bin余弦距离直方图上沿，确定性误差上界为0.00048828125。直方图、样本距离和cell count只在Phase1内存中存在，不进入bundle。

## KNN生命周期

- K1：单中心；因无法构造self-excluded support反事实，Stage2-B radius和Stage2-C碰撞boundary强制关闭，仅保留无参数纯余弦安全基线；新类radius候选仍须通过全部registered support守门才允许激活。
- K5：mean/medoid统一候选，LOO q80半径与prior在平方空间收缩。
- K10/K20：增加固定robust-trim候选，LTO q80半径平方收缩；K20复用K10规则。
- Stage2-B将真实sealed enrollment `package_root_sha256`固化为旧support capsule root；Stage2-C必须显式重传并精确匹配，不能仅凭类名和K值替换旧support。
- K≥5的旧类radius逐类通过self-excluded纯余弦→radius反事实守门后才激活；Stage2-C新类radius还要逐个及组合通过全部registered support守门。旧类prototype、radius、score路径和`radius_active`掩码位级锁定。
- 新类先全部生成再append；每个新类从全部其他old/new原型中选择至多1个rival，class-name tie-break避免输入顺序偏置。
- 碰撞边界必须在全部registered LEO_weak support上同时满足每类accuracy不下降和最差true margin不下降，否则该边界或组合整体off。

## 本地变更与验证

|文件|用途|状态|
|---|---|---|
|`项目.md`|授权domain20 core+R3 residual+P90 radius v2|提交`ba9c7d1`|
|`analysis/d21_knn_prototype_lifecycle_and_phase1_geometry_design_20260717.md`|完整多路线设计|提交`d373e7e4`，后续资源数字待补提交|
|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|v2 codec、validator、只读Phase2 API|提交`dc8f8bda`|
|`code/cvsrffi/phase1_geometry_streaming.py`|Phase1两遍有界内存中心/P90聚合|提交`dc8f8bda`|
|`code/cvsrffi/stage2_prototype_lifecycle.py`|K1/5/10/20旧snapshot和append-only注册|提交`735232d5`|
|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|真实ADV3B02两遍Phase1 export-only入口|提交`dc8f8bda`|
|`code/scripts/run_d21_support_only_lifecycle.py`|联合checkpoint+int8 bundle强绑定的support-only运行器|提交`0e724ebd`|

验证记录：

- `conda run -n ssr-gpu python -m pytest -q tests/test_phase1_center_lowrank_prototype_bundle.py`：10项PASS。
- `conda run -n ssr-gpu python -m pytest -q tests/test_phase1_geometry_streaming.py tests/test_phase1_center_lowrank_prototype_bundle.py`：21项PASS。
- `conda run -n ssr-gpu python -m pytest -q tests/test_stage2_prototype_lifecycle.py`：14项PASS。
- `conda activate ssr-gpu; python -m pytest -q tests/test_stage2_prototype_lifecycle.py tests/test_run_d21_support_only_lifecycle.py`：独立修复后21项PASS。
- `conda activate ssr-gpu; python -m pytest -q tests/test_phase1_center_lowrank_prototype_bundle.py tests/test_phase1_geometry_streaming.py tests/test_export_adv3b02_center_lowrank_radius_component.py tests/test_stage2_prototype_lifecycle.py tests/test_run_d21_support_only_lifecycle.py`：联合回归49项PASS。
- 第二轮对抗修复后同一联合回归为51项PASS；D21 lifecycle/runner聚焦回归为23项PASS。
- 加入联合bundle的最终联合回归为61项PASS；joint bundle聚焦回归为10项PASS。
- 联合bundle运行器主agent在`ssr-gpu`串行复核为24项PASS；基础环境误调用一次因无`pytest`立即作废，不计测试结论。
- 五个D21核心Python文件`py_compile`与本轮Markdown `git diff --check`均PASS；exporter `--help`确认不再存在detached-signature参数。
- pytest退出后存在本机已知临时junction `PermissionError`清理噪声；测试退出码均为0。

独立代码审计发现并已修复两轮正式阻塞：旧support现在绑定实际内容SHA、authority receipt与before/after capsule root；基础new prototype在append前逐类及组合检查旧类support accuracy/worst margin；K1旧/新radius与boundary统一关闭；公开评估器不能自报sealed；资源上限按含COMMIT的实际落盘总字节计算。当前只声明`internal-target-score-lock`，不声称DALI已接入。Phase1组件同时改为`formal_phase2_eligible=false`且`outer_bundle_signature_required=true`的pending状态；只有真实checkpoint+v2组件+代码/配置经外层联合seal后，正式Stage2 loader才允许打开。

联合bundle进一步固定8个部署成员，禁止raw `.pth`、source路径、sample feature/count和cache spec；内置固定authority公钥验签，调用方不能注入空verifier。loader复算TorchScript archive/state结构并拒绝extra file或未授权buffer；class binding采用有序语义SHA，seal/envelope与包内member均在同一字节流上完成哈希和解析。当前尚无外部authority生产签名，故只完成unsigned signing request和formal loader实现，不声称正式bundle已经发布。

## N607 Phase1 export-only计划

- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 数据：原B02 Phase1授权ManySig有标签train split，seed=`392002`；预计约8,064–8,400个样本、32–33个batch；不读取target。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；cwd=`/home/szu2070436088/2510044040/CV-SincNet`。
- GPU、精确命令、PID、log、输出路径和同步SHA将在本地exporter验证、Git提交、N607 preflight后补写。
- export-only输出必须只有共同bundle所需v2 payload、pending manifest/hash和完整离线审计；不得产生全精度中间PT/JSON，也不得以伪造或自循环detached signature冒充正式组件。随后另行建立真实checkpoint+component的外层联合seal。

### 2026-07-17 13:43只读预检与split漂移

- direct N607 preflight PASS；项目根`/home/szu2070436088/2510044040/CV-SincNet`可见，GPU0～7均为0%利用率、10MiB占用；只见系统`unattended-upgrade-shutdown` Python进程，无训练进程。
- 远端磁盘`/home`可用7.6TiB；checkpoint为8,582,116B，SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；`Dataset_WigSig/ManySig.pkl`为2,359,341,461B，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- checkpoint不可变`split_info`记录`labeled_size=8400`、`unlabeled_size=58800`、`source_val_size=16800`。按当前`项目.md`公式，`8400/(8400+58800)=0.125>0.1`；当前loader因此在重建前直接拒绝。该历史checkpoint可保留为历史诊断，不能绕过guard冒充当前正式Phase1 lineage。
- 用户已选择公式优先路线：先留出`0.30` source validation，再将剩余训练池按`0.10/0.90`切为L/U，对全池等价为`0.07/0.63/0.30`；正式checkpoint固定epoch200 final权重，source validation只作健康审计和报告。该决定已同步`项目.md`并提交`fe8421b3`。

## 直接算法开发入口与旧v1边界

- N607已保存D18母资产覆盖5个receiver×6个seed×3场景、旧6类＋new20、每类每场景40个唯一LEO_weak观测。正式切分不是mother数组前20/后20，而是按`SHA256(somph-offline-split-v1|receiver|seed|role|TX|sample_id)`确定性排序；每类前K条为嵌套support，固定rank20–39为query，可直接切K=`1/5/10/20`和new=`5/10/20`，不重建信道。
- 2026-07-17 14:25 direct preflight再次PASS，GPU0～7均空闲。已将`rx20-1/seed713101`三场景母cell拉取到`E:\type10-7\automation_reports\CV-SincNet\d21_knn_prototype_lifecycle_20260717\dev_cache_rx20_1_seed713101`，总计约10.8MiB；远端资产保持只读，传输后`ssh.exe=0`且N607 TCP22连接数为0。
- 本地另有已密封的`rx20-1/seed713101/K10/new5`before/after胶囊，可先用旧runtime做算法开发冒烟；正式确认必须在用户指定final runtime及联合int8 bundle重建后执行。
- 旧v1 125行仅含2个seen-new类且使用dense query graph，按当前协议只能作历史诊断。其25-cell聚合为：K1 old/new/H=`67.60/58.80/61.10%`，K5=`80.03/75.60/76.93%`，K10=`83.04/81.87/82.07%`，K20=`85.67/84.87/85.04%`；对应最差旧类floor仅`6.67/25.00/20.00/35.00%`。因此本轮主攻逐类floor和注册后竞争遗忘，而不是复用v1的query图。

## 2026-07-17固定接收IQ表征与极轻量适配开发结果

开发单元固定为`receiver=20-1`、`seed=713101`、K10、5个真实seen-new TX和三个互斥物理样本集合的LEO_weak场景。每个物理样本只有一份固定接收IQ；`FFT96`、`RF32`和差分相位统计均由该IQ确定性计算，不生成额外LEO状态或独立训练子样本。predictor先输出truth-free逐样本预测，独立scorer随后连接truth；下表均为开发结果，不是确认矩阵或正式成功声明。

### Query即测试集的硬边界

本轮起把query严格视为最终测试集：方法结构、超参数、support代理选择、早停、回滚和候选排名必须在query打开前仅依赖support完成并锁定；随后只允许对锁定候选生成一次不可变prediction，再由隔离scorer读取query truth计算测试指标。query及其标签、角色、类别数、quota、顺序和测试结果不得回流到训练、适配、校准、阈值选择、方法选择或下一轮调参。此前M1b对15个软融合点生成的query Pareto统一降级为`SEALED_TEST_DIAGNOSTIC_ONLY`：只能描述已观察到的测试行为，禁止据此选择beta、offset或设计后续机制；它不构成可晋升证据。

### 候选演进

|candidate|机制|参数/epoch|old before|old after|old floor|seen-new|new floor|H|forgetting|状态|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|L0|z_id160单质心identity|0/0|65.00%|40.83%|11.67%|53.67%|1.67%|46.38%|24.17pp|基线|
|L2|z_id160 support-LOO old guard|0/0|64.72%|48.33%|11.67%|51.33%|1.67%|49.79%|16.39pp|低性能对照|
|L4|z_id160每类2原型|0/0|62.78%|50.00%|13.33%|42.33%|15.00%|45.85%|12.78pp|低性能对照|
|L5|`normalize([z_id160,8*FFT96])`、每类top1 support|0/0|83.33%|73.06%|51.67%|77.33%|63.33%|75.13%|10.28pp|0参数表征基线|
|L5q|L5逐support向量int8＋FP16 scale|0/0|83.06%|72.78%|51.67%|76.67%|61.67%|74.67%|10.28pp|28,380B部署态|
|L6q|L5q＋256参数support-only对角metric|256/20|88.89%|77.78%|61.67%|79.00%|51.67%|78.38%|11.11pp|当前old-floor Pareto|
|L7q|`theta_B→theta_C`、类CVaR＋旧pair保持＋侵入hinge|256/20|88.89%|78.89%|58.33%|79.33%|51.67%|79.11%|10.00pp|当前mean/forget Pareto|

L6q的变换后support码同样使用逐向量int8＋FP16 scale，对角scale以FP16持久化；量化版与其FP32预测逐项一致，Stage2-C逻辑状态28,892B。L6q每query为28,160次support点积MAC＋256次metric缩放MAC；L7q状态和query MAC相同。L8在L6q上比较`alpha*top1+(1-alpha)*class-mean`，三场景support-LOO锁定`alpha=1`，证明简单mean融合无floor收益并自动回退L6q。

### 紧凑数学描述符4-arm消融

|arm|维数|old after|old floor|seen-new|new floor|H|forgetting|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|A0 z＋FFT96|256|72.78%|51.67%|76.67%|61.67%|74.67%|10.28pp|最强固定描述|
|A1 z＋FFT96＋RF32|288|73.06%|50.00%|72.33%|60.00%|72.69%|9.17pp|low-elev old floor降至25%|
|A2 z＋FFT96＋DP32|288|66.11%|41.67%|75.33%|58.33%|70.42%|13.06pp|support→query排序失配|
|A3 z＋FFT96＋RF32＋DP16|304|73.06%|48.33%|73.67%|56.67%|73.36%|10.83pp|维数增加但性能下降|

因此不再继续堆叠RF/DP描述；固定A0＋极轻量metric是后续主表征。独立floor-aware实现对A0得到old/new/H=`75.00/82.00/78.34%`、old/new floor=`60.00/56.67%`，但遗忘13.61pp，进一步确认均值、floor和遗忘必须联合优化。

### 正式seed哈希切分的K/new规模快速矩阵

首个矩阵误把mother原数组前20/后20当作support/query，已在artifact首行标为`SPLIT_MISMATCH_DIAGNOSTIC`，禁止引用、比较或晋升。修复矩阵严格复现formal capsule三场景support/query逐行post-channel IQ SHA集合，6项均为`EXACT_SET_MATCH`；K10/new5 pooled old/new/H精确复现`0.7305556/0.7733333/0.7513360`。

|K|new5 H|new10 H|new20 H|new5 old/new|new10 old/new|new20 old/new|
|---:|---:|---:|---:|---:|---:|---:|
|1|39.28%|36.83%|30.02%|39.44/39.67%|36.11/38.00%|30.28/30.33%|
|5|67.01%|62.87%|60.26%|66.11/68.67%|62.78/63.83%|61.67/59.42%|
|10|75.08%|72.21%|71.31%|73.06/77.33%|69.72/75.00%|68.61/74.25%|
|20|81.52%|78.98%|78.53%|81.39/81.67%|78.33/79.67%|77.50/79.58%|

K20的new5/10/20平均遗忘仍为7.78/10.83/11.67pp，最差旧类floor为55/45/40%，最差新类floor为55/40/30%。这说明增加K可稳定改善性能，但单纯top1 support注册无法消除多新类竞争，也不能满足K1非负适应增益。

### 遗忘保护、稳健注册与快速模型适配

下表均使用同一开发测试单元。M1b曾对多个融合点打开query，现已统一封存为`SEALED_TEST_DIAGNOSTIC_ONLY`，不得用于选择或后续调参；表中只保留其support预锁点。M5两条路线则严格先用support锁定，再只对锁定候选执行一次隔离测试。

|candidate|机制|参数/epoch|old before|old after|old floor|seen-new|new floor|H|forgetting|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|M1|双状态old/new metric|512/20|86.67%|85.56%|75.00%|63.33%|33.33%|72.79%|1.11pp|旧类保护强，但新类塌缩|
|M1b lock|support预锁`beta=1,offset=.02`|512/20|86.67%|86.39%|75.00%|53.00%|25.00%|65.70%|0.28pp|测试诊断封存，不晋升|
|M2|rank4低秩残差metric|2,304/20|86.94%|78.06%|55.00%|77.00%|63.33%|77.52%|8.89pp|遗忘/new-floor Pareto|
|M3|bagged2稳健局部注册|0/0|78.33%|70.83%|45.00%|79.00%|60.00%|74.69%|7.50pp|5,720B但old floor不足|
|M4|合法单观测D1全类轻量头|3,456/20|74.72%|60.00%|38.33%|72.67%|45.00%|65.73%|14.72pp|support过拟合，NO-GO|
|M5-lite|norm-affine sparse delta|1,136/5|83.06%|72.78%|51.67%|76.67%|61.67%|74.67%|10.28pp|更新无效，NO-GO|
|M5-key|input-proj sparse delta|22,080/5|82.78%|72.78%|51.67%|76.33%|61.67%|74.51%|10.00pp|非Pareto，NO-GO|

M5-lite固定SGD、momentum=0、5epoch=5step，实际非零FP16 delta为1,135/1,136，patch仅2,272B/场景，适配2.87–3.50s，峰值显存223.24MiB；delta合并回原层后推理新增MAC为0。它只改变7/660个预测，support loss仅下降约0.1%，属于适配无效而非典型过拟合。M5-key预锁定22,080参数`input projection`白名单，FP16 patch为44,160B，patch＋int8 head共72,760B，单次适配最大2.68s，峰值GPU显存69,401,600B，合并后新增MAC同样为0；相对identity-only top1，new下降0.33pp、H下降0.16pp且状态增加44,160B，不构成Pareto改进。

这两条实验确认“support-only sparse key-layer delta”在资源上完全可部署，但当前更新位置没有提供有效泛化增益。后续不得依据这批query继续修改白名单或超参数；若继续快速梯度路线，必须只用尚未打开的support证据预先锁定机制，并把新的query测试留到方法锁定之后。

### 当前artifact

- 主开发runner与完整loss/score：`local_artifacts/d21_capsule_fast_adapt_dev_20260717/`。
- 4-arm描述符、720条完整20epoch loss trace与逐类结果：`local_artifacts/d21_floor_explore/final4arm_smetric/`。
- 双状态与封存软融合诊断：`local_artifacts/d21_m1_dual_state/final/`、`local_artifacts/d21_m1b_soft_fusion/final/`。
- M5-lite norm-affine快速delta：`local_artifacts/d21_m5lite_norm_affine/final/`。
- M5 key-projection快速delta：`local_artifacts/m5_support_only_sparse_delta_k10_new5_20260717/`。
- 正式切分快速矩阵：`automation_reports/CV-SincNet/d21_knn_prototype_lifecycle_20260717/l5_fast_dev_formalsplit_20260717/`。
- 错误切分禁用标记：`automation_reports/CV-SincNet/d21_knn_prototype_lifecycle_20260717/l5_fast_dev_20260717/STATUS.md`。

当前未达正式门槛，不启动125确认矩阵。M1–M5已经完成并固化正负证据；任何历史多LEO副本D1结果不得回填，也不得重复打开当前query做机制或超参数优化。

### 三轮后回顾与下一轮锁定原则

本次回顾重新对齐活动目标、`项目.md`第7.1–7.3节、项目会话索引和M1–M5完整日志。结论如下：

1. 域适应和新类注册必须在同一row、同一锁定机制中等权评估，继续同时报告注册前old、注册后old/new/H、逐类floor和forgetting；只优化old或只优化new的路线均不晋升。
2. M1证明冻结旧状态可以抑制遗忘，但跨状态校准会牺牲新类；M2证明低秩metric可改善遗忘/new floor，但仍无法修复low-elev/rain旧类floor；M3的5,720B稳健注册足够轻，却同样缺少旧类域适应能力。
3. M4的support高分没有转化为query泛化；M5-lite和M5-key满足≤5epoch、低状态和合并后0新增MAC，却分别表现为更新无效和非Pareto。因此下一轮不能再靠扩大层白名单、学习率网格或重复测试来寻找偶然增益。
4. 当前开发query已经作为测试集打开，现永久封存；其结果只用于本轮最终报告，不得进入后续机制、损失、超参数、阈值、早停、回滚或候选排名。M1b多点query诊断也不得作为后续设计依据。
5. 下一轮只能在support内部使用self-excluded或class-balanced fold构造训练/验证证据，并在打开任何新query前锁定完整机制。候选仍须满足单一LEO_weak观测、无clean/source样本、无query truth/role/quota/global assignment、adapter≤50k、≤5epoch/50step快速梯度、状态≤256KB。

下一机制预注册为`M6 support-fold low-rank id-projection delta`：仅在精确`id_proj`白名单内训练rank-2/4低秩差分，使用恒等近端、旧类pair保持、新类分离和逐类CVaR的support-fold目标；rank和损失权重只能由support-fold预锁，FP16低秩patch可合并回原层，部署新增MAC为0。M6开发阶段禁止打开任何query；只有support门同时覆盖old/new floor和遗忘代理后，才允许在全新且此前未打开的测试单元执行一次隔离测试。

### M6 support-only停止结果

M6严格只读取`after/enrollment_only`中的三场景registered support IQ与标签。四个预注册候选`rank∈{2,4}×{balanced,old_guard}`在3场景×2个class-balanced fold上均未通过support门：基线old/new/H为48.33/46.67/47.31%，适配后为49.44/45.33/47.00%，H下降0.32pp，old/new最差类floor均为0→0。训练loss下降但held-out H下降，判定为support-fold非泛化/轻度过拟合，状态锁为`NO_GO_SUPPORT_GATE`。

因此M6未执行full-support refit，未物化最终patch/head，未生成prediction或score，也未打开任何query。实际rank2/4可训练参数为800/1,440，FP16因子状态1,600/2,880B；每fit固定5epoch=5step，24个fold fit总计59.66s，峰值显存73.02MiB；若合并回原`id_proj`层，更新原参数25,760、部署新增MAC为0。独立闭包审计31项负测全部通过：runner CLI仅允许`enrollment_root/output_dir`，manifest必须精确为runtime、method lock、overlay与3个support文件共6成员；额外query/truth/scorer/apply-only、绝对路径或`..`成员均fail closed，17个query访问/拟合/选择字段均为false。

证据位于`local_artifacts/d21_m6_support_fold_lowrank/`、`local_artifacts/d21_m6_query_unreachable_audit/`及`analysis/d21_m6_support_fold_lowrank_design_20260717.md`。M6证明support门可以在不消耗测试集的前提下淘汰无泛化收益的快速梯度路线；该路线不进入query测试或125确认矩阵。

## 成功与停止条件

1. Phase1真实导出必须得到84个有效域×类cell、domain20中心、非零真实P90半径、严格allowlist和可复核资源审计。
2. support-only先覆盖3个LEO_weak场景与K=1/5/10/20；L1/L2若任一旧类或新类support floor/margin退化则安全回退，不打开query。
3. support门通过后才共同封存并进入开发query；随后按5/10/20类和独立确认矩阵扩展。

# D21 KNN原型生命周期与Phase1压缩几何

- experiment ID：`d21_knn_prototype_lifecycle_20260717`
- timestamp：2026-07-17
- operator：Codex
- objective：在正式Stage2-B/C `LEO_weak-only`协议下，以极轻量KNN原型生命周期同时推进目标域适应和新类注册，利用与ADV3B02共同封存的Phase1压缩中心、域偏移和P90半径改善旧类floor，并抑制5/10/20个新类下的旧类遗忘与新旧混杂。
- current status：Phase1 exporter提交`dc8f8bda`、D21 lifecycle提交`735232d5`已落地；ADV3B02 runtime+v2组件联合bundle、外置seal/signing request与formal loader本地实现完成，61项联合回归PASS。N607只读预检已完成，但旧ADV3B02 split receipt不满足当前`rho_label`公式，正式重训/export仍等待协议口径选择。

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
|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|v2 codec、validator、只读Phase2 API|本地未提交|
|`code/cvsrffi/phase1_geometry_streaming.py`|Phase1两遍有界内存中心/P90聚合|本地未提交|
|`code/cvsrffi/stage2_prototype_lifecycle.py`|K1/5/10/20旧snapshot和append-only注册|本地未提交|
|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|真实ADV3B02两遍Phase1 export-only入口|本地未提交|
|`code/scripts/run_d21_support_only_lifecycle.py`|sealed support materialization与生命周期support-only审计|本地未提交|

验证记录：

- `conda run -n ssr-gpu python -m pytest -q tests/test_phase1_center_lowrank_prototype_bundle.py`：10项PASS。
- `conda run -n ssr-gpu python -m pytest -q tests/test_phase1_geometry_streaming.py tests/test_phase1_center_lowrank_prototype_bundle.py`：21项PASS。
- `conda run -n ssr-gpu python -m pytest -q tests/test_stage2_prototype_lifecycle.py`：14项PASS。
- `conda activate ssr-gpu; python -m pytest -q tests/test_stage2_prototype_lifecycle.py tests/test_run_d21_support_only_lifecycle.py`：独立修复后21项PASS。
- `conda activate ssr-gpu; python -m pytest -q tests/test_phase1_center_lowrank_prototype_bundle.py tests/test_phase1_geometry_streaming.py tests/test_export_adv3b02_center_lowrank_radius_component.py tests/test_stage2_prototype_lifecycle.py tests/test_run_d21_support_only_lifecycle.py`：联合回归49项PASS。
- 第二轮对抗修复后同一联合回归为51项PASS；D21 lifecycle/runner聚焦回归为23项PASS。
- 加入联合bundle的最终联合回归为61项PASS；joint bundle聚焦回归为10项PASS。
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
- `项目.md`5.1另写`0.1/0.6/0.3`，其数学上同样给出`rho_label=0.1/(0.1+0.6)=1/7>0.1`，与第5节公式存在内部口径冲突。当前优先分析`0.08/0.72/0.20`的合法ADV3B02从头重训练路线；在解决该冲突前不启动正式export。

## 成功与停止条件

1. Phase1真实导出必须得到84个有效域×类cell、domain20中心、非零真实P90半径、严格allowlist和可复核资源审计。
2. support-only先覆盖3个LEO_weak场景与K=1/5/10/20；L1/L2若任一旧类或新类support floor/margin退化则安全回退，不打开query。
3. support门通过后才共同封存并进入开发query；随后按5/10/20类和独立确认矩阵扩展。

# Phase1 P1回归根因与DualGuard16验证方案

## 协议边界

本次结论只覆盖Phase1 source-only地面域泛化训练。训练集仍为`ManySig.pkl`，`rho_label=0.08`，无真实unknown、目标接收机或Stage2 query参与训练、选模和阈值标定。以下结果能说明闭集DG、独立星地压力鲁棒性、known特征几何、proxy风险和`endpoint_accept_v1`导出质量，不能声明真实unknown FAR、FPR95、Stage2 old/new调和性能或部署成功。

## 当前批次不是评估误差，而是训练塌缩

截至停止队列，共有64个完整候选和16个中断候选。64个完整候选覆盖575877行stdout和12788个epoch记录；未发现Traceback、OOM、fatal或关键训练NaN，final权重与冻结held-out评估使用的checkpoint SHA一致。63/64个候选在E24-E43从约98.6%的早期source-val准确率跌至16.67%，因此测试结果长期相同是分类器收敛到近随机预测，而不是测试缓存、checkpoint错载或评估函数没有更新。

历史同协议结果排除了“本任务本来只能达到低性能”的解释：

|实验/候选|overall|strict UDU|receiver floor|satellite floor|open-set风险摘要|
|---|---:|---:|---:|---:|---|
|ADV3B02|89.18|84.89|75.55|74.18|proxy_vaccept 0.4074，source overflow 0.4593，p95/p99 54.26/79.16|
|OSFIX_PROXY_A|90.19|86.40|79.70|75.90|legacy proxy_vaccept 0.6409，source episode overflow 0.9739，legacy bridge 1.0|
|V2FIX8中位数|89.30|84.76|71.67|77.63|p95/p99约53.54/80.10，source overflow约0.762，legacy proxy约0.663|
|当前64条中唯一未完全塌缩的G0_FULL_DG_PROTECT_S2|49.76|44.81|35.59|23.03|p95/p99 90/90，source episode overflow 0.8996，legacy proxy 0.6052，legacy bridge 1.0|

当前批次相对历史基线下降约39-73个百分点，已超出seed方差和正常机制权衡范围。

## 直接根因

### 1.新激活的source-episode结构损失先于direct metric失控

64条曲线的中位数显示：`source_episode`在E18启动，weighted loss从E18的8.94升至E24的116.26、E30的226.66、E32的489.19和E200的3634.7。source-val从E22的98.40%降至E28的66.53%和E30的19.67%。`direct_metric`到E28才启动，E28和E30的weighted loss仅约0.08和0.23，不能解释已经发生的回归。

7月10日前的V2FIX路径虽然输出receiver-local component统计，但这些结构项没有真正进入反向传播。P0/P1修复首次激活`compact/invariant/inter/accept/density`后，矩阵沿用了未经幅值验证的高权重。这解释了为什么旧V2FIX保持约89% overall，而新P1矩阵在相同teacher和数据协议下大面积塌缩。

### 2.局部密度项存在无界小分母

旧实现计算`normalized_radius=own_angle/local_batch_radius`，其中radius来自当前batch内class×receiver/day cell的p95，随后只以`1e-4 rad`为下限并detach。少样本或近重合component可产生极小radius；单个tail样本经平方和CVaR后形成数百至数万量级的raw loss。模型无法通过调整detach边界消除处罚，只能旋转、分裂或拉远特征。

其表面结果是局部overflow下降，但全局p95/p99扩张到90°、known core accept接近5%、分类准确率降至随机水平。该方向压低了动态局部代理，却没有形成跨receiver/day稳定的invariant core。

### 3.clean/satellite结构压力被直接相加

`concat_sa`路径分别计算clean和satellite source episode后直接求和。同一lambda在启用星地视图后等效放大约两倍；satellite增强早期又使局部cell半径和中心更不稳定。星地视图必须参与几何优化，但应按active-view权重归一化，不能因为增加一个必要视图而改变总梯度预算。

### 4.梯度控制启动晚、只有下限、还可削弱closed梯度

`phase1_v2_os_eff_all_phases=true`在旧trainer中没有真正生效。控制器仍被`epoch>=direct_metric_start_epoch`限制，E18-E27的source episode不受控制。旧控制器只保证open梯度占比不低于目标，没有上限；当open梯度已经占主导时不会降权。若下限无法达到，它还允许把closed scale降至0.32-0.65，进一步削弱TX CE、teacher和satellite锚点。

### 5.矩阵目标不可达，优化器长期处于高罚而无可行解

旧矩阵将direct p95/p99目标设为15/40°，而历史稳定模型约为54/79°，当前训练初期也远高于这些值。source overflow目标0.40同样远低于历史0.74-0.97。高权重、不可达阈值和动态batch边界叠加后，优化器持续收到同向大梯度，却没有保持DG的可行下降路径。

### 6.终局readiness与final-only协议互相矛盾

P1明确要求只保留`final_ssdg.pth`并禁止tail rollback，但终局P0 readiness仍要求`tail_rollback_enabled=true`。这使成功训练也会被标记为`NON_PROMOTABLE_P0_DISABLED`。该冲突不造成准确率下降，却会让实验无法形成endpoint/prototype闭环。

## 已实施修复

|修复|实现效果|对应失败|
|---|---|---|
|有界局部结构目标|每个compact/invariant/inter/accept项使用可微`tanh`上界；density使用Smooth-L1、CVaR和独立cap|防止source loss升至数千|
|物理半径下限与support gate|结构component至少2个样本；loss分母使用3-4°下限，诊断radius保持原值|消除退化batch小分母|
|clean/sat active-view归一化|两视图按权重平均，同时保留各视图raw/bounded分解|保留`concat_sa`且不双倍放大|
|source结构独立慢启动|leave-domain从E30启动；local结构从E40启动并用40轮ramp|先稳定RFF身份/DG锚点，再塑造拒识几何|
|双侧梯度预算|open占比低于下限时补强，高于上限时降权；上限触发时保留closed梯度并投影冲突open梯度|避免open吞没CE/KD/sat|
|全阶段控制|任一open loss有有效梯度即启用，不再等待direct metric|覆盖E18-E27盲区|
|全局梯度裁剪|新矩阵使用`max_grad_norm=5`|阻断异常batch的一步破坏|
|source-val DG健康响应|相对best下降3pp时把open scale降至0.15；下降10pp或低于70%时停止并阻断promotion|final-only条件下避免继续训练塌缩模型|
|完整结构遥测|输出clean/sat raw loss、bounded loss、radius min/p95、floor rate、term upper bound、pre/post gradient share|确保loss变化能追溯到最终几何|
|final-only readiness修正|要求`checkpoint_selection=final_only`且rollback关闭|解除控制面自相矛盾|

单元和回归测试直接覆盖退化小半径、multiview归一化、open梯度上下限、冲突投影优先级以及旧Phase1启动器兼容性。

## 10小时DualGuard16矩阵

新矩阵使用8个机制单元×2个paired seeds，共16条；每GPU固定2条并发，一次性启动，无后续长队列。每条120 epoch，source-val heavy在E20-E100每20轮一次，最后20轮每5轮一次，held-out receiver/day和六个独立full-physics satellite场景只在final冻结后评估一次。scheduler具有10小时硬deadline，超时仅终止本run精确子进程并保留partial artifact。

|单元|目的|主要变量|
|---|---|---|
|C0_DG_ANCHOR|恢复历史DG上限|关闭source/direct结构，仅保留EPOC/ADV3式DG、satellite和U_s基础路径|
|C1_LEAVE_DOMAIN_ONLY|验证旧leave-domain项是否安全|source=0.008，local结构全关|
|C2_LOCAL_NO_DENSITY|密度项消融|启用compact/invariant/inter/accept，density=0|
|C3_LOCAL_BALANCED|稳定主候选|source=0.010，五个有界local项均启用|
|C4_LOCAL_STRONG_PROTECT|激进强度上界|source=0.016、direct=0.006，teacher更强，open上限0.18|
|C5_SAT_GEOM_STRONG|星地几何重点|sat source权重1.5，增强channel invariance、sat KD和sat consistency|
|C6_U_DOMAIN_QUAR|无标签最佳安全利用|加强U_s domain/ADV/sat/invariance/quarantine，不启用U direct|
|C7_FULL_JOINT|完整推进候选|bounded local、direct proxy、U_s三态/direct/quarantine、DG/satellite联合优化|

所有16条均保留`concat_sa`，并同时对satellite视图使用TX CE、teacher KL、domain/ADV、channel invariance和几何损失；训练增强族为`leo_*_weak`，评估族为独立`clear_leo/low_elev_leo/rain_leo/storm_mp/geo_clear/mixed_orbit`，scenario、family、config hash和channel implementation均要求无重叠。

## 判定标准

矩阵首先验证“不会再塌缩”，其次比较open-set代理。任何open代理改善若伴随known core reject-all或DG下降，均判为失败。

1.训练健康：16条均不得出现source weighted loss无界增长；`source_episode_loss<=source_episode_loss_upper_bound`；无fatal/OOM/关键NaN；source-val不得触发10pp健康停止。
2.泛化主门槛：C0应恢复overall≥88、strict UDU≥84、receiver floor≥72、satellite floor≥74；完整候选相对C0的overall/strict/satellite floor下降均不超过2pp。
3.拒识代理：完整候选的fixed/legacy p99不得扩张到90°；source episode overflow应低于0.90并优先达到≤0.85；legacy proxy_vaccept应相对同seed C0下降≥0.05并达到≤0.60；bridge、tail、overflow accept和radius/inter必须同向下降，不能只改善dynamic DM字段。
4.U_s利用：C7的U direct active epoch和weighted loss必须大于0；三态必须同时记录trusted_core、ambiguous_tail和outside_reject，outside rate≤0.90；C6用于判断domain/sat/quarantine是否比U direct更稳定。
5.星地目标：C5/C7的独立satellite mean/floor不得通过牺牲clean strict UDU获得；弱receiver和最差satellite stress必须同时报告。
6.导出闭环：主候选必须生成与final SHA绑定的`endpoint_accept_v1`和receiver-aware local prototype artifact；动态DM门控仍仅是训练代理。

即使所有代理门槛通过，Phase1仍只产生Stage2真实unknown评估候选。真实unknown FAR/FPR95和old/new校准结论必须由后续Stage2 query实验给出。

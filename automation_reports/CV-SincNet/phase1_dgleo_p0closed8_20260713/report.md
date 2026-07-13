# phase1_dgleo_p0closed8_20260713

- 协议：Phase1 source-only；ManySig；`rho_label=0.10`；120epoch；held-out test仅E120；source-val重评在E10-E100每10轮一次、最后20轮每2轮一次；默认测试增强为`leo_weak`三场景。
- 目标：同时保护clean/strict UDU/receiver floor/satellite floor，并直接改善fixed p99、source_episode_overflow、legacy proxy/bridge、tail/overflow accept和radius/inter。
- 声明边界：不声明真实unknown FAR/FPR95、Stage2成功或真实在轨泛化。

## 机制

1. `virtual_detach=true`固定virtual negative，`gate_reference_detach=false`保留拒识gate几何梯度。
2. open梯度拆分为`boundary/source/invariant/u_geometry`四组，先执行分目标份额控制，再执行总open预算和closed冲突保护。
3. pseudo CE使用`confidence∩trusted_core`；U direct使用trusted core；U身份域不变性使用all-valid source-unlabeled。
4. tail绝对阈值继续阻断promotion/export，但不提前停止训练；finite metric reference允许在绝对阈值未达标时建立。
5. `eval_sat_on=all`导出逐receiver satellite指标和seed；scheduler仅在全部子实验`COMPLETE/0`时返回0。

## 矩阵

|GPU|candidate|侧重点|boundary/source/invariant/U|open预算|
|---:|---|---|---|---|
|0|`P0C_C0_BALANCED`|稳定联合|0.35/0.25/0.25/0.15|0.14-0.22|
|1|`P0C_C1_SOURCE_HEAVY`|source episode|0.25/0.35/0.25/0.15|0.14-0.22|
|2|`P0C_C2_INVARIANT_HEAVY`|receiver/day invariant core|0.25/0.25/0.35/0.15|0.14-0.22|
|3|`P0C_C3_BOUNDARY_ALIGNED`|fixed boundary|0.40/0.20/0.25/0.15|0.14-0.22|
|4|`P0C_C4_U_GEOMETRY`|U_s三态几何|0.30/0.20/0.25/0.25|0.14-0.22|
|5|`P0C_C5_SAT_INVARIANT`|clean-sat不变性|0.25/0.20/0.40/0.15|0.14-0.22|
|6|`P0C_C6_INTEGRATED_AGGRESSIVE`|source+boundary激进|0.30/0.30/0.25/0.15|0.18-0.26|
|7|`P0C_C7_DG_PROTECTED`|closed保护|0.30/0.25/0.30/0.15|0.14-0.22|

## 成功标准

- 8/8完成E120；U direct活跃≥80%，U invariance活跃≥95%；四组open梯度均可审计且非零。
- `source_episode_overflow`相对0.973下降至少0.05；fixed p99≤82.38°；legacy proxy/bridge与tail/overflow/ratio至少三项同向改善。
- clean overall/strict/receiver floor及sat strict floor相对上一批J5下降不超过1.5pp。
- 绝对tail、reference→final扩张或readiness不通过时继续fail-closed，不导出正式endpoint。

## 本地验证

- `ssr-gpu`环境下核心/控制/sat/launcher聚焦测试84 passed。
- Phase1 P1协议、post-stage、旧launcher和U_s回归45 passed。
- 新launcher专项3 passed；`py_compile`与`git diff --check`通过。
- 预计运行5-6.5小时，wall limit为10小时。

远端sync、SHA、命令、PID、GPU和终局结果在发布后补充。

## 占用保护

启动前N607存在其他任务。旧v1排队器要求全机GPU compute与已知launcher全部退出，错误地把外部任务存在性当成本批业务依赖；该策略已于16:15审计后废止。v2只使用每GPU实时容量，每张卡固定绑定1个本批candidate，在总训练进程数低于2且连续2次确认后独立启动；外部PID只计入容量，永不计入本批candidate、epoch、完成率或状态。

## N607发布

- 提交：`1972022`、`a6dce14`。
- 远端快照：`code/snapshots/phase1_dgleo_p0closed8_20260713_pre_sync`。
- 8个同步文件SHA逐项一致，远端`py_compile`通过。
- queue PID=`3849944`；初始状态`WAITING_FOR_EXISTING_JOBS`，compute=10、blockers=13；本批trainer尚未启动。
- 状态文件：`logs/phase1_dgleo_p0closed8_20260713_queue_state.json`；stdout：`logs/phase1_dgleo_p0closed8_20260713_queue.out`。

## 2026-07-13 14:50历史估计（16:15撤回）

- 120epoch不变。held-out test仅在E120执行；source-val重评调度为E10-E100每10轮一次、最后20轮每2轮一次。该调度原本由基础launcher继承，本次在P0 wrapper中显式固定并增加回归测试，防止后续基础配置漂移。
- N607现有DRIFT约E173/200；Stage2B完成/跳过约265/500行；Stage2C约39/500行。Stage2C的`orthogonal_incremental`单行中位耗时约1024秒，是当前排队主瓶颈。
- [已撤回]当时按外部任务吞吐估计等待7-8小时及2026-07-14 03:00-05:30完成窗口；该方法混入非本批进程，不作为当前ETA证据。
- 显式调度提交：`e3e769d`；相关launcher/protocol测试15 passed，`py_compile`通过。远端同步后SHA256=`0b1b2deb7531f3831cf060c2d32e1b266c85042eac95dca8a6fa5ea626e2246b`，dry-run确认四个调度参数，queue仍为`WAITING_FOR_EXISTING_JOBS`。

## 2026-07-13 16:15进程归属纠正

- PID/CWD/cmdline复核证明：本批仅有旧queue PID=`3849944`，Phase1 trainer为0/8；其余13个GPU compute均属于独立Stage2C publication/accelerator，不是本批进程。
- 14:50基于外部Stage2C行进度推算本批完成时间的方法无效，原预计窗口撤回。后续ETA只依据本批candidate的实际`launched_at`、epoch吞吐和终局评估耗时。
- v2排队器状态字段分别记录`own_running_count`、`pending_count`、`own_terminal_count`和每GPU`external_count`；`foreign_processes_count_as_candidates=false`为硬约束。
- 当前容量快照：GPU0-2各有1个外部训练进程，可作为本批第二槽位；GPU3-7各有2个外部训练进程，等待其中一个槽位释放。v2将分卡启动，不再等待全机清空。

## 2026-07-13 16:22 v2发布与首轮核验

- 提交`3084999`；queue专项与Phase1协议测试16 passed。远端同步前快照为`code/snapshots/phase1_dgleo_p0closed8_20260713_capacityv2_3084999_pre_sync/`，本地/远端SHA256均为`007f108b4086a140e3008d7826ea9a4bc2e06cca9eb0cedfbd27aa99c1ee9c7c`，远端`py_compile`和dry-run通过。
- 旧v1 queue PID=`3849944`在确认0/8 trainer、run/log目录未创建后定向终止；v2 queue PID=`3925391`。GPU0-2分别启动`C0/C1/C2`，根trainer PID=`3925530/3925662/3926124`；GPU3-7的`C3-C7`只处于`PENDING_GPU_SLOT`。
- queue权威状态为`PARTIAL_RUNNING_WAITING_GPU_SLOTS`：本批3/8运行、5/8等待、0终止。DataLoader worker虽继承相同cmdline，但不是独立candidate；本批计数只使用queue登记的根trainer和candidate ID。
- E2实测约154秒/epoch。C0的四组open有效梯度均非零：boundary/source/invariant/U约`6.68/2.82/3.56/2.86`；U direct selected约13.32，U invariance active=1，三态比例约`core/tail/outside=0.119/0.058/0.823`。
- E2只证明机制已激活，不能声明几何改善：C0的source_episode_overflow仍约0.971，legacy proxy_vaccept约0.655，bridge_accept=1.0，source-episode p95/p99/tail-CVaR约`63.79/83.00/74.98°`。需看E10及后续同口径趋势。

## 2026-07-13 18:23全量启动与中期趋势

- v2 queue状态为`ALL_LAUNCHED_RUNNING`，8/8运行、0等待、0终止。C3-C7分别于17:48-17:50获得GPU3-7槽位，根trainer PID为`3987099/3984711/3985185/3986531/3984200`；C0-C2仍为`3925530/3925662/3926124`。
- 进度：C0/C1约E45，C2约E55，C3-C7约E12-E13。除GPU2外，每GPU当前均为1个本批trainer+1个外部trainer；GPU2只有本批trainer。没有超过每GPU2个训练进程。
- E10相对E2时，三组source-episode p95下降约6.2-6.5°、p99下降约6.4-7.0°、tail-CVaR下降约6.3-6.7°，但overflow仍约0.971-0.973、legacy bridge仍为1.0，只能说明tail角度早期收缩，不能说明接收边界闭合。
- E20时legacy proxy_vaccept进一步降至约0.620-0.624，但source overflow升至0.975-0.978，legacy radius/inter由约0.96反弹至约1.02。DM内部p95下降到约53°，同时DM proxy_vaccept、bridge和radius/inter恶化，继续支持“动态gate改善不等于固定边界改善”的判断。
- 当前较成熟行中，C1 E40的source overflow约0.9696但仍接近1；C2 E50的source-episode p95/p99/CVaR约`57.16/72.29/66.80°`，legacy proxy_vaccept约0.632、bridge=1、radius/inter约1.089。没有候选解决P0 overflow/bridge问题。
- source-val `leo_weak`均值/地板约`91.55-91.65/90.17-90.32%`，这是源验证视图，不是held-out sat-strict UDU或receiver floor，不能用于目标达标声明。
- 按各自实际`launched_at`和epoch吞吐，C0-C2预计21:30-23:30完成，C3-C7预计2026-07-14 00:30-02:30完成；终局E120评估可能使窗口后移，仍在各自10小时wall limit内。

## 2026-07-13 22:30终局完整分析

### 训练健康与证据边界

- 8/8候选均连续完成E1-E120；每份`metrics_epoch.csv`均为120行且无缺失epoch，每份stdout均为5402行并已完整扫描。
- 8份stdout均未发现Traceback、OOM、Killed、RuntimeError或fatal；`train_loss`与`train_loss_tx_labeled`120轮均为有限值。日志中的`nan`来自未激活分支、缺失domain probe和首轮梯度遥测，不是总损失NaN。
- 8/8均执行final-only冻结held-out与三种`leo_weak`评估，最终权重存在。终态均为`NON_PROMOTABLE_GUARD_BLOCKED`、exit code 5；scheduler的`CHILD_FAILURE`是非零子状态汇总，不是训练崩溃。
- 这是Phase1 source-only代理诊断。训练未使用真实unknown，不能声明真实unknown FAR、FPR95、Stage2 old/new准确率或星上未知拒识达到目标。训练和评估属于同一`leo_weak`增强族的独立随机压力，只支持同族鲁棒性，不支持跨信道族泛化声明。

### 泛化与星地性能

|candidate|overall TX|strict UDU|clean receiver floor|sat mean|sat floor|sat strict mean|sat strict floor|sat receiver floor|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|C0 Balanced|89.813|85.962|74.658|77.954|76.915|72.469|71.415|56.158|
|C1 Source Heavy|89.843|85.942|74.658|77.900|76.876|72.399|71.360|56.242|
|C2 Invariant Heavy|89.862|86.017|74.575|78.010|76.957|72.556|71.497|56.292|
|C3 Boundary Aligned|89.660|85.685|73.967|77.809|76.782|72.231|71.188|56.125|
|C4 U Geometry|89.850|86.005|74.483|77.998|76.956|72.533|71.463|56.342|
|C5 Sat Invariant|89.825|85.863|74.242|77.967|76.933|72.479|71.435|56.267|
|C6 Integrated Aggressive|**89.953**|**86.183**|74.633|77.991|**76.972**|**72.575**|**71.543**|**56.550**|
|C7 DG Protected|89.777|85.938|74.592|77.953|76.929|72.483|71.457|56.267|

- C6为本批DG联合最好，但相对上一批J5仅overall+0.082pp、strict+0.176pp、sat strict mean+0.826pp、floor+1.050pp，clean receiver floor反而-0.400pp。差异均为同seed小幅变化，不能称稳定提升。
- 目标`sat-strict UDU≥78%`仍差5.425pp；目标`sat receiver floor≥73%`仍差16.450pp。最弱点是unseen-day receiver，尤其rx8在low-elev/rain约56%-57%；平均星地性能的微增没有修复receiver floor。
- 所有候选的source-val最佳记录在E118附近，final-only模型未出现大幅闭集回落；但最终held-out差异过小，四组梯度配比没有形成清晰机制响应。

### Fixed endpoint代理几何

|candidate|p50/p95/p99/CVaR(°)|source overflow|proxy_vaccept|bridge/low-density|tail/overflow accept|radius/inter|core accept|
|---|---|---:|---:|---:|---:|---:|---:|
|C0|3.27/34.43/65.19/51.29|0.414|0.357|0.183/0.077|0.712/0.586|4.005|0.474|
|C1|3.01/31.61/61.79/47.99|0.408|**0.355**|0.235/0.078|0.650/0.618|4.256|0.485|
|C2|2.99/31.80/63.15/48.68|0.406|0.367|0.238/0.077|0.681/0.620|4.266|0.487|
|C3|3.39/36.27/66.45/53.05|0.420|0.366|**0.172**/0.079|0.730/0.582|**3.856**|0.475|
|C4|2.96/34.78/65.74/51.81|0.410|0.359|0.166/**0.075**|0.735/0.615|4.034|0.478|
|C5|3.01/31.69/61.81/48.11|0.410|0.382|0.285/0.082|0.678/0.598|4.180|0.492|
|C6|**2.79/28.41/57.29/44.03**|**0.403**|0.360|0.260/0.082|**0.640**/0.624|4.378|**0.503**|
|C7|3.11/33.42/64.44/50.34|0.411|0.357|0.198/0.077|0.700/0.594|4.096|0.481|

- 相对上一批J5，C6的fixed p95/p99/CVaR由55.39/82.79/68.91°显著降至28.41/57.29/44.03°，tail accept由0.850降至0.640；这证明local endpoint评估和高open预算能收紧局部主体与极端角度。
- 改善没有闭环：168个receiver/day/channel局部组件形成并集，`radius/inter`由上一批约2.50恶化至3.86-4.38；overflow accept仍约0.58-0.62。局部半径变小的同时组件间类间距离塌缩，形成过度碎片化和并集膨胀。
- C6最能压p99，却把fixed bridge推到0.260，并把legacy proxy_vaccept推到全批最差0.669；C3的bridge最好但DG、p99、tail最差。没有同一候选同时优化tail、proxy、bridge、overflow和ratio。
- E118→E120的fixed p99增量为-0.954至+0.094°，均未触发后期扩张门控；本轮解决了“训练末期继续扩tail”的观测问题，但没有解决绝对边界不安全。

### Legacy/source-episode与动态DM错位

|candidate|source-episode overflow|source p95/p99/CVaR(°)|legacy proxy|legacy bridge|legacy ratio|DM overflow/proxy/bridge|B_os_eff|
|---|---:|---|---:|---:|---:|---|---:|
|C0|0.971|56.62/72.03/66.52|0.646|1.000|1.229|0.760/0.347/0.075|0.140|
|C1|0.964|54.71/70.06/64.44|0.654|1.000|1.294|0.753/0.358/0.089|0.140|
|C2|0.962|55.02/70.68/64.96|0.654|1.000|1.306|0.750/0.366/0.089|0.140|
|C3|0.975|57.89/73.22/67.85|0.640|1.000|1.196|0.763/0.332/0.068|0.140|
|C4|**0.956**|57.76/73.14/67.62|0.653|1.000|1.228|0.738/0.364/0.084|0.140|
|C5|0.967|54.75/69.98/64.48|0.658|1.000|1.296|0.754/0.364/0.088|0.140|
|C6|**0.956**|**52.84/67.37/62.03**|**0.669**|1.000|**1.369**|0.749/0.340/0.102|0.180|
|C7|0.967|55.97/71.56/65.90|0.648|1.000|1.257|0.755/0.357/0.081|0.140|

- `source_episode_overflow`仍为0.956-0.975，未达到相对0.973下降0.05的成功标准；legacy bridge全批始终1.0。局部组件没有形成跨receiver/day/channel共享的身份核心。
- C6将open预算从0.14提高到0.18后，source p95/p99/CVaR最好，但legacy proxy和ratio最差，直接暴露“压类内角度却把类间低密度/虚拟未知纳入接收”的硬冲突。
- 动态DM final proxy为0.332-0.366、bridge为0.068-0.102，与legacy proxy 0.640-0.669、bridge=1完全错位。动态gate仍可随batch和表征共同移动，不能作为最终边界。

### U_s、梯度预算与身份域泄漏

|candidate|receiver/day/channel leakage excess|U direct活跃epoch比例|U direct活跃batch中位|open预算达标epoch比例|
|---|---|---:|---:|---:|
|C0|0.622/0.160/0.402|0.325|0.000|0.543|
|C1|0.607/0.191/0.402|0.317|0.000|0.443|
|C2|0.599/0.194/0.401|0.242|0.000|0.440|
|C3|0.605/0.169/0.403|0.317|0.000|0.765|
|C4|**0.595**/0.175/**0.394**|0.267|0.000|0.664|
|C5|0.613/0.184/0.402|0.258|0.000|0.517|
|C6|0.604/0.164/0.401|0.283|0.000|0.440|
|C7|0.603/0.177/0.399|0.267|0.000|0.296|

- `U direct selected`中位约7.3-9.2，但活跃batch比例的epoch中位仍为0，只有约24%-33%的epoch出现过至少一个活跃batch。selected数量与实际梯度路由不一致，C4提高U权重没有提高活跃率。
- open预算控制器长期贴住0.14/0.18下边界，达标epoch比例仅0.296-0.765；C6即使提高预算仍有约56%epoch略低于0.18。当前控制器是滞后的软比例器，不是稳定的每步梯度约束。
- receiver/day/channel probe excess全部显著非零；C4虽相对最好，幅度不足以证明身份/域解耦。`z_id`仍携带大量receiver和channel信息，宽known域依赖没有被真正替换为RFF不变特征。

### 候选决策

|candidate|泛化结论|拒识潜力|主要风险|Stage2真实unknown评估|下一步|
|---|---|---|---|---|---|
|C6|本批DG联合最好|fixed tail角度最强|legacy proxy最差、ratio/bridge恶化、overflow≈0.956|仅可diagnostic dry-run，不可promotion|作为tail/source压力锚点|
|C1|DG稳定|fixed proxy最佳且p99较低|bridge、overflow和ratio未闭合|仅diagnostic|作为边界平衡锚点|
|C4|DG稳定|U/channel leakage相对最好|U direct仍空转，fixed tail较差|仅diagnostic|作为U路由修复锚点|
|C3|DG同批最差|fixed bridge和ratio相对最好|p99/tail/source overflow最差|否|保留为bridge负例|
|C0/C2/C5/C7|没有独立DG增益|只有单项小幅变化|无同一行联合改善|否|不继续原配比扫参|

### P0/P1结论与下一轮约束

**P0**

1. 将最终接收边界改为“共享身份核心AND局部密度支持”，禁止168个局部球直接取并集；local component只描述nuisance残差，不能独立授予known acceptance。
2. 对同TX跨receiver/day/channel组件中心加入显式收敛损失，对异TX最近组件中心加入绝对角度margin；`radius/inter`拆成半径和最小类间距离两个硬目标，防止仅缩半径或分母塌缩。
3. source episode以leave-one-domain core交集为正样本，tail/outside降权隔离；直接优化跨domain core coverage、tail quarantine和outside rejection，不能继续用单一全局球覆盖全部source样本。
4. U_s direct移除pseudo-CE/temporal gate交集依赖，按三态连续加权：trusted core用于身份核心，ambiguous tail只做隔离，outside reject只做负接收；逐batch记录真实反传样本数和有效梯度。
5. 将fixed frozen endpoint风险纳入周期性训练校准：固定seed、EMA reference、stop-gradient阈值和同一endpoint实现；动态DM仅作辅助梯度，promotion只看fixed endpoint。

**P1**

1. 对receiver/day/channel泄漏采用probe-guided GRL与同TX跨域组件中心对齐的联合目标；每类泄漏分别设梯度预算，不能全部混入closed总损失。
2. 星地分支加入source receiver worst-group DRO和clean-sat paired identity-core一致性，重点优化rx8/rx11类的source代理弱组；仍不得使用held-out target receiver训练。
3. open预算改为每步投影约束并设置0.15/0.20控制目标，避免数值贴下界；closed梯度保护只在实测冲突时触发，不能长期覆盖open更新。
4. 非promotion候选仍导出带`diagnostic_only=true`和reason code的endpoint/prototype包，供Stage2真实unknown只读评估；正式promotion继续fail-closed。

当前实验对Phase1的贡献是：fixed局部tail几何和sat strict性能均有可测改善，并证明高open预算能压p99；同时证实局部组件并集、动态DM、U_s路由和总预算控制仍未构成最终拒识闭环。

当前不能声明的是：真实unknown拒识提升、Stage2成功、跨信道族泛化、`endpoint_accept_v1`正式可用或达到sat receiver floor目标。

最主要风险是：`source_episode_overflow≈0.96-0.98`、legacy bridge=1、legacy proxy未降、fixed radius/inter≈4、receiver/channel泄漏和sat receiver floor≈56%。

本批不存在promotion候选；C6/C1/C4分别只作为tail-source、boundary和U路由机制锚点。

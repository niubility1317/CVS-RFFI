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

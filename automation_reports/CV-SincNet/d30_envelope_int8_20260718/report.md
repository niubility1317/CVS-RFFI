# D30双包络int8组内校准support-only实验

## 目标与边界

- 实验ID：`d30_envelope_int8_20260718`；阶段：正式Stage2-B/C之前的development support-only筛选；状态：`LOCAL_VERIFIED_READY_FOR_N607_PREFLIGHT`。
- 目标：在D27-B的25步轻头上复用B3已经显示有效的z160+FFT96+RF32辅助主导拼接几何，真实接入密封Phase1 int8旧类聚合原型，仅改变old-old排序；随后用最大新类包络保持校准仅改变new-new排序。域适应与新类注册在同一run中保存注册前/后证据。
- 本轮只读取已保存、唯一LEO_weak信道观测的support；不重建IQ、不增加另一信道overlay、不从一个clean样本派生多信道view，也不重复扩展前期数据准备。FFT96/RF32/z160均由同一份固定接收IQ确定性提取。
- query保持完全未打开：0行、0标签、0分数；不参与适应、校准、选择、回滚或排序。逐样本在全部已注册类上决策，无query角色Oracle、真实batch类别数、类别配额或全局分配。
- int8组件只允许作为与ADV3B02 checkpoint共同绑定的不可变Phase1模型知识；不读取原始IQ、样本级源特征、clean派生信号或可逆索引，不在Phase2更新。

## D27-D29 retrospective后的单一新假设

D28的共同平移不能改变new-new排序；D29逐类boost虽然能改变new-new排序，但同时提高`max_new`，所有严格新类收益均碰撞旧类安全门。D30将两个组内校准写成包络保持算子：

```text
old: u_old = s_old + e_int8
     s'_old = u_old - max(u_old) + max(s_old)

new: u_new = s_new + b_new
     s'_new = u_new - max(u_new) + max(s_new)
```

因此每个样本的`max_old`和`max_new`保持不变，old/new组判定及跨组遗忘相对D30基础头应严格不变；只允许修复old-old和new-new身份混淆。若某新类的`new-group reachable ceiling`低，说明主要错误为new→old，本机制无法修复，下一轮应转向class-balanced/CVaR表征适配，而不是放宽query权限。

## 预注册候选与选择

- 对照固定为Z0、B3、C0；D30-A/B/C均采用B3辅助主导拼接几何+D27-B `15 Stage2-B + 10 Stage2-C`步轻头。
- D30-A/B/C只允许在预注册的int8旧类证据强度和新类support目标优先级上不同；任何额外超参数必须在support打开前写入candidate lock。
- 旧类int8重排门：support-held旧类总体、逐类和floor均不得下降；失败则该分支原子旁路。新类包络门：support-held新类总体、逐类和floor均不得下降，且至少一项严格改善；失败原子旁路。
- `K=1`精确透传基础头；`K=2~4`不伪造OOF；`K>=5`仅用shot-rank support OOF拟合与安全选择。开发seed只允许在统一K=10工作点选择一次候选，正式K1/5/10/20和独立确认矩阵不得再次调参。
- 选择优先级：先满足旧类逐类非退化，再最大化新类floor/最差20%均值，其次seen-new总体和H；不允许用query选择。

## 资源预算

- adapter活动参数上限80k、适配/注册合计不超过30epoch、50 optimizer step、256KB持久状态；D30预期峰值仍为2,016活动参数、25步。
- int8旧类组件按完整实际payload单列，不把临时反量化锚计成可持久FP32 bank；新类包络仅保存`C_new`个FP32偏移量。
- 无dense query图；逐样本包络校准为线性类别复杂度。报告必须同时给出适配MAC、head MAC/query、batch1平均/P95时延、持久状态、峰值RAM/CUDA，以及相对identity-only单qKNN的同口径Pareto变化。缺少同硬件identity时延时保持为未完成证据，不扩展成端到端结论。

## 计划矩阵与证据

- development矩阵：6候选×3个LEO_weak场景×5个held-rank，共90行；每个D30行记录基础头、int8旧类组内重排、新类包络校准三个状态。
- 输出逐scene、逐旧类、逐新类、旧类floor、新类floor、遗忘、H，以及`old-correct/old-wrong-old/old-to-new`、`new-correct/new-wrong-new/new-to-old`和逐类组内可达上界。
- 保存完整training log、support audit、candidate selection、geometry audit、resource audit、run receipt、源代码SHA和N607启动记录。
- 本轮不是正式query确认矩阵；即使support-only为正，也必须先共同密封checkpoint+int8并重建正式method lock，之后才可进入5 receiver×至少5 seed×3场景×2/5/10/20真实seen-new TX的独立确认。

## 本地版本与执行记录

- 工作树在开始时已有大量与本轮无关的修改/未跟踪文件；本轮只管理D30核心、runner的D30最小增量、D30测试、launcher、追溯表和本报告，不触碰或提交无关脏文件。
- 根目录`E:\type10-7`不是Git仓库；本报告的Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，实验启动前镜像到根目录报告面。
- D30 max-new核心已落地：K1精确旁路、K2～4 fail closed、K≥5五折shot-rank OOF、最多两轮breakpoint坐标搜索、逐新类非退化硬门、reachable-ceiling审计；20新类为80B FP32偏移+32B固定部署元数据，0 optimizer step。
- 主agent独立复验：D30核心+D29相邻测试16/16通过；另外对2/5/10/20新类各500个随机state、每state 17行，共2,000个随机state验证old prefix、`max_new`、old/new组判定和旧类身份不变量，全部通过。
- runner已完成D30 candidate lock v8、6候选/90行分发、fold/full-K10资源与geometry、候选选择、full-K10旧类门、CLI、selection和receipt闭包。
- 本地回归：`py_compile`通过；D30 envelope、DALI、D29、compact head与runner相邻测试合计58/58通过；`git diff --check`无错误。独立2,000-state精确不变量压力测试通过。
- 本轮拟提交文件：`code/cvsrffi/stage2_max_envelope_calibration.py`、`code/scripts/run_d25_support_only_concat.py`、`code/scripts/launch_d30_envelope_int8_support_20260718.sh`、两份测试、D30追溯表与本报告。
- 源闭包SHA：runner `5bc4b2eb...bc36`；D30 core `72f933a5...a516`；DALI `c51e1c02...003e`；launcher `c416dbf0...793d`。本地脏`stage2_diag_cosine_exploration.py`不属于本轮且不得同步，launcher锁定已验证远端SHA `14ec9193...1ca`。
- 计划远端目录：`runs/d30_envelope_int8_20260718`与`logs/d30_envelope_int8_20260718`；命令`D30_GPU=0 bash code/scripts/launch_d30_envelope_int8_support_20260718.sh`；环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 实现提交：`cf988f25 feat(stage2): add D30 dual-envelope int8 screen`。提交只包含7个D30文件；工作树其余脏文件均为既有无关改动，未纳入提交。
- 2026-07-18 05:50 CST直连N607 read-only preflight通过：host `dell-DSS8440`，project root可见；8×RTX3090均0%利用率、10MiB占用。live inventory未发现GPU compute或active training process，允许在GPU0启动本轮单任务。
- preflight与inventory完成后本地`ssh.exe`和ESTABLISHED TCP22均为0。计划只同步runner→`code/scripts/run_d25_support_only_concat.py`、D30 core→`code/cvsrffi/stage2_max_envelope_calibration.py`、launcher→`code/scripts/launch_d30_envelope_int8_support_20260718.sh`；其他依赖只核验远端SHA，不覆盖。
- 远端既有依赖SHA全部匹配：DALI `c51e1c02...003e`、D29 core `68633a72...eae`、D28 core `dd9f06ba...db0a`、D27 core `553d6361...f1ff`、D25/D24/CIAF/control均匹配，保留的远端diag operator为`14ec9193...1ca`。
- 已按计划同步3个文件；远端runner/D30 core/launcher SHA分别为`5bc4b2eb...bc36`、`72f933a5...a516`、`c416dbf0...793d`。远端`py_compile`和`bash -n`通过，`REMOTE_VERIFY_PASS_OUTPUT_ABSENT`。
- 每次SSH/SCP后均复核本地`ssh.exe`和ESTABLISHED TCP22为0。启动命令：`D30_GPU=0 bash code/scripts/launch_d30_envelope_int8_support_20260718.sh`；PID `3683614`；GPU0；log `logs/d30_envelope_int8_20260718/support_screen_v1.log`；output `runs/d30_envelope_int8_20260718/output/support_screen_v1`。
- 运行中观测：约2分25秒时进程`Rl`、CPU约572%、CUDA显存552MiB、log仍为0B、receipt pending；这是五折support闭式坐标搜索阶段，尚无错误证据。继续短连接监控，不干预任务。

## 结果与决定

- 待本地实现、N607 90行矩阵和完整日志审计完成后填写。

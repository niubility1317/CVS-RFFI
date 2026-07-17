# D31全注册类新后缀轻量适应support-only实验

## 实验登记

- 实验ID：`d31_all_registered_suffix_20260718`。
- 时间：2026-07-18；operator：Codex。
- 阶段：正式Stage2-B/C之前的development support-only筛选。
- 探索节奏：D31是D27-D29强制回顾后的第2轮；D30为回顾后第1轮，因此本轮结束后仍可执行D32，第3轮完成后必须在D33前再次回顾。
- 当前状态：`V1_COMPLETE_V2_EVIDENCE_REPAIR_PENDING_LAUNCH`。
- 目标：把D30确认有效的B3辅助主导拼接几何压入活动上限，并补上其关键机制——Stage2-C使用全部old+new注册support，让旧support成为新类权重的负证据；同时用新类CVaR优化floor，用旧类margin降低注册后遗忘。
- 比较对象：同run内`Z0`、超步数诊断`B3`、`C0`与D31-A/B/C；同一候选必须联合报告注册前旧类、注册后旧类、seen-new、H、逐类floor和遗忘。

## 协议边界

- receiver `20-1`、seed `713101`、K=10、5个seen-new TX、3个LEO_weak场景；本轮预计6候选×3场景×5折=90行。
- 每个support physical sample进入Phase2前只有一个已经叠加的LEO_weak信道观测。z160、FFT96和RF32只从该同一固定接收IQ确定性提取并拼接；不重建clean IQ、不增加第二个信道overlay、不把一个clean样本派生为多个support view。
- query等同测试集，本轮不打开query：0行、0标签、0分数，不参与适应、校准、候选选择、回滚或排序。预测必须逐样本面向全部已注册类，不得使用query角色Oracle、真实batch类别数、类别配额或全局分配。
- Phase2不读取clean/source样本、样本级原始/全精度特征、可逆索引或未授权派生信号。用户授权的Phase1 int8域×类聚合组件只读且不可更新；当前仍是pre-formal development screen，不产生正式性能或部署声明。
- K=1必须执行质心注册和零梯度更新精确旁路；本轮统一K10选参后，正路线再进入K=1/5/10及正式多receiver、多seed矩阵。

## 机制与候选

共同特征为辅助主导拼接：

`f=normalize([z160,4*(FFT96||RF32)])`。

共同Stage2-B复用D26的15步compact diagonal old adaptation。Stage2-C冻结共享对角和全部旧类权重，仅更新new suffix；每个optimizer step都使用全部已注册old+new support的class-balanced全类交叉熵，最多7个新类分块更新，避免参数随20新类失控。

|候选|Stage2-C机制|Stage2-C步数|总步数|目的|
|---|---|---:|---:|---|
|D31-A|全注册类balanced CE|10|15+10=25|验证旧support负证据本身|
|D31-B|balanced CE+top20%新类CVaR，权重0.35|10|15+10=25|直接抬升最弱新类/floor|
|D31-C|B的floor机制+旧类margin权重0.75、logit margin0.9|15|15+15=30|强化注册后旧类保护|

三者均使用质心锚；B/C权重分别为0.02/0.05。活动参数峰值上限2,016，满足adapter≤80k；总step≤30，满足放宽后的≤50step和≤30epoch；无dense query图。

## int8 slim medoid口径

- 当前development screen仍必须只读核验完整历史84-cell int8组件，不能在target support打开后把它伪装成已经预先密封的slim bundle。
- 正式部署bundle重建目标：每个旧类只保留一个离线固定medoid的160维int8锚，并绑定每类scale与radius。6类的核心payload约为`6×160B+6×4B scale+6×4B radius≈1,008B`；加入句柄、量化元数据和对齐后按约1.34KB审计。
- 该1.34KB是目标部署口径，不是当前历史84-cell组件的实测常驻state。正式Stage2-B/C前必须在任何target访问之前与checkpoint共同生成、绑定和密封，且Phase2不可更新。

## 本地实现与验证

- 计划变更：
  - `code/cvsrffi/stage2_all_registered_new_suffix.py`：D31全注册support新后缀训练、K1旁路、support safety gate和资源审计。
  - `code/scripts/run_d25_support_only_concat.py`：`d31_v1`候选锁、90行分发、before/after、confusion、resource、selection与receipt。
  - `code/scripts/launch_d31_all_registered_suffix_20260718.sh`：N607源闭包和唯一输出启动入口。
  - `tests/test_stage2_all_registered_new_suffix.py`及runner邻接测试。
- D31核心、runner、DALI、D30 envelope和D26 compact邻接测试60/60通过；`py_compile`、launcher `bash -n`和`git diff --check`通过。
- 随机不变量压力测试覆盖2/5/10/20新类、K=1/5、A/B/C三种机制，共72个状态；旧对角阵和旧权重逐字节不变，注册后的旧分数前缀与注册前位级一致，new bias不为正，活动参数≤2,016且总步数≤30。该测试还促使推理改为旧前缀与新后缀分别点积后拼接，避免宽GEMM造成旧列末位浮点漂移。
- 已执行验证命令：显式使用`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile ...`；执行D31 core、DALI、D30 envelope、D26 compact与runner相邻测试共60项；执行`bash -n code/scripts/launch_d31_all_registered_suffix_20260718.sh`和`git diff --check`，均通过。
- launcher当前锁定共享文件的本地SHA快照：runner `282f7917...53d2`；D31 core `db657629...0116`。共享runner若在统一回归前继续变化，必须刷新SHA；不得带旧hash启动。
- diag operator是既有独立脏文件，本轮不修改、不上传；launcher只核验已经验证的远端SHA `14ec9193...1ca`。

## N607执行计划

- 2026-07-18 06:24 CST直接SSH preflight通过：host `dell-DSS8440`，项目根可见，8×RTX 3090均为0%利用率且10MiB占用；live inventory无active training process。检查后本地`ssh.exe=0`、到N607/bridge的ESTABLISHED TCP22连接=0。
- 本地Git提交：`f2745221 feat(stage2): add D31 all-registered suffix route`。
- 仅同步了runner、D31 core和launcher；远端SHA分别为`282f79179f51f70bc9d30eafb8db43ed4859cc9faac7d51589a65c2e5c6e53d2`、`db65762955bfcddee7ff22f5f154b142a654849a93eddf2a64e55e31cbdd0116`和`4f5cd6b5beb264f720659c80d7a941c90998fc136cade81d8d2aac2e0614ce8f`。
- 远端全部继承依赖SHA、`py_compile`、`bash -n`和唯一输出不存在检查通过；diag保持远端既有`14ec9193...1ca`，未同步。同步/核验后本地SSH进程和TCP22连接均为0。
- 本地验证和Git提交后，先运行`tools\n607_ssh_preflight.ps1`与live process/GPU inventory；若已有任务且未获授权干预，转为monitor-only。
- 计划只同步runner、D31 core和D31 launcher；其他依赖只读核验远端SHA，不覆盖本地或远端无关改动。
- 远端cwd：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 启动命令：`D31_GPU=0 bash code/scripts/launch_d31_all_registered_suffix_20260718.sh`。
- run：`runs/d31_all_registered_suffix_20260718`；output：`runs/d31_all_registered_suffix_20260718/output/support_screen_v1`；log：`logs/d31_all_registered_suffix_20260718/support_screen_v1.log`；GPU、PID和启动时间待preflight后回填。
- 实际启动：2026-07-18约06:27 CST，GPU0，PID `3722010`；启动命令成功返回，随后本地SSH进程和到N607 TCP22连接均为0。启动成功仅表示任务landed，不等于artifact完成或性能达标。
- 每次SSH/SCP后必须确认本地`ssh.exe`与到N607 TCP22连接均已退出。

## 成功标准与需回填证据

本轮development筛选首先要求相对D30提升联合H、09f8 floor和注册后旧类floor，并保持逐类安全；正式目标仍是K10下target-old总体≥92%、每旧类≥88%、5新类seen-new≥92%，K5每项下降不超过3pp。单个边缘指标、support训练准确率或不同候选的独立最大值不能拼接成成功声明。

实验完成后必须回填：

- 90/90行完整训练日志解析、无NaN/Inf/OOM审计和每阶段loss/梯度/支持集表现。
- 每候选同一行的注册前旧类、注册后旧类、seen-new、H、遗忘、最差fold旧/新floor。
- 三场景、逐旧TX、逐新TX尤其09f8的new→old/new→wrong-new混淆。
- D31机制启用/旁路、support safety gate、K1旁路、full-K10 refit状态。
- 峰值活动参数、optimizer step、适配MAC、逐query MAC/CPU延迟、完整历史组件与目标slim medoid两种state口径，以及相对identity-only单qKNN的Pareto变化。
- 合法receiver/TX/support清单、唯一LEO_weak view审计、query/clean/source不可达、远端源SHA、artifact SHA和自动selection/receipt。

## 结果与决定

`support_screen_v1`已在20.47秒内完成90/90行，但审计发现`selection.json`因D31分支遗漏而把C0 fallback误写成`selected_positive_route=true`，同一run的RECEIPT正确写false；资源表还漏计Stage2-B MAC且缺batch-1 latency。原v1 artifact保持不可变并标记口径缺陷，不用于正路线声明。

已在提交`7aef0776 fix(stage2): repair D31 selection and resource audit`中统一selection/receipt正路线集合、补入Stage2-B/总适配MAC与batch-1 latency；launcher改为唯一`support_screen_v2`输出，准备低成本证据修复复跑。v1初步联合指标表明D31-B为85.56/67.78/72.00/H69.06/遗忘17.78pp，D31-C为85.56/76.11/60.67/H66.80/遗忘9.44pp，均未超过B3联合性能且不达正式目标。

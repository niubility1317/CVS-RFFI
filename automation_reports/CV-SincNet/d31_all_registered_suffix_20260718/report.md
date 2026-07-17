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

## 完成状态与artifact

- `support_screen_v2`于2026-07-18约06:37 CST在GPU0、PID `3727726`完成；elapsed 19.23秒，90/90行。selection与RECEIPT均选择C0且`selected_positive_route=false`；性能字段与v1逐行一致。
- 本地artifact：`E:\type10-7\automation_reports\CV-SincNet\d31_all_registered_suffix_20260718\remote_output_v2`。SHA：training `928ac09a...6125e`、support `304d127b...281d`、selection `31f5d4cb...55b9`、resource `84017461...734a`、geometry `4c36b902...f98`、receipt `9f820422...e6e6`；全部与RECEIPT绑定一致。
- v1保留为不可变问题证据：其selection误写positive route、资源漏计Stage2-B MAC且缺batch-1 latency；不得用v1 selection作晋升声明。

## 联合结果

以下均为development support-held结果，不是query/formal性能。每一行保持同一候选的注册前旧类、注册后旧类、新类、H和遗忘。

|候选|注册前旧类|注册后旧类|seen-new|H|遗忘|最差折旧/新floor|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|0%/0%|控制|
|B3诊断|86.67%|73.33%|73.33%|72.65%|13.33pp|0%/0%|性能最高但60步、仅诊断|
|C0|71.67%|50.56%|54.00%|50.35%|21.11pp|0%/0%|自动fallback|
|D31-A|85.56%|65.56%|71.33%|67.35%|20.00pp|0%/0%|非正路线|
|D31-B|85.56%|67.78%|72.00%|69.06%|17.78pp|0%/0%|D31内联合最好，仍低于B3|
|D31-C|85.56%|76.11%|60.67%|66.80%|9.44pp|0%/0%|旧类保护增强但新类显著受损|

### 三场景D31结果

|候选|场景|注册前旧类|注册后旧类|seen-new|H|遗忘|pooled旧/新floor|
|---|---|---:|---:|---:|---:|---:|---:|
|A|clear|88.33%|70.00%|80.00%|73.94%|18.33pp|50%/60%|
|A|low-elev|78.33%|63.33%|66.00%|64.44%|15.00pp|50%/10%|
|A|rain|90.00%|63.33%|68.00%|63.66%|26.67pp|40%/40%|
|B|clear|88.33%|70.00%|80.00%|73.94%|18.33pp|50%/60%|
|B|low-elev|78.33%|66.67%|68.00%|67.19%|11.67pp|60%/10%|
|B|rain|90.00%|66.67%|68.00%|66.06%|23.33pp|40%/40%|
|C|clear|88.33%|80.00%|64.00%|70.70%|8.33pp|60%/30%|
|C|low-elev|78.33%|71.67%|64.00%|67.06%|6.67pp|60%/30%|
|C|rain|90.00%|76.67%|54.00%|62.65%|13.33pp|50%/30%|

### 逐类结果

|旧TX|注册前|D31-A注册后|D31-B注册后|D31-C注册后|
|---|---:|---:|---:|---:|
|20-15|96.67%|83.33%|90.00%|93.33%|
|8-20|90.00%|90.00%|90.00%|90.00%|
|14-10|83.33%|53.33%|53.33%|70.00%|
|14-7|80.00%|56.67%|60.00%|66.67%|
|6-15|86.67%|56.67%|56.67%|66.67%|
|20-19|76.67%|53.33%|56.67%|70.00%|

|新类handle前缀|D31-A|D31-B|D31-C|
|---|---:|---:|---:|
|09f8|36.67%|36.67%|36.67%|
|1c2a|83.33%|86.67%|50.00%|
|b8fb|70.00%|70.00%|70.00%|
|d3af|86.67%|86.67%|83.33%|
|f608|80.00%|80.00%|63.33%|

09f8仍同时存在new→old和new→wrong-new：D31-B为11/30正确、7/30判旧、12/30判错新类，不能只靠统一新旧门控解决。

## 完整训练日志诊断

- D31共45个fold、1,290个逐步snapshot：A/B各405，C 480。递归数值扫描非有限值0；严格token扫描NaN/Inf/OOM/Killed/Traceback/Exception均0。
- Stage2-B共同loss `1.02394→0.12399`，旧support准确率`94.44%→100%`、floor `80.83%→100%`。
- A/B的Stage2-C虽然loss分别降至3.437/3.389，新support约94.67%/92.83%，但安全bias施加前旧support准确率始终为0；最终旧类主要依赖均值约-7.07/-6.63的事后非正bias恢复。
- C把raw旧support提高到92.64%、floor 64.17%，但新support从91.17%降到77.50%、新floor从70%降到10.83%，且15/15轨迹均有loss反弹；统一old margin过强。
- DALI在45/45折启用，但只修正一个low-elev旧类内部错误，对新类、新旧组判断及full-K10状态均无改变；约贡献+0.56pp旧类总体。

## 资源与Pareto

|候选|峰值活动参数|总步数|完整适配MAC|query MAC|batch-1 CPU mean/p95|当前完整bundle state|slim投影state|
|---|---:|---:|---:|---:|---:|---:|---:|
|A|2,016|25|15,897,600|4,416|0.230–0.425/0.265–0.432ms|52,071B|16,340B|
|B|2,016|25|15,897,600|4,416|0.233–0.238/0.272–0.276ms|52,071B|16,340B|
|C|2,016|30|21,124,800|4,416|0.240–0.258/0.278–0.282ms|52,082B|16,351B|

- identity-only单qKNN基线为17,600 MAC/query与35,200B FP16 sample state。D31 query MAC下降74.91%；当前实际full-bundle state高47.93%，不在state Pareto；1.34KB fixed-medoid重封后的总state投影低约53.58%，但仍未落地，不能当部署实测。
- 头部运行是NumPy CPU FP32，`head_peak_cuda_memory_bytes=0`；此值不包含ADV3B02 backbone、FFT96/RF32特征提取。每场景110行特征提取的backbone为3946/344/349ms，FFT约22/16/16ms，RF约41/41/41ms。

## 协议与清单闭环

- receiver `20-1`、seed `713101`、K10；旧TX为14-10、14-7、20-15、20-19、6-15、8-20，新类为不可逆handle 09f8/1c2a/b8fb/d3af/f608。artifact没有新类原始可读TX名，正式合法TX表必须从sealed manifest补映射，不能猜测。
- 每场景60旧+50新=110行，总330行；330个唯一overlay token。每场景physical ID、parent IQ hash均110/110唯一，三场景两两重叠0；support view=1、row multiplicity=1、derived row=0、additional overlay=0。
- query opened/rows/labels均0；角色Oracle、真实batch类别数、类别配额、全局分配均false；clean/source/cache/control-flow不可达；int8组件只读不可更新。
- 组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，因此本轮仅是`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`，formal metric/performance claim均false。

## 决定

D31验证了全注册support+CVaR可相对D30小幅提升H约0.87pp，但没有超过B3，也没有解决floor。根因是训练时未使用最终安全bias分数面：A/B的旧类在raw训练面全崩，C又用过强统一margin牺牲新类。D32将把每类非正安全cap从step0放入每次forward，并按新类逐类控制bias预算；不继续扩大统一old margin。D32是本次回顾后的第3轮，完成后必须在D33前执行新的记录回顾。

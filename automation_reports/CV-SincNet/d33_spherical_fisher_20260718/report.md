# D33球面同尺度注册与Fisher快速适应实验

## 登记

- 实验ID：`d33_spherical_fisher_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`D33_COMPLETE_NEGATIVE_ROUTE_RESOURCE_EVIDENCE_CLOSED`。
- 前置回顾：D30-D32三轮回顾已完成并提交。D30静态envelope旁路、D31事后bias存在训练/部署面不一致、D32约-9内生cap在fit support安全但held泛化失败；D33停止继续扫描bias/CVaR/DALI权重。
- 目标：用old/new统一球面centroid、robust radius和`-d/r-log(r)`评分消除跨组标尺失配，同时以对角Fisher近闭式Stage2-B降低适配MAC；域适应与新类注册保持同run、同等优先。
- 比较：Z0、历史B3诊断、C0、D33-A/B/C、D33-B3-FAST；7候选×3场景×5折=105行。

## 方法锁

所有D33候选使用同一个288维拼接身份空间：同一已接收LEO_weak IQ的z160、FFT96、RF32经块归一化后拼接；不增加物理样本、信道overlay或support view。

对每个注册类`c`，在冻结Stage2-B对角阵`D`后计算：

`u_i=norm(exp(D)⊙x_i)`，`p_c=norm(mean_{i:y_i=c}u_i)`，`d_c(x)=1-u(x)^T p_c`。

K>1时只用support LOSO自类角距离，在固定36点网格`q∈{0.5,0.75,0.9}`、`rho∈{0.25,0.5,0.75,1}`、`cap∈{1.15,1.25,1.5}`中估计类半径：原始分位数向全类median收缩并限制半径比。推理逐样本对所有注册类计算：

`s_c(x)=-d_c(x)/max(r_c,1e-4)-log(max(r_c,1e-4))`。

|候选|Stage2-B|Stage2-C/注册|选择目标|优化步|
|---|---|---|---|---:|
|D33-A|D26 AdamW 15步compact diagonal|统一球面centroid+robust radius|LOSO overall优先|15|
|D33-B|同上|同上|LOSO harmonic balance优先|15|
|D33-C|同上|同上|LOSO逐类floor优先|15|
|D33-B3-FAST|对角Fisher近闭式+固定5点收缩LOSO|统一球面centroid+balanced radius|联合balance|0|

K=1不存在独立类内半径证据，所有类统一`r=1`，评分严格退化为常数平移的cosine；不构造伪LOO、不做梯度更新。D33不使用DALI进行prediction，授权int8组件仅保持sealed bundle可用状态，不计入active predictor。

## 本地实现与验证

- 新增`code/cvsrffi/stage2_b3_fisher_closed_form.py`：对角Fisher类间/类内方差比、严格分块零均值与`log(1.5)`盒投影；固定5点强度只用old support LOSO选择。
- 新增`code/cvsrffi/stage2_d33_spherical_registration.py`：36点class-symmetric LOSO radius、A/B/C策略、K1旁路、逐样本all-registered scoring。
- 新增两组核心测试与共享runner集成测试；D33、Fisher、runner和原compact相邻测试54/54通过，`py_compile`、launcher `bash -n`和`git diff --check`通过。
- 6旧类K10 Fisher：活动2,016标量，估算865,728 adaptation MAC，相对Adam15参考5,443,200降低84.10%。
- 6旧+20新球面状态：活动7,828参数、实际常驻8,848B、K10适配估算2,564,640MAC；0个Stage2-C优化步，无dense query图。所有old/new centroid统一存为per-class symmetric int8+FP32 scale，不常驻FP32 centroid；相对原FP32 centroid版31,208B降低71.65%。
- 共享runner已完成candidate lock v11、105行fold、K10 full state、完整trace、MAC/延迟/状态、selection/receipt统一positive helper和真实old-score alias语义。2-new合成fold/full audit四个候选全部跑通；FAST为0步，Adam支线为15步。
- 本地SHA：runner `930a565a...5b50a`；D33 spherical `af4da352...50423f`；Fisher `2cc05c0f...5d8ef`；launcher `e5f30c76...ed536`。

## 协议与门禁

- receiver `20-1`、seed `713101`、K10、5个新类、3个LEO_weak场景；沿用现有密封support包，不新增数据准备。
- 每个physical support只有一个已经叠加LEO_weak的IQ观测；z160/FFT96/RF32只是同一IQ的确定性数学描述，不计入K。
- query为测试集且本轮保持未打开；外层held support只做开发泛化评估，不进入训练、半径选择或checkpoint选择。
- 无query标签、query角色Oracle、真实batch类数、类别配额、全局分配；预测是逐样本all-registered argmax。
- clean/source/cache/control-flow不可达；int8 Phase1组件只读且不更新。当前仍是`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`，不能作为正式query或多receiver确认性能声明。
- 晋级前必须同时改善注册后old、new、H、forgetting和逐类floor，重点核查14-7、09f8、f608；未超过B3同行联合指标不得扩展正式5 receiver×5 seed×3场景矩阵。

## N607计划

- 本地runner和核心测试已全部通过；下一步执行直接N607只读preflight与live GPU/process inventory。
- 仅同步共享runner、D33 spherical core、B3 Fisher core和D33 launcher；不修改或上传`stage2_diag_cosine_exploration.py`，远端固定SHA必须保持`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`。
- 计划远端cwd：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU由preflight后记录。
- 计划输出：`runs/d33_spherical_fisher_20260718/output/support_screen_v1`；日志：`logs/d33_spherical_fisher_20260718/support_screen_v1.log`。

2026-07-18 07:48 CST直接preflight通过：host `dell-DSS8440`，项目根可见，8×RTX 3090均0%利用率且约10MiB显存；live inventory显示无active training process、无GPU compute context。检查后本地`ssh.exe`和TCP22 ESTABLISHED均为0。按每GPU最多2个训练进程规则，GPU0可用于本轮单个轻量support screen。

仅同步runner、D33 spherical core、B3 Fisher core和launcher到对应远端`code/scripts`/`code/cvsrffi`路径。远端SHA依次为`930a565a...5b50a`、`af4da352...50423f`、`2cc05c0f...5d8ef`、`e5f30c76...ed536`；远端`py_compile`、launcher语法、唯一输出不存在检查通过，diag仍为`14ec9193...1ca`。同步与核验结束后本地SSH/TCP22连接均为0。

## N607 v1完成结果

- 启动：GPU0、PID`3759170`；远端输出`runs/d33_spherical_fisher_20260718/output/support_screen_v1`，完成105/105行，用时20.464s。
- 本地证据镜像：`E:\type10-7\automation_reports\CV-SincNet\d33_spherical_fisher_20260718\remote_output_v1`。`training_log.jsonl`逐行及嵌套trace均为有限数；无OOM、Killed、Traceback或异常终止。
- RECEIPT状态为`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`；selection选择C0 fallback，`selected_positive_route=false`。本轮未打开query，不能形成正式性能声明。

|候选|注册前旧类总体|注册后旧类总体|seen-new总体|held均值H|遗忘|
|---|---:|---:|---:|---:|---:|
|Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|
|B3历史诊断|86.67%|73.33%|73.33%|72.65%|13.33pp|
|C0|71.67%|50.56%|54.00%|50.35%|21.11pp|
|D33-A|85.56%|71.67%|63.33%|66.15%|13.89pp|
|D33-B|85.56%|71.11%|63.33%|65.74%|14.44pp|
|D33-C|85.56%|71.11%|63.33%|65.74%|14.44pp|
|D33-B3-FAST|82.22%|70.00%|59.33%|62.19%|12.22pp|

D33-A逐场景为clear`73.33/70.00/H71.24/F15.00`、low-elevation`65.00/64.00/H63.00/F13.33`、rain`76.67/56.00/H64.21/F13.33`；顺序均为注册后old/new/H/forgetting，单位为百分比或pp。B/C策略在15/15折给出同一选择与同一结果，说明选择目标在当前球面几何上坍缩。FAST虽把遗忘降至12.22pp，但相对B3的H下降约10.46pp。

|TX类|B3旧类|D33-A旧类|
|---|---:|---:|
|20-15|90.00%|93.33%|
|8-20|90.00%|90.00%|
|14-10|70.00%|60.00%|
|14-7|66.67%|63.33%|
|6-15|60.00%|63.33%|
|20-19|63.33%|60.00%|

|seen-new类handle|B3|D33-A|
|---|---:|---:|
|09f8|40.00%|16.67%|
|1c2a|86.67%|90.00%|
|b8fb|76.67%|80.00%|
|d3af|86.67%|83.33%|
|f608|76.67%|46.67%|

所有D33候选最差单fold逐类floor均为0；D33-A的pooled场景old/new floor为clear`50/20`、low-elevation`50/0`、rain`60/30`。因此当前路线未满足support开发门禁，更不能进入正式5 receiver×5 seed确认矩阵。

训练过程本身稳定：A/B/C各15/15折loss单调下降，均值`1.02394→0.12399`，fit support总体/floor均达到100%；FAST 15/15折选择完整Fisher强度1.0。失败发生在Stage2-C：对称球面重建后，A/B/C/FAST的旧类support non-degradation均为0/15折，full-K10三场景也全部为false；旧Stage2-B参数保持冻结，但最终old score columns改变。结论是“全类对称球面重注册+单半径评分”破坏旧类决策面，而非梯度优化崩溃。后续保留FAST Fisher，停止晋升当前球面Stage2-C，转向冻结旧类决策面和碰撞局部修正。

## v1资源审计缺陷与v2修复

v1 artifact把query MAC记录为3,212，但该公式仅计11类×`(288+4)`，漏掉每个query的288维对角变换，因此v1性能、protocol和artifact哈希有效，资源数字不可作为最终证据。v1其余实测资源仅用于诊断：A/B/C为15步、2,016 peak trainable、约121–122ms support适配；FAST为0步、约103ms；CPU FP32 batch1 head约0.095–0.100ms。`head_peak_cuda_memory_bytes=0`只表示numpy head不占CUDA显存，不代表完整主干VRAM为0。

v2直接用`(u·q_int8)×1/||q_int8||`评分，不再逐query构造FP32反量化中心矩阵；每类额外密封一个FP32逆范数。统一资源公式改为：

`MAC_query=288+C×(288+1+4)=288+293C`。

其中计入对角变换、int8 centroid点积、每类逆范数缩放和radius score；11类为3,511MAC，相对同注册类数identity-only单qKNN的17,600MAC降低80.05%。K10 6旧+5新状态由4,408B改为4,452B；6旧+20新状态为8,952B，active参数7,854，Stage2-C适配MAC为2,572,128。该修改不改变support、特征、半径、标签权限或逐样本决策协议；直接int8评分与原临时反量化面在本地随机回归中的最大FP32差约`5.7e-6`。

v2本地19项D33/Fisher/runner集成测试、`py_compile`、launcher `bash -n`与`git diff --check`已通过。下一步只同步修订后的D33 core和v2 launcher，重跑唯一`support_screen_v2`，并验证逐行预测/指标与v1一致后关闭资源证据。

2026-07-18 08:04 CST v2直接preflight再次通过：host、项目根和8×RTX 3090均正常，live inventory无active training process和GPU compute context；检查后本地SSH/TCP22均为0。本地Git修复提交为`776a7ae0`；D33 core SHA为`b60ec8a2...630d4`，v2 launcher SHA为`f2b96f41...cb941`。本轮只允许以下同步映射：

- `code/cvsrffi/stage2_d33_spherical_registration.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/stage2_d33_spherical_registration.py`
- `code/scripts/launch_d33_spherical_fisher_20260718.sh`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_d33_spherical_fisher_20260718.sh`

runner、Fisher及全部旧依赖保持已核验SHA，不同步；diag必须继续保持`14ec9193...1ca`。远端启动命令锁为`D33_GPU=0 bash code/scripts/launch_d33_spherical_fisher_20260718.sh`，输出与日志分别为`support_screen_v2`和`support_screen_v2.log`。

## N607 v2完成与最终判定

- GPU0、PID`3766114`；18.246s完成105/105行。远端输出`runs/d33_spherical_fisher_20260718/output/support_screen_v2`，本地镜像`E:\type10-7\automation_reports\CV-SincNet\d33_spherical_fisher_20260718\remote_output_v2`。
- v1/v2按candidate、scenario、fold、held ranks、注册前/后逐类指标、H、forgetting、joint floor及全部claim字段比较，105/105行完全一致；v2优化未改变任何实验判断。
- RECEIPT仍为`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`，selection仍为`D25-C0-DIM-CONCAT`且`selected_positive_route=false`，query未打开。v2所有JSON递归有限，无NaN/Inf；无OOM、Killed、Traceback。
- artifact SHA闭合：training`86d1c333...d4549`、selection`9d336246...d6ad`、resource`8d87816b...62af`、support`4355acfd...40c4`、geometry`0302185b...9a98`；均与RECEIPT一致。
- support仍为3场景×110=330个唯一物理IQ观测，每场景6旧×K10+5新×K10；每样本multiplicity/view均为1，新增overlay/物理样本/derived row均为0，场景间physical ID、parent IQ SHA和overlay token交集均为0。query rows/labels均为0，clean/source权限全false，逐样本all-registered、无角色Oracle、真实batch类数、quota或global assignment。

|路线|优化步|peak trainable|适配MAC|MAC/query|常驻状态|batch1 mean/P95|
|---|---:|---:|---:|---:|---:|---:|
|D33-A/B/C|15|2,016|5,996,808|3,511|4,452B|0.0706–0.0719/0.0764–0.0781ms|
|D33-B3-FAST|0|0|1,419,336|3,511|4,452B|0.0700–0.0701/0.0758–0.0765ms|

四个D33候选的v2 batch1 mean均值为0.07070ms，相对v1临时反量化实现的0.09737ms下降27.39%；P95均值由0.11097ms降至0.07681ms，下降30.79%。相对17,600MAC/query的K10 identity-only单qKNN，D33 head为19.95%，即MAC下降80.05%。这是post-backbone numpy CPU FP32 head口径；`head_peak_cuda_memory_bytes=0`不外推为端到端模型显存。

最终机制判定：D33作为资源工程闭环成功，但作为性能路线失败。保留`D33-B3-FAST`的闭式Fisher Stage2-B组件，拒绝“旧新类完全对称球面重注册+单半径LOSO”Stage2-C。下一轮必须把旧类决策面冻结为bitwise不变，只对support识别出的old-new碰撞对增加局部有限分数，并把旧类non-degradation作为候选生成与LOSO排序硬约束；同时单独提升注册前旧头，因为FAST当前注册前旧类82.22%本身不足92%目标。

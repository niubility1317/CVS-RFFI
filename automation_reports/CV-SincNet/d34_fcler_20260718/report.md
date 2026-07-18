# D34-FCLER冻结碰撞局部注册实验

## 登记

- 实验ID：`d34_fcler_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`REMOTE_COMPLETE_NEGATIVE_NO_PROMOTION`。
- 目标：保留D33验证有效的FAST Fisher零梯度Stage2-B，完全冻结旧类score前缀；Stage2-C只在support证明存在混淆的old-new局部边上注册int8新类原型，优先消除注册遗忘和floor类侵入。
- 比较：Z0、C0、历史B3、D33-FAST负对照、D34-A/B/C；7候选×3个LEO_weak场景×5个独立held-rank折=105行。
- 边界：本轮仍是开发support-only screen，不打开query，不作正式receiver/seed性能声明。若D34不能同时优于D33-FAST的old、new、H、forgetting和floor，则不进入正式矩阵。

## 机制锁

同一已接收LEO_weak IQ只生成一行288维`[z160,FFT96,RF32]`拼接特征。FAST Fisher仅用旧类support闭式得到冻结旧头`g_i(x)`；Stage2-C不得修改旧log-diagonal、旧centroid或旧score：

`S_i(x)=g_i(x), i∈Y_old`。

每个新类用独立support生成mean/medoid shrink原型`p_j`，最终只保存int8中心、FP32 scale和inverse norm。碰撞图`E_ij`只由新类support在冻结旧头上的top-old频次生成。对旧锚`i`，所有相邻新类共享安全offset：

`alpha_i=max(0,max_{x∈G_i,j:E_ij=1}[z(x)^T p_j-g_i(x)]+1e-4+lambda*u_i)`。

逐样本先取冻结旧头winner`i*`，然后对全部注册类给出有限score：

`S_j(x)=z(x)^T p_j-alpha_i*`，若`E_i*j=1`；否则`S_j(x)=g_i*(x)-2`。最终只做逐样本all-registered argmax，无query角色、标签、真实batch类数、配额或全局分配。

|候选|每新类碰撞边|新类原型|不确定度保护|
|---|---:|---|---|
|D34-A|top-1|mean|`lambda=0`|
|D34-B|top-2|25% medoid shrink|`lambda=0.5`|
|D34-C|达到top-old频次80%覆盖的最小top-1至top-3|25% medoid shrink|`lambda=1`；低于旧类median accuracy的类再×2|

K1只使用单个独立物理support作为原型，不制造LOSO或派生shot；Fisher走identity安全分支，`u_i`使用K10 method lock的固定0.05。K5/K10只从本K可达support重算闭式状态；统一arm和超参数只能在开发K10锁定。

## 选择与门禁

每折分别执行old LOO侵入审计和new LOO注册审计。候选排名顺序为：旧score prefix逐bit不变→旧support逐类/floor non-degradation→old LOO零侵入→最差场景joint floor→new最低类LOSO→H→new总体→更少碰撞边/MAC。若新类不存在同时满足旧类安全与正new LOSO margin的边，必须输出`UNREACHABLE_COLLISION_EDGE`，不能被总体均值掩盖。

D34成功也只解决注册抗遗忘；D33-FAST注册前旧类held总体82.22%，冻结旧头不会自动达到92%。若注册层验证成功，下一轮必须保留D34 Stage2-C并单独增强Stage2-B旧头。

## 协议与执行计划

- receiver`20-1`、seed`713101`、K10、5个新类、3个LEO_weak场景；复用D33同一密封support包，不新增数据准备。
- 每个physical support只有一个已叠加LEO_weak的IQ观测；三个数学描述来自同一IQ，support multiplicity/view均为1。
- query rows/labels/features均为0；无clean/source、角色Oracle、真实batch类数、quota、global assignment或dense query图。
- 预计0 optimizer step、活动状态远低于50k；必须同时报告平均/最坏碰撞degree、MAC/query、适配MAC、延迟、状态、head显存口径和相对identity-only单qKNN变化。
- 本地验证后先Git提交，再执行直接N607 preflight、live inventory、最小文件同步、SHA闭合和唯一输出检查；计划GPU0单任务，输出`runs/d34_fcler_20260718/output/support_screen_v1`。

## 完成后回填

待回填本地测试、Git提交、远端命令/PID/GPU、105行完整日志、逐候选/场景/类矩阵、碰撞边与不可达类、资源审计、support/query清单、artifact SHA、selection/RECEIPT及下一轮判定。

## N607启动前闭环

- 2026-07-18 08:54 CST直接SSH preflight通过：host`dell-DSS8440`，项目根存在；8张RTX 3090均0%利用率、10MiB显存，live inventory未发现训练进程；`/home`可用7.6TB；目标输出不存在。
- 已只同步上节登记的3个文件。远端SHA闭合：runner`e7eea547f57fe9a15698273ebb7dc36a698ffe073e549c89567b4d7e4d0d91a3`，D34 core`63d38feaee0a899eb07c57d761b74b011442dde7d2da8b8082242361cdda4957`，launcher`65dfc262b33224ce2e517015f3a827a7d38b3d7c98c480779e71db9a74b38da3`，未同步diag仍为固定`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`；其余13个依赖也逐项匹配launcher锁。
- 远端`bash -n`、runner/core`py_compile`通过。计划命令：cwd`/home/szu2070436088/2510044040/CV-SincNet`，`D34_GPU=0 bash code/scripts/launch_d34_collision_local_20260718.sh`；Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；日志`logs/d34_collision_local_20260718/support_screen_v1.log`；输出`runs/d34_collision_local_20260718/output/support_screen_v1`。
- 启动后回填PID；GPU0最多新增本任务1个进程，预计输出105行。成功标准仍是同一行联合门而非单项最大值；任一旧类held侵入、不可达新类、协议字段缺失或资源审计不闭合都不得晋级。
- 08:55 CST启动成功：PID`3787913`，GPU0；首次14秒探针显示进程运行、约552MiB GPU显存，日志尚未写出结果，RECEIPT尚未生成；SSH客户端与TCP22连接均已退出。

## 本地实现与验证

- 新增`code/cvsrffi/stage2_d34_collision_local_registration.py`：A/B/C固定arm、FAST-adapted unit rows、同尺度`18×cosine`、int8新类原型、support-only top-L碰撞边、每旧锚共享安全offset、稀疏逐样本全类score、K1无伪LOSO、真实new LOO和offset-only old LOO审计。
- 共享runner新增candidate lock v12、7候选精确105行、FAST单次共享变换、外层held-rank真实old intrusion硬门、full-K10三场景审计、资源/几何矩阵和RECEIPT闭环。晋级不再只看候选ID：必须旧类安全、全部新类reachable，并在old/new/H/forgetting/joint floor上联合达到或超过B3与D33-FAST门槛。
- launcher唯一输出为`support_screen_v1`；runner/core SHA已填入，测试明确禁止`__D34_*`占位符。
- D34、D33、Fisher、D26和相邻compact路径77/77测试通过；`py_compile`、launcher`bash -n`和`git diff --check`通过。
- 当前本地SHA：runner`e7eea547...d91a3`；D34 core`63d38fea...a4957`。`stage2_diag_cosine_exploration.py`保持用户其他改动，不纳入本轮编辑；远端仍必须核验固定SHA`14ec9193...1ca`。
- 本轮runner与105行D34-v1执行仅覆盖K10；K1/K5在本报告中只是已锁定的设计边界，尚未纳入执行或性能证据。
- 实现提交：`a1ac74b6 feat(stage2): add D34 collision-local registration`；该提交精确包含D34 core、共享runner增量、launcher、3个相关测试文件和本报告，不包含工作区其他历史修改。
- 计划最小同步映射：`code/cvsrffi/stage2_d34_collision_local_registration.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/stage2_d34_collision_local_registration.py`；`code/scripts/run_d25_support_only_concat.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/run_d25_support_only_concat.py`；`code/scripts/launch_d34_collision_local_20260718.sh`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_d34_collision_local_20260718.sh`。除这3个文件外不向N607同步任何工作区改动。

## 资源预审

20个新类、K10合成规模回归的最终部署状态为：FAST旧状态8,064B+D34注册状态6,124B=14,188B；D34 active数值5,926，0 optimizer step。三arm约0.58–0.60s完成本地fit+完整开发LOO审计。

|arm|边数|部署注册MAC|开发LOO证据MAC|平均额外MAC/query|最坏总MAC/query|
|---|---:|---:|---:|---:|---:|
|A|20|479,280|87,092,800|966.67|3,176|
|B|40|1,055,280|181,332,800|1,933.33|7,816|
|C|20|1,055,280|180,404,800|966.67|3,176|

`部署注册MAC`只统计最终可部署support refit；开发K10 arm选择所需的完整old/new LOO单列，不能冒充星上每次适配开销。query口径复用FAST的一次288维变换和6旧类点积，再只计算当前旧winner相邻的新类int8点积；最坏总MAC已含FAST旧头2,016MAC。对20新类的identity-only K10单qKNN为41,600MAC/query，因此最坏arm B仍下降81.21%，A/C下降92.37%。该合成回归仅验证扩展性和审计公式，不是性能证据。

## 审计修复闭环

启动前只读审计发现并已修复：launcher SHA占位；positive route只看旧安全；new score漏乘temperature；K1未评估却写PASS；reachable只需单个正margin；state有限性检查不足；LOSO开发计算未与部署适配MAC分列。Core内部old LOO现明确是`leave_one_old_out_rebuild_offset`，但正式晋级只信runner外层未参与fit的held-rank old intrusion证据，避免把未重训FAST的内部offset审计冒充完整Stage2-B LOSO。

## N607 v1完成结果

- 运行：2026-07-18 08:55 CST，PID`3787913`，GPU0；105/105行完成，runner计时24.480s。日志`logs/d34_collision_local_20260718/support_screen_v1.log`；输出`runs/d34_collision_local_20260718/output/support_screen_v1`；本地完整镜像`E:\type10-7\automation_reports\CV-SincNet\d34_fcler_20260718\remote_output_v1`。
- RECEIPT状态`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`，`selected_candidate_id=D25-C0-DIM-CONCAT`，`selected_positive_route=false`。D34-A/B/C均未通过旧类held零侵入、全部新类reachable和联合比较门，未晋级。
- 105行矩阵精确覆盖7候选×3场景×5个held-rank折；每折用每类8个fit support、2个独立held support。以下均为开发support held-rank指标，不是query/正式性能声明。

|候选|注册前旧类|注册后旧类|新类|H|遗忘|最大遗忘|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|41.67pp|control|
|D25-C0|71.67%|50.56%|54.00%|50.35%|21.11pp|41.67pp|自动fallback|
|B3诊断|86.67%|73.33%|73.33%|72.65%|13.33pp|25.00pp|最强诊断对照|
|D33-FAST|82.22%|70.00%|59.33%|62.19%|12.22pp|25.00pp|负对照|
|D34-A|82.22%|71.11%|55.33%|60.94%|11.11pp|25.00pp|不晋级|
|D34-B|82.22%|71.11%|57.33%|61.98%|11.11pp|33.33pp|不晋级|
|D34-C|82.22%|71.11%|57.33%|62.23%|11.11pp|33.33pp|不晋级|

D34确实把D33-FAST的旧类均值提高1.11pp、平均遗忘降低1.11pp，但D34-C的新类仍低2.00pp，H仅提高0.04pp，且held旧类侵入和新类不可达均未消除；相对B3仍全面不足。

### D34逐场景

|arm/场景|注册前旧类|注册后旧类|新类|H|遗忘|held旧侵入数|零侵入折|不可达class-fold|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A/clear|81.67%|73.33%|56.00%|62.50%|8.33pp|5|2/5|20/25|
|A/low-elev|76.67%|61.67%|62.00%|60.87%|15.00pp|9|0/5|21/25|
|A/rain|88.33%|78.33%|48.00%|59.45%|10.00pp|6|1/5|25/25|
|B/clear|81.67%|73.33%|64.00%|67.78%|8.33pp|5|2/5|20/25|
|B/low-elev|76.67%|68.33%|56.00%|59.19%|8.33pp|5|2/5|24/25|
|B/rain|88.33%|71.67%|52.00%|58.97%|16.67pp|10|1/5|25/25|
|C/clear|81.67%|73.33%|66.00%|68.89%|8.33pp|5|2/5|20/25|
|C/low-elev|76.67%|71.67%|58.00%|62.99%|5.00pp|3|3/5|23/25|
|C/rain|88.33%|68.33%|48.00%|54.80%|20.00pp|12|1/5|25/25|

### D34-C逐类与floor

旧TX到密封handle的登记顺序为`14-10→75aa6d50`、`14-7→8b02d999`、`20-15→1f33441e`、`20-19→f8dfc2ed`、`6-15→a53ca128`、`8-20→33bbd165`。

|旧TX|注册前held均值|注册后held均值|变化|
|---|---:|---:|---:|
|14-10|73.33%|63.33%|-10.00pp|
|14-7|86.67%|76.67%|-10.00pp|
|20-15|96.67%|83.33%|-13.34pp|
|20-19|63.33%|56.67%|-6.66pp|
|6-15|83.33%|66.67%|-16.66pp|
|8-20|90.00%|80.00%|-10.00pp|

20-19仍是总体floor，14-7也只有76.67%，两者均远低于目标88%。full-K10旧support在clear/low-elev/rain为95.00%/91.67%/93.33%，注册前后逐类完全不退化且floor均80.00%；这说明D34只记住fit support安全边界，不能保证未参与fit的同类物理样本安全。

|新类handle前缀|held均值|15折中为0的折数|
|---|---:|---:|
|09f80039|16.67%|10|
|1c2ad882|80.00%|2|
|b8fbace5|53.33%|3|
|d3afb5d1|83.33%|2|
|f608a348|53.33%|3|

full-K10物理support LOO中，D34-C在clear与low-elev各只有1/5新类reachable，rain为0/5；09f8和f608在所有场景都不可达。A/B/C在15折分别累计66/69/68个不可达class-fold，证明失败不是单一arm超参数问题。

### 协议与证据完整性

- support清单：receiver`20-1`、seed`713101`、K10、旧类6、新类5、`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。每场景before为60个旧support，after为60旧+50新=110；总before/after provenance为180/330条，overlay token均唯一，场景间physical ID、parent IQ SHA和overlay token两两零重叠；旧support在注册前后精确复用。
- `one_physical_support_one_leo_channel_observation=true`、`support_view_count=1`、`support_row_multiplicity=1`、`derived_support_rows=0`、`additional_leo_overlay_count=0`。z160/FFT96/RF32仅是同一接收IQ的3个确定性描述，不增加K。
- query rows/labels均0，`query_opened=false`；逐样本全注册类决策；role Oracle、真实batch类数、类别配额、global assignment、dense query graph均false/0。clean/source样本、cache、control flow与未授权衍生信号均不可达。
- RECEIPT内training log、support、geometry、resource、selection五项SHA均与本地镜像逐字节匹配；全部JSON可解析，105行候选/场景/折无缺失。该历史int8组件仍标记`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，因此仅属用户授权的pre-formal support-only screen，不能产生正式性能声明。

### 资源Pareto

|路线|状态|适配MAC|optimizer step|最坏MAC/query|head延迟|备注|
|---|---:|---:|---:|---:|---:|---|
|identity-only K10单qKNN|35,200B FP16样本状态|0|0|17,600|未重测|11类×10-shot|
|B3|14,618B|28.166M|60|3,456|适配约90ms|峰值CUDA 21.67MB|
|D33-FAST|4,452B|1.419M|0|3,511|约0.073ms|support闭式约72ms|
|D34-A|9,613B|0.999M|0|2,596|0.126–0.144ms|support闭式261–309ms|
|D34-B|9,613B|1.143M|0|3,176|0.132–0.135ms|support闭式262–264ms|
|D34-C|9,613B|1.143M|0|2,596|0.130–0.144ms|support闭式266–271ms|

D34-A/C相对identity qKNN最坏query MAC下降85.25%，D34-B下降81.95%；状态下降72.69%。相对B3，D34更省状态、适配MAC、query MAC且零梯度；相对D33，D34 query MAC更低，但状态、闭式适配时间和实测head延迟更高，只形成部分Pareto，尚不满足“全面更轻且性能更高”。完整开发LOO计算未计入部署适配MAC，已在资源artifact单列。

## 根因与D35方向

根因已经收敛：D34把每个新类只连到其fit support最常见的1–3个旧winner，而逐样本预测只检查当前旧winner相邻的新类。held新样本一旦漂移到未连接旧winner，就被固定`old_winner_score-2`压制；LOO margin中大量约-2的值正是该机制指纹。增加top-2或medoid不能解决跨winner漂移。另一方面，offset只用fit旧support的正确样本取界，能保证fit non-degradation，却无法覆盖held旧样本尾部，所以仍有2–5个full-K old LOO侵入和每场景3–12个外层held侵入。

下一轮不再扩大普通top-L碰撞图，而采用“全局可达、局部安全”的D35：所有新类在每个query上都有int8原型有限score；旧winner仅提供classwise安全阈值/半径归一化，不决定新类是否可见。用old LOO和new LOO共同选取每个新类的support-only校准项，floor旧类使用更保守的尾部分位数；旧FAST score前缀仍逐bit冻结。D35仍限制3个固定arm、0梯度、<=50k状态，先复用同一密封support执行105行K10筛选；若不能同时消除old intrusion并使09f8/f608 LOO转正，则不扩K1/K5。

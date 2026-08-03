# D130轻型DA×精简D92联合代理矩阵报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|d130_joint6_proxy84_20260803_r1|
|时间|2026-08-03（Asia/Hong_Kong）|
|状态|ARTIFACTS_COMPLETE / ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE|
|目标|修复D129真实数值范围的仿射编译缺陷后，运行同一最小完整source-held LOCO联合矩阵|
|主责任|主agent负责方法、协议、结果分析与晋级；唯一Terra Max runner负责N607落地、运行和artifact回收|
|与D129关系|DA候选、42折、K1/K5、六臂、seed和评分门完全不变；只修改Full/Lite仿射state的共同数值尺度|

本轮是7个receiver×6个Phase1已见held class×K1/K5=84原子行/候选，两候选合计168条candidate-row prediction。它是方向性代理，不是Target25真实新类实验，不输出正式N/H_old_new。

## 2.失败证据与修复原理

D129 r3已LANDED并完成真实smoke、prepare，但predict在0/168处因affine FP16 intercept is not representable确定性退出，最大真实预量化截距达到485329，超过FP16上限。r3未打开truth、未运行score，结论为NO_PERFORMANCE_RESULT。

D130对同一仿射state的全部类别共同乘正2次幂alpha=2^e，再进行原有逐类INT8 weight＋FP16 scale/intercept量化。e只由全部类别weights/intercepts的有限峰值决定，使截距和weight scale进入FP16范围；若任何非零weight行的scale会低于FP16最小正规数则fail closed。禁止逐类clip/scale、fallback qKNN或读取query。

共同正缩放在量化前严格保持类别argmax与类别置换等价，但INT8/FP16本身有损，因此receipt明确设置argmax_equivalence_scope=prequantized_common_positive_scaling_only，并设置quantized_any_query_argmax_equivalence_claim=false。wire仍为160C INT8 weights＋C FP16 scales＋C FP16 intercepts，即164C字节，query MAC不变。

## 3.冻结方法与判据

- 候选保持CSPAR-2和SRDH-2，不更新checkpoint全参数，不执行Phase2 optimizer/backward。
- 公共R0只拟合一次并跨候选复用；三头仍为qKNN、D92-Full160代理和对角OAS D92-Lite160。
- K1的F/L严格alias Q；K5固定比较R1Q-R0Q、R0L-R0F、R1L-R1F。
- 每项必须ΔH_retained_held_proxy>0且总正确数严格增加，并且ΔA_retained、ΔA_held_proxy、ΔF_retained非负。
- 完整负收益候选立即关闭且不调参；两候选均失败则本revision结束。

## 4.本地版本与验证

|文件|本地SHA256|用途|
|---|---|---|
|code/cvsrffi/stage2_d129_joint6_heads.py|338cdcd63831efd3dace1ba4bbc0658c71a0eea7cfc0febd275df78bab7e2db0|共同2次幂缩放、下溢门与诚实receipt|
|code/scripts/run_d129_joint6_proxy_matrix.py|4377dce3e4be5dba2271c9aad438a00bec63a5c26fa708a943b808eaa47f5e36|把336份数值audit写入resources|
|configs/d130_joint6_method_lock_20260803.json|本地LF字节905da107dd0e02795ebf458b204b436639171a767a12a3f689f0af6b57e83acf；Git归档CRLF字节619d407ba1faf0c291d31d70521bf8a7964e52f31fd521b4769c40d1d2b9e851|D130冻结方法锁；远端只认归档字节|

- 本地环境：C:\Users\lh594\.conda\envs\ssr-gpu\python.exe。
- 37项聚焦测试全部通过，两个修改入口py_compile通过，git diff --check无错误。
- 独立Terra Max复审：P0=0、P1=0。确认共同尺度、下溢fail closed、query零fit、168矩阵、wire/resource和argmax声明边界均合法。
- 同一SHA的D104 588条真实档案和本地LF method lock 905da107…完成168/168无truth prediction：package SHA=805efb2ac43e294851b1327c6f1dd589b2629d0bd5f85b7b44ec1993acc7938c；prediction SHA=4ff10ca1e938ad20bc584c748c9daef55dd7f018b937d71fd7bdb86f8ddfffcf；resources SHA=b9e32f717219f61b18ef10a20006549f16f3761fad301b3e0f78d6a8f7ba8750；truth_loaded=false。远端Git归档只改变JSON换行字节，不改变解析内容。
- 336份K5 affine audit中指数分布为e=0:327、e=-1:6、e=-2:1、e=-3:2；最大截距由485329缩到63479；非零截距cast-zero=0、subnormal=0；没有任意query量化严格等价声明。
- 上述只是完整功能证据，不是性能结果；本地未打开truth、未运行score。

根目录E:\type10-7不是Git仓库；本报告在Git工作树和根报告面保持字节镜像。发布提交由本报告进入Git后在runner handoff记录。

## 5.N607输入与路径

|项|冻结值|
|---|---|
|run root|/home/szu2070436088/2510044040/CV-SincNet/runs/d130_joint6_proxy84_20260803_r1，创建前必须ABSENT|
|Python|/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|
|D104档案|/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz|
|archive SHA|dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d|
|checkpoint SHA|2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98|
|fixture SHA|d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669|
|method lock SHA|619d407ba1faf0c291d31d70521bf8a7964e52f31fd521b4769c40d1d2b9e851，以Git归档解包字节为release truth|
|capsule/split|d106_ls588_proxy_dd315295 / d104_source_seed104713_v2|
|资源|OMP/MKL/OpenBLAS各1线程，CUDA_VISIBLE_DEVICES空，GPU allocation=none|

从精确Git提交生成tar.gz并同步到run root/input/release.tar.gz，fixture同步到input/d106_fixture.json，解包到source。先核验bundle、fixture、D104、checkpoint、D130 method lock、7个D129/D130源文件和py_compile，再执行。

## 6.冻结命令

所有命令使用绝对PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，并设置PYTHONDONTWRITEBYTECODE=1、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、CUDA_VISIBLE_DEVICES=。

~~~text
<PY> source/code/scripts/run_d129_joint6_real_archive_smoke.py --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --held-receiver AUTO_FIRST --held-class AUTO_FIRST --run-id d130_joint6_proxy84_20260803_r1-smoke --output <RUN_ROOT>/smoke/smoke.json

<PY> source/code/scripts/run_d129_joint6_proxy_matrix.py prepare --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --method-lock <RUN_ROOT>/source/configs/d130_joint6_method_lock_20260803.json --method-lock-sha256 619d407ba1faf0c291d31d70521bf8a7964e52f31fd521b4769c40d1d2b9e851 --capsule-id d106_ls588_proxy_dd315295 --split-id d104_source_seed104713_v2 --run-id d130_joint6_proxy84_20260803_r1 --output-dir <RUN_ROOT>/prepare

<PY> source/code/scripts/run_d129_joint6_proxy_matrix.py predict --package <RUN_ROOT>/prepare/predictor_package.npz --package-sha256 <PACKAGE_SHA> --output-dir <RUN_ROOT>/predict

<PY> source/code/scripts/run_d129_joint6_proxy_matrix.py score --prediction <RUN_ROOT>/predict/predictions.json --prediction-sha256 <PREDICTION_SHA> --plan <RUN_ROOT>/prepare/plan.json --plan-sha256 <PLAN_SHA> --truth <RUN_ROOT>/prepare/truth.json --truth-sha256 <TRUTH_SHA> --output <RUN_ROOT>/score/score.json
~~~

prepare、predict、score必须是不同进程。predict detached启动并写logs/predict.pid，立即核验PID/CWD/cmdline/run-root和日志增长；predict不打开truth，只有168条完整prediction封存后score才允许打开truth。

## 7.健康、停止与回收

仅P0协议/安全违规、错误checkout/hash、输出覆盖、launcher确定性故障，或至少两个不同row在prediction前出现同一确定性异常指纹时停止本run自有进程树。不得因accuracy、H、floor或候选表现差停止。禁止覆盖D129 r1/r2/r3；若D130再出现技术失败，保存partial artifact并返回，不自动创建新run。

预期artifact：smoke、prepare receipt/plan、predictions、resources、score、四份日志、PID/进程/GPU/哈希清单。truth文件默认不回传主分析面。

### 7.1实际执行闭环

- Git提交：`e7b1e006c2d9126112340accb6ef30201e4f6985`；bundle SHA=`97c756c082360ed1a847c251f11547935fb9909b6ffd50b9aec99845b11bde3b`。
- N607状态链：`LOCAL_VERIFIED -> LANDED -> RUNNING -> ARTIFACTS_COMPLETE`。真实smoke两候选通过；prepare封存package/plan；predict一次完成168/168，`truth_loaded=false`，query fit/selection/state update均为0。
- 首次score因父目录尚未创建，以`D129ProxyMatrixError: score output must be a new absolute file`退出；当时没有score输出。经主agent明确授权，仅创建冻结的空`score/`目录并对同一封存prediction执行一次独立score，未重跑smoke/prepare/predict，恢复日志完整保留。
- prediction SHA=`839ec215686bd51b289a05fb62cacb4cd1b7f14335eef2f13e736f40335d636b`；resources SHA=`4ffe5b1aed62a6624f423152c5a5213b3d1a9683dc009bf422614773bcfa0799`；score SHA=`a0021a80bedb0cbe74ce276dc307fc85fd0bfc78623d191d46b0ed96f1e446bf`。
- 最终无run-owned进程，8块GPU均空闲；所有SSH/SCP连接关闭。truth只在prediction完整封存后的独立score进程中打开，未回传主分析面。

## 8.结果与联合判定

本矩阵是7个receiver×6个Phase1已见held class的42折source-held LOCO方向性代理。以下百分数来自同一candidate、同一K、同一arm的完整2268条query池；`F_retained`为六个retained class中的最低准确率。它不包含Target25真实新类，不能输出正式`N/H_old_new`或注册成功声明。

### 8.1完整同臂结果

|candidate_id|K|arm|A_retained|A_held_proxy|H_retained_held_proxy|F_retained|总正确数/2268|
|---|---:|---|---:|---:|---:|---:|---:|
|CSPAR-2|1|R0Q/F/L|82.593%|82.540%|82.566%|65.079%|1873|
|CSPAR-2|1|R1Q/F/L|81.323%|82.011%|81.665%|56.825%|1847|
|CSPAR-2|5|R0Q|87.037%|87.037%|87.037%|66.667%|1974|
|CSPAR-2|5|R0F|85.661%|87.037%|86.344%|67.937%|1948|
|CSPAR-2|5|R0L|86.508%|86.508%|86.508%|66.667%|1962|
|CSPAR-2|5|R1Q|86.720%|86.243%|86.481%|67.619%|1965|
|CSPAR-2|5|R1F|84.603%|87.302%|85.931%|72.063%|1929|
|CSPAR-2|5|R1L|86.243%|86.243%|86.243%|66.349%|1956|
|SRDH-2|1|R0Q/F/L|82.593%|82.540%|82.566%|65.079%|1873|
|SRDH-2|1|R1Q/F/L|82.328%|82.011%|82.169%|63.175%|1866|
|SRDH-2|5|R0Q|87.037%|87.037%|87.037%|66.667%|1974|
|SRDH-2|5|R0F|85.661%|87.037%|86.344%|67.937%|1948|
|SRDH-2|5|R0L|86.508%|86.508%|86.508%|66.667%|1962|
|SRDH-2|5|R1Q|87.037%|87.037%|87.037%|66.667%|1974|
|SRDH-2|5|R1F|85.714%|87.037%|86.371%|73.333%|1949|
|SRDH-2|5|R1L|86.508%|86.508%|86.508%|66.667%|1962|

### 8.2预注册K5主比较

|candidate_id|主比较|ΔA_retained|ΔA_held_proxy|ΔH|ΔF_retained|Δ总正确数|判定|
|---|---|---:|---:|---:|---:|---:|---|
|CSPAR-2|DA：R1Q−R0Q|−0.317pp|−0.794pp|−0.556pp|+0.952pp|−9|失败|
|CSPAR-2|Lite：R0L−R0F|+0.847pp|−0.529pp|+0.164pp|−1.270pp|+14|失败|
|CSPAR-2|联合：R1L−R1F|+1.640pp|−1.058pp|+0.312pp|−5.714pp|+27|失败|
|SRDH-2|DA：R1Q−R0Q|0.000pp|0.000pp|0.000pp|0.000pp|0|失败|
|SRDH-2|Lite：R0L−R0F|+0.847pp|−0.529pp|+0.164pp|−1.270pp|+14|失败|
|SRDH-2|联合：R1L−R1F|+0.794pp|−0.529pp|+0.137pp|−6.667pp|+13|失败|

三项主比较都要求`ΔH>0`、总正确数严格增加，并且`ΔA_retained/ΔA_held_proxy/ΔF_retained`均不为负。CSPAR-2产生真实但总体负迁移；SRDH-2在K5 qKNN上退化为零效应。Lite头的平均与H小幅改善来自retained组，伴随held-proxy和最差类下降；它不是满足“域适应后模型与D92联合提升”的正版本。

### 8.3资源结果

|比较|部署数值状态|query head MAC/样本|拟合解析MAC|显式峰值工作集|K5拟合时延中位数|证据边界|
|---|---:|---:|---:|---:|---:|---|
|D92-Full160|984B|960|9,113,600|668,208B|约12.21–12.30ms|同160维表示的head因果对照|
|D92-Lite160|984B|960|22,400|62,768B|约0.75ms|拟合MAC减少99.754%，工作集减少90.607%；部署wire与Full160相同|
|历史formal D92-288|4,692B|1,728|未在本代理同机重测|未在本代理同机重测|未在本代理同机重测|仅系统级资源参考，表示管线不同|
|CSPAR-2＋Lite160|1,320B|960|头为22,400|头为62,768B|头约0.75ms|相对formal参考状态减少71.867%，不构成性能因果比较|
|SRDH-2＋Lite160|1,650B|960|头为22,400|头为62,768B|头约0.75ms|相对formal参考状态减少64.834%，不构成性能因果比较|

数值可表示性修复有效：336份K5仿射audit中共享指数为`e=0:327、e=-1:6、e=-2:1、e=-3:2`，最大预缩放截距485329经共同正2次幂缩放后不超过63479，非零截距cast-zero和subnormal均为0。此结论只说明D130不再因FP16溢出退出，不改变负性能判定。

### 8.4最终决定

- `CSPAR-2`与`SRDH-2`均记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并关闭。
- 不运行G0 588、fresh63、Target25或125，不修改本轮layer/rank/step/view/seed/shrinkage/阈值，不用边际正值复活候选。
- D130保留两项可复用工程结论：共同正2次幂仿射缩放解决真实FP16表示范围；Lite160显著降低拟合计算与工作集。但二者都不是性能晋级证据。
- 下一研发轮最多允许一条原理不同候选；必须先从参数空间局部残差、Fisher锚定和尾部保持头的可辨识性出发完成设计，再冻结更小的必要矩阵。

## 9.与D62、D91、D92、SVRN的证据分层对比

D62/D92/SVRN是旧定义下的历史target-capsule 125诊断矩阵，D91是单receiver、单seed的15行development单元；它们都不是后来定义的Target25，也不是D130的source-held LOCO代理。以下只在各自原始同row口径内报告，不能把不同矩阵的绝对值直接写成配对增益。

### 9.1D62与D92历史125同slice结果

|方法|slice|B_old|A_old|N|H|forgetting|结论|
|---|---|---:|---:|---:|---:|---:|---|
|D62|K10/new5|86.02%|76.33%|73.57%|74.60%|9.69pp|完整125诊断，非晋级|
|D92|K10/new5|86.111%|76.189%|74.133%|74.803%|9.922pp|H小幅增加，旧类/遗忘未改善|
|D62|K10/new10|86.02%|71.53%|66.75%|68.84%|14.49pp|完整125诊断，非晋级|
|D92|K10/new10|86.111%|72.533%|66.353%|69.106%|13.578pp|旧类与遗忘改善，新类下降|
|D62|K10/new20|86.02%|68.68%|68.78%|68.56%|17.34pp|完整125诊断，非晋级|
|D92|K10/new20|86.111%|71.333%|68.150%|69.555%|14.778pp|相对D62约`A_old+2.653pp、N−0.630pp、H+0.995pp、forgetting−2.562pp`|
|D62|K5/new20|81.32%|61.39%|59.28%|60.03%|19.93pp|完整125诊断，非晋级|
|D92|K5/new20|81.267%|63.711%|58.883%|60.955%|17.556pp|相对D62旧类/H改善，新类下降|
|D62|K1/new20|68.14%|44.03%|27.15%|33.41%|24.11pp|K1整体fallback|
|D92|K1/new20|68.144%|44.033%|27.150%|33.410%|24.111pp|逐值不变，没有K1功能增益|

D92 Role-Oracle另有完整125上界，在K10/new20把`A_old/N/H`提高到`83.31%/71.43%/76.91%`，但它读取query old/new role，明确为`LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE`，不能作为NEXT-R1合法基线或方法组件。

### 9.2其他路线与D130

|方法|矩阵与语义|关键同row结果|资源摘要|对NEXT-R1的约束|
|---|---|---|---|---|
|D91|Rx20-1、seed713101、K10/new5、15行development|`B/A/N/H/forgetting=92.78/82.22/84.67/82.62/10.56`，与D62 development的15/15 outer prediction相同|2159参数；40 optimizer steps；约25.427GMAC适应＋0.249GMAC额外crossfit；query 6624MAC；持久14,399B＋瞬时ground 116,304B|内部support目标或几何变化不等于分类功能；不再扩展consensus/sigma margin|
|SVRN-qKNN-BCRR r4.2|5receiver×5seed×5slice=125，完整375场景|overall `B_old/A_old/N/H/forgetting=73.10/43.03/23.46/29.25/30.07`；125/125的N和H均低于D62；配对`ΔA_old=-21.36pp、ΔN=-35.64pp、ΔH=-31.84pp、Δforgetting=+12.96pp`|0训练；持久状态均值126,981B、最大208,069B；评分矩阵每query均值0.07868ms|大分支状态和BCRR零范数修补不能弥补表示错配；该冻结revision关闭，不推广为所有DA无效|
|D130 CSPAR-2|42折×K1/K5 source-held LOCO代理|K5 DA`ΔH=-0.556pp、正确数-9`|DA 336B；与Lite160联合1,320B；head query 960MAC|共享PSD表示变换发生负迁移，关闭|
|D130 SRDH-2|42折×K1/K5 source-held LOCO代理|K5 DA全部delta=0|DA 666B；与Lite160联合1,650B；head query 960MAC|共享非线性响应在真实矩阵上无决策作用，关闭|
|D130 Lite160|同160维Full160因果对照|`ΔH=+0.164pp、正确数+14`，但`ΔA_held=-0.529pp、Δfloor=-1.270pp`|拟合MAC 22,400、工作集62,768B、时延中位数约0.75ms；部署wire仍984B/960MAC|保留效率实现，下一头必须加入类对称尾部保护，不能把平均增益当联合成功|

### 9.3资源口径限制

- D62完整125报告的适应计算均值约26.056GMAC，持久状态均值约15,194B，query为6,624–15,264MAC，CUDA峰值约21.4–22.0MB。
- D92最终retry2没有完整资源artifact；外部分析只能给出约11.153–11.741GMAC拟合上界、7.46–16.11KiB状态和3,168–7,488 compiled query MAC，属于上下文估计。
- D130的22,400拟合MAC、960 query MAC和1,320/1,650B联合状态只覆盖160维代理head＋DA数值资产，且runtime中backbone forward为0。它证明组件轻量，不能直接宣称相对D62/D91/D92端到端同比降幅。
- 各历史报告未提供统一的独立TX split字段；不得补写或猜测。所有跨版本主结论必须保留各自matrix、receiver、seed、K/new_count和claim语义。

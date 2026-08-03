# D130轻型DA×精简D92联合代理矩阵报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|d130_joint6_proxy84_20260803_r1|
|时间|2026-08-03（Asia/Hong_Kong）|
|状态|LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED / NO_PERFORMANCE_RESULT|
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

## 8.待回填结果

|candidate_id|机制|42折/K|A_retained|A_held_proxy|H_retained_held_proxy|F_retained|DA效应|Lite效应|联合效应|资源摘要|结论|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|CSPAR-2|rank-2接收机残差|完整K1/K5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|
|SRDH-2|rank-2共享响应字典|完整K1/K5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|

只允许同候选完整矩阵联合判定，不拼接边际极值。方向性胜者才进入G0 588、一次fresh63和单seed Target25；不运行125，不重复D62/D92/SVRN。

# D129轻型DA×精简D92联合代理矩阵r3报告

## 1.实验身份与目标

|字段|值|
|---|---|
|实验ID|d129_joint6_proxy84_20260803_r3|
|时间|2026-08-03（Asia/Hong_Kong）|
|状态|LANDED / STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT|
|目标|用最小完整source-held LOCO矩阵联合比较CSPAR-2、SRDH-2与精简D92|
|责任|主agent负责方法、协议、分析；唯一Terra Max runner负责N607落地、运行和artifact回收|
|修复边界|r1修正Git归档哈希；r2确认服务器无ssr-gpu；r3显式使用已验证项目环境CVS-RFFI。科学代码、矩阵、seed、判据均未改变|

本轮固定为7个receiver×6个Phase1已见held class×K1/K5=84原子行/候选，两候选合计168条candidate-row prediction，每条六个逻辑臂。它只是方向性代理，不是Target25真实新类实验，不输出正式N/H_old_new。

## 2.冻结方法与决定规则

- CSPAR-2：Phase1接收机效应rank-2极分解基＋K5全类共享scatter低秩残差。
- SRDH-2：Phase1冻结rank-2非线性响应字典＋support类别置换不变共享摘要。
- 两者均不更新checkpoint全参数，不执行Phase2 optimizer/backward；公共R0只拟合一次并跨候选复用。
- 三头为qKNN、D92-Full160代理、对角OAS D92-Lite160。K1的F/L严格alias Q。
- K5固定比较R1Q-R0Q、R0L-R0F、R1L-R1F。每项要求ΔH_retained_held_proxy>0且总正确数严格增加，并且ΔA_retained、ΔA_held_proxy、ΔF_retained非负。
- 完整负收益候选立即关闭且不调参；两候选都失败则本revision结束。

## 3.版本、输入与硬门

|项|冻结值|
|---|---|
|方法实现提交|7824295a2f4d7897d6ba4cd9370e97bce5988171；r3预注册修复提交由runner handoff记录|
|本地验证|ssr-gpu内35项聚焦测试通过；两个入口py_compile通过|
|独立方法复审|P0=0，P1=0，LOCAL_CORE_VERIFIED=YES|
|真实档案smoke|archive SHA=dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d；receipt SHA=263bb7945a438a70905d639b0300e423b0fd00ddcb5e0d966708ff3b88d354d9；420/30/54，两候选PASS|
|method lock release SHA|73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f，只认Git归档解包字节|
|checkpoint SHA|2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98|
|fixture SHA|d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669|
|capsule/split|d106_ls588_proxy_dd315295 / d104_source_seed104713_v2|

本地代码测试环境仍为C:\Users\lh594\.conda\envs\ssr-gpu\python.exe。N607未注册ssr-gpu，因此r3远端实验环境明确冻结为/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python；只读验证为Python 3.10.19、NumPy 2.2.5、PyTorch 2.1.0+cu121。不得用裸conda，不得安装或修改环境。

## 4.N607路径与命令

- run root：/home/szu2070436088/2510044040/CV-SincNet/runs/d129_joint6_proxy84_20260803_r3，创建前必须ABSENT。
- D104档案：/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz。
- 环境变量：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=；GPU allocation=none。
- 输出：smoke/smoke.json、prepare/{predictor_package.npz,truth.json,plan.json,prepare_receipt.json}、predict/{predictions.json,resources.json}、score/score.json和四份日志。

所有命令使用绝对PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python：

~~~text
<PY> source/code/scripts/run_d129_joint6_real_archive_smoke.py --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --held-receiver AUTO_FIRST --held-class AUTO_FIRST --run-id d129_joint6_proxy84_20260803_r3-smoke --output <RUN_ROOT>/smoke/smoke.json

<PY> source/code/scripts/run_d129_joint6_proxy_matrix.py prepare --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --method-lock <RUN_ROOT>/source/configs/d129_joint6_method_lock_20260803.json --method-lock-sha256 73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f --capsule-id d106_ls588_proxy_dd315295 --split-id d104_source_seed104713_v2 --run-id d129_joint6_proxy84_20260803_r3 --output-dir <RUN_ROOT>/prepare

<PY> source/code/scripts/run_d129_joint6_proxy_matrix.py predict --package <RUN_ROOT>/prepare/predictor_package.npz --package-sha256 <PACKAGE_SHA> --output-dir <RUN_ROOT>/predict

<PY> source/code/scripts/run_d129_joint6_proxy_matrix.py score --prediction <RUN_ROOT>/predict/predictions.json --prediction-sha256 <PREDICTION_SHA> --plan <RUN_ROOT>/prepare/plan.json --plan-sha256 <PLAN_SHA> --truth <RUN_ROOT>/prepare/truth.json --truth-sha256 <TRUTH_SHA> --output <RUN_ROOT>/score/score.json
~~~

prepare、predict、score必须是不同进程。predict detached启动并写logs/predict.pid，立即核验PID/CWD/cmdline/run-root和日志增长；predict不打开truth，只有168条完整prediction封存后score才允许打开truth。

## 5.健康、停止与回收

仅P0协议/安全违规、错误checkout/hash、输出覆盖、launcher确定性故障，或至少两个不同row在prediction前出现同一确定性异常指纹时停止本run自有进程树。不得因accuracy、H、floor或候选表现差停止。r3之后不再自动创建新run；若再次出现非科学发布缺陷，保存partial artifact并返回主agent。truth文件默认不回传分析面。

## 6.待回填结果

|candidate_id|机制|42折/K|A_retained|A_held_proxy|H_retained_held_proxy|F_retained|DA效应|Lite效应|联合效应|资源摘要|结论|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|CSPAR-2|rank-2接收机残差|完整K1/K5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|
|SRDH-2|rank-2共享响应字典|完整K1/K5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|

只允许同候选完整矩阵联合判定，不拼接边际极值。方向性胜者才进入G0 588、一次fresh63和单seed Target25；不运行125，不重复D62/D92/SVRN。

## 7.r3运行结论

r3的bundle、fixture、D104档案、checkpoint、7个D129源文件和method lock归档哈希均匹配，远端CVS-RFFI绝对Python编译通过。真实archive smoke完成，两候选均PASS且truth_loaded=false。prepare以独立进程完成，predictor package SHA256为402f223eab2705c003d76b62a4ad39920c249bfa08bcfc380cdeb2627d5da691；truth只保留在远端receipt绑定中，未读取或回传。

detached predict PID 517676在产生任何prediction前确定性退出：RuntimeWarning提示FP16 cast overflow，随后stage2_d129_joint6_heads.py的D129Joint6HeadsError报告affine FP16 intercept is not representable。predict目录未创建，完整性为0/168，score未调用。因此本run没有性能结果，不得把两个smoke PASS或任何partial状态解释为正收益。

run root、smoke、prepare、日志和PID证据均保留；无存活D129进程，GPU已释放，SSH连接清理为0。r3不得续跑或覆盖。下一研发动作仅允许在本地证明一个全类共享正logit缩放的数值编译修复，确保argmax不变后建立新scientific revision；禁止clip单类截距、fallback qKNN、改阈值或读取query truth。

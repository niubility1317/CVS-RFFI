# D129轻型DA×精简D92联合代理矩阵r2报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d129_joint6_proxy84_20260803_r2`|
|时间|2026-08-03（Asia/Hong_Kong）|
|状态|`LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED / NO_PERFORMANCE_RESULT`|
|主责任|主agent负责方法、协议、分析与晋级；唯一Terra Max runner负责N607落地和证据|
|目标|运行CSPAR-2与SRDH-2的最小完整source-held LOCO方向性矩阵，并联合比较精简D92|
|r1关系|r1在任何D129进程启动前因EOL字节哈希混用停止；r2只修release hash和解释器发现，不改方法、矩阵、seed或判据|

本轮不是Target25，不产生正式真实新类`N/H_old_new`。固定矩阵为7个receiver×6个Phase1已见held class×K1/K5=84原子行/候选；两候选合计168条candidate-row prediction，每条六个逻辑臂。K1的F/L严格alias Q。

## 2.冻结方法与判据

- CSPAR-2：Phase1接收机效应rank-2极分解基，K5用全类共享scatter估计低秩残差。
- SRDH-2：Phase1冻结rank-2非线性响应字典，support只形成类别置换不变的共享摘要。
- 两者均不更新checkpoint全参数、不执行Phase2 backward或optimizer。
- 公共R0只拟合一次并跨候选复用；头为qKNN、D92-Full160代理和对角OAS D92-Lite160。
- K5主比较固定为`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`。每项必须`ΔH_retained_held_proxy>0`且总正确数严格增加，并且`ΔA_retained`、`ΔA_held_proxy`、`ΔF_retained`非负。
- 任一候选完整矩阵失败即关闭，不调参；两候选都失败则本revision结束。

## 3.版本与已完成硬门

|项|证据|
|---|---|
|方法实现提交|`7824295a2f4d7897d6ba4cd9370e97bce5988171`；r2报告修复提交将在发布handoff记录|
|35项聚焦测试|全部通过|
|入口编译|`run_d129_joint6_real_archive_smoke.py`与`run_d129_joint6_proxy_matrix.py`通过|
|独立复审|`P0=0，P1=0，LOCAL_CORE_VERIFIED=YES`|
|真实档案无truth smoke|archive SHA=`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`；receipt SHA=`263bb7945a438a70905d639b0300e423b0fd00ddcb5e0d966708ff3b88d354d9`；420/30/54，两候选PASS|
|method lock内容|`configs/d129_joint6_method_lock_20260803.json`|
|method lock release SHA|`73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f`，以Git归档解包字节为唯一release truth|
|r1旧错误值|`fd47cd9f52d4ae29100ebcaff5e2a64c5397294b72394990e2f2040a16cbedd7`仅为Windows工作树CRLF字节哈希，r2禁止使用|

根目录`E:\type10-7`不是Git仓库；本报告在Git工作树和根报告面保持字节镜像。r1远端run root及partial artifact保持不变，r2不得覆盖或复用。

## 4.服务器输入与路径

|项|值|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d129_joint6_proxy84_20260803_r2`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D104 L_s archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|
|archive SHA|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|fixture SHA|`d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669`|
|capsule/split|`d106_ls588_proxy_dd315295`/`d104_source_seed104713_v2`|
|线程/GPU|`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=`；GPU allocation=none|

远端非交互shell不得依赖裸`conda`。runner只允许在`$HOME/miniconda3/envs/ssr-gpu/bin/python`、`$HOME/anaconda3/envs/ssr-gpu/bin/python`、`$HOME/.conda/envs/ssr-gpu/bin/python`中做只读存在性检查，并用`sys.prefix`、NumPy导入和Python版本验证唯一可执行解释器。零个或多个不等价候选均停止，不临时安装环境。

## 5.发布与命令

从r2精确Git提交生成`tar.gz`并同步到新run root的`input/release.tar.gz`，fixture同步到`input/d106_fixture.json`；解压到`source`。先核验bundle、fixture、7个D129源文件和method lock Git归档字节哈希，再执行`py_compile`。所有Python命令均用上节解析出的绝对`<SSRGPU_PYTHON>`，并置于固定线程环境中。

```text
<SSRGPU_PYTHON> source/code/scripts/run_d129_joint6_real_archive_smoke.py --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --held-receiver AUTO_FIRST --held-class AUTO_FIRST --run-id d129_joint6_proxy84_20260803_r2-smoke --output <RUN_ROOT>/smoke/smoke.json

<SSRGPU_PYTHON> source/code/scripts/run_d129_joint6_proxy_matrix.py prepare --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --method-lock <RUN_ROOT>/source/configs/d129_joint6_method_lock_20260803.json --method-lock-sha256 73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f --capsule-id d106_ls588_proxy_dd315295 --split-id d104_source_seed104713_v2 --run-id d129_joint6_proxy84_20260803_r2 --output-dir <RUN_ROOT>/prepare

<SSRGPU_PYTHON> source/code/scripts/run_d129_joint6_proxy_matrix.py predict --package <RUN_ROOT>/prepare/predictor_package.npz --package-sha256 <PACKAGE_SHA_FROM_PREPARE_RECEIPT> --output-dir <RUN_ROOT>/predict

<SSRGPU_PYTHON> source/code/scripts/run_d129_joint6_proxy_matrix.py score --prediction <RUN_ROOT>/predict/predictions.json --prediction-sha256 <PREDICTION_SHA> --plan <RUN_ROOT>/prepare/plan.json --plan-sha256 <PLAN_SHA> --truth <RUN_ROOT>/prepare/truth.json --truth-sha256 <TRUTH_SHA> --output <RUN_ROOT>/score/score.json
```

prepare、predict、score必须是不同进程。predict detached启动，PID写`logs/predict.pid`，立即核对CWD/cmdline/run-root和日志增长；predict只读predictor package，不打开truth。只有168条prediction完整封存且`rows_complete=true`后，score才允许打开truth。

## 6.健康与停止规则

仅在P0协议/安全违规、错误checkout/hash、输出覆盖、launcher确定性故障，或至少两个不同row在prediction前出现同一确定性异常指纹时停止本run自有进程树。绝不因accuracy、H、floor或候选表现差停止。fresh-run retry未授权。技术失败保存全部partial artifact并标记`NO_PERFORMANCE_RESULT`。

预期artifact：`smoke/smoke.json`、`prepare/{predictor_package.npz,truth.json,plan.json,prepare_receipt.json}`、`predict/{predictions.json,resources.json}`、`score/score.json`、四份日志、PID/进程/GPU/哈希清单。truth文件默认不回传主分析面。

## 7.待回填结果

|candidate_id|机制|42折/K|A_retained|A_held_proxy|H_retained_held_proxy|F_retained|DA效应|Lite效应|联合效应|资源摘要|结论|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|CSPAR-2|rank-2接收机残差|完整K1/K5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|
|SRDH-2|rank-2共享响应字典|完整K1/K5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|

只允许同候选完整矩阵的联合判定，不拼接边际极值。方向性胜者才顺序进入G0 588、一次fresh63和单seed Target25；不运行125，不重复D62/D92/SVRN。

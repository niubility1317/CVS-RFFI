# D129轻型DA×精简D92联合代理矩阵报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d129_joint6_proxy84_20260803_r1`|
|时间|2026-08-03（Asia/Hong_Kong）|
|主责任|主agent：方法集成、协议解释、结果分析与晋级决定|
|服务器责任|唯一Terra Max runner：N607落地、启动、健康检查、artifact回收；不得改方法或按性能重跑|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|目标|用最小完整方向性矩阵比较CSPAR-2与SRDH-2，并同时检验DA、D92-Lite替代和联合效应|
|比较对象|同一原子row内共享R0的`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`|
|正式声明边界|本轮是Phase1已见类source-held LOCO代理，不是Target25真实新类注册，不输出正式`N/H_old_new`|

## 2.假设与冻结矩阵

两条候选均不更新checkpoint全参数，也不做Phase2反向传播。CSPAR-2用Phase1接收机效应rank-2极分解基和K5全类共享scatter估计低秩残差；SRDH-2用Phase1冻结的非线性响应字典和支持集置换不变共享摘要更新残差。两者都与精简D92联合：`Full160`只作同表示机制代理，`Lite160`使用对角OAS后直接编译为INT8仿射头。

矩阵固定为7个receiver×6个held seen class×K1/K5=84原子行/候选，共168条candidate-row prediction；每条包含六个逻辑臂。K1的F/L严格alias Q，不重复拟合。K5三个主效应必须同时满足`ΔH_retained_held_proxy>0`、总正确数严格增加，并且`ΔA_retained`、`ΔA_held_proxy`、`ΔF_retained`均非负。任一候选完整矩阵失败即关闭，不调参复活；两者都失败则本revision结束。

## 3.本地实现与验证

### 3.1代码与锁定文件

|文件|SHA256|用途|
|---|---|---|
|`code/cvsrffi/stage2_d129_joint6_da.py`|`13d21b06d098c934985ca41f30835975c18e42e91c9e8df20e81b725b320a114`|两条轻型DA候选与Phase1资产|
|`code/cvsrffi/stage2_d129_joint6_heads.py`|`c78863a670bcd4f9856ae8da6af88abd04965e447cd09afd4aaf317618906448`|公共R0、qKNN、Full160、Lite160|
|`code/cvsrffi/stage2_d129_joint6_matrix.py`|`282952c91a09cee669ea9f0395bbde77f8c0ab067d7fe402b434b523ee45678f`|84行/候选矩阵及物理ID绑定|
|`code/cvsrffi/stage2_d129_joint6_runtime.py`|`2d98f78cbc9103b69e04be30d86ec997598f3d5815877cb9e95a2d6cdadbd27c`|六臂运行与只读query receipt|
|`code/cvsrffi/stage2_d129_joint6_scorer.py`|`e575e55bbcacd24cc64b3f2da140e9c3e4f4dfe8c4a13580e737e48aa131b9ba`|prediction封存后的独立评分|
|`code/scripts/run_d129_joint6_real_archive_smoke.py`|`119f9e32ef80ec1a54f5fb65a21ccd2692665de19c3be1cac0d6ba20250977c9`|真实D104六成员归档无truth smoke|
|`code/scripts/run_d129_joint6_proxy_matrix.py`|`c0f41d81b887c1b21fbba0317052d06465cbeea0bc6ec472c47e88be528b2879`|prepare/predict/score三进程边界|
|`configs/d129_joint6_method_lock_20260803.json`|预注册工作树字节`fd47cd9f52d4ae29100ebcaff5e2a64c5397294b72394990e2f2040a16cbedd7`；Git归档字节`73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f`|冻结方法与矩阵；r1错误混用两种EOL字节哈希|

Git工作树为`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`，分支`codex/stage2-da25-r1`，预注册父提交`949f0385f75b90fdb0f2873717774f6cde34e1cf`；最终发布提交由本报告进入同一提交后记录在runner交接和完成更新中。根目录`E:\type10-7`不是Git仓库，本报告在该Git工作树和根目录报告面保持字节镜像。

### 3.2验证结果

- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- 聚焦测试：D129的DA、heads、matrix、runtime、scorer、真实smoke和代理runner共35项，全部通过。
- 两个入口通过`py_compile`，`git diff --check`无错误。
- 独立Terra Max复审：`P0=0，P1=0，LOCAL_CORE_VERIFIED=YES`。
- 真实档案smoke：输入SHA256=`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`，fixture SHA256=`d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669`，receipt SHA256=`263bb7945a438a70905d639b0300e423b0fd00ddcb5e0d966708ff3b88d354d9`。
- smoke结果：`truth_loaded=false`、`performance_result=false`、Phase1-fit=420、support=30、query=54，CSPAR-2与SRDH-2均通过，公共R0候选重拟合数为0。

## 4.输入、版本与同步

|输入|固定值|
|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D104 L_s真实档案|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|
|档案SHA256|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|capsule_id|`d106_ls588_proxy_dd315295`（source proxy标识，不是Target capsule）|
|split_id|`d104_source_seed104713_v2`|
|方法seed/盐|仅使用method lock和代码中冻结盐；不增加调参seed|

提交后从精确Git提交生成只读release bundle并同步到`<run_root>/input/release.tar.gz`，解压到`<run_root>/source`；fixture同步到`<run_root>/input/d106_fixture.json`。不覆盖共享服务器代码。远端所有源文件必须与提交内容和上述哈希一致。

## 5.N607发布方案

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d129_joint6_proxy84_20260803_r1`
- 工作目录：上述run root。
- 环境：`conda run -n ssr-gpu`；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=`。
- GPU：不使用GPU；runner只记录发布时GPU占用，不占训练槽。
- 日志：`logs/smoke.log`、`logs/prepare.log`、`logs/predict.log`、`logs/score.log`，主PID写入`logs/predict.pid`。
- 输出：`smoke/smoke.json`、`prepare/{predictor_package.npz,truth.json,plan.json,prepare_receipt.json}`、`predict/{predictions.json,resources.json}`、`score/score.json`。

冻结命令顺序如下。所有路径在执行时展开为绝对路径，且输出目录必须预先不存在：

```text
conda run -n ssr-gpu python source/code/scripts/run_d129_joint6_real_archive_smoke.py --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --held-receiver AUTO_FIRST --held-class AUTO_FIRST --run-id d129_joint6_proxy84_20260803_r1-smoke --output <RUN_ROOT>/smoke/smoke.json

conda run -n ssr-gpu python source/code/scripts/run_d129_joint6_proxy_matrix.py prepare --archive <D104_LS_ABS> --archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --fixture <RUN_ROOT>/input/d106_fixture.json --fixture-sha256 d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669 --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --method-lock <RUN_ROOT>/source/configs/d129_joint6_method_lock_20260803.json --method-lock-sha256 fd47cd9f52d4ae29100ebcaff5e2a64c5397294b72394990e2f2040a16cbedd7 --capsule-id d106_ls588_proxy_dd315295 --split-id d104_source_seed104713_v2 --run-id d129_joint6_proxy84_20260803_r1 --output-dir <RUN_ROOT>/prepare

conda run -n ssr-gpu python source/code/scripts/run_d129_joint6_proxy_matrix.py predict --package <RUN_ROOT>/prepare/predictor_package.npz --package-sha256 <PACKAGE_SHA_FROM_PREPARE_RECEIPT> --output-dir <RUN_ROOT>/predict

conda run -n ssr-gpu python source/code/scripts/run_d129_joint6_proxy_matrix.py score --prediction <RUN_ROOT>/predict/predictions.json --prediction-sha256 <PREDICTION_SHA_FROM_PREDICT_STDOUT> --plan <RUN_ROOT>/prepare/plan.json --plan-sha256 <PLAN_SHA_FROM_PREPARE_RECEIPT> --truth <RUN_ROOT>/prepare/truth.json --truth-sha256 <TRUTH_SHA_FROM_PREPARE_RECEIPT> --output <RUN_ROOT>/score/score.json
```

prepare、predict、score必须是不同Python进程；predict只读predictor package，不导入或打开truth文件。predict以detached方式启动，runner立即验证PID、CWD、cmdline、run root和日志增长；完成168条prediction后才允许score打开truth。

## 6.健康、停止与成功标准

仅在P0协议/安全违规、错误checkout/hash、输出覆盖风险、launcher确定性故障，或至少两个不同row在产生prediction前出现同一确定性异常指纹时停止本run自有进程树。不得因准确率、H、floor或任一候选表现差而停止。自动fresh-run重试未授权；技术失败须保留全部partial artifact并返回主agent决定新run ID。

成功条件：smoke至少一候选有功能，prepare封存无truth predictor package，predict完整产生168条且`rows_complete=true`，独立score验证全部同键、公共R0和K1 alias闭合并生成两候选完整评分。smoke若单候选无功能，只关闭该候选；两候选都无功能则`NO_PERFORMANCE_RESULT`。

## 7.待回填结果

|candidate_id|机制|receiver/TX split|K|A_retained|A_held_proxy|H_retained_held_proxy|F_retained|DA效应|Lite效应|联合效应|资源摘要|结论|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|CSPAR-2|rank-2接收机残差|42折完整|1/5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|
|SRDH-2|rank-2共享响应字典|42折完整|1/5|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待判定|

本表只允许同候选、同完整矩阵的联合行；不拼接边际最大值。若两候选均负，本revision关闭并保存完整负结果；下一研发轮最多新增一条原理不同候选。

## 8.r1运行结论

runner于2026-08-03 19:28:54 CST经直连N607完成预检。run root创建前不存在；bundle、fixture、D104档案、checkpoint及7个D129源文件哈希均匹配。解包后method lock实际SHA256为`73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f`，而预注册错误要求工作树CRLF字节哈希`fd47cd9f52d4ae29100ebcaff5e2a64c5397294b72394990e2f2040a16cbedd7`。这是发布清单混用工作树与Git归档字节的确定性P0技术缺陷。

runner在任何D129进程启动前停止：`smoke/`和`logs/`为空，`prepare/`、`predict/`、`score/`不存在，无PID、无prediction、无性能结果。8张GPU均为0%/1MiB，所有SSH/SCP短连接结束后本机`ssh.exe=0`、N607:22连接数为0。远端run root及partial artifact保持不变。下一次只允许使用新run ID，并以Git归档字节哈希作为release truth；不得覆盖、续跑或重标r1。

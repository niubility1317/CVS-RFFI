# GEOFF/r2 Phase1双表征归档与coverage技术run报告

## 1.身份、目标与最终状态

- run_id:`rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2`
- 预注册时间:`2026-07-22T23:48:22+08:00`；runner:`/root/geoff_r2_n607_runner`；完成回收时间:`2026-07-23`。
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`。本run未改变received IQ、physical ID、receiver/TX、scene、K、support/query split或schema，未重新验证数据。
- objective:在同一Torch/CUDA/device合同下生成strict ADV3B02 base/candidate dual runtime、base parity receipt/vector、8400行Phase1 single-observation dual archive/manifest及只读coverage receipt。
- hypothesis:首个JIT边界前设置并回读`graph_executor_optimize=false`可消除r1的冷/热TorchScript计划漂移，同时维持batch1/8/256、每batch三次调用、三输出`maxabs≤1e-5`。
- final:`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。唯一启动后archive入口因调用签名不匹配自然退出；没有prediction、target/query/held/125访问、性能结果、promotion或bundle声明。retry=`NO`，未修改方法、未重启、未删除远端内容。

## 2.冻结版本、范围与本地证据

- 方法commit:`ca5d0c4bcf8fb295cdfb70e067f9009617bb3a5f`；release prereg输入:`d45f4cc22ac379c287ad09baed53fe07cdb791d2`；handoff:`26efcdb1461c44aaac4ec7bec06e27a3f75ffcc5`。
- 观察到的版本异常：输入的release prereg完整SHA在当前Git对象库不可解析；handoff对象存在，其父为`d45f4cc29e6b94fbcd8e348a570af61441605c0d`。当前正式repo检出HEAD为`15d3266e3f59f82f753729ed2ec0af2211b52d27`且有既有未跟踪文件；均未改动。
- handoff对象中的镜像报告与原root报告在回填前逐字节同为SHA256`0beb8f1d789cf490f61025c27b29ba627f60de3e5b3727f54ecdda54fe8d7167`。
- 冻结ZIP:`code/snapshots/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/source_ca5d0c4b.zip`，SHA256`5adbef8a1ebf2f0846132226f702e95648c99334a0ba5296b7487e45095e4778`，33,007,669B、4436成员。
- 冻结wrapper:`run_pipeline.sh`，SHA256`e1f497a757d54cef95a9559ac3de910a26cf2d9a3d0407d3cc865b628847afcf`。
- 预注册本地验证为`ssr-gpu`的`py_compile`通过、GEOFF专项35项及相邻回归共48 passed；本次远端再次完成6个冻结成员`py_compile`与wrapper`bash -n`。
- 运行范围仅为既有Phase1 source-validation received-IQ cache、冻结checkpoint、adapter和salt；禁止访问target/query、clean/source runtime sidecar、held prediction或125。

## 3.direct预检、落地与唯一启动

|项目|证据|
|---|---|
|route/preflight|本地`tools/n607_ssh_preflight.ps1`通过；direct`N607`，host=`dell-DSS8440`，项目根可见，8张RTX3090可见；连接后均检查，无本机遗留SSH/TCP22。|
|远端首次存在性|`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2`首次探测不存在；随后只创建该root的`input/`、`logs/`，权限均700。|
|GPU容量/分配|启动前没有compute process；选择GPU0，未超过每GPU两个compute process限制。|
|环境/CWD|Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/source_ca5d0c4b`。|
|sync与安全解包|仅SCP ZIP→`input/source_ca5d0c4b.zip`和wrapper→`input/run_pipeline.sh`；两项SHA匹配，ZIP安全检查4436成员，无绝对路径、`..`或symlink。|
|唯一launch|`nohup env CUDA_VISIBLE_DEVICES=0 bash .../input/run_pipeline.sh > .../logs/pipeline.log 2>&1 < /dev/null &`；PID=`357994`，CWD与cmdline均为冻结wrapper。|

### 冻结远端资产及源码核验

|资产/成员|SHA256|结果|
|---|---|---|
|checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|PASS|
|adapter|`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`|PASS|
|cache_set|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|PASS|
|selection_salt|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|PASS|
|6个冻结源码成员|报告预注册的6项SHA逐项匹配|PASS|
|远端`py_compile`与`bash -n`|6个成员及冻结wrapper|PASS|

## 4.自然退出、receipt与失败定位

- `pipeline.exit=1`；`PIPELINE_ARTIFACTS_COMPLETE`计数=0；自然退出后GPU0无compute process。
- export receipt:`cvs.phase1.adv3b02_dual_feature_torchscript_export.v2`，`status=PASS`，base/candidate runtime SHA分别为`6f9e424816be4189b8484731106fa8ec600b8ded25c058de02478db2cd3bb257`和`1d8e02ef104de9794d949963d07d438d35b048dbe31cb16f4bda2ab0c41f4631`。
- execution contract:`graph_executor_optimize=false`，getter/setter均为`torch._C`，readback=`false`，Torch=`2.1.0+cu121`、CUDA=`12.1`、device=`cuda:0`、max_abs=`1e-5`、contract SHA=`4a9ca47a435e5ac59a22fa12d335b5425569c51f818e1734a4fff8cd84147d19`。
- export的最大差为`2.384185791015625e-06≤1e-5`；base/candidate在batch1/8/256的每batch三次调用，三输出均为0。
- base parity receipt:`cvs.phase1.adv3b02_dual_runtime_checkpoint_parity_receipt.v2`，`status=PASS`，base runtime SHA闭合，max_abs=0，batch=`[1,8,256]`、calls_per_batch=3、vector root=`86a9a69ed8879f3c203b9393bd1018221536eb87dd7f719b52751c2fef5c44a4`。
- 失败发生在archive脚本的实际调用：`TypeError: export_phase1_singleobs_dual_feature_archive() got an unexpected keyword argument 'cache_set'`。因此archive/manifest、coverage receipt、`sha256sums.txt`均未生成，不能把已通过的runtime/parity当作完整技术完成。

## 5.archive与coverage状态

|项目|冻结要求|本次证据/状态|
|---|---|---|
|archive manifest/内部verify|v2并内部verify|未生成；调用在进入导出前失败。|
|coverage schema|`cvs.phase1.singleobs_dual_feature_coverage_receipt.v1`|未生成。|
|coverage常量|8400行、8400 unique physical、8400 unique observation、class/receiver/day/scenario=`6/7/4/3`、168 cells、zero=0、min>10|仅为预注册合同，未有本次coverage artifact，不能报告为通过。|
|registry/K余量|class=`[14-10,14-7,20-15,20-19,6-15,8-20]`；scene为3个`leo_*_weak`；K1/K5/K10余量为`min_cell-K`|未生成coverage receipt，余量未被runtime验证。|
|访问边界|coverage应只读元数据数组，`feature_arrays_read=[]`，不选held fold|archive步骤未进入；无target/query/held访问证据。|

## 6.已回收artifact及SHA256

回收目录:`E:/type10-7/automation_reports/CV-SincNet/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/retrieved/`。logs在该目录平铺；runtime在`output/runtime/`。8项本地SHA与远端清单逐项一致。

|相对路径|SHA256|状态|
|---|---|---|
|`pipeline.exit`|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|已回收，值1|
|`pipeline.pid`|`85c436453c99c1293a610b18ab44ff69805c94439cbc839c30c410e068e72d58`|已回收，值357994|
|`pipeline.log`|`41e443a795fb62554744decce95e955c0b3cf88b0ab8949a995dc5cd37a6c22f`|已回收，含TypeError|
|`output/runtime/base_dual_runtime.pt`|`6f9e424816be4189b8484731106fa8ec600b8ded25c058de02478db2cd3bb257`|已回收|
|`output/runtime/candidate_dual_runtime.pt`|`1d8e02ef104de9794d949963d07d438d35b048dbe31cb16f4bda2ab0c41f4631`|已回收|
|`output/runtime/dual_export_receipt.json`|`5054282b41e096e7269d9694d7c0c984e3f25de482f38b170854fb71d41f4d4b`|PASS receipt|
|`output/runtime/base_parity_receipt.json`|`43689bd2d665d652518c11d36d1522e3e498ea0d273f945fe48faa9da8c6beea`|PASS receipt|
|`output/runtime/base_parity_vector.json`|`c667fff8d33a4ec447ac58bf179a08d515df71a65bd1116074b9a4fe7929bde7`|PASS vector|

未生成且未伪造：archive NPZ/manifest、coverage receipt、`sha256sums.txt`、completion marker、prediction。

## 7.结论与后续边界

本run停在`ANALYZED / TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。技术原因是冻结wrapper与冻结archive exporter的调用签名不兼容，而非远端资产、GPU、执行合同、export parity或数据协议失败。修复只能在新的本地Git版本、独立预注册报告和全新run ID中完成；本run不允许补跑、覆盖或复用。

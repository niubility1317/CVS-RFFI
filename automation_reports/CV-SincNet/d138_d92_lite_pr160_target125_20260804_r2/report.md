# D138 D92-Lite-PR160 Target125实验报告r2

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r2`
- 当前状态：`LOCAL_VERIFIED / DEPENDENCY_CLOSURE_REPAIRED / REMOTE_GATE_PENDING / NO_TARGET_PERFORMANCE_RESULT`
- r1终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；r1只完成同步、TorchScript加载、compile和prepare尝试，smoke、shard、truth、score均未执行。
- r1失败原因：隔离source未同步D108基线依赖`stage2_d108_matrix_protocol.py`及`stage2_d108_target125_inputs.py`，prepare在业务逻辑前报`ModuleNotFoundError`。r1run root和证据永久保留，不覆盖、不续跑。

## r2修复边界

- 方法、候选、method lock、矩阵、数据、checkpoint、extractor和协议均不变；只扩大本地预注册的source依赖闭包。
- D138仍只测`DA0_REG0=before`与`DA0_REG1=after`；`DA1_REG0/DA1_REG1`为单运输臂候选范围外，不生成四状态DA因果结果。
- 远端Torch 2.1 extractor load gate已在r1通过，但r1的prepare未通过；r2必须重新验证compile、load、prepare和真实checkpoint no-query smoke后才允许shard。

## 冻结身份

- Git基础commit：`b27e088302d6f6bf4a6ae88e8357d626a2336236`；r2只增加发布依赖清单和本报告，不改变D138代码。
- candidate：`D92-Lite-PR160/r1`。
- method lock SHA256：`019dd59780de735af3026b091ef88b600c07d75c48f96aad0c2de34d49e8cee7`。
- source sealed runtime SHA256：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`。
- extractor SHA256：`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；大小`4618957`字节。
- protocol：`p2_min_v1`；矩阵：125 outer、375 scene、750 before/after surface、8 shard；GPU绑定为`CUDA_VISIBLE_DEVICES=i`+`--device cuda:0`。

## 新增source依赖闭包

下列文件来自本地Git工作树，按hash逐项SCP到r2的`source/code/cvsrffi`；不包含任何`stage2_next_r2_*`实验文件：

|文件|SHA256|
|---|---|
|`__init__.py`|`13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2`|
|`stage2_d108_matrix_protocol.py`|`d3e333f46a77f5b7f3f4d91378bfb9b3bd4eda16723c2234487897339925007f`|
|`stage2_d108_target125_inputs.py`|`c7f4afa728aaf1a210c07b2b5c15769edff019168efcbf5d4a733c55efecc661`|
|`stage2_d108_truth_scorer.py`|`ee64b32359599acba152487b8673ebae386f7d63e2d095ee8186275e5efad766`|
|`stage2_d92_lite_target125.py`|`f736cbb809f525e775a52151a84935eb0a0c9c7c2d20ab728f1f45b62643850a`|
|`stage2_d92_lite_target125_core.py`|`cfdd787ffb4b5c6a6e43435d268acacc2d5979cc272762d994a72d3efa1732e0`|
|`stage2_d129_joint6_heads.py`|`338cdcd63831efd3dace1ba4bbc0658c71a0eea7cfc0febd275df78bab7e2db0`|
|`stage2_adv3b02_ts_drqknn_bcrr.py`|`1765749bd97c2958f9e9632dd9a1aa9453375e2b1f0cd68a8bd5f30742c43e76`|
|`stage2_zid_student_t_qknn.py`|`f7bc2ab7e6f9457085973099431db934edfa840ba37e904288ff4720726101e2`|
|`stage2_next_r1_tsl.py`|`3abe3dfaf87021aea4719f8871ced8ee26ac9d98efbf12837d9001d7f2caacc0`|
|`stage2_diag_cosine_exploration.py`|`a12fd07b4c348970b958a0856783afef58dba1d3801f6bdf18fcebb3fbb307f2`|
|`stage2_predictor_runtime.py`|`989c7304895b11f7213dbe25158166b135654edd0dfe725bada666a0e2eed2e0`|
|`stage2_d42_unified_shrinkage_lda.py`|`2624b3600ec912b4031ad1f0dbc260b08a1329c40b2ac23ed393f7109005391b`|
|`stage2_d108_d92_core.py`|`d25a2f5a75476df944f519603900de9aae3450750989b81af3ff1e8991bb813f`|
|`somph_diagnostic_bundle_loader.py`|`b337351e85c376bb5c2bdc826a8e1a0f788947e6d4924bb96d8f3962ae616f1c`|
|`somph_predictor_bundle.py`|`49a05c6f1f809fc221e3cb64fffe0c2f11b1b252e6cdbe86449303f8fb5def48`|

## r2N607路径与门禁

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r2`，首次创建且不可覆盖。
- 输入和既有D92路径沿用r1报告；extractor放在`run_root/input/d92_pr160_extractor_runtime.pt`。
- 先执行directN607只读预检；同步D138代码、配置、报告和上述依赖；逐文件hash与compile必须通过。
- 远端TorchScript加载、`prepare`、真实checkpoint row0/clear no-query smoke是连续硬门；任一失败即停止r2并保留证据，不启动shard。
- smoke通过后才运行完整8shard；只有125/375/750闭包后才merge、validate、truth-open和score。r1partial不进入任何性能分析。

## 本地证据与版本

- D138本地全套回归：`36 passed`；`py_compile`和`git diff --check`通过。
- r1runner停止报告已提交为`d241a171`；r2依赖闭包报告待本地提交后交给唯一runner。
- 不push、不上传、不修改N607项目根、不触碰无关`stage2_next_r2_*`文件。

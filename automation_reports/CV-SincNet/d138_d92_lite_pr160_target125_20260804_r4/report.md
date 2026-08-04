# D138 D92-Lite-PR160 Target125实验报告r4

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r4`
- 状态：`LOCAL_VERIFIED / REMOTE_LAUNCH_PENDING / NO_PERFORMANCE_RESULT`
- 操作员：Codex主agent；N607唯一release runner：Luna/max
- 用户在r3prepare目录生命周期故障后明确要求立即修复并启动实验；r4为新的不可覆盖run ID，不复用或覆盖r3。

## 目标与冻结比较

- 目标：在`p2_min_v1`下执行冻结候选`D92-Lite-PR160/r1`的完整Target125矩阵。
- 假设：同一sealed pre-ReLU160表示与K1 qKNN/K5/K10共享对角仿射头可在完整125矩阵上形成可验证的D92-Lite结果；本run只测方法，不调参。
- 对比：同run内`DA0_REG0=before`与`DA0_REG1=after`；单一`M_JOINT`运输臂。`DA1_REG0/DA1_REG1`不在本候选范围内。
- 矩阵：125 outer、375 scene、750 before/after surface、8 shard；完整闭合后才允许merge、truth-open和score。

## 本次修复

r3失败原因是runner在唯一prepare前预创建了空的`prepared`目录，而prepare实现对该目录采用不可覆盖保护。r4固定修复为：run root只预创建`control`、`source`和`input`；禁止创建`prepared`及其子项；由唯一prepare命令首次创建并写入prepared产物。若`prepared`在prepare前存在，立即停止并保留证据，不删除、不覆盖、不重试。

方法代码、配置、checkpoint、extractor、数据和矩阵不变；复用r3已通过的31项source闭包、compile和Torch 2.1 load证据，不重复数据复验。

## 版本与映射

- D138代码基线：`f4903584`；r3终态文档：`84148f88`。
- method lock SHA256：`019dd59780de735af3026b091ef88b600c07d75c48f96aad0c2de34d49e8cee7`。
- extractor SHA256：`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`。
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r4`。
- 远端source CWD：`RUN_ROOT/source`；远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；Torch预期：`2.1.0+cu121`。
- 远端输入：`RUN_ROOT/input/d92_pr160_extractor_runtime.pt`；prepared目标：`RUN_ROOT/prepared`，prepare前必须不存在。
- 远端控制：`RUN_ROOT/control/source_hashes.out`、`compile.out`、`load.out`、`prepare.out`、`smoke.out`、`shard_*.out`、`merge.out`、`score.out`。
- GPU：8个shard分别绑定GPU0–7，CLI设备统一为`cuda:0`并设置`CUDA_VISIBLE_DEVICES=i`。

## 硬门与停止规则

先做有界N607预检、逐文件hash、远端compile和Torch 2.1 extractor load；然后唯一prepare和真实checkpoint row0/scene0 no-query smoke。smoke通过后直接启动完整8 shard。P0协议/安全违规，或两个不同outer row产生相同确定性异常且尚未生成prediction时停止本run；不依据准确率停止，不从partial产物推导性能，不自动重试。

## 结果

启动前状态：尚未产生prediction、truth、score或性能结果。完成后在本报告追加同row候选表、异常、解释和下一步。

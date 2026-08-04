# D138 D92-Lite-PR160 Target125实验报告r4

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r4`
- 状态：`REMOTE_GATE_REPAIR_IN_PROGRESS / NO_PERFORMANCE_RESULT`
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

## smoke门修复记录（2026-08-04）

- prepare已通过并生成完整125/375/3000输入闭包；首次真实smoke在forward前失败，退出码`1`，GPU未使用，未生成smoke输出或prediction。
- 具体异常：`ModuleNotFoundError: No module named 'cvsrffi.phase2_runtime_contract'`，由`stage2_diag_cosine_exploration.py`导入；该模块仅包含Phase2 contract常量/校验，无数据、Torch或旧runner导入。
- 本地修复验证：`ssr-gpu`环境下`python -m py_compile code/cvsrffi/phase2_runtime_contract.py`通过；SHA256=`792c3eda489679bebdde08825e437ef93c460e8f64fd6279b166909b3ab90a78`。
- 修复动作：将该已存在的本地验证模块同步到当前r4的`source/code/cvsrffi/`；保留既有prepared产物，不重做prepare，不删除或覆盖任何历史run。同步后重新执行smoke；若仍有硬门异常则停止并保留证据。
- 递归闭包复核又确认并同步6个同组运行依赖：`somph_predictor_runtime.py`、`stage2_d38_strong_b3_quantized.py`、`stage2_predictor_bundle.py`、`stage2_predictor_runtime.py`、`stage2_d108_cbrrc.py`、`stage2_d108_smme.py`；6个文件均在本地`ssr-gpu`环境py_compile通过，未改变D138方法或数据。
- smoke第三次仍只在导入阶段发现直接模块`somph_runtime_trust.py`未同步；该文件已在本地`ssr-gpu`环境py_compile通过，SHA256=`4b1dee1d8ffdc793f48c46c21a11b0fdf8b6ef6e3b253807cc1138011dc1f9fc`，补齐后继续同一prepared输入的smoke。
- 第四次smoke已进入真实材料化，确认before/after的60个old support received-IQ逐元素相同、physical ID和label完全相同，但重复GPU forward产生最大`8.18e-05`数值抖动，触发原有逐元素断言；GPU仍未产生prediction。
- 本地修复`code/cvsrffi/stage2_d92_pr160_runtime.py`：按support received-IQ的shape+SHA缓存首个160维forward结果，仅在同一physical support IQ重复出现时复用；不放宽ID/数据断言、不增加forward、不缓存query状态。新SHA256=`235c2e16f8eaab470fadda7693010ed89389e8453f044417310f04f1e7be16db`。
- 本地验证：`ssr-gpu`下py_compile通过；D138/D92/D108选定回归测试全部通过（pytest退出码0）。该修复已提交后再同步到r4 source，保留prepared输入闭包。

## 结果

启动前状态：尚未产生prediction、truth、score或性能结果。完成后在本报告追加同row候选表、异常、解释和下一步。

# D92 E0_FULL_ONLY完整Target125确认实验

|字段|内容|
|---|---|
|run ID|`d92_e0_full_only_target125_20260812_v1`|
|日期|2026-08-12|
|状态|`RELEASE_READY`|
|协议|`p2_min_v1`；复用`VALIDATED_ONCE`数据|
|候选|`E0_FULL_ONLY`，candidate=`d92_e0d_e0_full_only`|
|目标|验证完整Target125上是否同时保持或提高性能，并显著缩减D92注册计算|
|历史对照|原D92 retry2同排125结果，`row_metrics.csv` SHA256=`bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62`|

## 1.假设与方法锁

`E0_FULL_ONLY`保留D92的288维联合特征A、ground-spectrum Cauchy robust center B、task-balanced covariance C和F0逐query全注册类分类头；关闭注册态Fisher/Pareto E，并把注册态D从full/block K折LOO-soft fusion缩为单次full主拟合。K1/K2继续走原D92 exact alias。

Hard12-v3的10个fresh performance outer上，`E0_FULL_ONLY−D92_FULL`得到H`+0.2439pp`、old BA`+0.4722pp`、old floor`+1.0000pp`、seen-new`+0.1333pp`、forgetting`−0.4722pp`，paired median wall下降`97.63%`。该结果只用于提出完整125确认假设，不替代本实验结论。

本次只跑一个冻结候选，不在Target125结果返回后选择arm、阈值、receiver、seed、K或new-count。query逐样本面对全部注册类；query truth、role、真实batch类别数、class quota、fit、update、selection和global reassignment全部禁止。

## 2.完整矩阵

完整Target125为5receiver×5seed×5slice=125个outer；每个outer固定3个互斥LEO弱场景，共375个scene-arm单元。

|维度|冻结值|
|---|---|
|receiver|`20-1`、`3-19`、`7-14`、`7-7`、`8-8`，各25outer|
|seed|`713102`–`713106`，各25outer|
|slice|`K1/new20`、`K5/new20`、`K10/new5`、`K10/new10`、`K10/new20`，各25outer|
|场景|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|context|`target125_context.json`，SHA256=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|
|source packages|原D92 retry2的125个sealed package，不重做数据验证|
|shard|8；`outer_index mod 8`|
|smoke|`rx_20_1__seed_713106__k_1__new_20`，arm标识保持`E0_FULL_ONLY`|

运行产生固定适配开启条件下的`DA1_REG0`与`DA1_REG1`预测；本实验不新增DA关闭臂，因此`DA0_REG0/DA0_REG1`为`N/A`。新类准确率和H只在`DA1_REG1`报告。

## 3.预注册判据

### 3.1技术闭合

- 125/125job完成，8/8shard为`PASS`，failed=0；
- 125份score与job receipt、250份prediction/COMMIT/fit-audit/resource-audit齐全；fit-audit包含750个state-scene row；
- 真实checkpoint truth-free smoke先于shard，所有query禁止访问字段为`false`；
- 不覆盖既有输出；同一prediction前确定性异常指纹在2个不同outer出现时共享停派；fresh retry=false；
- 禁止按中间性能停止。

### 3.2完整125性能确认

新结果与原D92 retry2的125个相同`receiver/seed/K/new-count`行配对，使用两者score的同口径总体值，不把不同场景或不同outer的单项极值拼接：

- K>2的100行mean`ΔH_old_new>0`，且至少80/100行`ΔH_old_new≥0`；
- 全125行mean`Δold_acc≥0`、mean`Δold_floor≥0`、mean`Δseen_new_acc≥0`；
- 全125行mean`Δforgetting≤0`；
- 25个K1行必须保持D92 alias语义，单独报告同排差异；
- 必须同时给出receiver、seed、K/new-count、场景和per-old-class floor分解；任一总体门失败即`NO_TARGET125_PROMOTION`。

### 3.3计算与状态

- K5/K10的two-state component fit精确为2，`DA1_REG1`actual component fit精确为1；K1 exact alias计数为3；
- query MAC和永久state不因删除Fisher/LOO而增加；
- 完整125报告`DA1_REG1`注册wall、CPU time和增量peak的分布；历史D92 CSV没有同口径资源receipt，因此不虚构125行paired wall比例；
- D92的理论two-state fit为K5`48`、K10`88`，本候选均为`2`，对应组件拟合次数分别减少`95.83%`与`97.73%`。Hard12-v3的paired wall/peak只作为既有独立资源证据。

## 4.发布登记

|项目|冻结值|
|---|---|
|本地Git仓库|`E:\type10-7\code\snapshots\d92_125wt`|
|本地环境|`ssr-gpu`|
|N607 Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_only_source_snapshot_20260812_v1`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_only_target125_20260812_v1`|
|远端logs root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_only_target125_20260812_v1`|
|GPU|GPU0–7各一个shard，每child CPU threads=2|
|代码commit|`ba1aeb7a`（runtime method/runner）|
|runtime archive|`E:\type10-7\code\snapshots\d92_e0_full_only_runtime_closure_ba1aeb7a.tar.gz`；4987764B；SHA256=`899e409d742c2135a2a5a09bdfb5055e918dd86d5704ac014c9c606ed92ca1b0`|
|config SHA|`13709fb300239526b1d7885bb5ceb90257ff70a0ac29d7f8e6c2a04b2f11c2c1`|
|launch SHA|`7646376b4f3e2860552ac3a084d90af4f3533e5c51c6f95a3361638d61ec9ab2`|

唯一启动命令预注册为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_only_source_snapshot_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

发布前只要求：实际Git入口、聚焦协议负测、独立`P0=0/P1=0`、不可变run路径、真实checkpoint truth-free smoke和N607资源预检。完成后在本报告追加完整同排结果表、资源分布、异常和最终裁决。

## 5.本地实现与验证

|文件|用途|
|---|---|
|`configs/stage2_d92_e0_full_only_target125_v1.json`|单臂方法锁、125矩阵和晋级门|
|`code/cvsrffi/stage2_d92_e0_full_only_target125.py`|完整Cartesian matrix builder与严格身份校验|
|`code/scripts/run_d92_e0_full_only_target125.py`|prepare、真实smoke、8shard执行和共享技术停派|
|两份对应测试|125/375覆盖、arm/path篡改负测、smoke前置、正常分派与distinct-outer停派|

`ssr-gpu`下新封装9项通过；连同E0D slim/query、D92 probe和既有E0OCF runner的相关回归共80项通过，`py_compile`、CLI help和`git diff --check`通过。config SHA256=`13709fb300239526b1d7885bb5ceb90257ff70a0ac29d7f8e6c2a04b2f11c2c1`；selection SHA256=`e2d7a22c3f6968a661e9fc28a4b4259b33c286e1eb944a4d20bb42f0c49da67c`。

独立release review结论为`APPROVE`，`P0=0，P1=0`。审查确认单臂身份、K1 smoke、125×3覆盖、源包/seal、预测后独立评分以及跨outer共享技术停止均已闭合；N607 preflight与真实smoke属于下一执行步骤。

同步映射固定为：runtime archive→`source_root/d92_e0_full_only_runtime_closure_ba1aeb7a.tar.gz`；config→`source_root/configs/stage2_d92_e0_full_only_target125_v1.json`；launch→`source_root/launch.sh`。归档来自Git commit`ba1aeb7a`的完整`code/`树，共1296个成员，已核对包含`code/cvsrffi/__init__.py`、目标builder/runner及复用的E0OCF closure入口，且不存在`code/code`层级。

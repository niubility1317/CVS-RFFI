# 官方CSIL到PyTorch/CVS的移植路线

## 判断

官方MATLAB仓库应作为论文忠实复现的源参照。当前移植策略是：

1. `paper_reproduction/CSIL`保留ADS-B论文原始层，只对齐官方MATLAB机制和论文图表，不声明CVS Stage2或卫星部署成功。
2. `paper_reproduction/cvs_aligned`只接收经过CVS协议重定义后的扩展实验，例如把CSIL增量头接到CVS`z_id`特征上。
3. 服务器运行前先完成本地PyTorch机制校验、报告和同步清单；不把DataPort数据作为当前任务前置。

## 已据官方仓库修正的实现语义

|官方机制|官方MATLAB证据|当前PyTorch对接|
|---|---|---|
|zero-bias fingerprint层不是普通`bias=False`线性层|`zeroBiasFCLayer.m`中归一化权重、归一化输入，输出`normMag*cosine+normMag`，`normMag=5`|`ZeroBiasCosineClassifier`默认输出`5*cosine+5`|
|扩展后EWC只比较旧参数区域|`calcFisherLoss`对扩展层使用旧尺寸切片|`compute_ewc_penalty`在当前参数更大时切到`previous_params`形状|
|KD是旧类fingerprint/logit MSE，不是KL蒸馏|`modelGradientsOnWeightsEWC`中比较旧网络输出和新网络旧类输出|`compute_csil_loss`要求KD响应shape一致并对旧响应`detach`|
|mask必须作用到完整SGD+L2+momentum更新|`sgdmFunctionL2`先算`grad+2*L2*param`和momentum，再乘`gradientMasks`|`csil_masked_sgd_step`对完整velocity乘mask，避免冻结块被weight decay或旧momentum移动|
|扩展层不能造成device/dtype漂移|MATLAB在GPU数组上构造mask/参数|`expand_for_stage`保持原模块device/dtype|

## 官方MATLAB机制中仍需移植的部分

|优先级|事项|理由|
|---|---|---|
|P0|实现完整五阶段训练入口|当前仍是协议dry-run和组件测试，未产出Fig.7/Fig.8/Fig.9|
|P0|接入官方主控脚本语义|主控默认不是裸`CSIL.m`，而是`CSILLockOldFPsChessBoardPast5000`系列变体|
|P0|实现Fisher估计器|官方使用`mean(log(prediction))`梯度并取`exp(grad^2)`近似FI，当前只接收外部Fisher映射|
|P0|复刻或明确替代`Past5000`旧样本策略|官方默认变体会加入最多5000个旧样本；如果论文文字要求“无historical data”，必须在报告中解释代码与论文叙述差异|
|P1|补`fc_bf_fp`和`Fingerprints`两层分离|当前骨架把embedding和classifier概念化，仍不是官方MATLAB层级|
|P1|实现baseline/消融脚本|`noCSI*.m`、`Fixrep*.m`对应Fig.8/Fig.9/TableII所需对照|
|P1|输出DoC、old/new/overall、forget/stage的统一CSV|图表复现和子agent逐项监督需要结构化artifact|

## CVS对接边界

CSIL接入CVS时只能作为`cvs_extension=true`扩展，不继承ADS-B论文数据协议。最小CVS字段必须包括：

`stage`、`dataset_family`、`source_receiver_ids/source_receiver_labels`、`target_receiver_ids/target_receiver_labels`、`target_old_tx_ids/labels`、`target_new_tx_ids/labels`、可选`target_unknown_tx_ids/labels`、`k_shot`、`support_split`、`query_split`、`support_query_disjoint=true`、`target_channel_view`、`target_channel_scenarios`、`threshold_scope`、`unknown_query_scope=evaluation_only`、`sample_strategy`、`receiver_label_alignment`、`claim_boundary`、`old_acc`、`seen_new_acc`、`H_old_new`。

禁止项：

- 不得把ADS-B论文复现写成CVS Stage2-A/B/C结果。
- 不得把clean view或ADS-B地面数据写成satellite/LEO deployment success。
- 不得在Stage2-B声称seen-new identity accuracy。
- 不得用`Y_unknown`替代`Y_new`，也不得用unknown query调Phase2阈值。

## 服务器运行建议

服务器上优先跑两层任务：

|层级|目标|数据|输出|
|---|---|---|---|
|paper-faithful ADS-B|复现官方CSIL五阶段、baseline、消融|DataPort ADS-B数据或用户服务器已有等价文件|Fig.7/Fig.8/Fig.9/TableI/TableII CSV和报告|
|CVS-aligned extension|把CSIL增量头接到CVS`z_id`或特征导出上，评估Stage2-C target-old+seen-new|WiSig/ManySig协议样本，`R_s/R_t`不相交|`cvs_extension=true`同row指标表|

任何N607或其他服务器执行前，应先在本地生成报告，记录本地Git状态、同步文件、官方参考commit、命令、环境、数据路径和声明边界。

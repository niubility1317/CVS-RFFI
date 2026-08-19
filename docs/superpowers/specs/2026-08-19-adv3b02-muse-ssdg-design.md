# ADV3B02-MUSE-SSDG设计说明

## 1.目标与边界

本设计在`ADV3B02_CORE90_SOFT_E200`的模型主干、`160D z_id`、`z_dom`、源域开放集几何和部署接口上，重构Phase1半监督训练机制。新候选命名为`ADV3B02_MUSE_SSDG_E200`。

方法遵守当前Phase1数据协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；四个角色的物理样本ID互斥；训练、校准和选择期间不得访问target receiver；`U_s`只提供source-domain标签，不提供TX标签；`V_cal`只用于校准，`V_select`只用于选择；checkpoint采用`final_only`。

`U_s`不得生成proxy unknown，不得校准开放集半径、能量阈值或尾部边界。开放集几何和proxy unknown只能由`L_s`构造并由`V_cal`校准。

## 2.方法结构

### 2.1 保留的ADV3B02能力

- 保留现有CV-SincNet双表征主干和`z_id/z_dom`语义。
- 保留有标签clean+satellite训练、CosFace/分类损失、源域episode、紧致性和开放集边界损失。
- 保留现有EMA、卫星信道模拟、域头、GRL和prototype memory的可复用接口。
- 保持Phase2部署bundle接口兼容，不把训练期局部头、自监督投影头或扰动回归头导出。

### 2.2 新增的训练期模块

1. `MUSESchedule`：按epoch产生EMA、未标注分类、卫星采样、GRL和prototype动量参数。
2. `MUSEEvidenceFusion`：融合全局头、source-domain局部低秩头和分类原型头。
3. `MUSEReliabilityRouter`：根据置信度、top1-top2间隔、三头JS分歧、原型距离和时间稳定性，将未标注样本路由到`U_H/U_M/U_L`。
4. `MUSECandidateSetLoss`：为`U_L`构造累计概率质量至少`0.75`且最多3类的候选集合；无法满足时不产生身份分类梯度。
5. `MUSESelfSupervisionHead`：对身份安全的weak变换计算训练期自监督损失。
6. `MUSENuisanceHead`：从`z_dom`回归卫星模拟器产生的SNR、CFO、相位噪声、K因子、时延和AGC等扰动参数。
7. `MUSEClassificationPrototypeBank`：仅吸收连续稳定至少3次的`U_H`，未标注更新权重限制在`0.05-0.10`。

## 3.多证据融合与可靠度

EMA教师只读取weak/source视图。三路概率分别为全局分类概率、source-domain局部头概率和分类原型概率，默认以几何平均方式按`0.50/0.25/0.25`融合。局部头只读取样本已知的source-domain标签，不读取TX真值，并且不进入部署bundle。

融合概率执行source-domain类别先验对齐，默认强度`gamma=0.35`，校正比截断到`[0.5,2.0]`。可靠度由以下量共同确定：融合top1置信度、top1-top2间隔、三头JS分歧、最近prototype距离和跨epoch预测稳定性。

- `U_H`：硬标签CE；可参与小权重分类prototype更新和跨receiver对齐。
- `U_M`：soft-target CE，按连续可靠度归一化加权；可参与跨receiver对齐，不更新开放集几何。
- `U_L`：候选集合损失或仅域、自监督、扰动预测；禁止硬标签CE和熵最小化。

所有未标注损失按有效样本权重和归一化，避免固定batch分母导致稀疏路由时梯度随覆盖率异常缩小。

## 4.训练日程

训练共200个epoch，每个epoch由完整遍历一次`U_s`定义；`L_s`加载器循环使用。默认`B_l=32`、`B_u=96`。

|阶段|Epoch|行为|
|---|---:|---|
|S1|1-16|`U_s`参与域分类、GRL、自监督和扰动预测；不产生未标注身份分类梯度|
|S2A|17-40|启用EMA三证据融合；`U_H`硬监督、`U_M`软监督；`lambda_u:0->0.2`，`p_sat:0->0.25`|
|S2B|41-68|启用`U_L`候选集、跨receiver对齐和稳定`U_H`小权重prototype更新；`lambda_u:0.2->0.5`，`p_sat=0.5`|
|S3A|69-160|完整H/M/L联合训练；默认`lambda_u=0.6`|
|S3B|161-180|分类prototype动量由`0.95`提高到`0.99`，提高稳定性要求|
|S3C|181-200|冻结伪标签memory、阈值统计、未标注prototype和局部教师头；`lambda_u=0.25`，集中巩固有标签分类与开放集几何|

EMA默认衰减为Epoch 1-16使用`0.99`，Epoch 17-68使用`0.995`，Epoch 69-200使用`0.999`。GRL从`0.02`平滑增长到`0.10`；若首轮诊断显示域分类崩塌，最高只允许在后续候选中调整到`0.15`。

## 5.卫星视图与双几何保护

每个`U_s`样本的学生路径按`hash(physical_sample_id,epoch)`确定为strong或一个satellite视图，避免同时计算三种学生视图。卫星采样概率遵循训练日程，教师始终只使用weak/source视图。

分类prototype允许稳定`U_H`以低权重进入；开放集prototype、半径、能量、尾部和known/unknown边界严格排除全部`U_s`。训练结束导出时只保留模型主干和从合法Phase1知识形成的部署组件。

## 6.首轮四级矩阵

首轮使用相同物理样本、split、seed、优化步数、卫星参数和最终checkpoint策略。

|候选|机制|
|---|---|
|M0|同协议ADV3B02训练机制控制组|
|M1|M0+从Epoch 1利用全部`U_s`的域/GRL/自监督/扰动预测|
|M2|M1+三头融合、可靠度连续加权和H/M/L路由|
|M3|M2+未标注卫星学生、跨receiver prototype对齐和双几何保护，即完整MUSE-SSDG|

首轮采用单seed最小可证伪实验。若M3未达到预登记科学门槛，先分析失败机制，不直接扩大到多seed或完整S0-S8矩阵。

## 7.测试、日志与完成条件

代码验证包括：日程边界、三头融合数值稳定性、H/M/L互斥完备性、候选集上限、低置信无身份梯度、时间稳定memory冻结、未标注prototype更新上限、开放集几何拒绝`U_s`、基于样本ID的卫星视图确定性、训练期头不进入deployment state。

每个完成训练的候选必须使用其最终checkpoint评测并分别保存：

- clean测试；
- `leo_clear_weak`测试；
- `leo_low_elev_weak`测试；
- `leo_rain_weak`测试。

训练完成但缺少任一测试、checkpoint身份、评测配置或逐场景日志时，不得标记为`ARTIFACTS_COMPLETE`或`ANALYZED`。

## 8.首轮观测指标与停止规则

除clean和逐LEO场景准确率外，记录最差类别、receiver floor、`U_H/U_M/U_L`覆盖率、有效未标注权重、伪标签precision、三头分歧、prototype更新量和`z_id->receiver`泄漏探针。`U_s`隐藏TX标签只可由训练外诊断器在不影响状态和选择的条件下计算precision，不得进入训练分支。

仅协议越权、错误split/seed/场景、输出覆盖、不能产生checkpoint或prediction、确定性执行异常等技术问题允许停止。低性能只触发分析，不触发技术停止。

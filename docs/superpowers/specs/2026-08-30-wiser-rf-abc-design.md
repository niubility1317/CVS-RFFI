# WISER-RF ABC实现设计

日期：2026-08-30

## 1. 决策

同一代码发布与最小pilot包含五个实验臂：冻结ADV3B02基线、A/WB-FT、B/WB-FT+VSW、C/WB-FT+模型反演、ABC联合。A与B使用正式`p2_min_v1`权限；C以及包含C的联合臂标记为`DIAGNOSTIC_MODEL_INVERSION_NON_FORMAL`，不能用于正式晋级。

## 2. 正式Phase1摘要

正式ADV3B02优先复用与checkpoint SHA匹配的`int8_domain_class_prototypes.npz`：6类、26个source域、84个有效域×类聚合中心，文件5,363B。每类14个有效域中心直接组成经验虚拟源分布。若checkpoint只携带`int8_domain_class_center_lowrank_residual_radius_v2`，则按需反量化低秩方向构造sigma points。两种格式都不得恢复或持久化源样本级embedding、读取source数据路径或在线更新摘要。

对类别$c$，从量化中心$a_c$、域残差基$u_{c,j}$、域系数尺度和类半径构造：

$$
v_{c,0}=\operatorname{Norm}(a_c),\qquad
v_{c,\pm j}=\operatorname{Norm}(a_c\pm\tau_{c,j}u_{c,j}).
$$

$\tau_{c,j}$由冻结域系数的稳健RMS与冻结类半径共同确定。VSW只比较合法target-old support特征与这些固定虚拟源特征。

## 3. A/WB-FT

冻结`id_backbone.cls_head.head`、完整domain分支、domain/adv/aux heads、DAC/PA局部分支和Sinc前端。训练主身份时频路径、投影、fusion和identity projection。模型保持eval模式以关闭Dropout/MixStyle；可训练参数仍保留梯度，GroupNorm仿射参数按阶段更新。

损失为：

$$
\mathcal L_A=\mathcal L_{\mathrm{src-head}}+0.5\mathcal L_{\mathrm{LOO-proto}}+\lambda_{SP}\mathcal L_{\mathrm{L2-SP}}.
$$

不创建、训练或持久化target head。

## 4. B/VSW

$$
\mathcal L_B=\mathcal L_A+\lambda_s\mathcal L_{\mathrm{VSW}}.
$$

首轮固定32个确定性投影方向；方向由run seed生成，与query无关。虚拟源摘要不可训练。VSW梯度只能进入当前阶段已解冻的identity参数。

## 5. C/模型反演诊断

C从随机噪声初始化IQ，使用冻结checkpoint和冻结源分类头优化出类别条件伪IQ，不读取真实source IQ、source loader或样本级source feature。伪IQ只存在于诊断run root，不能加入正式deployment bundle，不能把结果用于`p2_min_v1`晋级。

## 6. 渐进解冻

- Stage0：映射与零更新基线。
- Stage1：`t3/f3`、时频投影、fusion、identity projection。
- Stage2：加入`t2/f2`。
- Stage3：加入`t1/f1`和主身份前端；Sinc保持冻结。

阶段与最终状态只由预登记步数和support目标决定，query指标不得选阶段、早停、回滚或调参。

## 7. 评测与晋级

同一历史pilot的三个LEO场景先生成P1冻结源头、P2冻结源原型、P3 old-only D92 prediction，再由独立truth-last scorer评分。正式晋级只比较B0、A和B；C/ABC-C只报告诊断上界。首轮不发布完整125。

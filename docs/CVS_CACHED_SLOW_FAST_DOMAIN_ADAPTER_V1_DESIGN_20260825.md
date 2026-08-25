# CVS Cached Slow-Fast Domain Adapter V1设计冻结

日期：2026-08-25

## 1.目标与证据起点

本设计落实用户提供的“地面慢参数＋目标域快参数”方案。冻结`ADV3B02_CORE90_SOFT_E200`主干和地面类原型，Phase1.5只在source侧缓存特征上学习类无关域偏移基；Phase2只用合法旧类support估计不超过24个快参数，随后冻结状态并逐条处理query。

此前10个Meta-Adapter Target5候选已经证明：扩大少量层更新或增大logit scale可以改变score，但多数候选不能改变top-1；scale16虽产生52次决策变化，clear weak floor下降5pp。V1不再让target support更新完整低秩矩阵，而是把可更新自由度限制在地面已学习的域方向内。

## 2.结合CVS实际的校正

1. 特征维度不硬编码为附件示例的256。运行时以冻结原型宽度为准；当前ADV3B02正式`z_id`和原型均为160维。
2. Phase1.5缓存属于地面训练临时资产，不进入Phase2 bundle。部署bundle只保存慢参数、快参数初始化/步长、原型、类别映射和基座checkpoint标识。
3. K=1不把同一物理样本的数学view当作第二shot。V1在K=1固定回退`DA0_REG0`，待存在独立物理support后再允许快更新。
4. 不依据历史target truth挑“最差row”。首次诊断固定同一`K10/new10`、receiver`20-1`、seed`392002`，在三种LEO weak场景各一行。
5. Phase2不使用source cache、clean IQ、query truth/role或query反馈，不训练分类头、协方差或D92式持久判决头。

## 3.统一接口

冻结身份特征为`z∈R^d`，地面原型为`P^g∈R^(C×d)`。所有候选输出

```text
z_adapt = normalize(z + delta(z; slow, fast))
```

support损失直接使用正式判决：

```text
CE(scale * normalize(z_adapt) @ normalize(P^g).T, y_support)
```

这保证support更新空间与query判决空间一致。

## 4.三个候选

### 4.1 COMMON_SHIFT_R4

地面按类中心化各source domain均值，对公共域残差做SVD，得到`U4`。目标support以岭回归估计4维系数`a`，使用`delta=-U4 a`消除公共偏移。它没有梯度更新，是判断公共低秩平移是否存在的闭式基线。

### 4.2 FAST_FILM_R8

地面学习`U,V∈R^(d×8)`、受约束残差强度`rho`和统一快参数步长。Phase2只更新`gamma,beta∈R^8`：

```text
h = V.T @ layer_norm(z)
delta = rho * U @ ((1 + gamma) * h + beta)
```

`U,V,rho`在Phase2冻结，快参数总数16。

### 4.3 FAST_LOWRANK_R8

在FAST_FILM_R8基础上增加8维方向可信门`q`：

```text
delta = rho * U @ (sigmoid(q) * ((1 + gamma) * h + beta))
```

Phase2只更新`gamma,beta,q`，总数24。`q`按保守小残差初始化，不能绕过trust-region。

## 5.Phase1.5缓存与训练

缓存只读取Phase1 source角色，字段固定为`z_id/label/receiver/day/scene/physical_sample_id/view`。support/query物理ID保持互斥；clean/LEO配对仅用于地面pair loss，不增加K。

慢参数目标由冻结原型CE、同物理样本clean/LEO余弦配对、smooth worst-class和区间trust组成。floor权重固定0.2；位移半径未超过`r_max`时不惩罚。episode从`K={1,2,5,10}`采样，概率固定`0.1/0.2/0.5/0.2`，query使用不同物理样本。

## 6.Phase2安全选择

K≥2时，在`lambda={0,0.25,0.5,0.75,1}`中用support leave-one-out macro、floor、margin和位移半径选择。候选必须同时不劣于lambda=0的LOO macro和floor，并满足位移上限；否则回退lambda=0。K=1固定lambda=0。

选定状态后立即冻结。query打开后不再选择步数、lambda、阈值或候选，不更新模型、原型或任何buffer。

## 7.最小实验

首次矩阵为三个候选×三种LEO weak场景，共9个同输入诊断row；operating point固定`K10/new10`。每个row输出`DA0_REG0/DA1_REG0`prediction，独立scorer在prediction闭合后连接truth。

进入完整Target5的预注册门槛为：三个场景旧类宏平均变化≥+1.0pp、全体类floor变化≥+0.5pp、任一旧类退化不超过5pp。未达标记`SCIENTIFIC_FAILURE_NO_PROMOTION`，不扩大到Target5/Target25。

## 8.实现边界

V1不实现`z_dom`条件hypernetwork、类条件Adapter、二阶MAML、新类support参与域更新、中间时频层Adapter或跨query联合估计。这些路线只有在V1证明公共域方向可迁移后才有单独研究价值。

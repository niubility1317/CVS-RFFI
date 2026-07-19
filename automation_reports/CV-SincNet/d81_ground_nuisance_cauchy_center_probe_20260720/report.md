# D81地面扰动谱稳健target原型实验报告

## 1.实验登记

|字段|值|
|---|---|
|实验ID|`d81_ground_nuisance_cauchy_center_probe_20260720`|
|候选|`ground_nuisance_cauchy_center`|
|operator|Codex`/root`|
|状态|`PREREGISTERED_NOT_IMPLEMENTED`|
|目标|高效利用全部地面压缩原型估计support样本的跨域扰动可靠性，同时让query判别几何完全由target support决定|
|开发单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|
|formal状态|当前ground组件资格false/UNVERIFIED，D81仅development diagnostic|

## 2.假设与创新点

D77-D80已经排除了把ground质心、低秩投影或ground协方差直接放进query距离/协方差的路线：这些方法能保护部分旧类，却把新类身份方向误当域噪声。D81把ground的作用前移到注册阶段，只回答“同一target类中哪个support样本更像受到已知跨域扰动”，再以target support自己形成稳健类中心。

该设计有三个隔离性质：

1. 地面old6类不提供任何类别锚点或query score，只提供类无关扰动方向；
2. 每类共同平移保持类内残差和target协方差不变，因此不会重写D62的target度量；
3. 权重在每个OOF fit内重算，held support和query均不可见。

## 3.锁定公式

从84个地面domain-class类中心构造`r_dc=g_dc−mean_d(g_dc)`与协方差`G`。对正特征值`lambda_j`计算：

`r_eff=(sum_j lambda_j)^2/sum_j lambda_j^2`，`r=ceil(r_eff)`。

固定保留前`r`个方向，并令`pi_j=lambda_j/sum_{l<=r}lambda_l`。对当前fit可见的target类`c`：

`e_ci=sum_{j<=r} pi_j [u_j^T(z_ci−mean_i z_ci)]^2`

`raw_w_ci=1/(1+e_ci/mean_i e_ci)`，`w_ci=raw_w_ci/sum_i raw_w_ci`

`mu_robust_c=sum_i w_ci z_ci`

`z'_ci=z_ci+(mu_robust_c−mean_i z_ci)`。

若能量为0则等权；K1显式identity，K2因两个中心残差互为相反数而严格等权identity。只变换z160，FFT96/RF32保持bitwise不变。禁止rank、尺度、温度、平移系数或场景/类别权重扫描。

## 4.协议与资源边界

- 数据状态沿用D18`VALIDATED_ONCE`；方法变更不触发重建/重验。
- 单一固定`LEO_weak`观测；support-only；query独立一次评分；无clean/source/query truth/role/quota/global assignment。
- target-old/new完全相同公式；不访问类ID语义、old/new角色、receiver handle或scene handle。
- 使用全部84个ground cell估计谱；当前组件无sample radius/count，不伪造这些统计。
- 预计新增适配复杂度为每次fit`O(N*r*160)`，其中`r`由ground effective rank自动确定；新增参数/optimizer step/query MAC均为0。
- 持久状态仍为D62单affine＋25,428B ground组件，≤256KB。

## 5.联合晋级门

相对同row D62：总体`A/N/H/J/min-class B/A/N`不得下降、`F`不得上升；每个场景`A/N/H`不得下降、`F`不得上升；`old→new`、`new→old`、`new→wrong-new`均不得增加；且至少一个联合指标严格改善。INT8/FP32不得发生outer argmax或margin-sign翻转。任一失败即停止，不启第二seed、125或N607。

## 6.版本状态

根目录`E:\type10-7`非Git仓库。实现、trace和本报告先进入独立Git worktree`E:\type10-7\code\snapshots\d81wt`，基于主发布分支提交`4dcf066b`；完成本地验证后再以精确commit闭环回主发布分支。服务器暂不使用。

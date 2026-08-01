# D111-LOO-GAT评分核实现记录

## 结论

`D111-S01`至`D111-S03`已完成，独立Terra Max复审结论为`GO / P0=0 / P1=0 / P2=0`。实现验证了D111具有非恒等、可回退、query-independent的计算路径；它没有读取accuracy，因此不构成正收益证据。

## 计算路径

评分核从正式D111 bundle和现有typed qKNN support bank构建一次性enrollment状态。六个旧类各自形成归一化support均值(m_c)和三维残差(r_c=U(m_c-g_c))。每个旧类只用其他五类残差，执行固定32步、阻尼(1/2)的Weiszfeld迭代。实现从最终迭代点构造满足零和与单位范数约束的dual变量；只有dual可行、primal-dual gap不超过封存(epsilon)、且至少3/5残差位于封存(B)内时，该类才取得正的(ho_c)。其余情况严格令(ho_c=0)。

K=1时目标均值方差使用封存(v_s)。K>1时，(S_c^2)定义为围绕归一化(m_c)的无偏逐坐标chord scatter，再使用(max(v_s/K,S_c^2/K))。Phase1 `class_radii`已锁定为逐坐标RMS，故(v_g)、(v_s)、(S_c^2)和((6B+epsilon)^2/160)量纲一致。

评分先调用原M0 identity Student-t qKNN。D111强制`kernel_volume_gamma=1`，在support和anchor两项中恢复同一完整Student-t归一化常数，完成总质量为1的`logaddexp`混合后，再减去全类共同常数回到M0 logit原点。未合格旧类和全部新类直接保留M0列；不存在old bias、额外先验或裸`K+1`平均。

## 验证与资源

|项目|结果|
|---|---:|
|D111评分核定向测试|9 passed|
|D111 bundle＋评分核|19 passed|
|连同Phase1 center-lowrank与D105相邻回归|217 passed|
|独立复审|P0=0/P1=0/P2=0/GO|
|新增持久数值态（7类示例）|约4.7KiB|
|enrollment投影|2,880 MAC|
|固定Weiszfeld标量工作量|2,880步内标量维操作|
|每query额外上界|960 MAC|
|query-dependent持久态|0 byte|

测试覆盖固定32步与dual gap、3/5边界、严格M0回退、K1非恒等、新类(ho=0)、完整单位质量公式、query顺序与拆批不变、类置换、坐标置换、稠密正交实数层等变、深度只读、receipt绑定和资源回执。INT8量化不声称bit级稠密旋转不变。

## 下一步

生产authority公钥和真实formal bundle尚未生成。完成这项外部资产后，只需一个真实588 tap的无truth G0，同时覆盖K1/K5/K10，检查资格数、正(ho)数、anchor变化、score/margin变化和`argmax_changed_count`。任一K完全恒等即淘汰D111；三K均有功能变化才发布一个source-held G1。不会运行125矩阵，也不会扫描rank、(B)、(ho)或阻尼。

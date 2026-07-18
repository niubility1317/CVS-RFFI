# D36-CJIC可编译联合int8校准设计追踪

## 1. 研发目标

D36同时解决两个同等重要的问题：Stage2-B旧类目标域适应不足，以及Stage2-C注册后新旧类混杂。D33–D35表明，只改注册层无法越过FAST注册前old=82.22%的瓶颈；稀疏hard visibility会造成新类不可达，全局max-residual门又会造成held旧类被新类大量侵入。因此D36不再调整二值可见性，而是联合学习轻量表征算子、对称int8注册头和连续新旧margin校准器。

开发输入仍只有已准入的LEO_weak support capsule及密封Phase1 deployment bundle。query保持关闭。每个物理support只有一个LEO接收观测；`z160`、`FFT96`与`RF32`均从同一固定接收IQ确定性提取并拼接为一行，不增加view、physical sample或K。

## 2. 单观测拼接表征

令

\[
\phi(x)=N\left([N(z_{160}),4N([FFT_{96},RF_{32}])]\right)\in\mathbb R^{288}.
\]

该公式与当前表现最强的B3拼接几何完全一致，等价于z160块能量`1/17`、FFT96+RF32联合辅助块能量`16/17`；不引入未经实验支持的新块权重。权重不按receiver、scene、fold、K或结果重选。FFT96和RF32只读取当前密封IQ，不调用LEO channel simulator，也不产生新的overlay provenance。

## 3. 极轻联合适配器

适配器固定为

\[
A_\theta=\operatorname{diag}(e^d)+UV^\top,\qquad r=2,
\]

其中`d∈R^288`，`U,V∈R^(288×2)`，瞬时可训练参数共1,440。D36-A令`U=V=0`；D36-B/C启用rank-2。低秩残差使用固定shot收缩

\[
h(K)=\frac{K-1}{K+3},
\]

所以K1自动关闭rank-2残差；K5/K20不得重新选择收缩、epoch或学习率。

Stage2-B只使用target-old support，全批SGD 6epoch/6step：

\[
L_B=L_{CE}^{old}+0.5\operatorname{CVaR}_{top2}\{L_c^{old}\}
+0.1\lVert d-d_{Fisher}\rVert_2^2
+10^{-3}(\lVert U\rVert_F^2+\lVert V\rVert_F^2).
\]

`d_Fisher`来自当前target-old support的闭式对角初始化，不是source Fisher。Stage2-C再用全部已注册target-old与target-new support执行6epoch/6step：

\[
L_C=0.5L_{CE}^{old}+0.5L_{CE}^{new}
+0.5\operatorname{CVaR}_{top2}\{L_c\}
+L_{preserve}^{old}+10^{-3}\lVert\theta-\theta_B\rVert_2^2,
\]

\[
L_{preserve}^{old}=|S_o|^{-1}\sum_{x\in S_o}
[m_B(x)-m_C(x)+0.1]_+.
\]

总计12epoch/12 optimizer step，低于活动目标的20epoch、50k参数上限。角色均衡只使用注册support标签；query角色和标签从不进入损失、选择或回滚。

## 4. 对称int8旧类校正与新类注册

对每个已注册类使用球面Huber中心与medoid收缩形成一个target原型：

\[
t_c=N\!\left((1-\alpha_c)\operatorname{HuberMean}\{A_\theta\phi(x)\}
+\alpha_c\operatorname{Medoid}\{A_\theta\phi(x)\}\right).
\]

`α_c`由类内稳健半径按固定公式确定；K1时mean=medoid。旧类target原型和新类target原型都必须量化为对称int8+FP16 scale，并保存FP16稳健半径；禁止持久化FP32 target prototype。

D36-B/C允许读取同一密封Phase1 bundle中的只读int8旧类锚。只在z160块进行不确定度融合：

\[
w_{g,c}=\operatorname{clip}\left(0.25\frac{u_{t,c}}{u_{t,c}+u_{g,c}},0,0.20\right),
\]

\[
p_{c,z}=N((1-w_{g,c})t_{c,z}+w_{g,c}g_c).
\]

`u_t`由target support稳健半径与有限K惩罚构成，`u_g`只读取密封bundle中的int8 radius；FFT96/RF32保持纯target。地面锚不更新、不按query选择，也不产生source replay或source-target分布损失。新类始终是纯target独立注册。

## 5. 编译式部署

适配完成后把算子编译进每类权重：

\[
\widetilde p_c=A_\theta^\top p_c,
\qquad q_c=Q_{int8}(N(\widetilde p_c)).
\]

星上预测器只保存全部旧/新类的`q_c`、scale、radius、registry和少量校准标量；不保存optimizer，也不在query路径执行对角或rank-2矩阵运算。每个query只需对全部注册类各做一次288维int8 dot，随后逐样本argmax。

5新类时共有11类，dot-MAC为`11×288=3,168`，比B3的3,456下降8.33%；20新类时共有26类，dot-MAC为7,488，相对K10 identity-only单qKNN的41,600下降82.00%。26类int8 head约7.8KB；加scale、radius、registry、校准器和约13.6KB密封Phase1组件，预计总状态低于32KB。

## 6. 连续新旧margin校准

所有新类始终具有有限score。对单个样本定义old/new top-score、次高score和最近原型半径标准化距离，构造6维特征：

\[
\psi(x)=[1,n_1-o_1,o_1-o_2,n_1-n_2,\rho_o-\rho_n,\min(\rho_o,\rho_n)].
\]

D36-C使用class-balanced ridge IRLS固定5次Newton更新得到`w∈R^6`：

\[
\Delta(x)=\operatorname{clip}(w^\top\psi(x),-2,2),
\]

\[
S_i^{old}=\ell_i,\qquad S_j^{new}=\ell_j+\Delta(x).
\]

校准训练数据来自outer-fit support内部固定4折交叉拟合：每个inner fold按每类rank留2个物理support，adapter、原型和半径只由其余support重建，再给inner-held样本生成`ψ`和support old/new二元标签。这样校准器不会使用对同一样本拟合出的自相似分数。D36-B只拟合一个class-balanced常数offset，D36-A不加新旧offset。

query输入schema只有单样本拼接特征、注册头、半径和校准状态；不包含query truth、query old/new角色、真实batch类数、类别quota、query排序或global assignment。校准器输出只对当前query生效，不更新任何状态。

## 7. 候选和否证门

- `D36-A-DIAG-TARGET-INT8`：对角联合适配+纯target old/new int8单原型，无新旧offset。
- `D36-B-R2-ANCHOR-CONST`：rank-2+只读地面int8锚+support交叉拟合常数offset。
- `D36-C-R2-ANCHOR-MARGIN6`：B+6维连续margin校准，预登记主候选。

K10 support-only outer-held门固定为：

1. 注册前old不低于B3的86.67%且目标为至少88%；注册后forgetting不超过3pp。
2. 注册后old/new/H分别严格优于B3的73.33%/73.33%/72.65%，并同时报告逐场景和逐类结果。
3. 每个outer fold任一旧类不得较其注册前退化超过10pp；重点旧类14-7、20-19不得由其他类均值掩盖。
4. 09f8至少达到50%，f608不低于73.67%；全部新类必须有正的physical LOO margin。
5. 活动参数≤50k、总epoch≤20、optimizer step≤20、持久状态≤50kB、无dense query图；query MAC必须低于B3。

任一主门失败即标记support-only负路线，不打开query、不生成正式性能声明、不扩正式125矩阵。只有K10锁定正路线后，才使用同一candidate和超参数依次执行K1/K5/K20压力测试；K5各主指标相对K10下降不超过3pp，K1必须报告注册前后old和`old_adaptation_gain≥0`。

## 8. 实现与证据边界

D36新增独立core、单测和共享runner候选集；不得修改或同步当前有他人改动的`stage2_diag_cosine_exploration.py`。完整训练trace必须逐step保存Stage2-B/C loss、CVaR、preserve loss、梯度范数、support准确率、量化误差、inner cross-fit校准trace和outer-held结果。资源审计必须把适配MAC、编译MAC、query dot-MAC、scalar ops、状态、延迟和峰值显存分开。

本设计没有修改`项目.md`的科学或数据协议；它实现现有Stage2-B/C、LEO_weak-only、query evaluation-only、逐样本全注册类和授权int8 bundle约束。

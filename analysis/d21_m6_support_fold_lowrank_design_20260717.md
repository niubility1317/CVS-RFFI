# D21-M6：support-fold恒等近端低秩适应设计

日期：2026-07-17
状态：`DESIGN_ONLY_SUPPORT_ONLY_QUERY_UNREACHABLE`
适用范围：Stage2-B域适应与Stage2-C新类注册；开发seed锁定统一rank和epoch后，确认seed只应用该锁。

## 1. 设计边界

M6只使用已注册K-shot support及其标签。设计、拟合、折叠选择、停止、回退和状态导出均不得打开query IQ、query特征、query token顺序、query预测、query标签、query角色或query类别计数。M6不引用M1b的query Pareto，也不使用任何历史query结果决定rank、损失权重、epoch或门限。

本设计把候选集合固定为：

\[
\mathcal A=\{I,\;M6(r=2),\;M6(r=4)\}.
\]

`I`是严格恒等候选。它既是比较基线，也是所有门禁失败时的原子回退状态。M6不增加descriptor分支，不构造同一物理样本的第二个LEO信道观测；输入特征来自每个固定received IQ的单一已登记特征向量。

## 2. 符号与注册状态

令\(x_i\in\mathbb R^d\)为L2归一化support特征，当前A0取\(d=256\)。Stage2-B只含\(C_o\)个旧类；Stage2-C含同一批\(C_o\)个旧类和\(C_n\)个seen-new类。每类恰有\(K\)个support，故

\[
N_B=C_oK,\qquad N_C=(C_o+C_n)K.
\]

旧/新身份来自已注册support registry，而不是query角色。模型对任一样本始终计算全部已注册类的分数。

## 3. `id_proj`低秩delta

### 3.1 有界残差映射

对rank \(r\in\{2,4\}\)，定义

\[
A_r=I_d+\Delta_r,
\qquad
\Delta_r=U\operatorname{diag}(s)V^\top,
\qquad
s=\epsilon\tanh(a),
\]

其中\(U,V\in\mathbb R^{d\times r}\)，列向量通过薄QR保持正交，\(a\in\mathbb R^r\)，固定\(\epsilon=0.125\)。变换后的部署特征为

\[
g_r(x)=\frac{A_rx}{\lVert A_rx\rVert_2}.
\]

由于\(\lVert\Delta_r\rVert_2\le\epsilon\)，有

\[
1-\epsilon\le\sigma_{\min}(A_r)
\le\sigma_{\max}(A_r)\le1+\epsilon.
\]

该界阻止小support把表征旋转成任意新空间。初始化采用确定性方法种子生成的正交\(U,V\)，并令\(a=0\)，所以初始状态严格满足\(A_r=I_d\)。Stage2-C从Stage2-B的\((U_B,V_B,a_B)\)初始化，不重新随机启动。

训练参数数目为

\[
P_r=2dr+r.
\]

当前\(d=256\)时，rank2为1,026参数，rank4为2,052参数，均远低于50k上限。

### 3.2 support分类分数

折内训练集记为\(T\)。对验证样本\(i\)，仅用\(T\)中第\(c\)类构造prototype：

\[
\mu_c^{(T)}=
\operatorname{norm}\left(\frac{1}{|T_c|}
\sum_{j\in T_c}g_r(x_j)\right),
\qquad
q_{ic}=g_r(x_i)^\top\mu_c^{(T)}.
\]

训练样本的prototype必须排除该样本自身；若折内某类只有一个样本，则该候选不可训练并回退恒等映射。交叉熵按类先平均，再按old/new分支平均，避免类数较多的一侧主导选择。

令

\[
L_c=\frac{1}{|T_c|}\sum_{i\in T_c}
-\log\frac{\exp(q_{iy_i}/\tau)}
{\sum_j\exp(q_{ij}/\tau)},
\qquad \tau=0.07.
\]

Stage2-B的中心损失为

\[
L_{eq}^{B}=\frac{1}{C_o}\sum_{c\in O}L_c.
\]

Stage2-C严格给予old/new相同总权重：

\[
L_O=\frac{1}{C_o}\sum_{c\in O}L_c,
\qquad
L_N=\frac{1}{C_n}\sum_{c\in N}L_c,
\qquad
L_{eq}^{C}=\frac{L_O+L_N}{2}.
\]

### 3.3 floor、CVaR与旧类保持项

对分支\(G\in\{O,N\}\)，取该分支最差25%类的平均损失：

\[
L_{tail}^{G}=\frac{1}{m_G}
\sum_{c\in\operatorname{TopLoss}_{m_G}(G)}L_c,
\qquad
m_G=\max\left(1,\left\lceil0.25|G|\right\rceil\right).
\]

Stage2-C的tail项同样old/new等权：

\[
L_{tail}^{C}=\frac{L_{tail}^{O}+L_{tail}^{N}}{2}.
\]

令冻结的Stage2-B模型在旧类support上的正确类margin为

\[
m_i^B=q_{iy_i}^B-\max_{c\in O,c\ne y_i}q_{ic}^B.
\]

Stage2-C的旧类遗忘训练代理为

\[
L_F=\frac{1}{|T_O|}\sum_{i\in T_O}
\operatorname{ReLU}(m_i^B-m_i^C).
\]

恒等近端由两部分组成：delta幅度和相对Stage2-B的漂移。

\[
L_{prox}^{B}=\frac{\|s_B\|_2^2}{r\epsilon^2},
\]

\[
L_{prox}^{C}=\frac12\frac{\|s_C\|_2^2}{r\epsilon^2}
+\frac12\frac{\|\Delta_C-\Delta_B\|_F^2}{r\epsilon^2}.
\]

固定训练目标采用已锁定的0.63/0.30/0.07权重，不把这些系数加入rank搜索：

\[
L_B=0.63L_{eq}^{B}+0.30L_{tail}^{O}+0.07L_{prox}^{B},
\]

\[
L_C=0.63L_{eq}^{C}+0.30L_{tail}^{C}
+0.07\left(\frac{L_F+L_{prox}^{C}}{2}\right).
\]

CVaR进入可微训练目标，但逐类floor只用作折外硬门禁。这样不会用一个不可导的最小准确率驱动小样本振荡。

## 4. class-balanced support folds

### 4.1 确定性分折

开发阶段设置

\[
F=\min(5,K).
\]

每个类内部按`SHA256(method_lock || scenario || class_handle || support_token)`排序，再round-robin分到\(F\)折。每折对每类保留相同数量的验证样本；不能整除时，各折同类样本数之差不超过1。K=10时每折每类验证2个样本，训练8个；K=5时每折验证1个，训练4个。每个support样本恰好产生一次out-of-fold预测。

Stage2-B和Stage2-C使用相同的旧类fold编号，因此旧类遗忘代理是配对比较。每个scenario独立建折，但rank和最终epoch在三个scenario间统一锁定。

### 4.2 小K规则

- K=1：强制`I`，训练参数为0，epoch为0。单样本类无法形成self-excluded训练或验证证据。
- K=2–4：禁用rank4。rank2只有在所有硬门禁均通过时才可保留，否则回退`I`。
- K≥5：比较`I`、rank2和rank4。
- 任一折出现空类、重复token、非有限特征或fold清单不平衡：整行fail closed，不生成适配状态。

该规则保证K=1仍有可部署结果，同时不把单样本拟合伪装成域适应收益。

## 5. 纯support选择指标

将每类K个OOF预测合并，得到Stage2-B旧类准确率\(a_{c,B}\)和Stage2-C准确率\(a_{c,C}\)。定义

\[
BA_B^O=\frac1{C_o}\sum_{c\in O}a_{c,B},
\qquad
BA_C^O=\frac1{C_o}\sum_{c\in O}a_{c,C},
\qquad
BA_C^N=\frac1{C_n}\sum_{c\in N}a_{c,C},
\]

\[
J_{eq}=\frac{BA_C^O+BA_C^N}{2}.
\]

逐类floor为

\[
f_B^O=\min_{c\in O}a_{c,B},
\quad
f_C^O=\min_{c\in O}a_{c,C},
\quad
f_C^N=\min_{c\in N}a_{c,C}.
\]

选择期的旧类遗忘代理只比较同一旧类、同一support OOF样本：

\[
F_{old}^{OOF}=\frac1{C_o}\sum_{c\in O}
\max(0,a_{c,B}-a_{c,C}).
\]

离散CVaR错误率定义为最差25%类错误率的平均值：

\[
R_{CVaR}^{G}=\frac1{m_G}
\sum_{c\in\operatorname{Worst}_{m_G}(G)}(1-a_{c,C}).
\]

这些统计只来自support OOF；最终query scorer不得把任何数值返回selector。

## 6. 硬门禁与rank选择

### 6.1 相对恒等候选的逐scenario门禁

rank候选\(r\)必须在每个scenario分别满足：

1. `before-old floor`不低于恒等候选：\(f_{B,r}^O\ge f_{B,I}^O\)。
2. `after-old floor`不低于恒等候选：\(f_{C,r}^O\ge f_{C,I}^O\)。
3. `seen-new floor`不低于恒等候选：\(f_{C,r}^N\ge f_{C,I}^N\)。
4. old/new CVaR均不恶化：\(R_{CVaR,r}^O\le R_{CVaR,I}^O\)且\(R_{CVaR,r}^N\le R_{CVaR,I}^N\)。
5. 旧类遗忘代理不恶化：\(F_{old,r}^{OOF}\le F_{old,I}^{OOF}\)。
6. 至少\(\lceil0.8F\rceil\)个fold上，old和new的fold-balanced accuracy均不低于恒等候选。

任一scenario失败即淘汰该rank；不允许用clear场景的提升抵消rain或low-elevation场景的floor下降。

### 6.2 统一rank的排序

通过硬门禁后，按以下字典序选择一个统一rank：

\[
\max\Big(
\min_s\min(f_{B,s}^O,f_{C,s}^O,f_{C,s}^N),
-\max_s\max(R_{CVaR,s}^O,R_{CVaR,s}^N),
\min_s J_{eq,s},
-\max_sF_{old,s}^{OOF},
-r
\Big).
\]

最后一项明确偏向较小rank。rank4只有在最差scenario的\(J_{eq}\)至少比rank2提高一个OOF判决分辨率时才可胜出：

\[
\delta_K=\min\left(\frac{1}{2C_oK},\frac{1}{2C_nK}\right),
\qquad
\min_s(J_{eq,s}^{r4}-J_{eq,s}^{r2})\ge\delta_K.
\]

如果没有非恒等候选通过门禁，输出`selected_rank=0`和严格恒等状态。selector不得选择“最不差的失败候选”。

## 7. 防小样本过拟合的停止条件

每个fold最多训练20epoch，每个epoch只在该fold的support validation上计算选择元组

\[
T_e=(f_{min,e},-R_{tail,e},J_{eq,e},-F_{old,e}^{OOF},-\|\Delta_e\|_F).
\]

训练采用以下停止与回滚规则：

1. patience=3：连续3个epoch未按字典序改善\(T_e\)，停止并回滚该fold最佳有效epoch。
2. 任一loss、梯度、参数或特征出现NaN/Inf，立即废弃该候选并回退`I`。
3. 连续2个epoch触及\(\|\Delta\|_2\ge0.99\epsilon\)且validation \(J_{eq}\)未改善，提前停止，防止长期贴边拟合。
4. 若训练集与validation的equal-weight balanced accuracy差
   \[
   G=J_{eq}^{train}-J_{eq}^{val}>\max(0.10,1/K)
   \]
   连续2个epoch成立，回滚到首次越界前的最佳有效epoch。
5. 少于\(\lceil0.8F\rceil\)个fold产生有效checkpoint，候选整体失败。
6. rank4若没有达到第6.2节的最小离散增益，自动降到rank2；rank2未过门则回退`I`。

开发seed锁定rank后，最终全support重拟合不再查看fold或query。部署epoch取所有有效fold最佳epoch的下四分位数：

\[
E^*=\max\left(1,\min\left(20,
\left\lfloor Q_{0.25}(e_{best})\right\rfloor\right)\right).
\]

下四分位数比均值更保守，避免少数晚收敛fold把全support训练推到过拟合区。Stage2-B训练\(E^*\)；Stage2-C从B状态初始化并训练同一\(E^*\)。确认seed不得重新选择rank、\(\epsilon\)、损失权重或epoch。

## 8. 参数、MAC与状态上界

### 8.1 adapter与分类MAC

单样本低秩变换依次计算\(V^\top x\)、\(\operatorname{diag}(s)V^\top x\)和\(U(\cdot)\)，MAC上界为

\[
M_{adapter}=2dr+r.
\]

all-registered-class单qKNN使用每类K个int8-dequant support code，分类MAC为

\[
M_{qKNN}=CKd,
\qquad
M_{query}\le2dr+r+CKd.
\]

当前\(d=256,C_o=6,C_n=5,K=10\)的上界为：

| 状态 | rank | 参数 | adapter MAC/query | qKNN MAC/query | 总MAC/query |
|---|---:|---:|---:|---:|---:|
| Before | 2 | 1,026 | 1,026 | 15,360 | 16,386 |
| Before | 4 | 2,052 | 2,052 | 15,360 | 17,412 |
| After | 2 | 1,026 | 1,026 | 28,160 | 29,186 |
| After | 4 | 2,052 | 2,052 | 28,160 | 30,212 |

没有dense query图，单个query不依赖其他query，也没有batch assignment。

### 8.2 持久状态

adapter参数以FP16保存，占\(2P_r\)B。每个256维support code采用int8向量和一个FP16 scale，占258B。总状态上界为

\[
S(C,K,r)=258CK+2(2dr+r)\;\text{B}.
\]

| 状态 | rank2 | rank4 |
|---|---:|---:|
| Before：6类×K10 | 17,532B | 19,584B |
| After：11类×K10 | 30,432B | 32,484B |

rank4最坏状态仍小于32KiB级别，且不保存训练样本的FP32特征副本。

### 8.3 最终适配MAC上界

以pairwise LOO相似度作为保守上界，单阶段20epoch适配MAC为

\[
M_{adapt}\le20\left[N(2dr+r)+N(N-1)d\right].
\]

当前rank4、K10单次最终拟合上界：

- Stage2-B，\(N=60\)：20,587,200MAC。
- Stage2-C，\(N=110\)：65,903,200MAC。

support-fold开发选择会训练临时fold模型，但这些模型不进入部署bundle；正式确认和星上应用只保留一次全support拟合及一个最终状态。

## 9. 必须落盘的support-only证据

实现M6时，每个开发行必须保存：

- capsule/support manifest SHA256、method lock SHA256、scenario/class/token清单；
- 每类确定性fold分配和fold root hash；
- `I/rank2/rank4`每折逐epoch完整loss、floor、CVaR、\(J_{eq}\)、旧类遗忘代理、\(\|\Delta\|_2\)；
- 每个硬门禁的通过/失败原因、统一rank排序元组和回退原因；
- 最终\(E^*\)、参数数目、MAC、FP16 adapter状态、int8 support code状态及各自SHA256；
- 明确的`query_access=false`、`query_fit=false`、`query_truth_opened=false`、`query_role_oracle_access=false`、`query_true_batch_class_count_access=false`、`query_class_quota_access=false`、`query_batch_global_assignment=false`；
- runtime access audit证明selector和trainer未打开任何query/scorer成员。

若上述任一证据缺失，M6只能标为`SUPPORT_SELECTION_EVIDENCE_INCOMPLETE`，不得生成query预测或性能声明。

## 10. 实现判定

M6的正向判定不是“rank2或rank4必须胜出”，而是support OOF证据能否在三个LEO_weak场景同时守住Before-old、After-old和seen-new的逐类floor/CVaR，并且不增加旧类遗忘代理。恒等回退是合法结果。该选择协议优先控制小K过拟合和旧类遗忘，再比较old/new等权目标；任何query指标都不参与这一决策链。

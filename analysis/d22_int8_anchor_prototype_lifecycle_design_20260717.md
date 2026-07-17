# D22：int8锚点驱动的prototype生命周期设计

日期：2026-07-17
状态：`DESIGN_ONLY_SUPPORT_ONLY_QUERY_UNREACHABLE`
目标：用联合密封Phase1 int8域×类锚点完成Stage2-B旧类域适应和Stage2-C新类注册，同时保持逐样本、全注册类、无role/quota的部署决策。

## 1. 边界与设计选择

D22只消费两类信息：联合密封部署bundle中的不可变Phase1 int8聚合锚点，以及当前目标接收机的LEO_weak K-shot support。Phase2不得更新、替换或独立装载int8锚点组件；不得访问clean/source样本、样本级源特征、可逆源索引或独立prototype sidecar。

本设计不训练神经adapter。旧类prototype shrinkage、新类注册、半径估计和old/new竞争校准均为闭式或固定次数的稳健统计，训练参数为0、训练epoch为0。它优先满足<5k参数、≤256KB状态和星上快速注册；80k参数、30epoch只保留为总协议上限，不作为D22的默认预算。

任何query均不参与设计、拟合、support-fold选择、门禁或回退。本文件不读取也不引用既有query结果。

## 2. 联合密封int8锚点表示

### 2.1 全局类中心与稀疏域偏移

设Phase1旧类集合为\(O\)，嵌入维数为\(d=160\)。对旧类\(c\)，bundle只保存：

1. 全局类中心int8向量\(q_c^0\in\mathbb Z_8^d\)及FP16尺度\(\alpha_c\)；
2. 每个训练域\(h\)的稀疏int8偏移：索引\(I_{hc}\)、值\(q_{hc}^{\delta}\in\mathbb Z_8^m\)、FP16尺度\(\beta_{hc}\)；
3. 域×类半径\(r_{hc}\)和全局类半径\(r_c^0\)，均为FP16；
4. 每个量化块的尺度、饱和计数和内容SHA256。

当前优先取\(m=16\)。当\(d\le256\)时，偏移索引使用uint8。重构公式为

\[
g_c=\alpha_c q_c^0,
\qquad
\delta_{hc}=\operatorname{scatter}_{I_{hc}}(\beta_{hc}q_{hc}^{\delta}),
\qquad
a_{hc}=\operatorname{norm}(g_c+\delta_{hc}).
\]

偏移只保留绝对值最大的\(m\)个坐标；被截断残差的L2范数并入\(r_{hc}\)。这比为每个域×类重复保存160维中心更紧凑，同时保留域方向、域半径和量化不确定度。

全局中心计算、top-m坐标选择、截断误差统计和int8量化均在任何目标support打开前离线完成，并与final checkpoint联合密封。Phase2只解码，不重新压缩或选择偏移坐标。

每个锚点的量化方差上界记为

\[
e_{hc}^2=\frac{d\alpha_c^2+m\beta_{hc}^2}{12}
+e_{hc,\mathrm{trunc}}^2.
\]

Phase2可在RAM中解量化并计算最终目标prototype，但不得把更新结果写回Phase1锚点组件。

### 2.2 support条件域聚合

对旧类\(c\)的目标support稳健中心\(t_c\)，计算每个域锚点的标准化距离：

\[
d_{hc}=\frac{1-t_c^\top a_{hc}}
{r_{hc}^2+e_{hc}^2+\varepsilon},
\qquad
w_{hc}=\frac{\exp(-d_{hc})}{\sum_j\exp(-d_{jc})}.
\]

聚合锚点和锚点方差为

\[
a_c=\operatorname{norm}\left(\sum_hw_{hc}a_{hc}\right),
\]

\[
v_{a,c}=\sum_hw_{hc}
\left(r_{hc}^2+e_{hc}^2+\|a_{hc}-a_c\|_2^2\right).
\]

该聚合只使用已登记类的support，不读取query域或query角色。softmax温度固定为1，不进入选择网格。

## 3. support稳健统计

对类\(c\)的归一化support特征\(z_{ci}\)，K≥5时使用3次固定Weiszfeld-Huber更新：

\[
t_c^{(0)}=\operatorname{norm}\left(\frac1K\sum_i z_{ci}\right),
\]

\[
u_i^{(j)}=\min\left(1,
\frac{\kappa_c^{(j)}}{\|z_{ci}-t_c^{(j)}\|_2+\varepsilon}\right),
\qquad
t_c^{(j+1)}=\operatorname{norm}\left(\sum_i u_i^{(j)}z_{ci}\right),
\]

其中\(\kappa_c^{(j)}\)是当前距离中位数。最终目标方差为

\[
v_{t,c}=\operatorname{median}_i\|z_{ci}-t_c\|_2^2+e_t^2,
\]

\(e_t^2\)是target feature/int8状态的量化方差下限。K=1时令\(t_c=z_{c1}\)，不从单样本估计类内方差。

固定3次更新不是训练epoch，不产生可学习状态。每类support权重只用于当前闭式中心，不持久保存。

## 4. Stage2-B：旧类受限prototype shrinkage

### 4.1 后验shrinkage

对旧类\(c\)，定义候选尺度\(\eta\in\{0.5,1,2\}\)，target中心权重为

\[
\lambda_c(\eta)=
\frac{Kv_{a,c}}{Kv_{a,c}+\eta v_{t,c}}.
\]

该式对应“锚点不确定度越高、K越大，越信任目标support；target类内离散越大，越收缩回锚点”。Stage2-B prototype为

\[
p_c^B=\operatorname{norm}\left((1-\lambda_c)a_c+\lambda_ct_c\right).
\]

若锚点与target中心的失配超过联合不确定度

\[
\|a_c-t_c\|_2^2>4\left(v_{a,c}+\frac{v_{t,c}}K\right),
\]

则该类触发`ANCHOR_CONFLICT_TARGET_ONLY`，设置\(\lambda_c=1\)。这是类局部安全门，不依赖query，也不改变其他类的K或权重。

### 4.2 旧类半径

受限prototype的部署方差为

\[
\sigma_{c,B}^2=
(1-\lambda_c)^2v_{a,c}
+\lambda_c^2\frac{v_{t,c}}K
+\lambda_c(1-\lambda_c)\|a_c-t_c\|_2^2.
\]

保存半径\(\sigma_{c,B}\)而不是support样本。Stage2-B输出仅含每个旧类的int8 prototype、FP16尺度、FP16半径和生命周期标记`old_adapted`。

## 5. Stage2-C：稳健新类注册与遗忘保护

### 5.1 新类prototype

seen-new类没有Phase1类锚点。K≥5时直接注册稳健中心：

\[
p_n^C=t_n,
\qquad
\sigma_{n,C}^2=
\max\left(v_{t,n},
\operatorname{median}_{c\in O}\sigma_{c,B}^2,
e_t^2\right).
\]

K=1时注册\(p_n^C=z_{n1}\)，但不从单样本声称类内半径；半径取Stage2-B旧类target残差方差的中位数：

\[
\sigma_{n,C}^2=
\operatorname{median}_{c\in O}
\left(\frac1K\sum_i\|z_{ci}-p_c^B\|_2^2\right)+e_t^2.
\]

这使K1新类可注册，却不会用零半径制造过度自信。

### 5.2 旧类冻结

Stage2-C不得重拟合旧类prototype或半径：

\[
p_c^C=p_c^B,
\qquad
\sigma_{c,C}^2=\sigma_{c,B}^2,
\qquad c\in O.
\]

因此新类注册只能通过新增prototype和一个全局竞争校准量影响旧类决策。旧类状态字节级冻结；任何旧类prototype hash在B/C之间变化都使该行fail closed。

## 6. 逐样本全注册类评分

对任一输入特征\(z\)，为每个已注册类计算半径归一化能量：

\[
E_c(z)=
-\frac{\|z-p_c\|_2^2}{2\bar\sigma_c^2}
-\frac12\log\bar\sigma_c^2,
\qquad
\bar\sigma_c^2=\max(\sigma_c^2,e_c^2,\sigma_{min}^2).
\]

Stage2-C只允许一个support-derived全局new bias \(\beta\)：

\[
S_c(z)=
\begin{cases}
E_c(z),&c\in O,\\
E_c(z)-\beta,&c\in N.
\end{cases}
\qquad
\hat y(z)=\arg\max_{c\in O\cup N}S_c(z).
\]

每个query独立面对全部已注册类。predictor知道注册表中哪些列是旧类或新注册类，但不知道当前query的真实old/new角色；它不读取batch类别数、类别配额、query顺序或其他query预测。

## 7. old/new竞争的闭式校准

### 7.1 support OOF竞争区间

在class-balanced support fold中，对旧类验证样本定义new入侵需求：

\[
u_i=\max_{n\in N}E_n(z_i)-E_{y_i}(z_i),
\qquad y_i\in O.
\]

对新类验证样本定义保留new所允许的最大bias：

\[
v_i=E_{y_i}(z_i)-\max_{c\in O}E_c(z_i),
\qquad y_i\in N.
\]

固定候选\(\alpha\in\{0.10,0.20,0.30\}\)，计算

\[
L_\beta(\alpha)=Q_{1-\alpha}(\{u_i\}),
\qquad
U_\beta(\alpha)=Q_{\alpha}(\{v_i\}).
\]

若\(L_\beta\le U_\beta\)，取闭式中点

\[
\beta^*(\alpha)=\frac{L_\beta+U_\beta}{2}.
\]

若区间为空，标记`OLD_NEW_CALIBRATION_INTERVAL_EMPTY`；该\((\eta,\alpha)\)候选失败，不能用query挑选一个折中bias。\(\eta\times\alpha\)共9个闭式候选，不含梯度训练。

### 7.2 遗忘代理

在相同旧类OOF样本上，Stage2-B和Stage2-C的逐类准确率分别为\(a_{c,B}\)和\(a_{c,C}\)。support-only遗忘代理为

\[
F_{old}^{OOF}=\frac1{|O|}\sum_{c\in O}
\max(0,a_{c,B}-a_{c,C}).
\]

此外记录旧类true-margin下降：

\[
M_F=\frac1{|V_O|}\sum_{i\in V_O}
\operatorname{ReLU}(m_i^B-m_i^C).
\]

prototype冻结保证几何状态不忘；\(F_{old}^{OOF}\)与\(M_F\)约束新增类带来的决策竞争遗忘。

## 8. support-fold选择与K行为

### 8.1 开发seed统一锁

K≥5时使用5折class-balanced OOF。每类support按`SHA256(method_lock || scenario || class_handle || support_token)`排序后round-robin分折：K=5每折每类验证1个，K=10每折每类验证2个。prototype、半径、域权重和\(\beta\)都只能从其余4折support计算。

开发seed在三个LEO_weak场景上统一选择一个\((\eta,\alpha)\)。确认seed固定\(\eta\)和\(\alpha\)，只从该seed自身support闭式计算prototype、半径和\(\beta\)；不得重新扩大候选网格。

### 8.2 K=1/5/10

| K | Stage2-B旧类 | Stage2-C新类 | 校准与门禁 |
|---:|---|---|---|
| 1 | 使用固定\(\lambda=0.25\)的锚点主导shrinkage；锚点冲突时最多放宽到\(\lambda=0.5\) | 单support直接注册；半径借用旧类target残差中位数 | 无合法OOF，不从本cell估计\(\beta\)，固定\(\beta=0\)；标记`K1_UNCERTIFIED_APPLY_ONLY` |
| 5 | 5折，每类4 train/1 validation；使用K10开发锁的\(\eta,\alpha\) | 3次稳健中心和半径 | 只做安全复核；失败则回退target-only prototype，不能重选超参数 |
| 10 | 5折，每类8 train/2 validation；开发seed在9个闭式候选中锁\(\eta,\alpha\) | 3次稳健中心和半径 | 完整old/new等权、floor、CVaR和遗忘门禁 |

K1仍执行逐样本全注册类预测，但不把单样本自拟合结果称为support-fold证据。K5的安全复核只能触发保守回退，不得形成新的K5专用方法锁。

## 9. 选择目标与硬门禁

### 9.1 old/new等权统计

对OOF逐类准确率定义

\[
BA_O=\frac1{|O|}\sum_{c\in O}a_{c,C},
\qquad
BA_N=\frac1{|N|}\sum_{c\in N}a_{c,C},
\]

\[
J_{eq}=\frac{BA_O+BA_N}{2},
\qquad
H_{ON}=\frac{2BA_OBA_N}{BA_O+BA_N}.
\]

floor与最差25%类CVaR错误为

\[
f_O=\min_{c\in O}a_{c,C},
\quad
f_N=\min_{c\in N}a_{c,C},
\]

\[
R_G^{CVaR}=\frac1{m_G}
\sum_{c\in\operatorname{Worst}_{m_G}(G)}(1-a_{c,C}),
\qquad
m_G=\max(1,\lceil0.25|G|\rceil).
\]

### 9.2 相对target-only基线的门禁

基线`T0`使用同一support和同一能量评分，但不读取Phase1锚点：旧/新类均由target稳健中心注册，\(\beta=0\)。D22候选必须在每个LEO_weak场景分别满足：

1. Stage2-B old overall和old floor均不低于T0。
2. Stage2-C old floor、new floor均不低于T0。
3. old/new CVaR错误均不高于T0。
4. \(F_{old}^{OOF}\le F_{old,T0}^{OOF}\)，且\(M_F\le M_{F,T0}\)。
5. \(J_{eq}\)和\(H_{ON}\)均不低于T0。
6. 至少4/5 folds中，old和new balanced accuracy同时不低于T0。
7. B/C旧类prototype与半径hash完全一致。

任一场景失败即淘汰该\((\eta,\alpha)\)。通过门禁后按以下字典序统一选择：

\[
\max\left(
\min_s\min(f_{O,s},f_{N,s}),
-\max_s\max(R_{O,s}^{CVaR},R_{N,s}^{CVaR}),
\min_sH_{ON,s},
\min_sJ_{eq,s},
-\max_sF_{old,s}^{OOF},
-|\eta-1|,
-\alpha
\right).
\]

没有候选通过时，D22回退T0；不得选择“最接近通过”的失败候选。单个旧类触发锚点冲突时允许该类局部回退target-only，因为该规则在query打开前已由support确定。

## 10. 参数、MAC与状态估算

### 10.1 Phase1锚点bundle

设有效域×类锚点总数为\(M\)，旧类数为\(C_o\)。稀疏偏移payload上界为

\[
S_{anchor}^{sparse}
=C_o(d+4)+(M-C_o)(2m+4)\;\text{B}.
\]

其中全局中心每类包含160B int8、2B scale和2B radius；每个域偏移包含\(m\)B int8值、\(m\)B uint8索引、2B scale和2B radius。

当前\(d=160,m=16,C_o=6,M=84\)：

\[
S_{anchor}^{sparse}=6\times164+78\times36=3,792\;\text{B}.
\]

即使保留全维int8偏移，payload也只有

\[
S_{anchor}^{full}=M(d+4)=84\times164=13,776\;\text{B}.
\]

这些数字不含manifest、签名和SHA字符串的文本开销；部署二进制payload仍远低于256KB。

### 10.2 Phase2目标状态

Stage2-C保存11个int8 prototype、各自FP16尺度和半径、一个FP16 \(\beta\)及11B生命周期标记：

\[
S_{target}=11(160+2+2)+2+11=1,817\;\text{B}.
\]

稀疏Phase1锚点加After目标状态为5,609B；全维偏移保守上界为15,593B。D22训练参数为0、epoch为0，无optimizer状态、无dense query图。

### 10.3 推理与适配MAC

单prototype能量评分的主项是每类一次160维点积：

\[
M_{query}^{B}\approx C_od=960\;\text{MAC},
\qquad
M_{query}^{C}\approx(C_o+C_n)d=1,760\;\text{MAC}.
\]

每类半径和bias只增加常数级标量运算。相同11类、K10的single-qKNN需要\(11\times10\times160=17,600\)MAC，因此D22 prototype head的主分类MAC约为其1/10。

开发K10的9候选、5折OOF闭式计算保守上界为

\[
M_{select}\le A\left[FC Kd+NCd\right],
\]

其中\(A=9,F=5,C=11,N=110,d=160\)，得到2,534,400MAC/scenario。确认seed不搜索9候选，只应用锁定\((\eta,\alpha)\)，计算量进一步下降。

## 11. 状态机与证据

D22生命周期固定为：

\[
\texttt{SEALED\_PHASE1\_ANCHORS}
\rightarrow\texttt{B\_OLD\_ADAPTED}
\rightarrow\texttt{C\_NEW\_REGISTERED}
\rightarrow\texttt{IMMUTABLE\_PREDICTOR\_STATE}.
\]

实现必须保存：

- 联合密封checkpoint+int8锚点bundle的整体SHA256、成员allowlist和pre-open验证；
- 全局中心、稀疏域偏移、半径、量化尺度、截断残差界和饱和计数；
- support token/class/scenario清单、5折分配、9候选逐折指标和每项门禁原因；
- 每类\(a_c,t_c,v_{a,c},v_{t,c},\lambda_c,p_c^B,p_c^C,\sigma_c\)及锚点冲突状态；
- \(L_\beta,U_\beta,\beta\)、old/new OOF逐类结果、CVaR、floor和遗忘代理；
- B/C旧类状态hash一致性、最终int8 target prototype状态和资源审计；
- `phase2_query_decision_policy=per_sample_all_registered_classes`；
- `query_access=false`、`query_fit=false`、`query_truth_opened=false`；
- `phase2_query_role_oracle_access=false`、`phase2_query_true_batch_class_count_access=false`、`phase2_query_class_quota_access=false`、`phase2_query_batch_global_assignment=false`。

scorer只能在不可变prediction落盘后连接query标签；scorer输出不得反馈prototype、半径、\(\beta\)、候选选择或回退。

## 12. 结论

D22把Phase1知识限制为不可变int8全局类中心、稀疏域偏移、半径和量化尺度。Stage2-B用目标support可靠度控制旧类anchor shrinkage；Stage2-C冻结旧类几何状态，稳健注册新类，并用support OOF闭式区间校准old/new竞争。当前配置为0训练参数、0epoch、After目标状态1,817B、联合状态约5.6KB、1,760MAC/query。若support-fold无法同时守住old/new floor、CVaR和遗忘代理，方法原子回退target-only prototype，不借助query补救。

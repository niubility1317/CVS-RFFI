# D118轻型快速域适应：可识别性边界与GN-ISF条件研发路线

状态：`THEORY_RESEARCH_COMPLETE / NO_IMMEDIATE_STAGE2_DA_CANDIDATE / GN_ISF_PHASE1_RESEARCH_ONLY / IMPLEMENTATION_NOT_STARTED / NO_NEW_PERFORMANCE_RESULT`

## 1.主裁决

当前`p2_min_v1`下，可立即实现并发布真实Stage2 G0的非恒等`target-domain state`候选数量为零。`θ=0`只是无适配基线，不是DA候选，也不满足G0三种K均产生prediction变化的功能要求。这不是因为缺少数学变换，而是现有合法观测只有共同封存的Phase1聚合知识与当前row有限support；它不能把共同receiver/domain状态、类×域交互、每个物理样本独立信道和K1采样噪声唯一分离。

两名独立Terra Max研究agent分别从正向条件定理和反向观测等价世界审查，结论一致：

1.公共平移、球面运动、协方差、带宽、view可靠度、raw-IQ逆滤和末端低秩metric已被D93/D94、C-id、D110、D112、D113、D114、D116、D117或SCXMAP覆盖并证伪/判为不可识别；
2.直接重启GRB-JP4或把同一support-ground残差改写为GroupNorm/FiLM系数，没有增加新observable，不能作为新的K1可识别DA；
3.在“不新增target observable、只研发新Phase1干预资产”的当前分叉中，唯一保留路线是`GN-ISF`：在真正早期非线性层预先学习≤2维GroupNorm/FiLM干预字典，并先证明它在receiver-held、class/TX-LOCO条件下恢复跨类共享变化、不过拟合TX且不等价于GRB-JP4/D102/末端PSD metric；
4.在该Phase1证伪通过前，GN-ISF只是条件研发路线，不是可发布Stage2 revision，更不能启动Target25/125。

这个裁决直接贯彻“性能弱就研发下一个方法，不浪费实验资源”：本轮不为旧信息通道重新建runner，不盲跑588/63/125，也不把闭式解、feature变化或参数非零冒充有效域适应。

## 2.问题形式化

### 2.1合法观测与状态

当前row中六个旧类support记为

\[
\mathcal S_O=\{(x_{ck},c):c\in\mathcal Y_{old},k=1,\ldots,K\}.
\]

Phase2公共状态只能写成

\[
\widehat\theta=A(\mathcal B,\mathcal S_O),
\]

其中`B`是与checkpoint共同封存的INT8 Phase1多样本聚合知识。`A`不得读取clean/source、query、query truth、old/new角色、真实类计数、class quota、跨query统计或全局重排状态。new support只注册新类；append new后旧公共状态必须逐字节不变。每条query只读自身固定received-IQ和query前冻结状态。

### 2.2无结构假设时的不可能性

设冻结表示为`φ`，希望从support恢复共同target状态`θ*`：

\[
z_{ck}=F_{\theta_*}(\mu_c)+\eta_{ck},\qquad z_{ck}=\phi(x_{ck}).
\]

若残差`η_ck`不受Phase1预先约束，则对任意`θ'`都可定义

\[
\eta'_{ck}=z_{ck}-F_{\theta'}(\mu_c),
\]

使`(θ',η')`产生完全相同的合法support。闭式方程有唯一数值解不等于target观测识别了receiver状态；唯一性可能完全来自prior。

更直接的两世界反例是：六类K1残差都等于`a`。世界A把它解释为共同receiver状态`θ=a`和零样本噪声；世界B把它解释为`θ=0`以及六个物理样本恰好各有噪声`a`。两世界向算法提供相同的support与sealed bundle，但未来query分别需要“统一补偿`a`”和“保持identity”。任何只读当前合法输入的算法都无法同时正确。

因此，轻型DA必须先限制模型族，而且限制必须来自target访问前的Phase1 receiver-held证据，不得由Target性能事后选择。

### 2.3局部可识别的最低条件

对预先固定的低维干预族`F_θ,θ∈R^r`，令

\[
J_{ck}=\left.\frac{\partial F_\theta(x_{ck})}{\partial\theta}\right|_{\theta=0},
\quad
\mathcal I_{\mathcal S}=\frac1{|\mathcal Y_{old}|}
\sum_c\frac1K\sum_kJ_{ck}^{\mathsf T}W_cJ_{ck}.
\]

必要但不充分的最低条件为

\[
\operatorname{rank}(\mathcal I_{\mathcal S})=r,
\quad E[\eta_{ck}\mid c]=0,
\quad \theta_*\text{跨类共享且不与TX/类身份混杂}.
\]

`Λ+I_S≻0`只说明MAP可解；必须单独报告data information是否满秩、最小特征值以及prior/data信息比。K1的六个不同物理support可贡献六组观测，但同一IQ的时间片、FFT bin、deterministic view或JVP坐标都不增加K。

进入任何设计冻结前还必须同时固定三项模型条件：Phase1学到的噪声与类交互分布能迁移到未见target receiver；`p(z|c,θ)`对`θ`单射且不能通过随`θ`重定义残差保持观测等价；GN/FiLM的函数规范已固定，排除后续层缩放、偏置补偿或通道置换形成的gauge。即使source receiver-held通过且`rank(I_S)=r`，缺少这三项仍只能把`â`称为support-conditioned参数，不能称为receiver/domain state。

## 3.为什么现有轻型路线不再重跑

### 3.1共同变换的qKNN边界

若support和query共同施加正交变换`Q`，则

\[
\|Qq-Qs\|_2=\|q-s\|_2,
\]

qKNN的邻居、距离和预测逐值不变。若施加非等距线性变换`T`，则

\[
\|T(q-s)\|_2^2=(q-s)^\mathsf T(T^\mathsf TT)(q-s),
\]

它等价于pullback PSD metric，属于D93/D94、D110或C-id已经覆盖的家族。只换rank、收缩或层名不构成新机制。

### 3.2现有路线去重表

|路线|理论问题|现有真实证据|裁决|
|---|---|---|---|
|D93/D94 post-feature低秩transport|六类张成空间外不受约束；非等距即PSD metric|D93 K1`B/A/Min/N/H/F=55.56/33.33/8.33/28.17/30.53/22.22`，D94收缩仍负|永久关闭|
|C-id/support covariance|K1无类内自由度|K1严格identity|不绕开边界|
|SCXMAP`z_dom→z_id`|`z_dom`含TX泄漏风险；连续变化不保证纠错|48/54状态非零，K5/K10 argmax全0；K1 wrong→correct`0`、correct→wrong`7`|永久关闭|
|D110共享PSD/Fisher metric|source nuisance相关性不等于target预测误差|K1注册H`-2.7957pp`、old floor`-10.0529pp`|永久关闭|
|D112 SEAM|共享球面运动未产生独立决策作用|63行`M_JOINT_SEAM=M_HEAD_GROUND`逐预测相同|关闭motion；保留head|
|D113公共加性校正|共同shift是假设，不是额外观测|真实G0 K1/K5/K10 argmax`0/1/0`|永久关闭|
|D114预测带宽|只是head浓度重参数化，不恢复domain state|K1 DA-at-base H`-6.1944pp`、floor`-19.1358pp`|永久关闭|
|D116 raw-IQ频响|receiver、channel、TX谱、CFO、相噪存在gauge自由度|独立审查`P0=4`|没有pilot/配对记录时关闭|
|D117 view可靠度|same-IQ view确定性相关，无法分离样本难度与view噪声|独立审查P0拒绝|关闭|
|GRB-JP4/r1-sealed|support-ground残差→4维网络干预；K1为identity|只有真实checkpoint无query可行性spike，无性能结果|已关闭，不得换名重入|
|GRB-JP4-CFM/r2-sharedK1|同一信息通道叠加CFM、ground多原型与D92|活动报告为`LOCAL_FEASIBILITY_REJECTED/NO_PERFORMANCE_RESULT`：state、MAC、fold与执行面未闭合；后存设计勘误又标`NOT_IMPLEMENTED/NOT_RELEASED`|以活动报告的已关闭、无性能结果为主，保留文档状态冲突|
|D102/MetaBias4|低维pre-ReLU干预，但RCN/`z_dom`有TX泄漏且非formal|均值收益仅`0.0358–0.0591pp`，9/42 LOCO fold为负|不得复用资产冒充GN-ISF|

## 4.唯一条件研发候选：GN-ISF

### 4.1机制

`GN-ISF`全名为`GroupNorm Intervention State Fitter`。它不从现有GroupNorm参数直接拟合，也不使用`z_dom`；它要求Phase1先在一个固定的早期非线性位置学习极低维干预方向。对固定层`l`：

\[
\gamma_l(a)=\gamma_{l,0}\odot(1+G_la),
\qquad
\beta_l(a)=\beta_{l,0}+H_la,
\qquad a\in\mathbb R^r,r\le2.
\]

令`f_a(x)`为施加该干预后的最终单位特征。Phase1共同封存`{G_l,H_l,μ_c,W_c,Λ}`。Phase2只用六个old类support解

\[
\widehat a=-\left(\Lambda+\mathcal I_{\mathcal S}\right)^{-1}
\frac1{|\mathcal Y_{old}|}\sum_c\frac1K\sum_k
J_{ck}^{\mathsf T}W_c\left[f_0(x_{ck})-\mu_c\right].
\]

随后固定`â`，用同一个模型重新forward old/new support及每条独立query。K1/K5/K10使用同一公式；不设K专属rank、步长、层选择或回退分支。

### 4.2它何时才不同于旧路线

只有同时满足以下条件，GN-ISF才是实质新候选：

1.干预位于足够早的非线性层，样本依赖Jacobian会改变后续激活路径；
2.其功能作用不能写成固定末端`T^T T`metric；
3.`J_GN(x)`不是旧GRB-JP4`joint_proj.0`Jacobian的固定线性重参数化；
4.方向资产不读取`z_dom`/RCN，不携带可预测TX身份的信息；
5.receiver-held与class/TX-LOCO共同表明，六类support拟合的同一状态能预测未参与拟合的类和新物理样本。

若不满足第1—3项，它只是换层的GRB/PSD；若不满足第4—5项，它只是support条件化分类器参数，不能记为DA。

### 4.3Phase1训练目标

Phase1只在source receiver/day上学习干预字典，不接触Target。对每个receiver-held episode：

- 用六个旧类的support子集闭式拟合`a`；
- 在物理ID互斥的held query上测量该状态是否降低类中心残差并增加正确决策；
- class/TX-LOCO时移除整个发射机类，检查同一`a`是否预测被留类变化；
- 在同一held split上附加TX身份可预测性probe，不再另跑一套阻塞矩阵；
- 与随机等能方向、旧GRB Jacobian span和最佳固定末端PSD进行同预算反证。

方向学习的目标不是最大化某个已开封Target指标，而是最大化source receiver-held的跨类可预测receiver变化，同时惩罚TX可预测性、class交互残差和prior主导：

\[
L=L_{held\_state}+\alpha L_{class\_loco}+\beta L_{receiver\_held}
+\gamma L_{tx\_leak}+\delta L_{equiv}.
\]

这些系数必须只由Phase1内部nested split固定；不得从Stage2 G0/G1/Target调整。

### 4.4轻型与快速边界

令选定GN层通道数为`C`、`r≤2`。数值载荷主项为`G,H∈R^{C×r}`：INT8方向约`2Cr`字节，另加逐方向scale、`2×2`先验与六类聚合锚。Phase2求解只涉及一个`r×r`系统；query无需optimizer、反向传播或跨query状态，但每条support/query需要一次带固定GN调制的正常forward。

具体`C`、字节、JVP次数和forward开销必须从实际checkpoint选层后精确报告；当前没有formal层选择和资产，不能虚构性能或资源数字。若JVP/full-forward成本超过现有轻型预算，应在Phase1候选阶段关闭，不通过降rank或换层扫描补救。

### 4.5当前资产实盘

本地资产实盘确认，当前Git快照和根目录历史报告中没有可直接部署的GN-ISF、GRB-JP4或D102正式Phase2资产，但根目录历史报告确实保留D102诊断payload：

- 未发现`phase1_grb_jp4_compact_component_v1.npz`、formal deployment manifest或GN-ISF资产；
- `code/cvsrffi/phase1_grb_jp4_bundle.py`只定义预期文件与`PENDING_OUTER_JOINT_SEAL/formal_phase2_eligible=false`状态，不能替代资产本体；
- `docs/D102_RB_METABIAS4_PHASE1_ANALYTIC_HELD_LOCK.json`为`PHASE1_ANALYTIC_INITIALIZER_HELD_DIAGNOSTIC_NON_PROMOTABLE`且`target25_authorized=false`；
- 根目录历史报告`automation_reports/CV-SincNet/d102_rb_metabias4_phase1held_target25_20260724/retrieved_d102_analytic_held_r6/output/`实际含`phase1_rb_metabias4_bundle.npz`、manifest、seal hash以及`phase1_jp4_tap_archive.npz`、manifest；bundle状态为`PHASE1_HELD_ASSET_PENDING_OUTER_JOINT_SEAL/formal_phase2_eligible=false`，tap状态为`DEVELOPMENT_ONLY_NOT_FORMAL/bundle_created=false/target25_release_authorized=false`；
- 当前快照内`automation_reports/CV-SincNet/d105_phase1_sourceheld_asset_20260731_r1/offline_controller/d105_d102_revocation_manifest.json`把同一D102 bundle/tap hash登记为`PHASE1_HELD_FALSIFIER_REJECT`；
- 实际存在的`d99_ground_bundle_dev.npz`状态为`PREREGISTERED_DEVELOPMENT_GROUND_AGGREGATE_NONFORMAL/formal_phase1_eligible=false`，只能作开发诊断，不能冒充GN-ISF sealed asset。

因此，下一步若继续GN-ISF，必须研发新的Phase1资产，而不是从当前目录挑一个旧NPZ改名接入。

## 5.最小证伪设计，不扩建实验gate

### 5.1Phase1科学证伪

这不是额外发布工程gate，而是判断该方法是否成立的科学实验。只运行一个冻结候选：一个固定早期GN层、`r≤2`、一个Phase1训练配方。必要输出为：

|检查|必须回答的问题|失败裁决|
|---|---|---|
|receiver-held K1/K5/K10|六类拟合状态能否预测未见物理样本|不能则关闭|
|class/TX-LOCO＋同split泄漏probe|共同状态能否跨类预测，且不主要编码TX/CFO/类身份|不能或泄漏则关闭|
|information audit|`rank(I_S)=r`且data information不被prior淹没|不满足则关闭|
|非等价证书|是否区别于GRB-JP4 Jacobian和固定末端PSD|等价则关闭|

只要其中一项失败，就不实现Stage2 runner、不调rank/层/正则、不跑Target。

### 5.2通过后才进入目标规定的最短路径

Phase1证伪通过后，严格回到目标文档顺序：

1.一次核心实现波次；
2.协议负测（含append new后旧公共状态逐字节不变）＋真实checkpoint无query smoke＋独立`P0=0/P1=0`；
3.真实588条K1/K5/K10 G0，仅检查feature/neighbor/margin/argmax/资源；
4.任一K argmax变化为0，立即`REJECT_REVISION_NO_FUNCTION`；
5.三K均非零，立即运行冻结`M0/M_DA/M_HEAD/M_JOINT`一次63行source-held G1；
6.`DA_AT_BASE`和`DA_AT_HEAD`均无独立正收益，永久关闭；
7.方向正确才运行单seed Target25，不先跑125、不补seed。

不新增authority、重复数据验证、通用runner、D92 scorer、125 executor或报告平台。数据`capsule_id/split_id/p2_min_v1`匹配且`VALIDATED_ONCE`时不重验。

## 6.当前真实性能账本

### 6.1完整历史基线

|方法|证据规模|before|old after|old floor|seen-new|H|forgetting|裁决|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|D62|完整125|81.51%|64.39%|35.15%|59.11%|61.09%|17.11pp|完整基线，未达目标|
|D92|完整125|81.55%|65.56%|36.81%|58.93%|61.57%|15.99pp|改善old/floor/forgetting，new略退，未达目标|
|SVRN-qKNN-BCRR|完整125|73.10%|43.03%|11.21%|23.46%|29.25%|30.07pp|全面弱于D62，关闭|
|D91|仅15个固定dev row|—|—|—|—|—|—|15/15预测与D62逐值一致，不能冒充125|

### 6.2最近因果证据

|方法/臂|证据|old BA变化|seen-new变化|H变化|old floor变化|裁决|
|---|---|---:|---:|---:|---:|---|
|D112`M_HEAD_GROUND`|K1登记42行source-held|+1.3228pp|+1.3228pp|+1.9736pp|+4.5855pp|最近K1四臂/登记证据中唯一明确正收益head组件；不是DA|
|D112 SEAM motion|完整63行|0|0|0|0|与head逐预测相同，关闭|
|D114`M_DA−M0`|K1登记42行|-3.7037pp|-3.7037pp|-6.1944pp|-19.1358pp|明确负收益，关闭|
|D114`M_JOINT−M_HEAD`|K1登记42行|-2.5573pp|-2.5573pp|-4.4375pp|-15.6966pp|明确负收益，关闭|
|GN-ISF|尚无formal资产/性能|—|—|—|—|`NO_PERFORMANCE_RESULT`|

严格回答“有正收益版本吗”：有，D112`M_HEAD_GROUND`在source-held K1登记上有明确正收益；BCRR仍保留K5的OTHER正信号，但二者都不是DA成功，也都没有证明Target25模型DA达标。旧GRB-JP4既不能称为性能失败，也不能称为成功，因为它没有性能结果；本轮拒绝的是把它换名成新的可识别DA，而不是伪造负性能。

## 7.Stage2-A与真正的新observable

独立Stage2-A无标签target reference是最小新增观测之一，但无标签边缘分布

\[
P_U(z)=\sum_y\pi_yP(z\mid y,\theta)
\]

仍会把类别混合比例`π_y`变化与domain状态`θ`混淆。它只有在物理ID与support/query互斥、类混合受控或统计量已证明类无关时，才能帮助识别。

而且按`项目.md`当前最严格边界，Stage2-B/C predictor state只能由Phase1 bundle与注册support决定。若要把Stage2-A reference写入Stage2-C状态，必须先显式修改跨阶段状态许可、物理ID闭包和immutable handoff语义；这属于科学协议范围变更，不能由本轮擅自执行。

另一个真正可识别的方向是新的Phase1 paired calibration：同一受控TX/pilot跨receiver配对，或已知接收机响应。它能消除`R_tH_i=(R_tG)(G^{-1}H_i)`的receiver/channel规范自由度。当前随机物理IQ没有这类观测。

## 8.工作流分工

|任务|模型/agent|边界|
|---|---|---|
|协议解释、可识别性、候选整合、最终晋级|主agent`gpt-5.6-sol/high`|保留最终判断|
|DA机制与Phase1干预字典|独立`gpt-5.6-terra/max`|作者不得自证|
|反例、P0/P1、等价性审查|独立`gpt-5.6-terra/max`|与作者分离|
|复杂核心实现|`gpt-5.6-terra/max`|文件所有权独立|
|数据、同row因果效应、paired CI和结果解释|`gpt-5.6-sol/high`|不交给机械agent|
|文件查找、状态字段、数量、hash、固定测试、表格整理|`luna_worker`|只做明确、可复核、可恢复任务；遇歧义立即升级|
|N607唯一runner|`gpt-5.6-terra/max`|只落地、监控和回收，不改方法|

主agent只把具体、独立、格式固定的任务交给Luna；不会把方法选择、协议冲突、性能解释、P0/P1或实验晋级交给低判断agent。

## 9.参考依据

1.Ben-David S,Blitzer J,Crammer K,Kulesza A,Pereira F,Vaughan JW.A theory of learning from different domains.Machine Learning,2010.
2.Ben-David S,Lu T,Luu T,Pál D.Impossibility theorems for domain adaptation.AISTATS,2010.
3.Fernando B,Habrard A,Sebban M,Tuytelaars T.Unsupervised visual domain adaptation using subspace alignment.ICCV,2013.
4.Sun B,Saenko K.Deep CORAL:correlation alignment for deep domain adaptation.ECCV Workshops,2016.

这些工作支持“域适应需要受限结构且边缘对齐本身不足”的一般边界；本文对K1、receiver/channel、query零更新和各D编号路线的结论来自本项目协议与真实artifact，不能把通用论文结论直接当作本项目性能证据。

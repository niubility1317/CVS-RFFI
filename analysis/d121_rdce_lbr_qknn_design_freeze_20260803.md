# D121-RDCE＋LBR-qKNN轻型域适应与局部竞争头设计冻结

状态：`DESIGN_FROZEN / IMPLEMENTATION_AUTHORIZED / NO_NEW_PERFORMANCE_RESULT`

日期：2026-08-03

## 0.结论

D121冻结为一个最小2×2因果实验：域适应端逐字复用D106已实现的`RDCE-r3-SCATTER02`，分类端新增无参数`LBR-qKNN`（Local Binary Rival qKNN）。本轮不再发明不可辨识的target域变换，也不把D106旧RCMR头带入新联合臂。

D121不是D106重跑。D106完整G1显示，RDCE相对M0的域适应平均主效应为：old BA`+0.2604pp`、seen-new`+0.3632pp`、H`+0.4447pp`、old floor`+0.2824pp`；但RDCE与RCMR旧头联合后old floor为`-0.2707pp`、all floor为`-0.5397pp`。这些是D106历史真实性能证据，只用于选择D121因果因素，不是D121新结果。D121要回答的是：保留已有正向RDCE，用新的局部二元竞争头能否避免RCMR的全局floor损失。

## 1.为什么不再研发新的D121域适应状态

正式`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`中的残余CFO、相位噪声、多径、AGC残差和AWGN均按物理记录独立采样；Phase2没有pilot、信道遥测、独立CFO truth或tap元数据。K1只给出每类一个独立物理样本，因而不能从support唯一识别160维任意target变换。盲两tap逆滤波、归一化自相关或无状态CFO/shift视图也分别与D116、SGC和`rx_light5`已有路线重叠。

因此本轮拒绝用不可识别的隐变量包装“新域适应”。复用D106冻结RDCE是最小、可审计且已有正向平均主效应的选择；新科学变量只放在分类头。若未来取得协议允许的pilot、校准遥测或Phase1成对校准聚合，再重开新的域适应路线。

## 2.LBR-qKNN数学定义

对当前row的全部注册类，support bank含`N=C×K`个support。令`y_i`为support`i`的注册类标签，`x_i`为其冻结表示。先只用support建立一次有向异类竞争图：

\[
n_i=\arg\min_{j:y_j\ne y_i}d(x_i,x_j).
\]

`d`逐字复用M0冻结度量。`n_i`只在support注册或D121联合臂完成RDCE变换后重建一次；query不得改变它。

对独立query`q`，先一次性计算M0对所有support的Student-t核logit`\ell_i(q)`，再做单跳局部竞争修正：

\[
\widetilde\ell_i(q)=\ell_i(q)+\log\sigma\!\left(\ell_i(q)-\ell_{n_i}(q)\right).
\]

数值实现必须使用稳定且非递归的等价式：

\[
\widetilde\ell_i(q)=\ell_i(q)-\log\!\left(1+\exp(\ell_{n_i}(q)-\ell_i(q))\right).
\]

所有`\ell_i`必须先由原始M0核一次性算完；不得把`\widetilde\ell_{n_i}`递归读回。最终类分数为：

\[
S_c(q)=\operatorname{LSE}_{i:y_i=c}\widetilde\ell_i(q)-\log K.
\]

同一公式同时用于旧类和新注册类。它没有温度、阈值、rank、混合权重、epoch或按类参数；K1仍有定义，K>1保持每类单位support质量。

## 3.确定性、类置换与失败语义

最近异类support若距离完全相等，先按不可变physical ID的SHA256排序，再按support内容SHA256排序；不得用类ID、registry位置、old/new角色或query结果破平局。若仍不能唯一确定，显式抛出`RIVAL_TIE_UNRESOLVED`，不得静默选择首项。

最终不同类的最大分数若bitwise相等，也不得用类ID或registry位置破平局；应抛出`CLASS_SCORE_TIE_UNRESOLVED`。类标签任意置换会同步置换support分组和输出列，但不会改变异类图的几何或预测语义。

多个support可以指向同一rival。这只是固定support图上的边界再加权，不生成额外样本、不增加K，也不形成跨query状态。

## 4.四臂与状态生命周期

|臂|域适应|分类头|用途|
|---|---|---|---|
|`M0`|无|原始qKNN|共同基线|
|`M_DA`|冻结D106 RDCE|原始qKNN|复核RDCE主效应|
|`M_HEAD`|无|LBR-qKNN|估计新头主效应|
|`M_JOINT`|冻结D106 RDCE|在RDCE后重建LBR图|检验组合交互|

支持新类注册时，原始bank加入合法K-shot support；`M_HEAD`重建LBR图，`M_JOINT`先用同一冻结RDCE变换全部注册support，再重建LBR图。RDCE的basis、scatter和衰减规则不因LBR、新类或query而重估。

query只读冻结bank、固定`n_i`和M0核配置；`query_rows_used_for_fit=0`、`query_state_updates=0`、`query_selection_count=0`。禁止读取clean/source运行时样本、query truth、old/new query角色、真实batch类计数、class quota或跨query全局分配。

## 5.与既有方法去重

- 不同于D106-RCMR：LBR不混合类级多view分数，不使用全局head权重；它对每个support只读取一个固定最近异类rival。
- 不同于SVRN-qKNN-BCRR：LBR没有可训练或累计的binary correction、reliability状态、阈值或回归残差。
- 不同于D14类rival路线：LBR是单support、单跳、固定异类边和log-sigmoid margin，不做类级pair bias或跨query更新。
- 不同于D112 ground head：LBR不引入ground anchor或第二专家；D112正收益仅作为历史参考，不在D121内融合。
- 2026-08-03刷新1285条项目会话索引后，未发现`l_i+log sigmoid(l_i-l_n_i)`、`sigmoid`或`l_n_i`的精确公式记录；仍承认可能存在不同符号的等价历史，因此实现报告须继续保留上述机制级去重边界。

## 6.资源上界

每个support只新增一个rival索引。Target25最大`N=260`时用`uint16`共`520B`。注册阶段的朴素异类距离上界为：

\[
N(N-1)\times160=260\times259\times160=10{,}774{,}400\text{ MAC}.
\]

query阶段复用M0已经需要的support核logit，只增加每support一个稳定`logaddexp`和类内LSE，复杂度`O(N)`；无反向传播、GPU常驻优化器、query batch拟合或额外Phase1资产。

## 7.最小验证与晋级规则

### 7.1本地硬门

仅要求：

1.新LBR核心及冻结四臂入口；
2.聚焦测试覆盖异类约束、确定性tie、标签置换、K1/K5/K10、非递归稳定公式和query零fit/零update/零selection；
3.真实checkpoint/真实archive的无query训练smoke；
4.独立审查`P0=0 / P1=0`；
5.不可覆盖run ID、本地Git提交和N607只读预检。

不重复Phase2数据验证，不新增authority/signature层，不搭建通用125平台，不以文档润色阻挡实验。

### 7.2G0：588查询无truth功能证伪

固定K1/K5/K10，每个K都在相同588个query上比较M0与`M_HEAD`，至少记录rival索引root、support核logit变化、LBR margin变化和argmax变化。G0禁止打开truth和任何性能字段。

若任一K的`argmax_changed_count=0`，立即关闭D121当前revision，不进G1、不修参数；仅当三个K的argmax变化均非零，才允许进入G1。

### 7.3G1：最小四臂source-held

G1只跑固定四臂，不扩展到125矩阵。晋级要求：

- `M_HEAD`与`M_JOINT`均不得降低old/new净正确数；
- old floor与all floor均不得降低；
- K1总正确数必须严格增加；
- 所有比较必须同row、同query、同truth-side scorer闭合。

任一条件不满足即判定该revision性能弱，保留完整负结果并转入下一个原理候选，不启动调参矩阵。

## 8.独立审查收据

独立Terra Max审查结论：`MERGE / P0=0 / P1=0`。审查特别要求：tie不得用类ID；基础`ell_i`一次性计算、禁止递归；`n_i`固定于support图；共同rival不增加K；局部过抑制属于G1科学风险而不是发布前工程gate。上述要求已全部写入冻结设计。

当前结论只授权最小实现与G0，不构成D121性能收益、Target25达标或正式矩阵晋级声明。

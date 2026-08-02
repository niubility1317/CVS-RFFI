# D123 LOO-CRES轻型可靠度适配设计冻结

## 1.裁决与定位

- 候选：`D123-LOO-CRES`（Leave-One-Out Cross-Class Residual Excess Shrinkage，留一跨类残差超额收缩）。
- 状态：`DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED`。
- 唯一目标：在D112静态ground单位质量head上，用当前receiver的合法old-class support降低不可靠Phase1 ground anchor的权重；不学习表示、不恢复receiver因果状态、不移动anchor。
- 证据动机：D112静态head为正；D122中`HEAD_AT_DA`均值为正，但K1 old/all floor分别下降`0.6918/1.6949pp`且18-2、19-2为负尾。D123只研究“更保守地信任ground”能否修复该尾部。
- 本冻结不构成性能结论；若一次固定G1无独立正收益，永久关闭，不调公式、不补seed、不运行125。

## 2.合法输入与old集合

冻结old集合为

\[
O=\operatorname{tuple}(\text{immutable Phase1 bundle.class\_registry})。
\]

实现必须验证`O`恰有六个唯一opaque handle且`O\subseteq bank.classes`。`O`只来自与checkpoint共同封存的Phase1 bundle，不读取`held_class`、query truth、评分角色、TX/receiver身份、类配额或batch类计数。source-held评分中的`held_class`仍保留在`O`中；实际Phase2 append的新注册类不在bundle registry，因此不会成为donor或获得ground head。新增类不得改变`O`。

证据边界：现有source-held runner中的`held_class`只是truth-side生命周期proxy，预测时该类仍拥有Phase1 ground asset；因此63行G1只检验D123在source-held旧类上的组件效应，不是实际`Y_new`无ground注册性能。非old列逐bit边界由显式`O+new`测试和无truth smoke验证，但真实Stage2-C新类收益仍须由合法Target矩阵另行确认。

## 3.冻结公式

identity臂与RDCE臂必须在各自坐标独立计算以下量，不得跨臂共享。对每个`c∈O`：

\[
s_c=\operatorname{Norm2}\left(\sum_{k=1}^{K}x_{ck}\right),\qquad
e_c=\|s_c-g_c\|_2^2/160。
\]

`v_{s,c}`、`v_{g,c}`和`g_c`逐字继承D112；RDCE臂逐字继承D122的support/ground Jacobian标量输运和`T(s_c)`中心语义。令合法donor集合`D_c=O\setminus\{c\}`，冻结

\[
\delta_{-c}=\max\left(0,\operatorname{median}_{j\in D_c}
[e_j-v_{s,j}-v_{g,j}]\right),
\]

\[
\rho_c^{CRES}=\frac{v_{s,c}}
{v_{s,c}+v_{g,c}+e_c+\delta_{-c}}。
\]

不新增`epsilon`、倍率、阈值、温度、rank或K专属参数。六个old类时每类固定五个donor。合法donor必须通过相应identity/RDCE坐标中D112/D122已有的有效性检查，且`s_j`、`e_j`、`v_{s,j}`、`v_{g,j}`全部有限；少于三个时，该类必须直接复用原D112/D122冻结的`rho`和打分路径，不通过重新计算`delta=0`近似fallback，也不得根据表现选择fallback。非old列的logit必须逐bit沿用对应`M0/M_DA`。

单位质量head保持不变：

\[
L_c(q)=\operatorname{logaddexp}
\left(\log(1-\rho_c)+L_c^{sup}(q),
\log(\rho_c)+\ell(q,g_c)\right)。
\]

## 4.理论边界

- `delta>=0`，所以`0<=rho_CRES<=rho_D112/D122<1`；D123不会比原head更强地信任ground。
- donor残差与方差均为每维chord-MSE，`delta`解释为跨old类共享的超额support-ground失配proxy。
- 五个类别不是receiver效应的独立重复；该proxy混合receiver偏移、类域交互、ground误差和K-shot噪声，并依赖old类近似可交换性。因此不得宣称识别receiver state或提供因果域校正。
- `rho`上界只保证连续收缩，不保证准确率、H或floor单调改善。
- 非old列逐bit不变不等于new准确率必然不变；old列变化仍会改变全类竞争。

## 5.资源冻结

- enrollment新增：六个160维残差、至多六次五元median和六个标量`delta`；无训练、无反传、无模型更新。
- 相对D112/D122新增query MAC为0；绝对head仍至多六个160维anchor核，约960MAC/query。
- query-dependent state为0；每个query独立在全部registered classes上竞争。

## 6.最小因果矩阵与停止规则

唯一G1矩阵仍为冻结63行四臂：`M0`、`M_DA=RDCE`、`M_HEAD=identity+CRES`、`M_JOINT=RDCE+CRES`。先封存全部prediction，再由独立scorer打开truth。报告`DA_AT_BASE`、`HEAD_AT_ID`、`HEAD_AT_DA`、factorial interaction、K分层、receiver负尾、old/new/H、old/all floor和同row正确数。

性能前只允许一次21个真实包无truth功能审计。出现任一项即不发布G1：读取role/truth/held_class；`O`随评分角色变化；非old列不逐bit等于基线；`rho_CRES>rho_D112/D122`；三个K的`delta`、`rho`和最终logit均与原head逐bit恒等。合法logit变化但argmax未变化不构成发布阻碍。通过后直接发布，不增加别的gate。

G1后若`HEAD_AT_DA`没有同时带来正的平均H、all-class floor和总正确数，或18-2/19-2负尾相对D122没有收窄，则关闭D123。任何负结果都不得触发`delta`倍率、median规则、donor数量、K规则或seed调参。

## 7.可行性摘要（20行内）

1.问题可观测：每个old类都有合法K-shot support和sealed ground anchor。
2.新增状态仅为六个support-ground残差及六个LOO标量。
3.量纲闭合：`e`、`v_s`、`v_g`、`delta`均为每维chord-MSE。
4.old集合只取immutable Phase1 registry，不读held/scorer角色。
5.新增类没有Phase1 asset，不进入donor且不获得ground head。
6.identity/RDCE分别重算，不跨臂复用proxy。
7.公式无可调超参数，K1/K5/K10共用同一规则。
8.`delta>=0`给出相对原head的确定性收缩上界。
9.不宣称receiver因果识别或floor保证。
10.实现可复用D112/D122状态、scorer和standalone runner结构。
11.无truth功能审计通过后只需一次63行四臂G1。
12.性能弱即关闭，不补125、seed或参数搜索。

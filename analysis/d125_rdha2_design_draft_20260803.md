# D125 RDHA-2设计草案

## 1.当前裁决

- 名称：`D125-RDHA-2`（Receiver-Held Dynamic Hyper-Adapter，rank-2）。
- 状态：`DESIGN_DRAFT / PHASE1_FALSIFIER_ONLY / STAGE2_IMPLEMENTATION_FORBIDDEN`。
- 独立终审：`REVISE / P0=0 / P1=6 / P2=1`。
- 当前不实现Stage2、不发布G0/G1/Target/125；只允许理论闭合后的一次source-only嵌套Phase1 falsifier。

## 2.为什么可能是新机制

D125不新增Phase2 observable；新增信息来自Phase1的receiver-held反事实响应监督：support set摘要必须预测一个对未参与拟合的receiver、TX/class和物理query仍有效的低维adapter响应。它严格禁止`z_dom`、source receiver/day bank、sample-to-bank matching、D102/D105 precision/codebank和support残差闭式ridge。

候选adapter放在joint projection之前的唯一早期冻结层：

\[
h_a(x)=h(x)+U\left[a\odot\tanh\left(V\operatorname{GN}(h(x))\right)\right],
\]

其中`U∈R^{C×2}`、`V∈R^{2×C}`，`a∈[-a_max,a_max]^2`。后续checkpoint网络保持冻结。样本依赖的`tanh/GN`必须改变后续激活路径；若可被固定末端线性变换、常量`B a`或PSD metric解释，立即按D102/D124重入关闭。

## 3.Phase2合法summary修订

作者原先使用old anchor/W的summary被拒绝。修订后，注册完成时只读取全部registered classes的合法support：

\[
u_c=\frac1K\sum_{k=1}^{K}\rho\left(\operatorname{GN}(h(x_{ck}))\right),
\qquad
s=\operatorname{Pool}_{c\in C_{registered}}(u_c),
\qquad
a=a_{max}\tanh(Qs+b)。
\]

`Pool`必须逐类等权、对class顺序置换不变；`rho/Q/b`只由Phase1学习并与checkpoint共同封存。old和new support共同生成唯一state；同一`a`重新forward全部old/new support并注册，随后每个query只做一次同adapter forward，零fit、零update、零selection。不得读取query、truth、held role、quota或query-batch统计。

`rho`维度、唯一层位、`r=2`和`a_max`尚未冻结，必须由嵌套Phase1完成选择；`a_max`优先由Jacobian/branch-norm稳定界确定，禁止读取任何Target或已开封source-held结果。

## 4.Phase1 episodic监督

每个outer fold同时held receiver与held TX/class。held实体必须从`U/V/rho/Q/b`训练、layer/rank/`a_max`选择、summary标准化、teacher响应优化和任何anchor/统计中完全排除。support、teacher-response query和outer eval query的物理ID互斥。

Phase1 teacher只可在inner source episode上用query标签求受限低维响应`a*`；student mapper学习`support summary→a*`，最终选择只看独立outer receiver-held×TX/class-LOCO query。Phase2和outer eval绝不运行teacher优化，也不打开query标签。

## 5.最小Phase1 falsifier

只运行一个预冻结source-only嵌套包，K1为主，并用同一公式覆盖一个常规K；不运行Target或125。

必须同时记录：

1.资产allowlist仅含int8/FP16`U/V/rho/Q/b`、尺度和稳定界；不得含source/clean/raw/cache、样本feature、物理ID、receiver/TX/day键或可检索bank。
2.outer receiver-held×TX/class-LOCO上的同rowheld-query BA/H、class floor和总正确数；系统性负收益或old/new单侧塌陷即关闭。
3.summary/系数TX与class probe不得超过预注册chance＋CI容许带。
4.K1二维support信息满秩、系数有限且非prior/bias常量；至少一个held package产生非零特征和预测效应。
5.联合标签置换后summary不变、logit列等变；INT8与reference forward预测等价。
6.固定末端线性、D102常量偏移和PSD拟合不能解释adapter输出；held样本Jacobian或激活mask必须随样本变化。

任一核心项失败即`CLOSE_D125_RDHA2_PHASE1_FALSIFIER`，不调rank、层、`a_max`、summary、seed或门限来复活。通过只允许进入一次fresh/development G1设计，不构成Stage2性能或真实新类结论。

## 6.资源边界

Phase2 enrollment预计每个support一次base forward生成summary，再一次adapted re-forward注册；每query只一次adapted forward。adapter额外约`2rC`MAC/样本，另计完整backbone；无反传、optimizer或query state。实际层通道数、payload字节、forward延迟和INT8误差必须在Phase1冻结后实测，不能沿用概算宣传。

## 7.可行性摘要（20行内）

1.D125新增的是Phase1 receiver-held响应监督，不是新的target observable。
2.早层样本依赖非线性adapter有机会区别于D102常量偏移和PSD metric。
3.Phase2禁止z_dom、source bank、闭式support残差solve和query反馈。
4.adapter系数只由all-registered support逐类等权summary生成。
5.old/new support和query使用同一冻结adapter state。
6.outer held receiver/TX/class从训练、teacher和选择完全排除。
7.K1可拟合2维系数不等于可辨识，必须由outer held迁移证明。
8.rank、层位、summary维度和a_max仍待Phase1嵌套冻结。
9.当前只允许一个Phase1 falsifier，不允许Stage2实现或N607性能矩阵。
10.falsifier失败立即关闭，不回退到D102/D105或PSD路线。

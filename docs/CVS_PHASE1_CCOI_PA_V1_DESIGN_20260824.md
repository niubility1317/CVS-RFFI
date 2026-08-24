# 从内容控制到条件系统辨识：Phase1挑战条件化PA算子辨识详细结合设计

日期：2026-08-24
设计代号：`CCOI-PA-V1`（Challenge-Conditioned Operator Identification with PA，挑战条件化PA算子辨识）
配套追踪表：`docs/CVS_PHASE1_CCOI_PA_V1_TRACE_20260824.md`
当前状态：完成优化设计、本地实现、协议负测和一次P0/P1定点修复；本地聚焦测试已通过，真实Core90 checkpoint smoke与N607实验尚待发布验证，因此技术交付状态为`LOCAL_VERIFIED`，科学增益仍为`UNKNOWN`。

## 0. 结论先行

### 0.1 对粘贴文本的吸收、修正与否定

本设计吸收了粘贴文本中最有价值的四层递进：第一，内容对齐只用于控制激励，不能宣称自动消除了信道与接收机；第二，把单个固定指纹向量改写为“挑战条件—局部响应—集合算子”；第三，用同挑战跨domain、异TX同挑战和DiD保持相对设备几何；第四，用留出挑战预测、负对照和coverage曲线区分“分类增益”与“系统辨识证据”。这些内容都已映射为可运行模块、损失或诊断。

本设计没有原样接受五类主张：没有可靠重复payload/前导元数据时，模型近邻只能叫M2代理挑战匹配，不能冒充M1真实语义对齐；内容对齐不能被解释为接收机/信道消除；V1不同时引入Soft-DTW、partial OT、多机制生成器和状态子空间；不把同TX样本无条件压成单点；不在主C4中人为删除挑战分区制造“未见挑战”，因为这会改变训练分布并破坏C3/C4同row归因。最后一项改为只读码本稀疏/未占用诊断，严格跨挑战OOD留给单独等预算实验。

这些取舍是合理的：它们保留了文本的物理核心和可证伪实验，同时删除了缺少真实元数据支持、会混淆因果归因或显著扩大V1变量面的部分。当前设计只能检验“受限PA挑战范围内的条件响应”，不能直接证明恢复了发射机的完整物理系统。

粘贴文本提出的方向与现有Phase1并不冲突，但不能把它理解成“再加一个内容对齐loss”。现有Phase1已经具有适合承载该设计的三个基础：`z_id/z_dom`双分支、记忆多项式PA/包络特征路径，以及按TX×接收域构造矩形批次的采样器。真正缺少的是一条明确的条件系统辨识闭环：先描述当前输入激励，再估计设备在该激励下的局部响应，最后跨多个挑战聚合为设备级算子，并用未参与估计的挑战检验该算子能否预测响应。

本报告建议的第一版不是重写现有网络，而是在冻结的`ADV3B02_CORE90_SOFT_E200`旁增加一条可关闭的PA算子侧路：

\[
x\xrightarrow{\text{content view}}q_t,
\qquad
x\xrightarrow{\text{fingerprint view}}h_t^{\mathrm{PA}},
\qquad
r_t=F_{\mathrm{PA}}(h_t^{\mathrm{PA}},q_t),
\qquad
\hat\theta_{\mathrm{PA}}=\operatorname{Pool}\{r_t,q_t,w_t\}.
\]

其中，`q_t`描述第`t`个局部片段提供了什么激励，`r_t`描述设备在该激励下产生了什么PA/包络响应，`\hat\theta_PA`是跨挑战聚合后的设备算子证据。基线CosFace身份头保持不变，算子头独立产生logits，只在决策末端做源域冻结的证据融合。这样可以同时满足四个要求：保留现有强基线、避免旧checkpoint失效、把增益归因到条件辨识而不是普通增参，并为Phase2合法的K-shot support聚合留下接口。

V1只做PA/包络机制，不引入Soft-DTW、partial OT、全波形生成器、多状态子空间或多机制联合建模。最小实验按C0–C4逐项增加“同容量集合头→挑战条件化→DiD→留出挑战预测”，只有在单seed同row证据达到预注册门槛后才进入多seed或更大矩阵。

## 1. 问题重述：内容对齐不是设备指纹

### 1.1 观测对象是条件响应，不是孤立常量

接收IQ可以抽象为：

\[
y=\mathcal R_j\circ\mathcal H_d\circ\mathcal F_i(x;s_i)+n,
\]

其中，`x`是发射内容或激励，`\mathcal F_i`是第`i`个发射设备的非理想响应，`s_i`是设备状态，`\mathcal H_d`是传播域，`\mathcal R_j`是接收机链路。即使两个样本内容相似，`\mathcal H_d`和`\mathcal R_j`也没有自动消失；反过来，如果完全忽略`x`，同一设备在不同幅度、频谱和时序激励下的PA响应也会被错误压成一个常量。

因此，合理目标不是寻找一个与内容完全无关的局部表示，而是估计条件设备响应：

\[
r_{i,t}=F_i(x_t,q_t),
\]

再从多个受控且足够多样的挑战中辨识稳定设备参数或低维算子`\theta_i`。这与物理层设备识别把观测建模为设备相关函数的思路一致；早期函数化RFFI工作已经明确把物理层过程作为待识别函数，而不是固定残差向量[1]。

### 1.2 “同挑战”和“多挑战”承担不同任务

- 同挑战比较用于公平：相同或近似激励下，设备间差异更可能来自硬件响应。
- 多挑战聚合用于可辨识：单一窄带、恒包络或低动态范围激励可能无法激活足够多的PA/IQ失真自由度。

这两者必须同时存在。若只有同挑战拉近，会得到内容簇而不是设备算子；若只有跨内容聚合，会把激励变化当作设备状态。可辨识性最终取决于挑战覆盖、动态范围和激励矩阵条件，而不是embedding维度是否更大。

### 1.3 为什么优先PA/包络机制

现有模型已经包含记忆多项式和包络门控，改造成本最低，也最适合做条件响应。公开实证显示，频谱再生长可由PA非线性主导，CFO更适合作为辅助特征[2]；2026年的Fisher信息分析进一步指出，在恒模PSK下I/Q不平衡参数可出现秩亏，而PA非线性贡献更强[3]。这些结果不能直接证明本项目WiSig划分上的收益，但支持“先选一个可解释、已有骨干支持的机制做最小辨识实验”，而不是一开始并行堆叠PA、CFO、I/Q、相位噪声和信道残差。

## 2. 现有Phase1能力与证据边界

### 2.1 现有方法可以直接复用的部分

当前Git承载面的Phase1实现已经具备以下结构：

| 现有能力 | 当前作用 | 在CCOI-PA中的复用方式 |
|---|---|---|
| Sinc/HF共享前端 | 从256点双通道IQ提取时频局部特征 | 保持原输入和基线路径，不重新定义主干 |
| identity/domain双分支 | 输出`z_id`和`z_dom`，承担身份与域因素解耦 | `theta_pa`只进入身份侧证据；`z_dom`继续承担接收域建模 |
| `MemoryPolynomialLift` | 显式构造1/3/5阶、记忆深度4的PA特征 | 暴露池化前时序特征，作为条件响应的fingerprint view |
| `EnvelopeGate1d` | 建模幅度相关响应 | 与挑战码共同形成FiLM条件响应 |
| CosFace身份头 | 维持已验证的源域判别几何 | 不替换；作为C0基线和晚融合主证据 |
| 同TX跨域MixStyle与域约束 | 提升跨接收域鲁棒性 | 继续保留，不把内容条件化误当作域消除 |
| `BalancedTxDomainBatchSampler` | 构造TX×domain平衡批次 | 直接采样同TX跨domain正对和异TX同domain负对 |
| clean+LEO_WEAK卫星视图 | 提供同物理IQ的已知配对扰动 | 作为挑战编码器跨域一致性的可靠同挑战锚点 |

这意味着新增设计应当是“侧路+少量接口”，而不是复制一套编码器或改变Phase1的默认训练配方。

### 2.2 必须冻结的基线

对照基线固定为`ADV3B02_CORE90_SOFT_E200`：clean与satellite concat；卫星分支只使用分类CE；`lambda_sat_cls=0.68`、`lambda_sat_cons=0`；E1–40使用`leo_clear_weak,p=0.30`，E41–90使用`leo_low_elev_weak/leo_rain_weak,p=0.60`，E91–200使用三类并集、`p=0.80`。CCOI-PA不得顺带修改这些参数，否则无法判断收益来自条件系统辨识还是新的卫星增强配方。

历史可引用锚点为：ADV3B02在既有正式记录中的E194结果为clean 86.09%、clear 72.56%、low-elev 69.87%、rain 69.27%、LEO均值70.56%。这只是历史锚点，不是新实验C0的替代物；C0–C4仍须从同一冻结checkpoint、同一数据划分和同一评估器产生同row结果。

### 2.3 最近候选没有替代本设计的证据

| 路线 | 已证实状态 | 对本设计的含义 |
|---|---|---|
| FastTrust多行伪标签 | 部分严格U卫星身份/类别上限配置出现小幅改善，但尚无多seed新默认 | 说明伪标签置信本身未解决“输入激励是什么”的辨识问题 |
| RC4质量预算 | P3/P5/P6已完成；P3在三者中最好，clean 85.500%、LEO均值73.979%、floor 58.558%；覆盖扩大没有继续提升 | 内容/质量选择需要可辨识性与覆盖联合约束，不能只追求更多伪标签 |
| HSID最小路线 | 完成但收益仅约+0.043个百分点LEO总体、+0.055个百分点严格子集，不晋级 | 早期拼接或极小侧信息不足以证明设备算子学习 |
| RC4 QB | QB0/QB1技术失败，QB2完成但未建立科学增益 | 不能把技术闭合当作性能闭合 |
| QB3 B/C/HPS | 当前正式状态为`RUNNING` | 不存在可用于替换Core90的完成性能结论 |

因此，本设计不把任何尚未闭合的近期候选当作新控制组，也不把现有小增益解释为内容条件辨识已经奏效。

## 3. 协议兼容性评估

### 3.1 Phase1数据边界

CCOI-PA只允许读取`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`的源域物理样本划分。`L_s`可用于TX身份损失、DiD和泄漏探针；`U_s`可用于同物理样本clean/satellite挑战一致性、自监督码本学习和无标签留出预测，但不能读取其TX真值；`V_cal`只校准融合、温度和证据门槛；`V_select`只做候选选择。`V_cal/V_select`不得反向传播或更新码本、归一化状态和设备原型。

目标域、Phase2 query、query role、query truth、真实batch类别数、全局配额和跨query重分配均不得进入挑战编码、匹配、融合或门控。Phase2部署时，`theta_pa`只能从合法K-shot support集合估计；每条query独立与注册算子原型比较，不能把多条query合成一个设备算子。

### 3.2 同一物理样本的多视图不增加K

clean与卫星增强、FFT、均衡或其他数学视图都来自同一固定物理IQ时，它们只是一个样本的多个视图。Phase1可用这些视图建立挑战不变性；Phase2中它们不能被计为多个K-shot证据。实现必须携带`physical_id`，集合池化先在同物理样本内部汇总token，再按独立物理样本汇总K个support，防止视图膨胀样本数。

### 3.3 Phase1完成标准保持不变

每个完成训练的候选都必须使用最终选定checkpoint评估clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，保存checkpoint身份、评估配置、分场景指标及日志。LEO均值不能替代三个场景。低性能不是技术停跑条件；只有数据越权、错误split/receiver/seed、输出覆盖、错误checkout、确定性执行故障、无prediction闭合或scorer连接错误才允许停止。

## 4. CCOI-PA-V1总体架构

### 4.1 组件边界

V1由五个新增组件构成：

1. `DualIQView`：从同一固定接收IQ产生内容视图`x_content`和指纹视图`x_fp`。
2. `PAChallengeEncoder`：把内容视图切成局部token，输出连续挑战码`q_t`、48类软码本分配和覆盖指标。
3. `PAConditionalResponseHead`：读取现有PA路径的时序特征，通过FiLM估计局部条件响应`r_t`。
4. `OperatorPool`：按匹配置信度和观测质量聚合局部响应，得到64维`theta_pa`。
5. `HeldoutChallengePredictor`与`ObservabilityGate`：检验算子能否预测未参与聚合的挑战，并报告证据是否充分。

新增模型应包装现有`DualCVSincNetDisentangle`，而不是在默认模型中无条件注册新参数。`ccoi.enabled=false`时实例化路径、state_dict键和输出必须与当前模型一致；`true`时wrapper额外返回`theta_pa`、`operator_logits`、挑战覆盖和留出预测统计。这样才能保证旧checkpoint严格加载，不用把整个Phase1历史重训当作实现前置条件。

### 4.2 双视图设计

内容视图的目标是稳定描述激励，指纹视图的目标是保留设备响应。二者来自同一个接收IQ，但处理强度不同。

| 操作 | 内容视图 | 指纹视图 | 设计理由 |
|---|---|---|---|
| 粗同步/固定裁剪 | 使用 | 使用相同边界 | token必须物理对应 |
| 包级RMS幅度归一化 | 使用，保留token间相对幅度 | 仅沿用现有最小预处理 | 降低绝对增益污染，但保留PA激励强弱 |
| 整包CFO/线性相位补偿 | 可用，参数stop-gradient | 不做或只使用现有处理 | 防止q主要编码频偏，同时不删除指纹侧相位特征 |
| 局部白化 | 有界使用 | 不使用 | q关注内容结构，不破坏响应幅相 |
| 粗均衡 | 仅在已有可靠估计时可选 | 不使用 | 避免把均衡残差当设备指纹 |
| 逐token独立幅度归一化 | 禁止 | 禁止 | 会消除挑战幅度动态和PA可观测性 |
| 高频细节抑制 | 内容视图可轻度使用 | 禁止 | 避免q直接携带微弱设备失真 |

这里的关键不是“内容视图越干净越好”，而是只去除会使挑战码学习接收域捷径的部分，同时保留决定PA响应的激励幅度、峰均比、局部谱占用和转移结构。

### 4.3 token化和挑战码

对长度256的IQ使用长度64、步长16的重叠窗口，得到：

\[
T=\left\lfloor\frac{256-64}{16}\right\rfloor+1=13
\]

个token。每个token编码为32维`q_t`，再对48个源域挑战原型做软分配。32维和48类是粘贴文本建议范围内的工程起点，需要通过码本占用、熵和泄漏探针检验，不是预先成立的物理类别数。

挑战码预训练使用五类信号：

- 同一物理IQ的clean/satellite视图：作为可靠正对，约束q对传播扰动稳定。
- 同一视图的轻度内容保持增强：约束局部一致性。
- 掩码内容统计重建：从未被直接观察的邻域上下文预测被掩码token的RMS、PAPR、谱质心、谱平坦度和相邻差分统计；目标由固定、stop-gradient的解析函数产生，解码器不能读取TX/RX标签。
- 局部时序预测：用当前token挑战码预测下一非重叠锚点的固定内容统计，防止q退化为单样本实例编号。
- 跨样本近邻：只在冻结挑战编码器后作为带置信权重的代理匹配，不参与端到端互相强化。

为防止q携带设备或接收域，预训练时使用梯度反转TX/RX探针，评估时另行拟合冻结q的TX/RX/day线性探针；同时进行四个负对照：随机码、batch内shuffle码、RX标签码和时间位置码。真实q必须同时优于这些对照的系统辨识指标，并保持TX/RX泄漏接近机会水平。这里的“接近”按归一化机会优势报告，例如`probe_acc−chance_acc≤0.05`，而不是对所有类别使用固定绝对准确率。

`challenge_dim`和`challenge_codebook_size`是两个独立量。V1固定`challenge_dim=32`、`challenge_codebook_size=48`，只验证一次码本占用和塌缩；不得同时扫描两者。原文的“32～64个粗挑战类型”对应码本规模，不等同于连续挑战向量维度。

### 4.4 条件PA响应

现有PA路径池化前特征记为`h_t^PA∈R^64`。挑战码经过小型MLP生成FiLM参数：

\[
(\gamma_t,\beta_t)=g(q_t),
\qquad
r_t=\phi\big((1+\gamma_t)\odot h_t^{\mathrm{PA}}+\beta_t\big).
\]

`gamma`初始为0附近，`beta`初始为0，使新头从近似恒等映射开始。挑战编码器在进入身份训练前冻结，匹配索引和权重stop-gradient；这能显著降低q与响应分支“合谋”产生任意设备编码的风险。

V1不从理想波形重建残差作为唯一输入。残差强依赖同步、均衡和信道估计误差，容易把估计器缺陷变成指纹。PA时序特征仍从实际接收IQ提取，q只解释激励条件。

### 4.5 集合级设备算子

局部响应按权重`w_t`聚合：

\[
u_t=\phi_r([r_t,q_t]),\qquad
a_t=\operatorname{softmax}_t(s(u_t)+\log w_t),
\qquad
\hat\theta_{\mathrm{PA}}=\rho\left(\sum_t a_tu_t\right).
\]

`w_t`由挑战匹配置信、有效幅度、饱和/截断检测和mask共同决定。V1采用DeepSets式带权attention池化，复杂度为`O(T)`；不使用Set Transformer或OT的`O(T²)`交互。池化必须满足排列不变性，并输出有效token数、挑战码本覆盖率、分配熵和最大单token权重，防止模型只选择少量容易片段。

单包Phase1先在13个token内得到`theta_pa`。Phase2若有合法K-shot support，则先对每个独立物理样本产生一个`theta_pa^(k)`，再做第二级集合汇总；同一样本的clean/satellite/FFT视图只能进入该样本内部，不能增加K。

### 4.6 晚期证据融合

算子侧使用独立归一化分类头：

\[
\ell_{op}=s_{op}\cos(\hat\theta_{PA},p_y^{op}),
\qquad
\ell=(1-\alpha)\ell_{base}+\alpha\ell_{op}.
\]

`alpha=0`必须严格复现当前基线。第一轮C1–C4建议固定`alpha=0.15`或仅在`V_cal`内比较`{0,0.05,0.10,0.15,0.20}`，再在`V_select`冻结选择；不得根据目标域或单个LEO结果调节。算子头与基线头保持独立，既能看清算子证据自身有效性，也避免早期拼接把它退化为普通增维。

## 5. 匹配、四元组和DiD几何

### 5.1 匹配证据分级

匹配必须区分三种证据，不得混为“内容真值”：

| 等级 | 证据 | 可用范围 | 可信度 |
|---|---|---|---|
| M0 | 同一`physical_id`的clean与卫星视图 | `L_s/U_s` | 已知同挑战锚点 |
| M1 | 数据元数据确认的重复前导、payload或受控波形 | 若数据审计确认存在 | 真实语义匹配评估 |
| M2 | 冻结q空间中的近邻/同码本 | 源域训练批次 | 代理挑战匹配，只能带权使用 |

当前最高风险是M1是否存在。若WiSig记录没有可靠内容标识，则可以训练M2，但报告必须写“代理挑战匹配”，不能声称真实内容precision/recall已经达到某值。模型置信度、低loss或下游准确率都不能替代M1验证。

### 5.2 正负对与四元组

现有平衡采样器可产生`TX_a/TX_b×domain_1/domain_2`矩形。对每个矩形，在冻结q中选择相近挑战token：

- 正对：`(TX_a,domain_1,q)`与`(TX_a,domain_2,q')`，其中`q≈q'`。
- 硬负对：`(TX_a,domain_1,q)`与`(TX_b,domain_1,q')`，其中`q≈q'`。
- 第二域对照：`(TX_a,domain_2,q)`与`(TX_b,domain_2,q')`。

局部条件对比损失只在匹配置信超过源域预注册门槛时计算；没有有效匹配时该项安全跳过并记录coverage，禁止为了凑足配额放宽到全batch全局重排。C1使用相同容量、相同temperature的普通TX SupCon；C2将正样本限制为同TX、跨domain且挑战相似，将有效困难负样本限制为异TX、同domain且挑战相似。二者的区别只能是挑战条件mask。

### 5.3 差中之差约束

设设备对差向量为：

\[
\Delta^{(d)}_{ab}=\hat\theta_{a,d}-\hat\theta_{b,d}.
\]

DiD损失约束跨域相对设备差保持一致：

\[
L_{DiD}=1-\cos\left(\Delta^{(d_1)}_{ab},\Delta^{(d_2)}_{ab}\right)
+\eta\left|\|\Delta^{(d_1)}_{ab}\|_2-\|\Delta^{(d_2)}_{ab}\|_2\right|.
\]

它不要求同一设备在不同域的所有表示完全相同，而是要求“设备A相对设备B的差”在域变化后保留。这比无条件域不变更符合接收信号仍包含信道和接收机影响的事实。

## 6. 留出挑战预测：把判别特征提升为算子证据

### 6.1 任务定义

每个样本只从窗口长度64、间隔64的非重叠锚点中划分support-challenge和holdout-challenge；步长16产生的其他重叠token只参与身份分类，不参与holdout预测。C4把support与holdout原始采样区间分别置入两个只读冻结Core90前向，避免PA卷积感受野跨区共享原始采样；`OperatorPool`只能读取support隔离视图得到`theta_pa`，预测头接收`theta_pa`与holdout的`q_t`，预测holdout隔离视图的冻结PA响应图：

\[
\hat r_t^{hold}=G(\hat\theta_{PA}^{support},q_t^{hold}).
\]

holdout目标由holdout隔离视图经过冻结Core90的PA池化前特征产生并强制stop-gradient；响应头和预测头不能改变目标。C1的FiLM、attention和holdout预测全程使用同一个可学习常量条件，不允许集合权重暗中读取真实q。损失使用归一化MSE和余弦误差，不直接要求重建完整IQ。若条件算子确实描述设备响应，C4应在未参与估计的挑战上比C1更准确。

### 6.2 不能接受的替代指标

- 分类准确率上升但holdout NMSE不改善：只能说明增参有判别收益，不能说明学到设备算子。
- 同一token自编码：输入已经泄露目标响应，不构成留出预测。
- 随机划分相邻高度重叠窗口：可能共享大部分采样点。实现应按非重叠中心区或挑战bin整体留出。
- 跨目标query聚合后预测：违反独立query边界，也引入batch构成信息。

### 6.3 最小判定

相对同容量非条件控制C1，C4的holdout响应NMSE至少降低5%，且`R²>0`；真实q还必须显著优于shuffle/random/RX/time-code。该阈值是V1预注册的工程判定线，不代表普适统计效应。若分类提升但留出预测失败，结果只能称“条件判别侧路有效”，不能称“系统辨识成立”。

## 7. 训练目标与阶段

### 7.1 保持不变的基线损失

`ADV3B02_CORE90_SOFT_E200`当前身份分类、域分类、GRL/域混淆、协方差正交、MixStyle和卫星CE均保持原配置。CCOI不重新调这些权重。

### 7.2 V1新增损失

第一版增加七项，其中挑战编码器预训练项与冻结后的身份侧训练项分阶段启用：

\[
L_{CCOI}=\lambda_qL_{q\_aug}+\lambda_mL_{masked}
+\lambda_tL_{temporal}+\lambda_cL_{cond\_cls}+\lambda_pL_{pair}
+\lambda_dL_{DiD}+\lambda_hL_{holdout}+\lambda_vL_{var}
+\lambda_bL_{code}.
\]

| 损失 | 作用 | 初始权重 | 边界 |
|---|---|---:|---|
| `L_q_aug` | 同物理IQ跨clean/satellite挑战一致性 | 0.25 | q预训练后冻结；不读TX真值即可用于`U_s` |
| `L_masked` | 从上下文预测被掩码token的固定内容统计 | 0.50 | 目标stop-gradient；不重建设备残差 |
| `L_temporal` | 预测下一非重叠锚点的固定内容统计 | 0.20 | 不允许相邻重叠窗口作为预测目标 |
| `L_cond_cls` | `theta_pa`独立身份分类 | 1.00 | 只在`L_s`使用TX标签 |
| `L_pair` | C1普通SupCon或C2–C4挑战匹配SupCon | 0.15 | 相同temperature和容量；无有效pair时安全跳过 |
| `L_DiD` | 跨domain保持设备对相对差 | 0.10 | 只在有效2TX×2domain矩形计算 |
| `L_holdout` | 预测未参与聚合的挑战响应 | 0.20 | `L_s/U_s`均可，但U不读TX标签 |
| `L_var` | 防止theta或码本塌缩 | 0.02 | 只做方差下界，不强迫均匀类别配额 |
| `L_code` | clean/satellite码本一致性、批次占用和单token置信度 | 0.25/0.05/0.005 | 只作用于源域q预训练，不使用类别配额或目标数据 |

这些数值是首个head-only smoke的工程起点，不是已验证最优超参数。初始训练冻结现有主干和挑战编码器，只训练条件头、集合池化、算子分类头和holdout预测头，建议新头学习率`3e-4`。若C2/C3达到门槛，再以`3e-5`解冻PA最后一个block及joint投影；不先全量解冻。

### 7.3 分阶段实施

| 阶段 | 变化 | 核心输出 | 进入下一阶段条件 |
|---|---|---|---|
| A 数据/物理审计 | 核查M1真实挑战元数据、幅度动态、截断和token覆盖 | 可辨识性审计表 | 明确M1存在或正式标注为代理匹配 |
| B 挑战编码器 | 双视图、掩码内容统计、局部时序、码本与泄漏探针 | 内容预测、q覆盖、熵、TX/RX/day探针 | 内容预测优于常量基线，不塌缩且不以TX/RX为主 |
| C 条件头 | 冻结骨干，对比C1/C2 | 分类、margin、覆盖、算子logits | C2相对C1达到最小收益且clean不显著退化 |
| D DiD | 在C2上加入矩形DiD | 跨域设备差保持率 | floor/跨域稳定性改善 |
| E 留出预测 | 在C3上加入holdout预测 | NMSE、R²、对照差 | 满足算子证据门槛 |
| F 小步解冻 | 仅解冻PA末block/joint | C4微调结果 | 只在前述证据成立后执行 |
| G Phase1确认 | 多seed或扩大场景确认 | clean+三LEO场景完整证据 | 达成晋级阈值后才进入 |

若B或C失败，不进入Soft-DTW/OT；应先判断挑战码是否被接收域污染、是否抹除了激励动态、或PA响应在当前波形中是否缺乏可观测性。

### 7.4 梯度冲突处理

第一轮只记录`L_cond_cls/L_DiD/L_holdout`对共享新增层的梯度余弦，不启用PCGrad。只有在多个epoch内出现稳定负余弦、且确实阻碍C2–C4收敛时再引入梯度投影。PCGrad的原始工作针对多任务冲突梯度提供投影方法[4]，但增加优化器复杂度本身不能替代对损失定义错误的诊断。

## 8. 最小实验矩阵与归因

### 8.1 C0–C4同row矩阵

| row | 基线主干 | 同容量PA集合头 | 挑战条件q | DiD | 留出预测 | 要回答的问题 |
|---|---|---|---|---|---|---|
| C0 | 冻结Core90 | 否 | 否 | 否 | 否 | 当前真实控制性能是什么 |
| C1 | 同C0 | 是 | 否 | 否 | 否 | 普通增参、集合池化和普通SupCon本身带来多少收益 |
| C2 | 同C0 | 是 | 是 | 否 | 否 | 挑战匹配SupCon是否优于同容量普通SupCon |
| C3 | 同C0 | 是 | 是 | 是 | 否 | 相对设备差的跨域保持是否改善floor |
| C4 | 同C0 | 是 | 是 | 是 | 是 | 表示是否具备留出挑战预测能力 |

所有row固定同一Git commit、数据split、seed、起始checkpoint、训练预算、LEO_WEAK课程和最终评估器。早期只运行单seed的关键最小矩阵，不把多seed、完整125或全部扩展消融作为V1前置门槛。

### 8.2 必须同时运行的代码对照

C2–C4至少加入四个不额外训练或低成本复算的q对照：

1. `q_random`：同分布随机码。
2. `q_shuffle`：batch内打乱真实q。
3. `q_rx`：显式接收域编码，检验是否只是domain shortcut。
4. `q_time`：仅使用token位置编码，检验固定时序模板捷径。

如果真实q与shuffle无差异，说明模型没有使用挑战条件；如果RX-code最好，说明所谓算子主要依赖接收域；如果time-code等价，说明挑战码可能只记住包内位置。

每个row还必须从冻结prediction计算三种距离：

\[
d_1=d(r_{i,d_1,q},r_{i,d_2,q}),\quad
d_2=d(r_{i,d_1,q_1},r_{i,d_2,q_2}),\quad
d_3=d(r_{i,d,q},r_{j,d,q}).
\]

目标关系为`d1<d2<d3`。若`d1≈d2`，内容不是主要类内变化来源；若`d1>d3`，域影响超过设备间隔；若三者同时下降，检查表示坍塌。该诊断只读取已产生的源域或冻结目标prediction，不增加训练row。

完成C4后只读报告源域码本占用、稀疏bin和未占用bin，不把未占用bin伪称为已有样本上的真实OOD性能。原先“主C4训练删除固定分区、评估保留分区”的方案被否定，因为它会使C4与C3使用不同训练分布并混淆holdout损失归因。严格跨挑战OOD改为后续独立等预算row，必须同时固定幅度/PAPR/谱统计分区且不用于目标域选模。

### 8.3 晋级阈值

V1先使用以下预注册门槛：

- C2或更高row相对同容量C1：LEO三场景均值增益至少超过`max(0.30个百分点,2×control复评标准差)`。
- receiver-cell floor增益至少超过`max(0.30个百分点,2×control复评标准差)`。
- clean下降不超过`0.50`个百分点。
- identity margin retention不低于C1的`0.995`，或统计上无明确下降。
- 三个LEO场景不得出现大于`0.50`个百分点的单场景退化而被均值掩盖。
- holdout响应NMSE相对C1降低至少5%，且`R²>0`。
- q的归一化TX/RX探针机会优势不高于0.05；真实q在holdout预测上优于四个负对照。
- coverage不能靠只保留极少token换取准确率；至少同时报告100%、75%、50%、25%覆盖点和AUC。

这些门槛用于V1工程决策，不将一次单seed跨线直接解释为新默认。单seed通过后再做多seed确认和置信区间；未通过则保留负结果，分析失败机制并决定是否进入下一候选。

### 8.4 科学不晋级条件

以下结果允许实验完整结束，但不晋级：

- 源域分类提高而三LEO场景下降。
- q能高准确率预测TX或RX。
- 真实q不优于shuffle/random/time-code。
- 只在极低coverage下取得优势。
- identity margin明显下降或跨域设备对差方向不稳定。
- holdout预测不优于同容量非条件头。
- 权重集中在少数高SNR、低幅度或固定位置token，形成“容易样本选择器”。

这些是科学判定，不是进程中途终止理由。

## 9. 指标体系

### 9.1 性能指标

- clean、clear、low-elev、rain逐场景准确率。
- LEO均值、receiver-cell floor、最差场景、跨receiver方差。
- 当前Phase1已有的身份margin、域混淆和遗忘/稳定指标。
- 单包与多独立物理包聚合曲线；多视图不得冒充多包。

### 9.2 表征与泄漏指标

- `Probe_TX(q)`、`Probe_RX(q)`、`Probe_day(q)`。
- `Probe_TX(theta)`与`Probe_RX(theta)`，验证theta增强身份而不是接收域。
- 相对设备差保持率：

\[
R_{stable}=\mathbb E\cos(\Delta^{d_1}_{ab},\Delta^{d_2}_{ab}).
\]

- margin retention：`R_margin=margin_Ck/margin_C1`。
- 真实q与四类对照码之间的性能差。

### 9.3 系统辨识指标

- holdout响应NMSE与`R²`。
- 挑战码本占用率、有效挑战数、分配熵。
- M0/M1/M2分级的match precision、recall、coverage；没有M1时不得报告“真实语义precision”。
- 最大token权重、有效样本量`1/Σa_t²`和coverage—accuracy曲线。
- 挑战激励矩阵条件数或最小奇异值，用于解释不可辨识区域。
- `rho_int`：设备差与域差的交互强度，判断算子是否仍被接收链主导。

### 9.4 状态稳定性

公开研究已观察到SDR发射机在掉电、FPGA重载后指纹发生状态变化[5]。因此，若本项目未来具备跨上电/跨时段元数据，应把`theta_pa`分解为稳定核和状态偏移，并分别报告设备间、状态内和状态间方差。当前WiSig划分中的receiver/domain标签不能替代设备状态标签，V1不宣称已经解决状态漂移。

## 10. 可辨识性门控

门控只判断“当前样本提供的挑战是否足以支持算子证据”，不决定是否从评估集中删除样本。输入包括：

- 有效挑战bin数和熵。
- token有效样本量与最大权重。
- 激励幅度动态、包络分位数和频谱覆盖。
- holdout预测不确定度。
- 码本到训练覆盖区域的距离。

门控输出`evidence_sufficient∈{0,1}`和连续置信度。Phase1已知类测试中，低置信或拒绝样本仍计为错误；同时报告按覆盖率分层的准确率。Phase2只能用合法support计算注册算子质量，每个query独立决策。不得按query batch类别配额调整门槛，也不得利用query truth回填。

## 11. 实现落点

### 11.1 文件级修改方案

| 文件 | 计划修改 | 兼容性要求 |
|---|---|---|
| `code/model.py` | 在现有aux中暴露PA池化前`pa_token_map` | 不新增参数，默认logits和state_dict不变 |
| `code/model_dual_cvsincnet.py` | 复用现有`aux_id`和原始`tx_logits`接口 | 无需修改 |
| `code/cvsrffi/ccoi_pa.py`（新） | 双视图、固定内容统计、挑战码本、FiLM响应、集合池化和隔离holdout预测 | 组件独立、无样本级持久状态 |
| `code/cvsrffi/ccoi_pa.py`（新） | 双视图、固定内容统计、挑战编码器、FiLM响应、集合池化、holdout预测、门控和诊断 | 组件独立、可单测、无全局状态 |
| `code/cvsrffi/ccoi_losses.py`（新） | 普通/挑战匹配SupCon、DiD、非循环holdout、三距离诊断 | 空匹配/空矩形时返回零损失并记录原因 |
| `code/train_phase1_ccoi_pa.py`（新） | 冻结真实Core90 checkpoint，执行挑战预训练、C0–C4训练、四场景prediction与指标导出 | 不改变默认`train_ssdg.py`；训练和评估只读源域角色 |
| `code/score_phase1_ccoi_pa.py`（新） | prediction闭合后独立连接truth并输出同row指标 | prediction流不含truth |
| `code/cvsrffi/balanced_tx_rx_sampler.py` | 仅增加四元组索引导出或复用现有矩形统计 | 不创建跨batch全局配额 |
| `code/tests/test_ccoi_pa.py`（新） | 组件形状、排列不变、mask、梯度和负对照测试 | CPU可跑 |
| `code/tests/test_ccoi_protocol.py`（新） | `physical_id`、split、无target/query、K计数负测 | 协议失败必须硬报错 |
| `code/tests/test_ccoi_checkpoint_compat.py`（新） | 旧checkpoint严格加载、alpha=0基线等价 | 输出逐项容差核对 |
| Phase1配置目录 | 增加C1–C4配置，固定基线参数和差异字段 | 每个row只改变表中声明机制 |

### 11.2 建议配置面

```yaml
ccoi:
  enabled: true
  mechanism: pa_envelope
  token_length: 64
  token_stride: 16
  challenge_dim: 32
  challenge_codebook_size: 48
  operator_dim: 64
  challenge_encoder_frozen: true
  match_mode: physical_pair_plus_frozen_q
  min_match_confidence: 0.70
  fusion_alpha: 0.15
  loss_q_aug: 0.25
  loss_masked: 0.50
  loss_temporal: 0.20
  loss_cond_cls: 1.00
  loss_pair: 0.15
  loss_did: 0.10
  loss_holdout: 0.20
  loss_variance: 0.02
  heldout_fraction: 0.25
  observability_gate: report_only
```

配置中所有阈值都必须记录为“V1工程起点”。`observability_gate: report_only`是首轮硬约束，避免门控通过拒绝困难样本虚增准确率。

### 11.3 checkpoint与部署bundle

CCOI使用sidecar checkpoint或带命名空间的追加状态：

- `base_checkpoint_id`：冻结Core90 checkpoint身份。
- `challenge_encoder`及冻结码本。
- `conditional_pa_head/operator_pool/operator_classifier`。
- 源域校准后的`fusion_alpha`、温度和门控阈值。
- 每类聚合算子原型及半径，只保存允许的聚合知识。

Phase1 deployment bundle仍保留现有特征、prototype、radius、energy和tail输出；新增算子原型是补充，不删除旧证据。Phase2不得携带源样本、样本级embedding或挑战token缓存，只能使用协议允许的冻结模型与聚合知识。

## 12. 测试与独立正确性审查

### 12.1 聚焦协议负测

实现后至少覆盖：

1. target/query记录进入挑战预训练、匹配、融合校准或门控拟合时立即失败。
2. 同一`physical_id`的多个视图不能被计为多个K。
3. `V_cal/V_select`参与反向传播、码本更新或BN状态更新时失败。
4. 跨query集合聚合或按batch类别配额决策时失败。
5. M2代理匹配被标为M1真实匹配时报告字段校验失败。
6. 最终结果缺少任一LEO场景时不得标记Phase1完成。
7. holdout support与target窗口原始采样区间重叠时立即失败。
8. holdout目标未detach或来自可训练响应头时立即失败。

### 12.2 模型单元与回归测试

- 256点输入产生13个token；边界mask正确。
- `OperatorPool`对token排列不敏感。
- 所有token无效时不产生NaN，且门控为证据不足。
- 同一q、不同设备响应可分；同一设备、同挑战跨domain相对距离缩小。
- q常量时C2退化为同容量非条件头，而不是获得额外信息。
- q shuffle后holdout预测下降。
- 普通SupCon、挑战匹配SupCon和DiD使用相同输入维度与temperature。
- `d1/d2/d3`诊断在手工构造几何上恢复预期排序。
- `ccoi.enabled=false`和`fusion_alpha=0`分别复现旧模型结构与旧logits。
- 旧真实checkpoint进行一次无query smoke并产生合法prediction。

### 12.3 一次独立P0/P1审查范围

审查只关注会直接导致下一次真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的问题。报告字段美化、未来OT能力、额外hash/seal/receipt或完整V2设计均为非阻断项。

## 13. 复杂度与资源评估

现有长度256输入产生13个token，V1的挑战编码器、FiLM和DeepSets池化均为线性token复杂度。相对主干，新增计算主要来自小型内容编码器和64维响应头，预期显著低于复制一套完整CV-SincNet。具体显存、训练时延和参数量必须由本地真实checkpoint smoke测量，当前不填写估算值冒充实测。

Soft-DTW需要二次时间和空间复杂度[6]，partial OT还引入运输比例、dummy point及选择偏差控制[7]，因此二者延后是由V1归因和资源约束共同决定，不是否定其研究价值。

## 14. 风险矩阵

| 风险 | 失败表现 | 最小诊断 | 处置 |
|---|---|---|---|
| q被TX污染 | `Probe_TX(q)`远高于机会，shuffle影响小 | TX探针、q可视化、随机码对照 | 加强视图标准化/泄漏约束；冻结q；不进入C3 |
| q被RX污染 | RX-code与真实q同样有效 | RX/day探针、跨receiver M0一致性 | 调整内容视图；禁止目标域校准 |
| 过度归一化删除激励 | 码本熵高但holdout预测差 | 幅度/PAPR与q互信息、取消逐token归一化 | 恢复包内相对幅度和谱结构 |
| 代理匹配自证 | M2训练loss低但M1未知 | 分开报告M0/M1/M2 | 没有M1时限制科学表述 |
| 只选容易token | 低coverage准确率高，最大权重接近1 | 有效样本量、coverage—accuracy | 权重上限/熵正则；全覆盖结果为主 |
| PA不可观测 | C2≈C1，holdout `R²≤0` | 激励条件数、幅度动态、码本覆盖 | 结论为PA-V1不成立；不盲目加复杂度 |
| 相对几何损伤 | clean或margin明显下降 | C1/C2/C3同row、梯度余弦 | 降低DiD权重或停止解冻主干 |
| 增参伪增益 | C1已获得全部提升 | C0/C1/C2容量匹配 | 不宣称条件辨识贡献 |
| 域捷径伪算子 | RX-code最好、跨域差不稳定 | DiD、RX探针、receiver floor | 不晋级；保留负结果 |
| 设备多状态 | 同TX跨时间theta多峰 | 时间/上电元数据审计 | 延后到状态子空间V2，不用domain代替状态 |

深度网络会主动利用强混杂因素，除非训练明确抑制这些捷径[8]。因此，q泄漏探针、同容量控制和对照码不是装饰性分析，而是判定“条件系统辨识”是否成立的核心证据。

## 15. 后续扩展的进入条件

### 15.1 Soft-DTW

仅当真实挑战序列具有相同顺序但局部时间偏移，且V1的离散token错配成为主要误差时引入。必须验证它没有把SFO、记忆效应或设备相关时序差异一起对齐掉。

### 15.2 partial OT

仅当M1子集证明局部重叠普遍存在、M2覆盖不足时引入。需同时报告transport mass、dustbin比例、真实match precision/recall和选择偏差，防止只运输容易token。

### 15.3 多机制算子

PA-V1达到分类、floor和holdout预测门槛后，才按“PA记忆→差分谱→I/Q/相位”逐机制加入，每次保留同容量控制和独立head。差分信号处理已有对接收机指纹的专门建模工作[9]，但其假设和本项目发射端识别边界需要单独验证，不能直接并入V1。

### 15.4 稳定核与状态子空间

只有获得跨时间、掉电、重载或温度状态元数据后，才定义`theta=theta_stable+theta_state`。没有状态标签时强行分解会把receiver/day或随机噪声误称为设备状态。

## 16. 预期科学结论的等级

本路线可能产生四档结论：

1. **条件判别有效**：C2优于C1，但holdout预测未过线。只能说挑战条件改善了判别。
2. **局部算子证据成立**：C2/C3优于C1，holdout预测通过，q负对照失败。可以说学到了受限PA挑战范围内的条件响应。
3. **跨域算子稳定成立**：DiD提高receiver floor和相对设备差保持率，三LEO场景均不受损。可以说PA算子证据具有源域到卫星弱场景的稳定性。
4. **开放世界可部署候选**：多seed确认后，算子原型进入Phase1 bundle，并在Phase2严格support/query边界下独立验证。此前不得声称已解决新类注册或目标域适应。

当前达到`LOCAL_VERIFIED`：新增代码、配置、launcher、独立scorer和聚焦测试已完成；本机Git Bash因路由错误记为`FAILED`，脚本将由N607远端`bash -n`验证。尚无真实checkpoint smoke、prediction、score或N607性能结果，因此科学增益仍为`UNKNOWN`，不得把本地单测写成方法有效性证据。

## 17. 推荐执行顺序

1. 完成数据可辨识性审计，覆盖接收功率、CFO、PAPR、谱平坦度、饱和、挑战覆盖和TX—RX—内容混杂，并优先解决M1真实挑战匹配证据是否存在；无法可靠估计的信道相干时间标为`UNKNOWN`。
2. 实现双视图、固定内容统计、掩码/时序挑战预训练及泄漏探针，先冻结q。
3. 用wrapper实现普通SupCon C1和挑战匹配SupCon C2；完成旧checkpoint兼容与协议负测。
4. C2通过后加入DiD形成C3；再加入holdout预测形成C4。
5. 完成本地真实checkpoint无query smoke和一次独立P0/P1审查。
6. 形成最小预登记报告，提交Git，执行N607 preflight、单release归档SHA核对和远端编译。
7. 先跑单seed C0–C4关键矩阵；prediction完整后由独立scorer连接truth。
8. 达到预注册科学门槛后再进入多seed确认；未达到则记录失败机制，不以流程工作替代下一候选判断。

## 18. 参考依据与核验说明

1. Merchant等，[Deep Physical-Layer Authentification of Wireless Devices via Function Learning](https://arxiv.org/abs/1901.05914)，将物理层设备过程作为函数学习问题，为“设备算子而非固定向量”的表述提供早期依据。
2. Sankhe等，[No Radio Left Behind: Radio Fingerprinting Through Deep Learning of Physical-Layer Hardware Impairments](https://arxiv.org/abs/2408.09179)，讨论PA频谱再生长等硬件特征及跨接收天线实验。
3. Alhazmi等，[Fisher Information-Guided RF Fingerprinting: Identifiability, Nonlinearity, and Robust Feature Design](https://arxiv.org/abs/2603.29766)，从Fisher信息与实测角度讨论不同调制下硬件参数可辨识性。
4. Yu等，[Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)，用于说明后续PCGrad的适用边界。
5. Jian等，[A Case Study on the Reliability of Radio Frequency Fingerprinting](https://arxiv.org/abs/2412.07269)，报告设备掉电/重载后的指纹状态变化风险。
6. Cuturi和Blondel，[Soft-DTW: a Differentiable Loss Function for Time-Series](https://proceedings.mlr.press/v70/cuturi17a.html)，说明Soft-DTW机制及其二次复杂度。
7. Chapel等，[Partial Optimal Transport with Applications on Positive-Unlabeled Learning](https://proceedings.neurips.cc/paper/2020/file/1e6e25d952a0d639b676ee20d0519ee2-Paper.pdf)，说明部分质量运输与dummy point路线。
8. Geirhos等，[Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780)，用于约束挑战编码器的混杂捷径风险。
9. Zhang等，[Receiver-Agnostic Radio Frequency Fingerprinting via Differential Signal Processing](https://arxiv.org/abs/2503.22378)，作为后续差分机制候选，不作为PA-V1有效性的依据。

粘贴文本中编号[2]的IEEE链接与其文字所述函数辨识论文存在映射不一致，本报告改用可核验的arXiv原始版本；编号[9]的IEEE页面未作为关键结论依据。所有外部文献只用于支撑方法动机和风险判断，不替代本项目同row实验。

## 19. 最终判断

这条路线值得实现，但最合理的实现方式是“有条件、可关闭、可证伪的PA算子侧路”，而不是把整个Phase1改造成内容匹配网络。它与现有方法的有机结合点已经明确：现有PA特征提供响应底座，clean/satellite同物理视图提供可靠挑战锚点，TX×domain矩形采样提供DiD结构，CosFace与`z_id/z_dom`保留原有强基线，部署bundle承接冻结算子原型。

决定路线成败的不是模型是否更复杂，而是三项证据能否同时成立：挑战码不携带TX/RX捷径；条件头优于同容量非条件头；由部分挑战估计的算子能预测未见挑战响应。只要其中任一项缺失，就应降低科学表述，而不是用更多模块掩盖证据缺口。

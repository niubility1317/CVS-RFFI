# CCOI-PA-V2完整设计—实现—实验验证报告

- run ID：`PHASE1_CCOI_PA_V2_S20260824_20260825A`
- 更新时间：2026-08-25
- 方法代号：`CCOI-PA-V2`
- 实现提交：`8a959d00da768d1134ce859bd366052f4ea9c109`
- 实验状态：`ANALYZED`
- 最终判定：工程闭环`VERIFIED`；科学结论`SCIENTIFIC_FAILURE_NO_PROMOTION`

## 0.摘要与结论

本研究检验一条比“内容相似样本对齐”更严格的Phase1路线：先估计局部波形提供了什么PA/包络激励，再提取冻结Core90主干在该激励下的条件响应，最后把多个挑战—响应token聚合为设备级算子证据。方法没有把内容对齐解释为信道或接收机消除，也没有改动Core90主干、源域角色、LEO弱场景和目标/query边界。

V2完成了三项由V1真实日志直接定位的修复：恢复原始`meta.rx_i`的receiver导出；把未标定的加性旁路改为源域尺度对齐的凸融合；增加有界soft码本覆盖约束。实现、35项聚焦测试、真实checkpoint无query smoke、C0–C4五行训练、四场景prediction/truth和独立评分均完整闭合。

实验没有建立可晋级的分类收益。C2–C4相对同容量C1的最佳LEO均值提升只有`0.0095`个百分点，最佳LEO receiver-floor均值提升只有`0.0361`个百分点，均远低于预登记的`0.30`个百分点。冻结C0的LEO均值`76.4047%`仍高于所有侧路row。C4内部真实q相对shuffle只改善`2.815%`；现有hard统计则是packet dominant code，只占用`4/48`个码，不能据此证明token级码本塌缩。证据表明：V2解决了receiver和融合接口问题，但尚未证明q是跨TX共享挑战、θ是跨记录稳定TX算子，也没有证明sidecar拥有超出Core90的互补纠错信息。

因此，本实验应被保留为机制定位清楚的负结果。它否定“只要修复receiver或放大条件头，分类收益就会出现”，但保留“受控PA响应预测具有一定可学习性”这一较弱结论。当前实现是原始研究路线的PA单机制最小近似，不是完整条件系统辨识器。

### 0.1针对提交`26ac49e0`深度复盘后的逐项修订

复盘指出的核心因果缺口成立，但其中部分表述超出了现有证据。报告按“已证明事实—竞争解释—后续实验”重新分层，避免把诊断症状写成根因。

| 复盘事项 | 修订判定 | 本报告处置 |
|---|---|---|
| sidecar没有分类增量 | 已证明 | 维持`SCIENTIFIC_FAILURE_NO_PROMOTION` |
| `0.01`个百分点属于真实正收益 | 不成立 | 改为统计上不可解释的小波动，下一审计增加配对分组bootstrap |
| receiver-floor差等于信息不可恢复 | 证据不足 | 只能证明receiver×channel异质性强，尚不能区分可校正偏移与信息损失 |
| `0.70`阈值有效筛选挑战 | 不成立 | 同TX跨receiver匹配率为`99.904%`，当前阈值对正关系近似无筛选 |
| `d3_count=0`直接证明q泄漏 | 证据不足 | q泄漏和无shuffle逐batch几何都可能产生该结果，必须做完整`V_select`全局审计 |
| 训练负样本和anchor已闭合 | 不成立 | 旧历史只保存正对和rectangle；实现提交`6134e9c5`为未来run补充`negative_pairs/anchor_count/anchor_fraction`，不追溯伪造旧数据 |
| hard码本塌缩是分类主因 | 不成立 | hard code不进入下游，且soft为token概率质量、hard为packet dominant，两者不是同一统计变量 |
| C4相对C1的92.5%NMSE改善证明算子学习 | 不成立 | C1预测器没有holdout监督；公平证据只保留C4内部real相对shuffle的`2.815%` |
| `1-NMSE`是标准`R²` | 不成立 | 全文改称“归一化能量拟合分数” |
| q等于理想baseband challenge | 不成立 | 改称`received-waveform excitation proxy`，仍可能携带TX、receiver、day和位置 |
| q预训练目标数学矛盾 | 表述过强 | 改称域不变与卫星统计重建之间的目标张力 |
| q token与PA token已物理对齐 | 不成立 | `adaptive_avg_pool1d`只对齐token数量，不保证原始窗口和感受野一致 |
| exact clean–satellite已约束operator | 不成立 | 当前精确配对只约束q一致性，没有直接约束response/operator一致性 |
| DiD是挑战条件化DiD | 不成立 | 当前是packet-level TX×receiver DiD，没有按挑战区域分层 |
| sidecar与Core90冗余 | 结构事实，统计量待测 | 两者读取同一`pa_token_map`；下一审计计算base/operator四格表和oracle ceiling |
| 48码物理上错误 | 尚未证明 | 48是未经物理辨识验证的超参数；先做token占用、位置、转移和clean–satellite一致性，不强制均匀 |
| 立即加入OT、Soft-DTW或解冻Core90 | 证据不支持 | 延期，先执行冻结V2因果审计 |

## 1.研究问题与科学边界

### 1.1从内容对齐到条件响应

接收IQ可抽象为：

\[
x=\mathcal R_d\!\left(\mathcal H_d\!\left(\mathcal T_y(s)\right)\right)+n,
\]

其中，`s`是发射内容或有效激励，`\mathcal T_y`是TX硬件响应，`\mathcal H_d`是传播链路，`\mathcal R_d`是接收机响应。控制相似内容只能减少激励差异，不能令`\mathcal H_d`和`\mathcal R_d`自动消失。真正需要估计的是设备在挑战条件下的局部响应：

\[
r_t=F_{\mathrm{PA}}(h_t^{\mathrm{PA}},q_t),
\qquad
\hat\theta_{\mathrm{PA}}=A\{q_t,r_t,w_t\}_{t=1}^{T}.
\]

`q_t`在本实现中只是从接收后IQ估计的`received-waveform excitation proxy`，不是已恢复的理想基带挑战。`h_t^{PA}`来自冻结Core90的PA时序图，`r_t`是条件响应，`\hat\theta_PA`是集合聚合后的候选算子证据。同挑战用于公平比较，多样挑战用于可辨识；二者不能互相替代。

### 1.2本实验能够回答什么

本实验回答四个受限问题：

1. PA挑战条件化是否优于同容量非条件集合头；
2. DiD相对几何是否改善跨receiver最差性能；
3. 从部分token聚合的算子是否能预测原始采样区间隔离的holdout PA响应；
4. 条件证据经源域校准后是否为clean和三类LEO弱场景提供稳定增量。

它不能证明已恢复完整TX物理传递函数，不能证明M2近邻等于真实语义内容，不能把WiSig/ManySig称为真实卫星数据，也不能把Phase1 source-only结果写成Phase2 few-shot适应、新类注册或Phase3在轨确权结果。

## 2.对粘贴文本的吸收、修正与否定

### 2.1已吸收并落地的内容

| 原文核心主张 | 本研究处理 | 实际落地 |
|---|---|---|
| 内容对齐只控制激励 | 保留信道、receiver和TX—域交互，不宣称完全域不变 | 冻结Core90主干；只在侧路建模条件PA响应 |
| 局部响应允许依赖内容，最终算子才应稳定 | 将`q_t→r_t→theta_pa`分成三层 | `PAChallengeEncoder`、FiLM响应头、`OperatorPool` |
| 同挑战比较，多挑战辨识 | clean/satellite同物理视图作为M0锚点，13个局部token形成挑战集合 | 双视图、token化、集合池化 |
| 内容编码器可能泄漏TX/RX | q预训练加入TX/RX梯度反转探针，随后冻结编码器 | `adversarial_probe_logits`和`freeze_challenge_encoder` |
| 同TX跨域正对、异TX同域困难负对 | C2按冻结q相似度限制pair mask | `challenge_pair_masks`和masked SupCon |
| 直接同类拉近可能过度不变 | C3加入跨域设备差的DiD约束 | `ccoi_did_loss` |
| 分类准确率不能证明系统辨识 | C4加入原始采样区间隔离的holdout响应预测 | `HeldoutChallengePredictor`及real/shuffle/random/constant对照 |
| 集合聚合比强制全序列对齐更适合随机载荷 | 使用线性复杂度、排列不变的attention pooling | `OperatorPool`及coverage/entropy输出 |
| 多机制早期拼接可能负迁移 | 只做PA/包络侧路，最终在logit层融合 | 独立operator classifier和源域凸融合 |

### 2.2经过修正后采用的内容

- 原文的“内容匹配”在当前数据上只能实现为M2代理挑战相似度。没有可靠payload、前导或理想激励真值时，q空间近邻不能写成真实内容对齐。
- 原文建议挑战覆盖均衡，但本实现没有强迫48个码严格均匀。V2只在soft有效码数低于`0.75K`或最大soft均值概率高于`0.10`时施加hinge惩罚，避免人为制造不存在的挑战类别。
- 原文建议可辨识性门控。本轮只把coverage作为只读诊断；所有低coverage样本仍进入准确率并按错误计数，不允许通过拒绝困难样本虚增性能。
- 原文建议完整留出挑战预测。本轮在同一物理记录内使用原始采样区间不重叠的token子集，验证局部PA响应预测；它不是跨payload分布或跨物理状态的严格OOD系统辨识。

### 2.3本轮明确否定或延期的内容

| 内容 | 处理 | 原因 |
|---|---|---|
| 把内容相似解释为信道/receiver已消除 | 否定 | 与复合接收链观测模型冲突 |
| 无条件把同TX所有样本压成单点 | 否定 | 会删除条件响应并造成表示塌缩 |
| RF32/FFT96等多特征早期拼接 | 否定 | 会改变容量和归因，且容易引入域捷径 |
| Soft-DTW或大尺度时间形变 | 延期 | 可能删除SFO、PA记忆和瞬态，且当前尚未证明局部错位是瓶颈 |
| partial OT、dustbin和全token二次匹配 | 延期 | 缺少M1真实对应，无法审计选择偏差 |
| PA、IQ、CFO、相位噪声多机制同时建模 | 延期 | 首轮必须保持单机制可归因 |
| 理想波形重构后硬相减 | 否定为主路径 | 信道、均衡和解调误差可能大于真实指纹 |
| 稳定核+设备状态子空间 | 延期 | 当前没有跨上电、重载、温度等真实状态标签 |
| 用目标域、query或receiver truth调融合与门控 | 否定 | 违反Phase1 source-only和独立评分边界 |

该取舍属于“原始路线的最小可证伪近似”，不是严格全量设计等价。严格等价仍需要M1语义匹配、独立probe、跨挑战OOD、多机制逐项消融和真实设备状态数据。

## 3.V1结果暴露的问题与V2修复假设

V1五行训练和1,632,000条/row预测均正常，说明执行链路健康；但C2–C4相对C1的LEO均值变化不足`0.01`个百分点。日志和artifact进一步暴露三项可定位问题。

### 3.1receiver元数据没有真正进入prediction

WiSig批次移动后形成`extra=(domain_tensor,meta_mapping)`，V1只在`extra`本身是mapping时读取`rx_i`，最终receiver回退为`-1`。总体准确率仍可计算，但receiver-floor是虚假的。V2递归读取tuple/list中的明确`meta.rx_i`；runner在receiver缺失或为负值时硬失败，scorer也拒绝主loader中的未知receiver。

### 3.2算子logit被量纲和小alpha共同压弱

V1实际融合为`base+alpha×operator`。新初始化算子logit的幅度小于冻结基线，`alpha≤0.2`进一步压缩其作用；所以“C2≈C1”既可能表示条件信息无效，也可能表示旁路没有真正进入判决。V2让operator classifier直接用自身logit训练，并在`V_cal`上标定两路去中心RMS比例，再做凸融合。

### 3.3soft均衡不能代表packet dominant占用

V1的48码已有明显集中。V2增加soft有效码数和最大均值概率约束，假设这能改善挑战覆盖。旧审计中的soft统计汇总所有token概率质量，hard统计却先对每包token概率求均值再取单个dominant code，因此`35.216/48`与`4/48`不是同一随机变量的soft/hard版本。该结果证明packet dominant高度集中，但不能证明token级码本塌缩，更不能将其直接写成分类零收益的根因。

## 4.方法设计

### 4.1总体结构

```text
收到的source IQ x
├─内容视图：包级去均值/RMS归一化→64点token→q_t(32维)+48码soft分配
└─指纹视图：冻结Core90→PA时序图h_t^PA
                         q_t条件化FiLM
                              ↓
                       局部响应r_t(64维)
                              ↓
             attention集合池化→theta_pa(64维)
                              ↓
                      operator logits

冻结Core90 base logits ──源域V_cal尺度标定凸融合──→最终logits
```

冻结主干保证C0是原有Phase1证据，C1–C4侧路保持相同参数容量。新增侧路没有保存样本级source状态，prediction完成后才写truth并交给独立scorer。

### 4.2双视图、token与挑战编码

内容视图只做包级去均值和RMS归一化，禁止逐token单位幅度归一化，从而保留包内相对幅度、PAPR和PA工作点变化。指纹视图保持原始IQ并沿用Core90处理。长度256的IQ按窗口64、步长16切分：

\[
T=\left\lfloor\frac{256-64}{16}\right\rfloor+1=13.
\]

每个token由两层一维卷积编码为32维L2归一化挑战向量`q_t`，再输出48维soft码本概率。固定、stop-gradient的9维内容统计包括I/Q均值和标准差、RMS、PAPR、幅度差分、幅度一阶相关和相位创新。

挑战编码器在`L_s/U_s`的clean—satellite同物理视图上预训练。总损失为：

\[
L_q=L_{cons}+L_{mask}+L_{temp}+0.1L_{var}+0.25L_{code-cons}
+0.2L_{balance}+0.005L_{conf}+0.1L_{TX-GRL}+0.1L_{RX-GRL}.
\]

`L_mask`从邻域预测被遮挡token的固定统计，`L_temp`预测下一非重叠锚点统计，`L_var`抑制连续q坍塌。`U_s`不读取TX真值。预训练结束后q编码器被冻结，避免匹配器与身份分类器端到端共谋。

### 4.3条件PA响应与设备算子

冻结Core90输出PA时序图`h_t`。q经线性层产生FiLM参数：

\[
r_t=W_ph_t\odot(1+\tanh\gamma(q_t))+\beta(q_t).
\]

FiLM权重以小增益初始化，使侧路从接近非条件映射开始。C1使用可学习常量条件，C2–C4使用真实q；因此C1与C2的差异主要是挑战条件和pair mask，而不是侧路容量。

`OperatorPool`在`[r_t,q_t]`上预测token权重，并对`r_t`的value映射做排列不变加权求和，得到64维`theta_pa`。它同时输出coverage和attention entropy，但本轮不删除任何低权重样本。

### 4.4配对几何、DiD与holdout

C1使用普通同TX跨domain正对和异TX同domain负对。C2使用q余弦相似度`≥0.70`限制相同关系，形成M2代理挑战匹配SupCon：

\[
L_{side}=L_{operator-cls}+0.15L_{pair}.
\]

C3增加`0.10L_DiD`，保持设备差向量在不同domain中的方向和尺度，而不是强迫同TX表示完全重合。C4再增加`0.20L_holdout`：从原始采样区间不重叠的support token估计`theta_pa`，结合holdout q预测冻结PA目标。real、shuffle、random和constant q在同一`V_select`上复算NMSE，用于判断模型是否真正使用条件。

### 4.5训练、选择与融合

Core90始终冻结。q预训练10个epoch；C1–C4各训练同容量sidecar 20个epoch，并按source `V_select`的operator accuracy选择sidecar状态。融合只在source `V_cal`完成。对每个样本去类别均值后的logit计算RMS：

\[
s=\operatorname{clip}\left(\frac{RMS(\ell_{base})}{RMS(\ell_{op})},0.25,20\right),
\]

\[
\ell=(1-\alpha)\ell_{base}+\alpha s\ell_{op}.
\]

`alpha∈{0,0.05,0.10,0.20,0.35,0.50}`，并列选择更小alpha。C0强制`alpha=0`并逐logit复现冻结基线。目标接收机、LEO测试结果和独立scorer指标均不参与训练、选模或校准。

### 4.6C0–C4归因矩阵

| row | 冻结Core90 | 同容量sidecar | 真实q条件 | q匹配SupCon | DiD | holdout预测 | 主要问题 |
|---|---|---|---|---|---|---|---|
| C0 | 是 | 否 | 否 | 否 | 否 | 否 | 冻结控制性能 |
| C1 | 是 | 是 | 否，常量条件 | 否，普通pair | 否 | 否 | 普通增参和集合头收益 |
| C2 | 是 | 是 | 是 | 是 | 否 | 否 | 条件匹配的独立贡献 |
| C3 | 是 | 是 | 是 | 是 | 是 | 否 | 相对几何是否改善floor |
| C4 | 是 | 是 | 是 | 是 | 是 | 是 | 局部算子预测是否成立 |

## 5.代码落地

| 文件 | 已落地内容 | 运行证据 |
|---|---|---|
| `code/model.py` | 在不新增主干参数的情况下暴露`pa_token_map` | 真实checkpoint严格加载，195个state tensor无missing/unexpected/mismatch |
| `code/cvsrffi/ccoi_pa.py` | 双视图、13-token、9维固定统计、q32/code48编码器、GRL探针、FiLM响应、集合池化、隔离holdout、soft码本正则 | 组件测试及真实q/sidecar artifact |
| `code/cvsrffi/ccoi_losses.py` | 普通/挑战pair mask、masked SupCon、DiD和`d1/d2/d3`诊断 | 合成几何测试及source audit |
| `code/train_phase1_ccoi_pa.py` | 冻结Core90重建、协议校验、q预训练、C0–C4训练、V_select选模、V_cal融合、四场景prediction与后置truth | 五行各1,632,000条prediction/truth |
| `code/score_phase1_ccoi_pa.py` | prediction闭合后连接truth；按场景、loader和receiver评分；拒绝无效receiver | 五个`metrics.json`均为`ANALYZED` |
| `code/scripts/launch_phase1_ccoi_pa_v2_20260825.sh` | 真实checkpoint no-query smoke作为launcher第一步，PASS后立即运行C0–C4并调用独立scorer | 监督日志完整出现SMOKE、PREDICTIONS和ANALYZED标记 |
| `code/tests/test_ccoi_*.py`、`test_phase1_ccoi_pa_*.py` | 双视图、token、梯度、holdout隔离、公式、receiver、scorer、协议和launcher负测 | `ssr-gpu`中35项聚焦测试通过 |
| `docs/experiments/PHASE1_CCOI_PA_V2_CONFIG_20260825.json` | 冻结矩阵、损失、码本、融合、source roles和truth连接方式 | dry-run与manifest读回一致 |

实现没有修改默认`train_ssdg.py`训练入口，没有把sidecar强制注册进所有Phase1模型，也没有将source样本级embedding写入sidecar checkpoint。V2侧路是可关闭、容量匹配、可独立归因的实验实现。

## 6.已实现与未实现边界

| 设计项 | 最终状态 | 说明 |
|---|---|---|
| PA单机制条件响应、集合算子、C0–C4、尺度融合、receiver评分 | 已实现并由N607验证 | 属于本次严格交付范围 |
| M0同物理clean/satellite锚点 | 已实现 | 同一物理样本的多视图不增加K |
| M2冻结q代理匹配 | 已实现 | 不能称为真实内容匹配 |
| M1真实payload/前导匹配precision/recall | 未实现，数据证据阻塞 | 当前最高风险缺口 |
| 独立冻结q的TX/RX/day probe最终数值 | 未闭合 | 训练存在GRL探针，但没有形成完整独立probe artifact |
| 二值ObservabilityGate | 未实现 | 当前只报告coverage，不拒绝样本 |
| 严格跨挑战分布OOD | 延期 | 当前holdout是同物理记录内原始区间隔离 |
| Soft-DTW、partial OT、多机制算子、状态子空间 | 延期 | 需要前序最小假设成立后逐项进入 |
| 完整`d1<d2<d3`实证 | 部分完成 | C4中`d1=1.2934<d2=2.2932`，但`d3_count=0`，不能声称完整排序成立 |
| margin retention、`rho_int`、单包/多包曲线 | 未闭合 | 不用于本轮晋级结论，必须在后续严格算子声明前补齐 |

因此，代码对V2最小修复和C0–C4矩阵是严格落地；相对于粘贴文本的完整条件系统辨识路线，它是有意缩小的PA-only近似。

## 预登记

- 状态：`ANALYZED`；工程闭环`VERIFIED`，科学结论`SCIENTIFIC_FAILURE_NO_PROMOTION`。
- 候选：`CCOI-PA-V2`，单seed最小矩阵`C0/C1/C2/C3/C4`。
- 科学对照：冻结`ADV3B02_CORE90_SOFT_E200`；沿用V1的同split、同seed、同训练/评估预算和四场景，C1–C4保持同容量。
- 修复范围：原始`meta.rx_i`接收机导出；算子独立分类与源域`V_cal`尺度对齐凸融合；有界码本有效数/集中度正则。Core90、source roles、场景和目标/query边界不变。
- Git实现提交：`8a959d00da768d1134ce859bd366052f4ea9c109`，分支`codex/phase1-ccoi-pa-v1-20260824`，远端OID已独立核对一致。
- 主要文件：`code/train_phase1_ccoi_pa.py`、`code/score_phase1_ccoi_pa.py`、`code/cvsrffi/ccoi_pa.py`、V2 launcher/config、聚焦测试及V2设计/追踪报告。
- 本地验证：`ssr-gpu`中35项CCOI聚焦测试通过；三个生产Python文件语法编译通过；C0–C4 dry-run通过；一次定点P0/P1检查闭合。
- 本机Git Bash：既有路由证据为`FAILED`，未在错误Bash通道执行launcher；发布后在N607运行`bash -n`。
- 源域协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`rho_label≤0.1`；目标域、query、query role和query truth不进入训练、校准或选择。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别输出。
- seed：训练和卫星扰动均固定为`20260824`，用于与V1同row比较。
- N607环境/CWD：普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00`。
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- GPU：`0`；2026-08-25 01:05直连预检显示8张RTX 3090利用率均为0、显存占用1MiB，无compute app和`train_phase1_ccoi_pa.py`进程。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A`；smoke使用同名`_REAL_CKPT_NO_QUERY_SMOKE`独立不可覆盖根。两者预检均不存在。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A.out`；launcher监督日志使用同run ID独立文件。
- release归档：`E:\type10-7\release_archives\phase1_ccoi_pa_v2_8a959d00.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00.tar.gz`；本地与远端SHA256均为`976bfe2919f4632e5b5b277b915ec418c7754866c9d2a058859439429eab5628`，传输状态`VERIFIED`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00 && ROOT=$PWD CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_20260825 RUN_ID=PHASE1_CCOI_PA_V2_S20260824_20260825A GPU=0 bash code/scripts/launch_phase1_ccoi_pa_v2_20260825.sh`。
- 预期artifact：`protocol_and_smoke.json`、挑战预训练历史、每row校准、sidecar、challenge audit、`prediction.jsonl`、后置`truth.jsonl`、`metrics.json`、matrix manifest和完整日志。
- 直接技术停止规则：仅在协议/query泄漏、错误checkout/CWD/run-root/GPU、输出碰撞、无法启动、prediction无法闭合，或至少两个row出现相同确定性预prediction异常时停止；不因低准确率停止。只处理该run绑定的进程树并保留全部局部artifact。
- 科学门槛：C2或更高row相对C1的LEO均值和receiver-floor分别至少提升0.30个百分点；clean下降不超过0.50个百分点；C4 holdout NMSE相对C1下降至少5%且归一化能量拟合分数`1-NMSE`大于0。未过线记负结果，不中止健康运行。
- 新run授权：本报告仅授权唯一run ID `PHASE1_CCOI_PA_V2_S20260824_20260825A`；不得重复启动或覆盖旧run。

## 运行更新

- 2026-08-25 01:07：release归档已同步；远端SHA256读回为`976bfe2919f4632e5b5b277b915ec418c7754866c9d2a058859439429eab5628`，与本地一致，传输状态`VERIFIED`。
- release目录已新建且未覆盖旧目录；三个V2生产Python文件远端编译通过，三个对应`.pyc`均完成独立读回；launcher远端`bash -n`通过。
- 启动前资源再次确认：无`train_phase1_ccoi_pa.py`进程、无NVIDIA compute app，目标run和smoke根均不存在。
- 2026-08-25 01:09：唯一launcher PID`2500917`已启动，PPID为1，CWD为release目录；正式训练PID`2501324`，完整cmdline、run-root、seed、GPU0和日志路径均与预登记一致。
- smoke已先行通过并由`protocol_and_smoke.json`独立读回：`Phase1_source_only`、source roles为5,880/52,920/12,600/12,600，比例`0.07/0.63/0.15/0.15`，`rho_label=0.1`；源/目标receiver交集为0；checkpoint严格加载`missing/unexpected/mismatch=0/0/0`，195个state tensor；PA图`[64,64,64]`、logits`[64,6]`且有限；`target_or_query_access=false`。
- launcher日志已依次出现`REAL CHECKPOINT NO-QUERY SMOKE`、`[CCOI-SMOKE] PASS`、`[CCOI-V2-SMOKE] PASS`和`FULL MATRIX`，正式矩阵已健康进入训练。
- 启动后GPU0出现另一条先于本run启动的无关meta-adapter训练PID`2498587`，占约486MiB；本run占约620MiB，两条训练进程未超过每GPU允许上限。该无关进程不属于本run，不做任何干预。

## 完成状态与完整性

- 2026-08-25 02:22：唯一launcher自然退出；监督日志依次出现`[CCOI-PREDICTIONS] COMPLETE`、`[CCOI-SCORE] ANALYZED`和`[CCOI-V2-LAUNCH] ANALYZED`，没有重复启动、强制停止或输出覆盖。
- C0–C4每行均有1,632,000条prediction和1,632,000条后置truth；远端`wc -l`独立读回共16,320,000行。五个`metrics.json`均为`ANALYZED`，且`truth_joined_after_prediction=true`。
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四场景全部闭合；receiver键为真实`0–11`，不再出现V1的`-1`聚合，receiver导出修复得到真实实验验证。
- 主日志与监督日志未发现Traceback、OOM、NaN/Inf、exception、killed、invalid receiver、target/query越权等异常指纹。运行健康性结论为`VERIFIED`。

## 同row结果

所有数值均为百分比；`LEO-floor均值`是三个LEO场景receiver-floor的算术均值。

| row | clean | clear | low-elev | rain | LEO均值 | LEO-floor均值 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 90.1402 | 78.4564 | 75.5686 | 75.1892 | 76.4047 | 56.8639 |
| C1 | 90.1235 | 78.4456 | 75.5338 | 75.1770 | 76.3855 | 56.8736 |
| C2 | 90.1255 | 78.4544 | 75.5490 | 75.1804 | 76.3946 | 56.9097 |
| C3 | 90.1201 | 78.4397 | 75.5422 | 75.1804 | 76.3874 | 56.8722 |
| C4 | 90.1250 | 78.4583 | 75.5431 | 75.1833 | 76.3949 | 56.8986 |

相对C1：C2的clean、LEO均值、LEO-floor均值分别变化`+0.0020`、`+0.0092`、`+0.0361`个百分点；C3分别为`-0.0034`、`+0.0020`、`-0.0014`个百分点；C4分别为`+0.0015`、`+0.0095`、`+0.0250`个百分点。clean保护全部通过，但LEO均值和receiver-floor的`+0.30`个百分点门槛全部未通过。

## 机制诊断

- q预训练10个epoch完整结束；总损失由`0.4697`降至`0.3113`，masked/temporal/variance损失分别降至约`0.00566/0.00553/0.00876`，未出现数值异常。
- V2 soft effective codes为`35.216/48`，最大soft均值概率为`0.0994`；packet dominant code只出现`4/48`个，计数为`30/11/8886/3673`，最大单码占`70.52%`。这只能证明包级dominant统计集中。token-level hard occupancy、每位置占用、每包code数和转移矩阵尚未计算，所以旧报告关于“离散挑战码塌缩”的因果表述撤回。
- C4真实holdout NMSE为`0.12593`，归一化能量拟合分数`1-NMSE=0.87407`。C4相对C1下降`92.50%`主要比较了受holdout监督预测器与未受该监督的预测器，不能作为设备算子证据。公平的C4内部对照中，真实配对相对shuffle只改善`2.815%`（shuffle NMSE=`0.12957`）。
- C1–C4的`V_cal`自动选择均为`alpha=0.1`，融合尺度约`1.416–1.502`；尺度修复使sidecar不再因量纲失配而数值失活，但最终分类增益仍只有约`0.01`个百分点，因此“融合失活”不是唯一瓶颈。

### 机制指标明细

| row | alpha | scale | real NMSE | shuffle NMSE | real相对shuffle | 解释 |
|---|---:|---:|---:|---:|---:|---|
| C1 | 0.10 | 1.4587 | 1.67857 | 1.67857 | 0.000% | 常量条件下shuffle无影响，符合对照预期 |
| C2 | 0.10 | 1.4853 | 1.46756 | 1.46772 | 0.0109% | 条件匹配带来的q特异性极弱 |
| C3 | 0.10 | 1.4160 | 1.48406 | 1.48421 | 0.0101% | DiD没有增强q依赖 |
| C4 | 0.10 | 1.5023 | 0.12593 | 0.12957 | 2.815% | holdout训练显著降低NMSE，但真实q只小幅优于shuffle |

C4的random和constant对照NMSE分别为`0.20387`和`0.18108`，说明预测器并非完全忽略条件输入；但shuffle只破坏样本—q对应关系而保持q分布，真实q仅领先2.815%，仍不足以证明稳定的样本级挑战辨识。

C4的条件距离为`d1=1.2934,d1_count=213,196`、`d2=2.2932,d2_count=204`、`d3=N/A,d3_count=0`。同TX跨receiver关系中`0.70`阈值的匹配比例为`213196/(213196+204)=99.904%`，说明阈值近似无筛选。`d3_count=0`既可能来自q的TX泄漏，也可能来自无shuffle的逐batch组成；因此旧结果不能支持`d1<d2<d3`，也不能单独判定泄漏根因。

C4按attention保留100%、75%、50%、25%token时，source `V_select`准确率分别为`98.4048%/98.3968%/98.3889%/98.3968%`。降低coverage没有带来增益，说明当前attention既没有发现强少数证据，也没有通过只选容易token虚增准确率；它更接近近似均匀聚合。

### receiver-floor明细

| row | clean floor | clear floor | low-elev floor | rain floor |
|---|---:|---:|---:|---:|
| C0 | 81.0375 | 58.5792 | 56.0292 | 55.9833 |
| C1 | 81.0667 | 58.5833 | 56.0375 | 56.0000 |
| C2 | 81.0750 | 58.6167 | 56.0875 | 56.0250 |
| C3 | 81.0250 | 58.5708 | 56.0708 | 55.9750 |
| C4 | 81.0708 | 58.6125 | 56.0708 | 56.0125 |

receiver-floor修复后可以确认，瓶颈集中在低抬升和雨衰弱场景，最差receiver准确率约56%。C2对C1的单场景floor改变量最大也只有`+0.0500`个百分点；DiD没有形成稳定的最差receiver保护。

## 暴露的问题

### 问题1：旧hard审计口径不能证明token码本塌缩

V2优化的是批次平均soft概率的熵和最大值。旧hard histogram则对每个包的13个token概率先求均值再取argmax，最终4个packet dominant code承担全部12,600个`V_select`样本，其中单码8,886个。后续既不能把soft effective codes当作挑战覆盖成功证据，也不能把packet dominant集中写成token级collapse；两类统计必须在同一token随机变量上重新计算。

### 问题2：可学习响应不等于可判别身份增量

C4的NMSE和归一化能量拟合分数表明模型可以根据support表示和q预测局部PA图，但这项能力没有转化为LEO分类或receiver-floor提升。预测目标可能主要包含所有设备共有的包络结构；预测器也可能主要利用q的分布信息。只有q-only、θ-only、正确/打乱/其他TX/同TX跨receiver/跨day的同容量分解才能判断θ是否包含设备稳定增量。

### 问题3：真实q的样本对应关系贡献过弱

C2/C3真实q相对shuffle只改善约`0.01%`NMSE，C4也只有`2.815%`。如果系统真正依赖“当前挑战—当前响应”配对，打乱q应造成更明显损伤。当前结果更像q提供了总体激励分布或位置先验，而不是稳定的样本级挑战身份。

### 问题4：融合修复后仍无分类收益

四个sidecar row均选择非零`alpha=0.1`，说明`V_cal`上存在极小正差；但最佳选择只相当于约1–2个样本的变化，LEO测试没有相应增益。尺度失配已被排除，继续手工放大alpha只会用源域微小波动覆盖负结果。

### 问题5：DiD所需几何在实际batch中证据不足

实际audit没有产生`d3`关系，意味着当前评估batch和匹配阈值无法同时提供足够的“异TX、同domain、同挑战”对。DiD代码和合成测试正确，不等于真实数据上的四元组几何充分。后续应先提高可审计的pair coverage，而不是增加DiD权重。

### 问题6：原始路线仍有关键证据缺口

M1真实语义挑战匹配、冻结q独立TX/RX/day probe、严格跨挑战OOD、margin retention、`rho_int`和单包/多包性能尚未闭合。因此本实验只能否定当前PA-only M2实现，不能否定“条件系统辨识”这一更广泛研究方向，也不能宣称完整路线已经实现失败。

## 科学判定与后续路线

- 工程判定：`VERIFIED`。V2真实checkpoint smoke、receiver修复、源域尺度校准、四场景prediction/truth和独立评分均完整闭合。
- 科学判定：`SCIENTIFIC_FAILURE_NO_PROMOTION`。虽然clean约束和C4 holdout拟合门槛通过，但C2/C3/C4对C1的LEO均值与receiver-floor增益均远低于`+0.30`个百分点，不晋级多seed或完整确认。
- 否定的解释：不能再把失败归因于receiver缺失或单纯融合尺度过小；两项已修复且真实验证。当前数据只证明条件信息对分类决策增量极弱，尚未定位为hard code、q泄漏、公共PA目标或sidecar冗余中的单一根因。
- 下一步不是锐化48码或重新训练分类矩阵，而是复用冻结Core90与C4完成`q泄漏→全局pair geometry→H0–H6/残差HR→base/operator互补性`因果审计。只有θ相对q-only、shuffle和其他TX均取得至少5%误差改善、分组置信区间不跨零，且source synthetic LEO oracle gain达到`0.30`个百分点，才设计残差V3。

对应实现提交为`6134e9c5fe11b3cbd01ea906eaab2fe1ed64f2a3`。它不修改旧V2权重、不重复C0–C4，只增加source-only冻结审计runner、token/packet双口径统计、完整`V_select`全局pair扫描、分组bootstrap、H0–H6与cross-fit残差HR小头、互补性四格表及不可覆盖launcher。若任一预登记停止条件失败，PA-M2路线停止，不进入Soft-DTW、OT、低秩解冻或多机制扩展。

## 最终结论

本轮工作完整实现并验证了一个与现有Phase1有机结合的最小PA条件算子侧路：它复用冻结Core90的PA时序响应和强基线，用source-only挑战编码、条件FiLM、集合聚合、DiD、holdout预测和晚期校准构成可关闭的实验链。receiver和融合接口已被真实实验修复，全部数据、prediction、truth、场景和评分artifact正常。

科学上，当前实现没有超过冻结基线，也没有达到相对同容量控制的LEO和floor门槛。最可信的结论是：局部PA图可预测不等于获得TX特异算子，当前sidecar也没有证明能纠正Core90错误。packet dominant code集中只是症状；q是否泄漏、θ是否跨记录稳定、PA目标是否主要由q-only公共映射解释，以及sidecar的oracle ceiling仍需冻结因果审计回答。

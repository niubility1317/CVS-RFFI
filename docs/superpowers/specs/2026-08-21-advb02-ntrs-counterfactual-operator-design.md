# ADVB02 NTRS-V4反事实条件算子设计

## 1.目标与边界

本版本在ADV3B02 CORE90成熟D1 checkpoint上继续采用adapter-only训练，身份骨干、域骨干和共享CosFace头保持冻结。训练与最终测试均只使用`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，seed固定为`392034`，不使用历史`mixed_orbit`。

本版本只解决Phase1中可模拟、可成对监督的星地信道扰动，不宣称消除完全未见接收机链路，也不引入Phase2目标域状态。

## 2.被淘汰的旧假设

A1/A2同checkpoint证据表明，当前`40维绝对统计q＋rank-8类共享加性平移＋平均CE`的修正方向不可靠：三场景均为负收益，Strict UDU退化更大，且`harmed>rescued`。因此不继续放大alpha，也不直接训练support gate。

## 3.新结构

### 3.1条件上下文

- `descriptor_normalized`：40维分组物理描述符先按固定分组尺度归一化，再编码为`q_iq`。
- `metadata_teacher`：训练时使用仿真器提供的9维归一化metadata编码`q_meta`，以stop-gradient教师方式监督`q_iq`。
- 推理只依赖IQ描述符；metadata教师不成为测试时必需输入。
- 提供`constant`、`shuffled`、`random_feature`对照，区分条件信息是否真正按样本起作用。

### 3.2q条件低秩算子

旧公共平移：

`delta=U a(q)`

新算子：

`delta=U[a(q) * (V^T z)]`

其中`z`为detach后的成熟身份嵌入，`U,V`为低秩正交基，`a(q)`末层零初始化。该结构允许同一信道条件对不同TX嵌入产生不同但低秩受限的旋转、缩放或剪切。

### 3.3反事实风险目标

- `pair_shift`：直接监督`delta`逼近同物理样本的`z_sat-z_clean`。
- `pair_cosine`：约束`z_sat-delta`接近`z_clean`方向。
- `harm`：对raw卫星预测已正确的样本，惩罚robust margin下降。
- `rescue`：仅对clean raw正确且satellite raw错误的信道诱发错误，推动robust margin转正。
- `clean_tail`：采用相对修正hinge/CVaR风格尾部约束，目标clean p95不超过0.2%；训练clean分支可配置exact bypass。
- `q_distill`：仅metadata有效时监督`q_iq`接近stop-gradient的`q_meta`。

`lambda_harm`必须大于`lambda_rescue`，防止多数raw正确样本被牺牲。

## 4.B0先行诊断

在source calibration pair上先运行无训练诊断：

- paired shift在rank4/8/16/32下的PCA累计解释率；
- rank投影transport error；
- full shift和rank投影oracle的rescued/harmed；
-当前A1/A2方向上`g=0,0.05,...,1`连续oracle上限；
-公共方差与TX×nuisance交互方差；
-q的有效秩、饱和率、metadata probe和TX泄漏。

B0是机制诊断，不读取query truth，不更新模型，也不替代最终独立测试。

## 5.实验矩阵

| 行 | 结构 | 唯一变化 |
|---|---|---|
| B0-PCA | 无训练诊断 | paired shift低秩可修正性 |
| B0-C | v4 operator | constant q |
| B0-S | v4 operator | batch-shuffled q |
| B0-R | v4 operator | frozen random-feature q |
| B1-M | v4 operator | metadata教师监督q |
| B1-N | v4 operator | 分组归一化IQ q，无metadata蒸馏 |
| B2-A | additive PCA | 冻结PCA公共平移 |
| B2-O | q-conditioned operator | 条件低秩算子 |
| B3-R | B2-O | harm/rescue反事实风险目标 |

首轮使用seed`392034`完成同row最小矩阵。B4 gate只在B3-R满足正净救回和连续oracle条件后运行；B5-RX属于Phase2，不在本轮发布。

## 6.晋级标准

- raw backbone/head参数与输出严格保持；
- LEO均值相对同checkpoint raw提升至少1.0pp；
- clean下降不超过0.5pp；
-三种LEO_WEAK单场景下降均不超过0.5pp；
- Strict UDU不下降；
- `rescued>harmed`且翻转精度大于50%；
- clean相对修正p95不超过0.2%或clean exact bypass；
- q相对constant/shuffled/random-feature有稳定增益；
- 最终检查点完成clean和三种LEO_WEAK独立测试。

## 7.协议冲突处理

外部分析中“正式轨继续使用mixed_orbit”的建议与当前`项目.md`冲突。按项目源事实执行：从2026-08-20起Phase1训练和测试默认均为三种LEO_WEAK，`mixed_orbit`仅保留为历史实验口径。

# 固定 Phase1 的信息保留审查

日期：2026-09-29。范围：代码与理论可行性。未读取任何 query IQ、query 成绩、结果段或评分索引，未运行新实验、加载实际 checkpoint、访问 source 样本或逐样本 source feature。仅新增本文件，不修改已冻结方法。

## 1. 结论

**可以推进一项逐 row 的 support-only 增量信息验证，暂不冻结新分类器。** 最有直接代码依据的检查对象是：同一冻结 identity backbone 已计算、但现有导出丢弃的融合前分支表示。它无需重训 Phase1 或读取 source，也不只是对现有 identity160 做新的可逆变换。

这与“已证明存在可用于 TX 判别的新信息”不同。宽表示被压缩到 160 维，说明有结构上的压缩瓶颈；实际 support 所在低维集合上是否丢掉有用差异、被丢弃部分是否主要是 RX/channel 信息，必须分别验证。K1 仍没有独立类内物理重复样本，不能据一个 support 的多 view 或多个中间层构造独立验证。

FFT96 明确丢弃了谱相位、绝对 DC/能量及部分频率分辨率，但 identity 分支本身会读取时序、复数 I/Q 和非线性结构。不能把 FFT96 的丢弃项泛化为整个 identity+FFT descriptor 完全没有这些信息。也不能把已有 RF32/高阶矩改名当成全新物理机制。

## 2. 已核对的原生路径与证据边界

`tools/cvs_native_artifacts.py:54` 以后调用原生 `build_exact_ssdg_model_from_checkpoint` 和 `identity_only_feature_forward`，输出 normalized z_id160 与 logits。原生 `cvsrffi/checkpoint_loading.py:32` 以后从 checkpoint args 重建模型，检查 missing/unexpected keys；本次只读该代码，没有实际读取权重或模型输出。

`experiments/adv3b02_xuc/code/cvsrffi/identity_only_forward.py:33` 已调用 `id_backbone(...,return_aux=True)`，第 40 行附近只取 `_pick_z_id(aux_id)` 与 logits。原生 `model.py:1875` 至 1893 的 aux 同时包括：

- `t_emb`、`f_emb`：池化/投影后的时间与频率分支；
- `dac_local`、`pa_local`：相应局部分支，关闭的分支为零；
- `base`、`feat_cls`、`feat_dac`、`feat_pa`、`feat_imp`、`feat_joint`：不同融合层级的表示。

实际训练 checkpoint 的 flags/width 在本审查中仍为 **UNKNOWN**。静态配置链只支持以下推导：matched recipe 没有覆盖 `model_variant/branch_ablation/id_feature_key`，而 `SSDG/train_ssdg.py:1164`、1178 至 1184 默认 `lite_d/no_dac/feat_joint`；`model.py:2003` 至 2020 的 lite_d 对应 emb_dim=160。该推导不能替代 exact loader 的实际模型属性核对。

因此不能把 native 代码中存在的 DAC 分支当作当前已训练的可用补充。按静态推导，当前候选主要是启用的时间、频率和 PA 路径及其融合前状态；未来应核对实际 `use_*`、`id_feature_key` 和输出宽度。不得为了获得新维度启用未训练分支、改变权重或加载其他 checkpoint。

## 3. 当前各层到底保留或丢弃什么

### 3.1 FFT96 的确定性丢弃

`code/cvsrffi/stage2_diag_cosine_exploration.py:191` 的 `spectral_logmag_sketch` 顺序为：原始 received IQ 转复数，减复均值，按 RMS 归一化，乘 Hann 窗，FFT，取幅度并 log1p，将 256 点插值为 96 点，减谱均值，再单位归一化。

由公式可确认：

- 幅度操作不保存该 windowed signal 的复谱相位；
- 减复均值不保留原始 DC；RMS 与末尾单位归一化不保留全局增益和绝对谱量级；
- 256→96 的插值压缩不保证可逆，尤其不能保证保留窄带局部细节；
- 全记录 FFT 汇总不是时间定位表示，不能从 FFT96 恢复同一频率结构发生于哪个时间位置。

注意 Hann 窗和有限边界使这条完整 descriptor 不能被概括为严格的时间循环移位不变。全局复相位旋转的幅度不变性在非退化数值范围内成立，但 CFO 是随时间变化的相位斜坡，会移动或改变加窗频谱，不会被取幅度自动消掉。

### 3.2 identity160 并非只有功率谱

原生 `model.py:1690` 至 1719 的时间路径读取配对 Sinc I/Q、可选非线性基和高频差分，再经一般实 Conv、GroupNorm、ReLU、时序卷积、池化与投影。它不是理论相位不变映射。`model.py:1484`、1565 以后的频率路径主要从功率/正负频谱比/不对称性压缩形成特征，但仍与时间路径联合进入 base。

PA 路径 `model.py:622` 的 `MemoryPolynomialLift` 包含带延迟的 `z*|z|^(p-1)` 类基函数，后接 envelope gate、扩张卷积与池化；它具备响应幅相耦合和有限记忆结构的能力。原生代码中的 widely-linear DAC 路径也会处理共轭分量，但静态配置推导为关闭，不能计入本次实际可用信息。

这些结构说明模型**可能**编码相位、局部时序及非线性响应；不能单凭名称证明最终 160 维保留了某项统计，更不能反向证明它一定丢弃了某项统计。`PhaseDeltaStabilityStem` 虽在代码中实现显式相位增量，默认 off；不应把一个未启用模块算作当前模型保证。

射频前端非线性记忆可以提供设备区分依据，已有原始研究专门分析匹配滤波与 PA 级联的记忆效应；这只支持核查相关冻结分支的物理动机，不能证明当前 received 数据上存在可迁移的 TX 增量。[Radio Frequency Fingerprinting Exploiting Non-Linear Memory Effect，IEEE TCCN 2022](https://ieeexplore.ieee.org/document/9913208/)

### 3.3 融合与导出的具体瓶颈

`model.py:834` 的 `_compute_features_for_head` 先形成 `feat_id`、可用的 defect features，并可对 identity 施加 gate；第 860 行附近把它们拼接后交给 `joint_proj`。`joint_proj` 在第 798 行附近定义为：

`Linear((1+defect_count)*emb_dim,emb_dim) -> ReLU -> Dropout`。

若 checkpoint 的 id_feature_key 为 feat_joint，最终返回的是这一步的 160 维结果，后续导出还做 L2 归一化。若输入宽度大于 160，线性层在完整输入空间必有非平凡 nullspace；ReLU 和归一化又增加多对一可能性。这是数学上的全空间压缩事实。

它不证明真实 support 流形穿过该 nullspace，也不证明相关分支存有最终 TX head 未使用的正确判别信息。必须分开记录：

1. 模型/导出结构存在更多冻结表示，且当前 cache 未存储它们；
2. support 上这些表示是否在数值上非零、非重复；
3. 其非重复部分是否改善 held physical support 的类判别，而不是主要追踪 receiver/channel。

当前仅第 1 项有代码证据，第 2、3 项尚未测量。

## 4. 单条 received IQ 的物理解释不能越界

实际残差路径 `experiments/adv3b02_xuc/code/leo_practical/residual.py:54` 至 98 包含 residual frequency、tracking phase、widely-linear image 和噪声；第 109 行附近只施加有界标量 AGC。这意味着 received 的相位/幅度结构混合了 TX、原接收机、传播和新增接收残差，不能直接叫纯 TX fingerprint。

单条 IQ 可计算相位差、复相关、非圆性或高阶幅相统计；它们不会增加物理样本数。对特定变换具有不变性也不等于能分离 TX 与 channel。例如相邻时刻的四阶闭合量

`z[n+1]*z[n-1]*conj(z[n])^2`

在共同增益归一化后可消去全局相位及线性相位斜坡，因为相位指数相消；但它仍受调制、滤波、加性噪声、非线性 phase tracking 和 IQ image 影响。该公式只用于说明 FFT magnitude 与联合时序高阶结构不同，不在本审查中立为新分类器或决定 lag 网格。

同样，256 点片段之外的真实开机瞬态无法恢复。没有 packet-start/完整符号边界证据时，不能把当前短窗解释为标准化的发射瞬态。循环时间移位还会改变有限窗边界；它不是无需说明就合法代表另一次物理信道实现的手段。

## 5. 与已有实现逐项去重

| 已有方法/代码 | 已使用的信息或作用 | 本审查排除的重复主张 |
|---|---|---|
| SFHead | identity160、support 标签、旧类 support teacher 条件项 | 换判别头不能恢复原出口未保留的分支 |
| SGJoint | identity+FFT、固定 ground 域变化归一化、support covariance | 改写 covariance prior/whitening 不算新信息 |
| MVKME | 四相位、局部窗口、Fourier view 均值、ridge | 再做同一 view 的矩近似不是新观测 |
| BNNA | 四相位 within-physical 方向与 gate | 再称 view covariance 为独立物理 variance 不成立 |
| OSC | 四相位有序循环匹配与共享 covariance | 同一 orbit 的重新排列不是新 descriptor |
| MVRidge | identity+FFT 的四 view 共同判别与一致性项 | 新的线性头目标本身不扩展输入信息 |
| 历史 RF32 | I/Q 均值与方差、相关、包络分位、2/3/4 阶复矩、lag 1/2/4/8 复自相关 | 简单追加这些统计不能说仓库此前没有使用过 |
| M23 RF-lite / identifiability stats | C40/C42、幅度比、相关强度、分段高阶/相位统计 | 把高阶矩改名不等于新结构 |

RF32 的具体实现见 `stage2_diag_cosine_exploration.py:220`；C40/C42 与 RF-lite 见 `stage2_m23_rfguard.py:315`；相位/分段高阶诊断见 `identifiability_stats.py:283`、331 以后。这里只读了实现，不读取其历史结果，不推断这些历史方法的效果。

融合前分支与上述只读取 z_id 的后处理不同：它们是另一个已冻结计算节点的输出，未必是 z_id 的函数。但只有 support 的非冗余和 held 判别验证通过，才有理由开发使用它们的方法。不能用“更多维度”本身证明更有效。

## 6. 最少必要的 support-only 验证

只推进以下三个范围明确的步骤，不先选择新分类器或建立宽参数搜索。

**A. 当前 checkpoint 的出口事实。** 复用既有已合规的 exact loader 与来源绑定，核对实际 `id_feature_key`、`use_time/use_freq/use_pa/use_dac`、各 aux 宽度和 eval 状态。比较同一次 forward 返回的原 z_id 与现有出口路径，确认数值未漂移；只读取合法 support。关闭或恒零分支排除，不开启新路径。这是确定实际输入，不是重做 dataset/capsule 验证。

**B. 逐 row 的数值信息审计。** 在同一批合法物理 support 上记录每个实际启用分支的 norm、非零比例、有限性、有效数值 rank，以及相对于 z_id 的条件残差。只比较固定导出节点，不做按类或按 receiver 选择。不能用训练集线性重构误差为零断言无新信息：K1 的 C 行通常可被高维线性模型插值；也不能因维度大或训练误差低断言有 TX 增量。

**C. K>=2 的最小 held-physical 增量检验。** 只在当前 row 内按已有 per-class physical folds 把物理样本完整留出。任何标准化、条件投影、相关性消除及线性探针都仅从 trainfold 拟合。使用预先指定的同一种简单探针和相同正则口径，比较 z_id 与预先固定的启用分支集合；不从几十个分支组合中挑最好结果。记录 OOF 改变量、classwise 误差及额外状态/计算。这个步骤可检验预测增量，但仍不是对真实 query 的证明。

K1 只执行 A/B 的输入和数值检查，没有独立同类 holdout。不能借另一个 row 的额外 support 训练 K1 state；不能将四 view 或网络不同分支当作额外验证样本。即使 K>=2 上有增量，K1 的收益仍是未证实假设。

本审查不执行这三个步骤。若后续验证发现实际启用分支全零、与现有输出等价，或新增部分缺少 held-physical 判别增量，应停止该方向，而不是再追加未知 descriptor 或借 query 选择特征。若只有 A/B 可测而 C 数据不足，记录证据不足，不输出正向性能结论。

## 7. 两种计算与传输成本口径

**未来同次导出。** 原生 identity helper 当前就调用 `return_aux=True`，模型已经计算实际启用分支；若在同一次 forward 把这些输出一起写入，通常不新增 backbone 前向次数。新增的是 tensor 取出、精度转换、CPU/GPU 传输、cache I/O，以及后续 support 拟合与 query 打分成本。不能把“无额外前向次数”等同于“零开销”。

**补现有 cache。** 当前 identity+FFT cache 没存这些分支，不能从 z_id 反解出来。若不能复用曾经保存的匹配 aux artifact，补提取需要对允许读取的 received IQ 重新运行冻结模型。当前授权的下一项仅为 support-only 验证，所以成本至少包括每个所需物理 support 的重新前向；本审查不得读取 query IQ。若未来正式方法需要 query 分支，必须在其授权的独立导出/推理阶段计入 query 再前向，不能报告本轮零新增计算。

设实际额外保存分支数为 B、每个宽度 160、view 数 V：float32 cache 增量为 `4*160*B*V=640*B*V B/物理样本`。原相位验证取 V=1，不需要自动扩成四相位。若实际宽度不同，按各 ndarray.nbytes 求和，不能强行套 160。

冻结模型已部署时不增加模型传输；不新增 source 样本、source feature 或 ground 摘要载荷。分支特征在卫星本地从 received IQ 产生，其 cache 不是上行 source payload。当前 ground v2 的 feature schema 仅绑定 z_id160，不得把它的矩阵或半径直接用于不同分支坐标。未来分类状态字节取决于尚未确定的方法，本文件不虚构一个新 head 或字节承诺。

日志应分开记录冻结前向、分支转换/cache 写入、support 探针、OOF 推理和实际 RSS；只能填实测值。本次均未测量。

## 8. 交付判定

可推进的是一个明确的信息问题：**当前冻结、实际启用的融合前表示是否含有超出最终 z_id 的 held-physical 判别信息？** 代码给出了合法、可获取、非纯重参数化的检查入口，尚未给出性能答案。

暂不推进的是：凭 FFT 丢相位便宣布模型没相位、启用关闭 DAC、从短窗推断开机瞬态、把 RF32/HOC 重命名为新机制、把 view 数当 K、或在未核对 aux 增量前先冻结一个新分类器。本次只交审查文档，保留全部冻结代码和运行安排。

# ADV3B02 FCR-V2完整实验矩阵设计

日期：2026-09-03

状态：已获用户口头确认，待书面规范复核
依据：提交`84c32ae819be04f7d7ea99c2ef5eaecc3bbff869`全面复核报告

## 1.目标与结论边界

本轮在不改变ManySig数据协议、source/target边界和ADV3B02成熟身份主干的前提下，重构FCR生成分支及训练控制，并发布报告定义的完整实验矩阵。实验首先消除额外训练预算、FCR身份路由和self reconstruction之间的混杂，再分别识别`z_f shared`、`z_s shared`和true swap的贡献，最后逐步恢复eta、物理解码、cycle、necessity、directed transplant和three-axis机制。

本轮不把性能提高等同于物理因子分解成立。只有对应诊断证明机制实际获得有效样本、产生非零训练信号并影响预期分支时，报告才声明该机制已闭合。target test仅用于所有训练完成后的最终评价，不参与损失调权、能力门控、checkpoint选择或候选筛选。

## 2.固定基线与统一初始化

所有新训练行从同一个ADV3B02 E200最终checkpoint初始化：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/
phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/
ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth
```

基线身份固定为：

- candidate：`ADV3B02_CORE90_SOFT_E200`
- split seed：`392005`
- checkpoint epoch：`200`
- clean：`76.2268%`
- LEO均值：`60.1397%`
- 四场景均值：`64.1615%`

C0直接复用该checkpoint及既有独立评分结果，不重复训练。C1–M6均训练200epoch，使用相同样本顺序和按样本键控的无状态增强随机流。所有行只采用最后一个epoch生成的`final.pth`，禁止target test选模。

## 3.FCR-V2结构

FCR-V2保留ADV3B02的160维身份接口，并将训练期生成路径拆分为五个职责明确的部分：

1. canonicalizer输出canonical IQ、可监督的信道参数`eta`和canonical坐标残差；
2. content encoder输出受限容量的`z_s`，避免以高容量内容码复制完整波形；
3. fingerprint路径将ADV3B02身份嵌入投影为`z_f,id`，并使其真实参与硬件响应生成，而不是只服务分类头；
4. nuisance encoder仅保留物理解码器能够消费的有效维度，并接受`eta`监督；
5. identity-initialized decoder以恒等信道为初态，随后学习IQ不平衡、多径、STO和SFO等可辨识扰动。

`cross_decode(source,target)`必须使用target提供的fingerprint和nuisance条件，不能继续复用source fingerprint。canonical residual统一定义在canonical坐标系。STO通过时间移位/可微采样实现，SFO通过随时间累积的相位斜率实现；complex Gram保留复数相位信息。Phase1结束后的部署路径仍只保留身份推理所需模块，decoder、nuisance和transplant模块不进入星上推理。

## 4.新增模块与集成点

新增以下独立模块，避免继续向旧FCR文件叠加互相耦合的分支：

| 模块 | 职责 |
|---|---|
| `code/cvsrffi/phase1_fcr_v2_metadata.py` | 定义并校验样本metadata、`eta_schema_version`、`eta`和`eta_valid_mask`；shape或schema不一致直接报错 |
| `code/cvsrffi/phase1_fcr_v2_pairing.py` | 构造确定性nuisance/content/fingerprint配对，记录逐TX覆盖率 |
| `code/cvsrffi/phase1_fcr_v2_factors.py` | 实现受限`z_s`、与生成路径耦合的`z_f`和无null dimension的`z_n` |
| `code/cvsrffi/phase1_fcr_v2_physics.py` | 实现canonical residual、正确STO/SFO、identity初始化decoder和complex Gram |
| `code/cvsrffi/phase1_fcr_v2_losses.py` | 独立计算identity、prototype、tail、self、shared、eta、swap、cycle、necessity、transplant和physical损失 |
| `code/cvsrffi/phase1_fcr_v2_schedule.py` | 实现分组学习率、EMA损失归一化、渐入权重和source-only能力门控 |
| `code/cvsrffi/phase1_fcr_v2_diagnostics.py` | 输出pair coverage、eta误差、latent leakage、swap敏感性、梯度比例、逐TX和资源诊断 |

`code/model_dual_cvsincnet.py`只负责组装V2模块和提供`forward_identity_only`；`code/train.py`只负责批次数据流、优化器参数组、损失聚合及artifact写出。旧FCR-V1路径保持可读，不把V2行为隐式覆盖到历史R1–R8。

## 5.数据流与metadata

训练样本至少携带：

```text
physical_sample_id
content_record_id
crop_offset
common_preamble_id
tx_id
rx_i
day_i
view_type
link_condition
excitation_bin
eta_schema_version
eta
eta_valid_mask
```

三种配对关系分别为：

- nuisance pair：相同TX和内容，不同信道扰动；
- content pair：相同TX和信道条件，低重叠的不同内容窗口；
- fingerprint pair：相同内容、接收机、日期、链路和激励分箱，不同TX。

增强器直接返回其实际使用的`eta`，训练代码不得根据tensor形状失败后静默生成全零mask。配对随机数由`seed+epoch+physical_sample_id+view_type`确定，因此各消融行共享同一批次和增强实现，不依赖进程内随机顺序。

## 6.损失与训练日程

FCR-V2-Core由`self+asymmetric z_f shared+small z_s shared+response shared+eta`组成。起始相对权重为：identity CE 1.00、prototype 0.10、tail/worst-class 0.075、self 0.10、`z_f shared`0.20、`z_s shared`0.05、response shared 0.05、eta 0.10、swap从0渐入至0.05、无标签FCR总权重0.35。

每个损失先使用EMA量级归一化，再监控其相对identity CE的梯度比例。调节只依赖source训练/验证诊断，不能读取target指标。TX1保护由source标注数据上的cosine prototype margin和class-tail/CVaR项实现，不针对target结果反向调参。

训练阶段固定为：

| Epoch | 阶段 | 行为 |
|---:|---|---|
| 1–20 | Head warm-up | 冻结成熟主干，训练identity-noop head、content、eta和self |
| 21–60 | Shared learning | 开启非对称`z_f shared`和小权重`z_s shared` |
| 61–100 | Nuisance learning | 训练eta及identity-initialized decoder |
| 101–130 | True swap | 在source-only能力诊断满足时渐入修正后的swap |
| 131–160 | Cycle/need | M2/M3及其后续行分别启用修正机制 |
| 161–200 | Identity refinement | 生成损失降至0.10–0.25，保留CE、prototype和shared |

主干早期层E1–20冻结，之后学习率`5e-6`；后两层使用`1e-5`至`2e-5`；身份head使用`2e-5`至`5e-5`；identity projection使用`5e-5`；新FCR模块使用`2e-4`。

## 7.完整实验矩阵

完整矩阵包含1个既有基线和14个新训练行：

| 行 | 机制 | 主要比较 |
|---|---|---|
| C0 | 既有ADV3B02 E200 | 固定基线，不重训 |
| C1 | 从C0继续训练E200，无FCR | C1−C0：额外训练预算 |
| C2 | FCR identity route，辅助损失全0 | C2−C1：路由/归一化效应 |
| C3 | C2+self | C3−C2：self效应 |
| S0 | shared-f=0，shared-s=0，swap=0 | shared组零点 |
| S1 | S0+`z_f shared` | S1−S0：fingerprint一致性 |
| S2 | S0+`z_s shared` | S2−S0：content一致性 |
| S3 | S0+`z_f shared+z_s shared` | 联合shared及交互 |
| S4 | S3+corrected true swap | S4−S3：true swap |
| M1 | FCR-V2-Core+eta+full-physics | nuisance/物理路径 |
| M2 | M1+corrected latent cycle | cycle增量 |
| M3 | M2+corrected necessity | necessity增量 |
| M4 | M3+directed transplant | transplant增量 |
| M5 | M4+complex physical feature | physical增量 |
| M6 | M5+three-axis | 完整三轴模型 |

矩阵使用同一umbrella run ID和不可覆盖的逐行输出目录。发布顺序为C1、C2、C3、S0–S4第一批，M1–M6第二批；每张GPU同时只分配一个本矩阵实验。第二批属于同一预登记发布，不根据第一批target性能筛选，只核对source-only机制能力和技术健康状态。

## 8.诊断、错误处理与机制闭合

以下错误属于技术失败：metadata/schema不一致、配对关系违反定义、输出目录已存在、checkpoint身份不匹配、NaN/Inf、prediction不完整、scorer连接失败、进程CWD或run-root不匹配。失败行保留全部产物并使用新run ID修复重发，禁止覆盖或原地热补丁。

诊断必须在deferred target evaluation返回前写出。每行至少记录：各pair active count与coverage、eta valid coverage和各分量误差、`z_n`对decoder均值的敏感性、swap前后输出差异、`z_f/z_s/z_n`泄漏probe、各损失相对CE的梯度范数与余弦、逐TX source指标、峰值VRAM和epoch耗时。

能力门控仅决定某一高级损失是否在预定epoch获得非零权重，不能中止完整矩阵，也不能读取target。若门控未满足，该行仍训练到E200，但必须报告`MECHANISM_NOT_ACTIVATED`，不得把结果解释为该机制的有效消融。

## 9.验证、测试与最终评分

本地验证包括：metadata严格失败测试、三种配对单元测试、相同随机键跨行一致性测试、`cross_decode`来源测试、identity decoder初值测试、STO/SFO物理作用测试、complex Gram测试、necessity相对误差测试、diagnostics-before-defer测试、真实ADV3B02 checkpoint无query smoke和完整launcher dry-run。

独立P0/P1审查只检查会导致真实实验跑错、越权、覆盖输出、误杀进程、无法启动或不能产生合法prediction的问题。修复后仅对原问题定点复审。

所有训练结束后，使用每行E200 `final.pth`执行truth-last评价：先对固定target输入生成prediction，再由独立scorer连接独立truth sidecar。每行报告clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`、LEO均值、四场景均值、最差场景、逐TX准确率、相对C0增量及机制诊断。禁止在训练期间、checkpoint选择或第二批发布决策中使用这些target结果。

## 10.N607发布与健康追踪

本地提交并push后构建单一release归档，执行一次本地/远端SHA一致性核对和一次远端编译。N607预检确认时间、项目路径、checkpoint和GPU可见性后发布不可覆盖的新run ID。启动后立即核对主PID、CWD、cmdline、run-root、GPU映射和日志增长。

实验发布后每30分钟进行一次短连接、只读健康检查，检查：活动行数量、训练epoch增长、日志更新时间、GPU显存/利用率、确定性异常指纹和输出目录增长。健康运行时不干预；只有协议/安全错误、错误checkout、输出冲突、无prediction闭合或重复确定性技术异常才允许停止对应run-owned进程树。低性能不构成停止条件。

训练结束不等于实验完成。只有14个新训练行均保存E200 checkpoint、四场景prediction、独立score和诊断后，umbrella run才标记为`ARTIFACTS_COMPLETE`；完成汇总分析并发布报告后标记为`ANALYZED`。

## 11.交付物

- FCR-V2七个新增模块及最小集成修改；
- 完整矩阵launcher、配置和聚焦测试；
- N607不可覆盖run root、训练日志、E200 checkpoint和诊断；
- truth-last prediction、独立score及逐场景/逐TX汇总；
- `automation_reports/CV-SincNet/<run-id>/report.md`完整实验报告；
- 对应Git提交、远端分支OID核对和发布记录。

# Phase2域适应/新类注册目标模式历史普查（2026-07-18）

## 1. 普查方法与判定口径

本次刷新`E:\type10-7\conversation_index`，得到997条项目相关记录；再以`Phase2`、`Stage2-B/C`、`qKNN`、`domain adaptation`、`new enrollment`、`active goal`、`D1-D36`等关键词检索，并反查原始rollout与实验报告。

纳入“方法研发目标”的条件是：对Stage2-B旧类域适应或Stage2-C新类注册提出/修改机制，并以性能、遗忘、floor或资源结果决定下一条技术路线。仅做协议加固、数据审计、monitor、runner健康检查、报告重建或解释的对话不单独计为研发目标。

## 2. 独立方法研发主线

| 层级 | thread/时间 | 目标与路线 | 结论 |
|---|---|---|---|
| 最初qknn8 | `019e6f48-e792-7230-bff8-b35bd5f703d5`，2026-05-28 | R8/R9/R10多接收机协作qknn8，协作数1～5 | exact-K缺少同事件五接收机覆盖；budget协作最佳old约35.4%、new约2.4%，结构性失败 |
| qKNNV92主线 | `019f36ac-3ee7-7c33-a7f9-e4717d9d26b3`，2026-07-06～09 | support int8 code、top-m KNN、prototype与旧类anchor；同时优化K5/K10旧域适应和新类注册 | 历史报告K5 old96.62%、new95.10%，K10 old96.40%、new95.92%；后续Oracle审计与现行单观测协议使其不能作为当前正式证据 |
| Stage2-B前史 | `019f3b7a-8668-7321-b837-a170cef4b2e6`，2026-07-07 | RIEI/DRIFT+ProtoNet-CDA旧类适应 | 低性能负证据 |
| Stage2-B修复 | `019f3bdf-ba4a-7f72-baf6-59e8720c8538`，2026-07-07 | source-logit校准、receiver-conditioned support head | source-logit失败；按receiver拟合提升约5.7～11.2pp，但old仍约55%～66% |
| ADV3B02基线 | `019f3f93-facb-7593-83b6-5e46ee786d98`，2026-07-08 | ADV3B02+ProtoNet-CDA，K5/K10 Stage2-B | old约72.5%，基本不优于冻结基线，`20-19`更差 |
| 第一次正式active goal | root `019f5fe9-b4ed-7c00-b935-91eb4657c1fc`，2026-07-14～16 | identity-only、qKNN+FFT96、BPJG/JG、SOMP-H到D1前身；统一轻型Stage2-B/C | 过多时间投入输入包/runtime/authority/Landlock；主报告第29节才进入真实D1/D2实验 |
| JG验证支线 | `019f6902-be97-73c0-a4d0-69690d06586f`，2026-07-16 | JG_R8_LR020，new5/10/20×3场景 | new5 old/new/H约57.8/61.0/59.2%，new20约50.8/20.7/29.3%，负结果 |
| 第二次正式active goal | root/current `019f699b-0853-7d32-bf63-ec18a92a6647`，2026-07-16～18 | D1-D36：单IQ拼接、多原型、低秩、ground int8、对角/Fisher、局部碰撞、安全门、连续校准 | 无达标路线；B3是当前开发support-held最强比较器，但仅before-old86.67%、after-old73.33%、new73.33% |

主要证据入口：

- 第一次active goal：[qknnv42 Stage2-B/C主报告](../automation_reports/CV-SincNet/qknnv42_stage2bc_extreme_light_route_20260716/report.md)
- 第二次active goal原始rollout：`E:\codex\home\sessions\2026\07\16\rollout-2026-07-16T14-26-40-019f699b-0853-7d32-bf63-ec18a92a6647.jsonl`
- 项目对话索引：`E:\type10-7\conversation_index\type10_7_conversations.json`

早期主线可复核路径：

| thread | 原始rollout |
|---|---|
| `019e6f48-e792-7230-bff8-b35bd5f703d5` | `E:\codex\home\archived_sessions\rollout-2026-05-28T23-51-49-019e6f48-e792-7230-bff8-b35bd5f703d5.jsonl` |
| `019f36ac-3ee7-7c33-a7f9-e4717d9d26b3` | `E:\codex\home\sessions\2026\07\06\rollout-2026-07-06T17-04-45-019f36ac-3ee7-7c33-a7f9-e4717d9d26b3.jsonl` |
| `019f3b7a-8668-7321-b837-a170cef4b2e6` | `E:\codex\home\sessions\2026\07\07\rollout-2026-07-07T15-28-33-019f3b7a-8668-7321-b837-a170cef4b2e6.jsonl` |
| `019f3bdf-ba4a-7f72-baf6-59e8720c8538` | `E:\codex\home\sessions\2026\07\07\rollout-2026-07-07T17-19-05-019f3bdf-ba4a-7f72-baf6-59e8720c8538.jsonl` |
| `019f3f93-facb-7593-83b6-5e46ee786d98` | `E:\codex\home\sessions\2026\07\08\rollout-2026-07-08T10-34-50-019f3f93-facb-7593-83b6-5e46ee786d98.jsonl` |

第一次active goal的root session未作为独立条目进入索引，但34个子agent session的`parent_thread_id`均指向`019f5fe9...`，并可由主报告恢复。

## 3. 目标模式连续对话清单

下列会话不是新的独立主线，但其用户目标明确修改或约束了Phase2方法研发，因此必须纳入目标重构：

| thread | 用户增量目标 | 对后续目标的影响 |
|---|---|---|
| `019f6573-45a7-7080-9b66-445907e310a1` | 分析ADV3B02最有效适配层与loss | 引入轻量关键层/adapter更新 |
| `019f6573-9453-7e90-b4b8-eabac68fe8e4` | K1适应必须明显优于直接ADV3B02 | K1成为强制压力点 |
| `019f6710-a7e4-7541-ba9f-fdb814a9f99c` | 不得偏离Stage2目标 | 研发优先于外围工程 |
| `019f6882-849d-74c2-8c0b-534ae0257c49` | 实测必须包含注册后新类性能 | 旧适应和新类注册同等重要 |
| `019f6a32-fff3-7e72-b390-2f6f21672c26` | 正式Stage2-B/C轻型逐样本目标 | 固化性能、矩阵、资源与证据框架 |
| `019f6aed-7425-7320-9dc1-d9610d9b7b85` | 重启同一正式goal | 继续统一K10选参与独立确认 |
| `019f6ba2-f075-7af1-9b30-e45245b6c83e` | 重跑125并修复问题 | 加入稳定性重跑和缺陷闭环 |
| `019f6cec-f02d-71b0-beb1-c1f8ed3e1ca6`、`019f6cfa-3f9d-70c2-aede-57b3b4088fe1` | 同一物理IQ不能生成三种LEO观测 | 建立单物理样本单观测协议 |
| `019f6de4-4c18-7a41-aea5-a08c7ae41c18`、`019f6de4-abf8-7092-858c-95c90a768186`、`019f6de5-062b-7cd3-bdce-6f94dd59b81b` | 合法Phase2样本一次保存，避免重复前审 | 建立`VALIDATED_ONCE`跨方法复用 |
| `019f6e2e-efb2-7ad3-8f01-3444309b5094` | 压缩ground域×类原型并核对真实存储 | 发展中心+偏移+半径及量化状态 |
| `019f71d8-0142-7a33-929e-d431eb5f34fe` | int8旧类锚+target old/new原型+对角/低秩更新 | 形成UFDR/D24及其后拼接路线 |
| current root | 加FFT96、数学表征、拼接、多机制、快速梯度、floor优化 | 形成D25-D36，并最终要求本次历史/协议/目标重构 |

## 4. D1-D36完整谱系

| 版本 | 方法/机制 | 结果与教训 |
|---|---|---|
| D1 | `z160+FFT96+RF32`的288D对角余弦头，辅助块权重4.0，20epoch | 开发单seed高；125确认K10/new5 after-old87.82%、new84.24%，new10 old84.46%、new80.97%；K1负迁移；同物理IQ跨三场景多观测不符合现行协议 |
| D2 | 288参数轻头 | 明显弱于D1 |
| D3 | 冻结旧score、support侵入安全门 | support侵入为0，但held旧floor和new失败；support安全不等于泛化安全 |
| D4 | 单LEO观测固定IQ变换/轻适配 | before-old76.39%、after-old62.22%、new68.67%，遗忘14.17pp |
| D5 | 低秩、谱收缩、多原型 | 低秩无效；谱收缩选择不稳；多原型有局部信号但无合法正路线 |
| D6 | support-only全局view/残差/margin选择 | 回退base；fresh after-old63.33%、new66.67% |
| D7 | 逐类IQ表征选择、局部对比边界 | 有类条件信号，但性能未闭环 |
| D8 | 二阶段锁与开发control | 主要转入authority/exact-K/封装，非性能突破 |
| D9-D10 | 两代盲IQ operator | 宽接口与证据绑定问题，floor不足 |
| D11 | 冻结旧adapter后追加prototype | 遗忘6.7～15pp，new46%～60% |
| D12 | 约2.4k参数联合old/new残差logit head | 局部信号，但仍遗忘5～11.7pp |
| D13 | support margin安全delta | 所有正guard失败，回退`delta0` |
| D14 | 稀疏pairwise Fisher/局部几何 | 0参数、约22KB，但逐类门失败 |
| D15-D16 | 共享/逐类非对称幅度floor修正 | 改善旧类会伤新类；正候选未过门 |
| D17 | pair-specific局部证据+canonical Z0 | 工程闭环增强，性能仍无突破 |
| D18 | 固定IQ相位/时移稳健化 | old约70%、new62%，旧/新floor均为0 |
| D19 | CIAF：ground int8 anchor直接融合target原型 | 强融合把new压至6.67%～22%；弱融合退回Z0 |
| D20 | ground int8旧类内部重排+IQ/FFT-RF轻头 | loader/schema、Torch/NumPy兼容失败，无性能结果 |
| D21 | 中心+低秩偏移+半径、KNN生命周期、M1-M6 | L7 old78.89/new79.33；M1 old强但new63.33%；无达标路线 |
| D22 | B0-B4 int8锚与单IQ288D头 | B3 before-old86.67、after-old73.33、new73.33、H72.65；floor为0，B4 int8无收益 |
| D23 | target-old/new FP32/FP16/int8统一bank | 完成存储与append-only基础，不是性能路线 |
| D24 | UFDR-160：ground int8+target-old不确定度融合、target-new独立注册、Diag/LowRank | 设计/核心实现，后被高维拼接吸收 |
| D25 | 288D能量拼接、ground-z融合、逐块半径 | C0 H50.35%；ground/半径更差；C3最好H约53.38% |
| D26 | 单new-group bias | v1旧类崩；v2 old78%～80%但new仅2.67%～8% |
| D27 | 每新类独立安全bias | 最佳old67.22/new47.33/H52.82，floor失败 |
| D28-D29 | 逐样本证据门、逐类安全释放 | 全部被门禁，等于D27-B |
| D30 | B3几何+双包络int8校准 | old85.56/new66.67/H68.19；包络旁路，遗忘18.89pp |
| D31 | 全部old+new support、CVaR/旧margin | B old67.78/new72/H69.06；C遗忘9.44pp但new60.67% |
| D32 | 训练期内生安全cap | old63.89%～68.33%、new60.67%～70.67%；support安全不泛化held |
| D33 | 球面centroid/radius+Fisher快速适配 | A old71.67/new63.33/H66.15；FAST70/59.33；资源成功、性能失败 |
| D34 | 冻结旧score、稀疏old-new collision局部注册 | C old71.11/new57.33/H62.23；大量新类不可达 |
| D35 | 新类全局可见、winner阈值/双原型 | old53%～55%、new55%～59%；大量旧类侵入 |
| D36 | 联合target-old/new int8头、对角+rank2、只读ground int8、连续margin ridge | 只有设计锁和core单测，无可引用性能 |

D19以后主报告：[D21](../automation_reports/CV-SincNet/d21_knn_prototype_lifecycle_20260717/report.md)、[D22](../automation_reports/CV-SincNet/d22_int8_anchor_lifecycle_20260717/report.md)、[D25](../automation_reports/CV-SincNet/d25_multimodal_concat_support_20260717/report.md)、[D30](../automation_reports/CV-SincNet/d30_envelope_int8_20260718/report.md)、[D33](../automation_reports/CV-SincNet/d33_spherical_fisher_20260718/report.md)、[D34](../automation_reports/CV-SincNet/d34_fcler_20260718/report.md)、[D35](../automation_reports/CV-SincNet/d35_dense_safe_20260718/report.md)、[D36](../automation_reports/CV-SincNet/d36_compiled_joint_int8_20260718/report.md)。

## 5. 排除项

以下不算独立方法研发目标：

- `019f60fd-7930-7751-87de-d6986c0e6c03`：qKNN报告重建和Oracle协议加固；
- `019f649b-1632-78d3-a6a1-c7aa8935879d`：V14结果审计，正式target结果为0；
- `019f5f8d-d807-71a3-91fd-292f0747836c`：严格CI比较矩阵，主要是方法验证；
- `019f6ad9-d80a-79b1-8ade-1529b8ff1cdb`：Stage2最小准入文档修订；
- monitor、runner health、报告补写和代码审查子agent：属于主研发线程的支撑任务。

## 6. 共同结论与路线纠偏

1. 两次active goal都过度投入协议工程。第一次主报告1773行、49个二级章节，第1～28节主要处理package/runtime/authority/Landlock等，直到第29节才启动D1/D2；第二次D4-D20又反复进入authority、capsule和schema闭包。
2. 当前最可信正信号是同一固定received IQ的规范化高维拼接和class-specific cosine head（B3），不是ground原型直接融合、半径评分或复杂hard gate。
3. B3本身仍未解决Stage2-B/C：before-old86.67%、after-old73.33%、new73.33%，所以不能只继续修注册层。
4. hard gate反复在“新类不可达”和“旧类侵入”两端摆动；support fit/LOO安全也不能代表held泛化。
5. 持续floor类为旧类`14-7`、`20-19`、`6-15`，新类`09f8`、`f608`，后续必须显式优化和报告。
6. ground int8适合做只读身份先验/正则/不确定度参考，不适合直接改写target原型。
7. 拼接维度不是越多越好；必须做分块归一化、能量控制和matched ablation。D36连续margin方向可继续验证，但在有性能数据前只能算未验证设计。

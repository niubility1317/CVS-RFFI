# FastTrust-QB3有界域混淆与伪标签利用设计

## 目标

在不改变Phase1数据权限、Core90初始化、`LEO_WEAK`拼接增强、seed、U batch和E200训练步数的前提下，修复QB0/QB1在E104–E106出现的域对抗非有限损失，提升`U_s`伪标签的有效利用，并用同row矩阵区分H、P-set和P-conditional的真实增量。

## 已有证据与项目调整

终态日志显示，首次非有限分叉前不仅U域对抗项异常，`train/loss_adv_labeled`也已分别升至约29.7和303.7。因此设计报告中的“替换U侧GRL CE”按项目实际扩展为：所有`z_id→domain`路径统一采用有界混淆，覆盖有标签样本和RC4无标签样本；`z_dom→domain`监督保持普通交叉熵。

`P-set`与`P-conditional`必须独立开关、独立权重、独立尾段退火和独立梯度诊断。N路由继续关闭，但P的APS阈值不再通过N的99%排除目标被隐式抬高。

## 机制冻结

### 有界域混淆

- 域判别器：对`z_id.detach()`预测source domain并最小化CE，只更新判别头。
- 身份编码器：冻结判别头参数，对`z_id`最小化`KL(p(domain|z_id)||Uniform)`，只更新身份表征路径。
- 有界混淆损失范围为`[0,log D]`，全程以float32计算；均匀分布为唯一最优点。
- `z_dom`域CE、`z_id`域判别CE和`z_id`有界混淆分别配置和记录。
- 旧方法默认仍保留`grl_ce`，仅QB3矩阵显式启用`bounded_confusion`，避免改变历史可复现性。

### RC4校准与路由

- 正确性校准保留现有七个truth-free基础特征。
- P集合安全校准新增集合大小、集合概率质量、集合边界间隔、集合内熵和跨视图集合一致性特征。
- P的APS目标固定为95%覆盖，并与N的99%排除目标解耦。
- QB3首先使用全局APS阈值，消除“按真类拟合、按预测类使用”的口径错配；旧分层模式保留为兼容路径。
- H和P阈值必须同时满足总体风险与最差source receiver风险；校准交叉拟合仍按source domain分组。
- H和P分别施加class×receiver有效权重质量上限；不再用一个总15%预算让P填充H剩余量。

### 路由预算和尾段

- H有效权重质量上限：0.05。
- P-set有效权重质量上限：0.10。
- P-conditional使用相同P样本，但独立损失系数上限为0.02，不重复计入P-set样本权重质量。
- N关闭，R只承担表征/一致性目标。
- E181–E200线性退火到：H=0.60、P-set=0.20、P-conditional=0；Core90原有增强日程不变。

### 诊断与恢复

- 在校准更新epoch和E181、E200记录`|g_L|`、`|g_H|`、`|g_Pset|`、`|g_Pcond|`、`|g_adv|`的`z_id`梯度范数及其比值。
- 每个epoch覆盖写入仅供恢复的`latest_finite_ssdg.pth`；E90额外保留`recovery_e90_ssdg.pth`。它们不得参与final-only选模。
- 首次出现非有限损失、梯度或参数时写入一次异常诊断包；系统性技术失败仍按预登记规则停止并保留全部产物。

## 训练加速

- 训练预算保持200epoch、U batch256、相同步数，不以减少训练样本或缩短epoch换取速度。
- 每张GPU仅运行一个训练进程，五行并行，避免同GPU双进程竞争。
- 评测batch由512提高到1024；source-heavy评测从每5epoch改为每10epoch，最后20epoch仍每epoch评测。
- 保留AMP、TF32、现有拼接前向和最终clean/三LEO完整评测。
- 梯度诊断只在预定epoch的首个batch执行，避免把监控变成显著训练开销。

## 正式E200同row矩阵

| 行 | 有界混淆 | H | P-set | P-cond | 非伪标签身份锚点 |
|---|---:|---:|---:|---:|---:|
| C0 | 开 | 关 | 关 | 关 | 关 |
| C1 | 开 | 开 | 关 | 关 | 关 |
| C2 | 开 | 开 | 开 | 关 | 关 |
| C3 | 开 | 开 | 开 | 开 | 关 |
| C4 | 开 | 关 | 关 | 关 | 开 |

C4使用冻结Core90对全体U clean view的`z_id`特征锚定，强度按C2的P-set名义梯度预算预注册，用于判断P是否只是提供一般身份稳定梯度。所有行均使用相同split、Core90、seed392002、E200、U256和训练步数。

## 验证与结论边界

- 聚焦RED/GREEN：有界性、梯度隔离、P/N APS解耦、集合特征、最差receiver阈值、分路预算、尾段退火、恢复包和矩阵传播。
- 相邻回归：FastTrust-RC4、训练集成、launcher与速度测试。
- 真实Core90 checkpoint无query smoke：只加载合法source资产，不读取target query/truth。
- 一次独立P0/P1审查。
- E200完成后，每行必须具有`final_ssdg.pth`、完整训练日志、clean和三种LEO weak场景指标后方可称为实验完成。
- 单seed结果只支持同row机制判断；target指标不得用于改参、补跑或候选重排。

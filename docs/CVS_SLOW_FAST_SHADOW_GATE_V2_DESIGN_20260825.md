# CVS Slow-Fast影子状态与连续风险门控V2设计

## 1.目标

本轮先解除提交`8c6ddd3a7adae16abc44b3c1a10a22caf40a8817`留下的诊断盲区：上一轮9个row全部由support-only gate回退到`lambda=0`，只能证明回退策略有效，不能判断被拒绝的非零Adapter是否在query上有收益。

V2保持冻结`ADV3B02_CORE90_SOFT_E200`、冻结决策原型、`p2_min_v1`和`VALIDATED_ONCE`数据不变。Phase2只能用旧类target support更新和选择状态；query逐样本只读，所有固定影子状态在truth未知时一次性输出，评分完成后不得反馈重跑或选择。

## 2.P0实现范围

### 2.1统一余弦logit

新增统一`prototype_logits(features,prototypes,logit_scale)`入口。Phase1.5静态目标、inner update、outer query目标、Phase2快更新和support cross-fit全部使用bundle中的`support_logit_scale`。prediction继续输出raw cosine，并在receipt显式记录`score_type=raw_cosine`。

### 2.2零中心方向门控

`FAST_LOWRANK_R8`将方向门控从`sigmoid(a)`改为`tanh(a)`。`a=0`时该方向关闭，并允许目标域support选择正向或负向修正。bundle schema升级，旧bundle由显式兼容转换加载，不能静默改变既有V1结果。

### 2.3分层cross-fit与连续风险

K10使用每类5／5双折并交换训练／验证折，按固定seed重复3次；K5使用3／2和2／3双折；K1保持DA0回退。每个训练fold内各类样本数严格相等。

对每个固定强度计算：

```text
R(lambda)=MacroCE+0.3*CVaR_class+0.1*MeanMove
```

候选必须同时满足：

```text
Macro(lambda)>=Macro(0)-0.02
Floor(lambda)>=Floor(0)-0.10
MaxMove<=trust_radius
R(0)-R(lambda)>=1e-4
```

在全部合格候选中选择风险最小者；风险并列时选择较小`lambda`。强度网格固定为`{0,0.125,0.25,0.5,0.75,1}`，不再选择“最大的可通过强度”。

### 2.4影子状态

每个row在query truth未知时输出：

```text
DA0_REG0
DA1_J{01,03,05,10}_A{050,100,200,400}_L{0125,0250,0500,0750,1000}_REG0
DA1_GATE_LEGACY_REG0
DA1_GATE_CF_REG0
```

`COMMON_SHIFT_R4`没有梯度步数，只输出五个固定lambda及两个gate状态。影子状态是预注册消融，不参与运行时选择；正式部署候选仍只有`DA1_GATE_CF_REG0`。

### 2.5完整审计

每个lambda保存cross-fit macro/floor、class-balanced CE、class CVaR、mean/min margin、mean/max feature move、support prediction flip、每类loss变化、拒绝原因和风险。计算量拆成`loo_fit_count`、`attempted_gradient_updates`、`committed_gradient_updates`。receipt同时记录快参数更新范数、实际logit scale、trust radius和raw cosine score类型。

## 3.P1、P2触发条件

P0 truth-last完成后按以下顺序决策：

- 至少一个非零影子状态改善query：Adapter存在上界，优先修gate／步数／步长，不重训Phase1.5。
- 非零状态改善support cross-fit但损害query：进入receiver-held-out gate与元训练一致性修正。
- 所有非零状态几乎不改变score：进入因子化receiver／LEO慢基、paired reduced-rank operator和clean identity loss的轻型Phase1.5重训。
- 非零状态显著移动但始终无稳定收益：P1失败后才允许测试time／freq／fusion中间层Adapter。

P1和P2不与P0同时发布，避免多机制叠加后无法归因。

## 4.项目化修正

- `COMMON_SHIFT_R4`只通过`rho`表达强度，`common_coeff`保持未缩放，三个候选共享同一强度语义。
- `trust_radius`取消默认值，所有正式调用必须显式传入。
- 原型／checkpoint一致性使用实际checkpoint加载、160维特征、class mapping和预测一致性进行数值核验。逐代码文件SHA、feature code SHA和额外seal属于`REJECTED_EXTRA_GATE`，不作为实现或发布条件。
- 所有正式结果使用`DA0_REG0`及带`DA1_*_REG0`的完整DA／registration状态名；REG0新类指标为`N/A`。

## 5.实验矩阵与结论边界

复用receiver=`20-1`、seed=`392002`、`K10/new10`和三种LEO weak场景。先运行单seed同row诊断，不进入Target25。晋级门槛保持旧类均值至少`+1.0pp`、floor至少`+0.5pp`且任一旧类下降不超过`5pp`。

本轮可以判断非零Adapter上界、gate有效性、步数／步长敏感性和support统计对query收益的预测能力；不能据此声明新类注册、unknown拒识、真实在轨性能或多seed稳定性。

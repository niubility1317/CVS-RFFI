# Phase1高泛化最小实验设计

版本：2026-08-08

执行模式：`GOAL_MODE=ACTIVE`

快速通道：`PHASE1_G0_FAST_RELEASE`

设计状态：`DESIGN_FROZEN/LOCAL_VERIFIED`

候选标识：`P1-GeoSat-Lite`

独立复核：2026-08-08，`STATUS=PASS`，`P0=0`，`P1=0`，允许按冻结四臂脚本发布。

## 1.目标与边界

本轮只验证一个最小问题：在不堆叠多种对齐、proxy训练、动态门控或事后适配的情况下，轻量角度几何约束与clean→LEO一致性是否能改善Phase1表征的跨接收机、跨日期和LEO弱信道泛化。

本轮不实现Phase3协同，不输出真实unknown结论，不读取目标query、目标support、confirmed unknown或运营身份。完整版v2 bundle与CARE-PoE均为后续项，不是本轮发布门。

## 2.数据角色

首轮使用ManySig的固定TX顺序建立4/1/1source-only开发切分：

```text
source_known_train_tx      = 14-10,14-7,20-15,20-19
source_known_validation_tx = 6-15
source_proxy_unknown_tx    = 8-20
```

训练入口只加载`source_known_train_tx`并在内存中连续重编号。后两组完全不进入训练、source physical-validation、逐epoch评估、checkpoint选择、阈值或fallback。checkpoint冻结后，才各进行一次只读held-TX审计。该4/1/1实验是Phase1开发证据，不是六类最终deployment bundle；候选通过后另行以全部六个旧类重训正式bundle。

同一TX不得跨组；所有接收机、日期、equalization和LEO view继承其TX角色。训练内仍使用`项目.md`规定的`0.07/0.63/0.30`物理样本划分，`rho_label=0.10`，并保持source/target receiver互斥。

## 3.唯一候选

模型复用ADV3B02结构和closed/SSDG配方，但四臂均从随机初始化训练。不能用已见全部六个ManySig TX的历史checkpoint初始化4/1/1开发实验，否则held-known与proxy TX已经通过权重泄漏。损失为：

\[
L_{Lite}=L_{ADV3B02}+\lambda_gL_{ang}(z_{id},y)+\lambda_sKL(\operatorname{sg}[p(x_{clean})]\Vert p(x_{leo}))
\]

其中`L_ang`直接复用现有`open_world_feature_space_loss`，只开启类内角度紧致、类间角度margin和sample margin；domain alignment、tail和vacuum全部为0。每个物理样本只产生一个预注册`leo_*_weak`训练view。

冻结参数：

```text
from_scratch=true
freeze_backbone=false
lambda_open_world_feat=0.0024
ow_feat_radius_deg=12
ow_feat_inter_margin_deg=55
ow_feat_sample_margin_deg=5
ow_feat_domain_align_weight=0
ow_feat_tail_weight=0
ow_feat_vacuum_weight=0
lambda_sat_cls=0
lambda_sat_cons=0.10
sat_cons_start_epoch=1
max_grad_norm=5.0
checkpoint_selection=final_only
```

legacy batch轮换proxy loss、mixup、source episode、direct metric bank、EVT、gradient surgery、U_s open loss和动态fallback全部关闭。`T_proxy`只在checkpoint与公式冻结后审计一次。

## 4.最小四臂矩阵

| arm | 角度几何 | clean→LEO一致性 | 用途 |
|---|---:|---:|---|
| A | 0 | 0 | ADV3B02匹配基线 |
| B | 0.0024 | 0 | 几何单因素 |
| C | 0 | 0.10 | LEO一致性单因素 |
| D | 0.0024 | 0.10 | `P1-GeoSat-Lite`完整候选 |

四臂使用相同TX清单、source物理划分、seed、epoch、checkpoint规则和LEO view seed。首发锁定一个seed与120 epochs，不扫参；只有四臂完整返回后才解释因果贡献。

## 5.最小发布门与结果门

发布前只要求：

1.TX三组互斥且训练receipt只含`T_train`；
2.角度损失和LEO一致性loss有限、非零且可反向传播，冻结参数不变；
3.真实checkpoint完成no-query smoke；
4.本地focused tests通过、Git提交、不可覆盖run ID和一次独立P0/P1核对。

不增加重复数据验证、额外签名层、通用权限系统或完整Phase3审查。

结果采用同排五门：训练无崩溃；known跨接收机准确率相对A下降不超过2pp；最低known类及每个receiver/day/三种LEO弱场景floor均不低于0.70且相对A下降不超过2pp；冻结后proxy unknown FAR或AUROC相对A出现明确正信号；真实checkpoint能导出基础deployment bundle。registered样本reject/defer均按错计，proxy结果只能写为source-only研发证据。

## 6.首发产物

```text
四臂不可覆盖checkpoint与日志
source TX角色receipt
训练与资源receipt
同排known/LEO/proxy结果表
真实checkpoint基础bundle smoke
```

首轮不要求v2本地证据schema、CARE-PoE、31节点子集、四状态DA/REG矩阵或Phase3授权桥接。

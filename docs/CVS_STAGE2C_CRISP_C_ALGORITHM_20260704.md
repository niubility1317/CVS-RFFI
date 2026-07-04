# CVS Stage2-C CRISP-C协同推理算法设计

## 目标

`CRISP-C`（Cooperative Residual-Interval Sketch Prototype）用于诊断并尝试修复当前`ADV3B02_CORE90_SOFT_E200+qknn8`在Stage2-C中的前融合吸收问题。AOR/KERA已经显示seen-new和unknown在evidence层大量被old吸收，因此CRISP-C不再只调event fusion顺序，而是在每个接收机本地构造旧类收缩包络、seen-new多原型、support residual和conformal p-value，再上传低带宽rank sketch做协同融合。

## 协议边界

| 项 | 约束 |
|---|---|
| 底座模型 | 冻结`ADV3B02_CORE90_SOFT_E200`特征，不做full fine-tune |
| 在轨方法 | qknn8风格support/prototype检索与轻量统计更新 |
| 旧类 | source old prototype作为锚点，target-old support只做有限shrinkage |
| 新类 | target-new support注册1到3个prototype/medoid |
| 未知类 | `target_unknown`仅用于最终评估，不进入prototype、阈值、profile或reliability选择 |
| 协同数量 | `M=1..receiver_count`，每行记录参与接收机、bytes和latency代理 |
| 成功声明 | 只有同一行同时满足old、seen-new、unknown和资源门控才可`target_pass=true` |

## 单接收机证据

对每个接收机`r`和query特征`z`，CRISP-C构造：

```text
P_old_y = normalize((1-alpha) * P_source_y + alpha * P_target_old_y)
P_new_c = medoid/support prototypes from target-new K-shot support

d_old(z)=1-max_p cos(z,p), p in P_old_y
d_new(z)=1-max_p cos(z,p), p in P_new_c

old_envelope_violation=max(0,d_old-tau_old)
seen_new_residual=max(0,d_new-median(D_support_new))
p_c(z)=(1+count(D_support_c>=d_c(z)))/(1+|D_support_c|)
```

上传字段为top old、top seen-new、old envelope violation、seen-new residual、conformal p-value、margin、reject score、receiver quality、bytes和latency。默认包长为`128B/receiver/query`。

## 融合逻辑

CRISP-C按以下顺序判决：

1. seen-new residual gate：seen-new得分显著超过old，且new residual和p-value通过，输出seen-new。
2. old shrinkage gate：old得分、old p-value和old envelope通过，输出old。
3. unknown envelope gate：old和seen-new均包络外，reject score达到阈值，输出`reject_unknown`。
4. 否则输出`defer`。

该顺序的目的不是牺牲旧类换unknown，而是显式检查“old是否仍在可信包络内”和“seen-new是否在support局部结构内”。如果真实unknown仍落入old包络，CRISP-C会保留负诊断，而不会通过后验阈值把unknown query变成调参依据。

## 当前本地诊断

在拉回的`features_proxy_mined.npz`上，CRISP-C可以生成完整`M=1..5`曲线，但仍未达目标。`crisp_old_guard,M=2`的本地结果为`old_acc=0.842697,min_old=0.5,seen_new_acc=0,unknown_reject=0,unknown_FAR=1,target_pass=false`。该结果说明多原型/top-k暴露了更多seen-new候选，但真实unknown仍有高old accept score，support残差不足以形成拒识边界。

## 结论边界

CRISP-C当前是`NON_DEPLOYMENT_DIAGNOSTIC`。它支持的结论是：仅靠部署侧support residual和rank sketch仍不足以解决当前真实target unknown拒识；下一步必须进入地面训练或特征生成阶段，加入source-heldout/open-set episodic训练、旧类prototype distillation和hard-negative feature repair，再由CRISP-C/AWARE类轻量协同层复验。

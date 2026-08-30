# ADV3B02-BiCAD-XR Phase1架构设计

日期：2026-08-30

状态：待用户书面复核后进入实现

适用阶段：Phase1 source-only weak-label/semi-supervised DG

## 1.目标与边界

本设计把ADV3B02的全局单因素DANN重构为类条件、多因素、双向解耦的域泛化训练框架，并增加跨接收机决策兼容、弱类别margin-tail和有限receiver方向外推。方法名称冻结为`ADV3B02-BiCAD-XR`，第一版默认候选为`ADV3B02-BiCAD-XDC-V1`。

本方法保留ADV3B02现有双骨干、共享Sinc/HF前端、身份分支、域分支、RCN、160维`z_id`、160维`z_dom`、CosFace身份头和Phase1 source-only边界。推理路径仍为：

```text
IQ -> shared Sinc/HF -> identity backbone -> z_id -> TX CosFace
```

所有receiver/day/channel判别器、TX反向对抗器、XDC donor heads、receiver tangent和训练诊断均只在训练期存在，不进入最终部署推理图。

本设计不访问Phase2数据、目标接收机、support、query或truth。`V_cal`和`V_select`保持source-only且只用于校准和选模，不参与反向传播或持久状态更新。

## 2.协议冻结

用户已确认采用现行`项目.md`协议，而不是报告中的`70% clean+30% mixed_orbit`单前向方案：

- `phase1_method=bicad_xr`不继承HCF-DG专用单前向例外；
- 训练使用ADV3B02同款`concat_sat_ce_only=true`；
- `lambda_sat_cls=0.68`，`lambda_sat_cons=0`；
- 卫星辅助CE从E80开始计入；
- E1–40：`leo_clear_weak,p=0.30`；
- E41–90：`leo_low_elev_weak,leo_rain_weak,p=0.60`；
- E91–200：三种`LEO_WEAK`并集，`p=0.80`；
- 最终checkpoint必须分别评估clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

报告中的clean/satellite配对一致性作为E3显式消融实现，默认V1关闭。E3直接复用拼接增强中同一物理source样本的clean/satellite成对特征，每4个optimizer step最多抽取12对，不额外恢复全batch第三次主干前向。

## 3.方案比较与选择

### 3.1选定方案：模块化扩展ADV3B02双骨干

新建`code/cvsrffi/phase1_bicad_xr/`，集中承载配置、因子化对抗头、条件交叉协方差、XDC、margin-tail、receiver tangent、调度和诊断。`model_dual_cvsincnet.py`只增加必要的中间特征出口和可配置训练头；`train_ssdg.py`只增加一个显式`phase1_method=bicad_xr`路由和训练钩子。

优点是checkpoint语义清楚、旧ADV3B02默认行为不变、候选之间可以做严格单因素消融，也不会继续扩大已经很长的`train_ssdg.py`主体。

### 3.2未选方案：直接把全部机制写入`train_ssdg.py`

该方案改动入口少，但会把XDC求解、梯度投影、tail状态和receiver tangent全部混入旧训练循环，难以证明候选开关真的只改变一项，也增加旧ADV3B02回归风险。因此拒绝。

### 3.3未选方案：在HCF-DG单骨干实现上改名

HCF-DG的主干、batch语义、训练日程和反事实结构均与报告要求不同。复用其通用小工具可以接受，但不能用HCF-DG模型替代ADV3B02双骨干。因此拒绝。

## 4.模型结构

### 4.1身份路径与类条件多因素DANN

身份分支输出`z_id∈R^160`和TX logits。对有标签`L_s`，条件向量固定使用真实TX one-hot，不使用预测概率：

```text
psi(z_id,y)=flatten(z_id outer onehot(y)) in R^(160*C)
```

`psi`分别进入receiver、day和channel三个共享条件判别器。每个判别器前使用独立GRL系数，但统一由身份任务保护控制器限制有效梯度比。`U_s`没有TX标签，因此V1不对`U_s`计算类条件DANN；它只能为域分支receiver/day/channel监督提供合法metadata。

channel标签来自当前clean/satellite视图和具体`LEO_WEAK`场景，不使用笼统`mixed_orbit`标签。clean记为独立channel类，三种LEO弱场景各自独立成类。

### 4.2域路径与双向净化

域骨干与RCN输出`z_dom∈R^160`。域路径使用receiver、day、channel三个正向分类头保存环境信息；`L_s`上的TX对抗头通过独立GRL降低`z_dom`中的TX可判别性。

V1保持严格二分解，不启用报告中可选的`z_e+z_int`三因素分解。只有后续证据同时满足“增强TX-GRL后域分类明显崩溃、`z_dom` TX probe仍高、弱类别margin下降”时，才允许另立候选实现低维交互支路。

### 4.3TX条件交叉协方差

普通单样本正交默认关闭。对batch中样本数不少于2的每个TX分别中心化`z_id`与`z_dom`，计算归一化交叉协方差Frobenius范数；有效TX取均值。无有效TX时返回与图连接的零损失，不制造NaN或伪样本。

默认搜索值为`lambda_cond_xcov∈{0.01,0.02,0.04}`，V1起点为0.02。`lambda_orth=0`。

### 4.4跨接收机决策兼容XDC

每4个optimizer step使用一次结构化micro-episode：`6 TX×4 RX×2 samples=48`。缺失TX×RX cell必须mask，禁止placeholder、重复样本、跨TX填充或伪造矩形。

每个receiver用自身support构造带正则的ridge线性分类器：

```text
W_r=Z_r^T (Z_r Z_r^T+lambda I)^(-1) Y_r
```

实现必须使用稳定线性求解，不显式求逆；条件数超过阈值、类别覆盖不足或自身support准确率过低的donor直接跳过。donor权重由自身support准确率、平均margin和条件数按统一公式产生，不允许按receiver ID或TX ID写特例。

query receiver使用其他receiver的`stop_gradient(W_r)`计算XDC交叉熵。donor共识经温度软化后蒸馏到公共CosFace头，ensemble目标停止梯度。

### 4.5margin-tail

样本风险定义为`softplus(delta-(logit_y-max_other_logit))`。维护`TX×receiver`、`TX×receiver×day`和`TX×receiver×channel`三类EMA风险，分别计算20% CVaR，并按0.6/0.3/0.1组合。

tail权重只作用于公共TX分类、XDC query分类和receiver tangent分类，不放大domain CE、GRL、conditional cross-cov或`z_dom` TX adversary。所有组使用同一公式和超参数，满足类别标签置换等变性。

### 4.6receiver tangent

F阶段维护融合层类条件receiver EMA中心`mu[y,r]`。对收缩后的`mu[y,r]-mean_r(mu[y,r])`堆叠矩阵做top-K SVD，默认`K=4`。

- F1：沿观测到的factual receiver shift训练；
- F2：在切向基内执行一步受限最坏margin方向上升，再对扰动特征训练TX CE和margin；
- F3：在F2上增加source-LORO低风险窗口SWAD。

receiver tangent在训练进度70%后才启用，不访问heldout receiver或目标接收机。

### 4.7梯度防火墙与任务保护

域正向CE回到共享Sinc/HF的梯度缩放为0.05；域骨干后半段和RCN正常接收域监督。身份CE、卫星TX CE和身份路径条件GRL仍可更新共享前端。

常驻GRL强度由滑动梯度范数控制，使身份路径对抗梯度比目标为0.15–0.25，域路径TX对抗梯度比为0.05–0.10。D6每4步在身份fusion、projection和最后时域块上执行一次任务保护投影；若对抗梯度与TX梯度冲突，则去除破坏TX的一维分量。不得对完整网络执行多损失梯度投影。

## 5.Batch与数据流

普通主batch占75%训练步，推荐`batch_size=96`，同时做TX平衡、receiver平衡和近似day平衡。条件DANN要求每个参与receiver尽量覆盖全部TX；不满足覆盖时mask缺失组合，不允许复制样本。

结构化XDC batch占25%训练步，固定目标形状为`6×4×2`，实际有效cell由mask决定。clean/satellite拼接由现有`ConcatSatChannelAugment`生成，卫星视图只进入TX辅助CE；只有E3每4步从已存在的成对输出中抽取8–12对计算identity cosine、prediction JS和域分支channel CE。

`L_s`用于TX监督、条件DANN、TX条件cross-cov、XDC、tail和TX对抗；`U_s`只用于无需TX真值的环境正向监督。V1关闭FastTrust、pseudo-label、CSD、HCF transport、26D content LODO、HDRO、proxy unknown、soft unknown MixUp、open-world feature loss、Fishr、generic MixUp和MixStyle。

## 6.训练日程

所有阶段由`optimizer_update/total_updates`决定，避免epoch长度变化造成机制错位：

| 阶段 | 进度 | 启用机制 |
|---|---:|---|
| Stage0 | 0–10% | TX CosFace、域正向头、在线判别器、`z_dom` TX判别器；GRL=0 |
| Stage1 | 10–35% | 条件receiver/day/channel GRL、`z_dom` TX-GRL、conditional cross-cov、gradient firewall |
| Stage2 | 35–70% | 每4步XDC和XDC-KD；E3可启用小规模pair |
| Stage3 | 70–90% | margin-tail；F候选启用receiver tangent |
| Stage4 | 90–100% | 条件DANN降至峰值的0.6；Sinc/shared stem LR降10倍；保留TX、XDC、tail；F3启用SWAD |

不在Stage3前冻结Sinc或第一个时域块。

## 7.候选矩阵

每个候选只能增加表中一项主要变化：

| 候选 | 定义 |
|---|---|
| D0 | 关闭FastTrust/open/unknown的CORE90双骨干对照 |
| D1 | D0+receiver/day/channel因素化域头 |
| D2 | D1+class-conditional CDAN |
| D3 | D2+`z_dom` TX adversary |
| D4 | D3+conditional cross-cov |
| D5 | D4+shared-stem gradient firewall |
| D6 | D5+task-protected adversarial gradient |
| E0 | 第一阶段冻结最佳候选，默认父候选D5 |
| E1 | E0+sparse XDC |
| E2 | E1+XDC ensemble distillation |
| E3 | E2+small paired satellite invariance |
| E4 | E3+class×receiver margin-tail |
| F0 | E4 |
| F1 | F0+receiver tangent factual shift |
| F2 | F0+receiver tangent worst-direction shift |
| F3 | F2+SWAD |

`ADV3B02-BiCAD-XDC-V1`是一个冻结配置别名：D5+E1+margin-tail，不包含D6、pair、receiver tangent或SWAD。这样保留报告推荐的第一版核心，同时允许D6/E2/E3/F1–F3独立验证。

快速筛选计划支持fold1/fold8、seed392001/392002/392003和5000 updates；方法确认支持5个source-LORO fold×3 seeds；最终确认只有在方法和超参数source-only冻结后才生成8-seed计划。当前工作只实现和验证代码，不启动N607矩阵。

## 8.诊断与artifact

每个训练row必须记录：

- 在线receiver/day/channel判别准确率和独立的条件receiver probe输入artifact；
- `z_dom` TX probe输入artifact以及域正向分类准确率；
- receiver donor→query XDC迁移矩阵；
- clean/satellite配对的`Delta z_id`、`Delta z_dom`、TX margin变化和flip rate；
- `Q0.1(margin)`以及最差TX×receiver、TX×receiver×day、TX×receiver×channel组；
- TX与两类对抗梯度比、梯度投影触发率、有效XDC donor数、ridge条件数；
- 吞吐、峰值显存、GPU-hours、额外前向比例和最终推理参数量。

Phase1 row只有在final checkpoint严格重建，clean及三种LEO弱场景分别完成并保存per-class/floor结果后，才能标为`ARTIFACTS_COMPLETE`。

## 9.错误处理

- 缺TX标签时不得调用条件DANN、TX cross-cov、XDC或TX adversary；
- 无有效TX或donor时返回可微零损失并记录原因计数；
- ridge求解失败、非有限条件数或非有限loss必须fail closed，不得`nan_to_num`后继续；
- 结构化batch不能通过重复样本补cell；
- candidate、fold、seed、update、receiver、day和场景必须由launcher与checkpoint runtime共同固化；
- 旧ADV3B02入口默认参数和checkpoint重建必须保持不变。

## 10.实现文件

计划新增：

```text
code/cvsrffi/phase1_bicad_xr/__init__.py
code/cvsrffi/phase1_bicad_xr/config.py
code/cvsrffi/phase1_bicad_xr/heads.py
code/cvsrffi/phase1_bicad_xr/losses.py
code/cvsrffi/phase1_bicad_xr/sampler.py
code/cvsrffi/phase1_bicad_xr/xdc.py
code/cvsrffi/phase1_bicad_xr/tangent.py
code/cvsrffi/phase1_bicad_xr/gradients.py
code/cvsrffi/phase1_bicad_xr/metrics.py
code/cvsrffi/phase1_bicad_xr/trainer.py
code/scripts/launch_phase1_bicad_xr_matrix_20260830.py
code/tests/phase1_bicad_xr/
```

计划修改：

```text
code/model_dual_cvsincnet.py
code/post_stage_common.py
code/SSDG/train_ssdg.py
```

修改遵守“小入口、大模块”原则：旧文件只提供必要feature出口、构造参数和显式路由。

## 11.验证策略

实现采用TDD：每组机制先写失败测试，再写最小实现。聚焦验证包括：

1.真实one-hot条件映射形状、梯度方向和无标签fail-closed；
2.因素化头及channel标签区分clean/三种LEO弱场景；
3.TX条件cross-cov对独立/相关表示的数值行为；
4.XDC稳定求解、donor stop-gradient、缺cell mask、标签置换等变性；
5.margin-tail分组、CVaR和不放大域损失；
6.receiver tangent只使用source receiver中心且F1/F2行为可区分；
7.gradient firewall和D6投影只影响登记模块；
8.Stage0–4边界、候选单因素差异和V1别名；
9.`concat_sat_ce_only+LEO_WEAK`协议负测；
10.旧ADV3B02构造与默认训练路径回归；
11.`py_compile`、聚焦pytest、launcher dry-run和`git diff --check`；
12.一次真实ADV3B02 checkpoint无query smoke，严格重建并完成clean+三LEO弱小评估。

实现完成后只进行一次独立P0/P1正确性审查；若发现直接P0/P1，修复后最多进行一次定点复审。

## 12.成功判据与声明边界

本阶段成功表示代码达到严格设计可达、聚焦测试通过、真实checkpoint smoke通过且Git发布完成，不表示性能已经提高。

性能结论必须来自后续source-LORO同row矩阵。目标接收机结果只能在方法和超参数source-only冻结后做一次性确认，不能反馈调参、选种、重训或选择性重跑。WiSig/ManySig仍是地面代理数据，LEO弱信道仍是物理启发压力代理，不构成真实在轨验证。

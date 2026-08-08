# P1-ManyTx-RealOE-v2冻结设计

状态：`DESIGN_FROZEN`；标签：`DEVELOPMENT_SOURCE_ONLY_NON_CONFIRMATORY`。

## 目的与边界

本轮只检验一项源侧训练机制：将真实、未标注的ManyTx发射机IQ作为outlier exposure，要求其logit energy高于已知ManySig样本的停止梯度能量锚点。它不训练未知类别分类器，不产生Phase3拒识阈值，不访问target query、query truth、query role或N607结果。结果即使完成，也不能构成独立unknown确认或部署性能主张。

v2修正ManySig与ManyTx原始receiver index不等于同一物理接收机的问题。共同source physical RX固定为`[1-1,1-19,14-7,18-2,19-2,2-1]`，source days固定为`[2021_03_01,2021_03_08]`，held target RX固定为`[20-1,3-19,7-14,7-7,8-8]`。任何raw index字符串、标签顺序漂移或source/target RX重叠均fail-closed。

20个既有authority-locked`target_new`身份首先永久排除：`1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6`。全150个ManyTx在共同physical-RX slice审计中有8个不足：`1-1,1-2,10-1,12-1,13-18,16-5,2-1,2-20`；其余142个通过。排除locked20与ManySig6后，eligible为116个，冻结为80个OE训练、20个proxy、16个reserve；partition root为`ca3ed65a533359d2abb022fa513c49101ad93235738a39b362b5cdd15879c3d1`。

```text
OE(80)=10-4,3-1,7-8,16-20,11-17,8-14,19-1,2-13,11-1,19-19,18-1,4-1,13-19,18-4,13-3,11-10,19-11,7-20,1-11,18-11,14-8,3-19,13-20,14-9,19-4,18-17,19-7,2-17,7-10,1-10,2-7,9-1,18-14,11-4,18-15,20-18,19-2,14-12,3-20,1-12,3-2,5-1,7-13,11-20,20-4,18-5,18-2,6-1,20-7,10-17,8-1,18-16,17-10,20-1,2-19,14-20,8-8,10-7,9-20,6-6,19-20,2-6,20-5,1-15,1-14,8-13,18-20,8-18,7-11,8-7,9-7,18-12,11-7,16-16,14-14,20-14,15-19,2-8,14-13,20-8
proxy(20)=20-20,20-16,19-3,1-19,3-18,19-12,5-20,7-14,12-7,7-9,17-11,20-3,12-20,16-1,18-7,2-3,19-10,18-9,2-4,15-6
reserve(16)=2-14,10-11,9-14,13-7,2-12,7-12,5-5,2-15,18-13,5-16,19-14,15-1,12-19,3-13,7-7,4-11
```

proxy、reserve与locked`target_new`只存在于静态分区回执；训练样本、epoch/检查点选择、阈值校准和候选选择均为零访问。训练loader仅允许索引OE(80)在上述两个日期、六个共同physical RX与`equalized=1`的IQ；每个TX必须通过`>=400`样本、两个日期和至少两个共同RX的覆盖检查。返回给模型的TX标签强制为`-1`，不进入CE、伪标签、对齐或类别统计。

## 机制

令`E(l)=-logsumexp(l)`，冻结`T=1`。每个已知训练batch从80个OE身份中均衡采16个TX、每TX采8个样本，得到128个真实OE样本。只有epoch61起启用，10个epoch线性warmup：

\[
L_{OE}=\frac{1}{|B_{OE}|}\sum_{x\in B_{OE}}\operatorname{softplus}\left(\frac{1-(E_{OE}(x)-\operatorname{stopgrad}(\overline{E}_{known}))}{1}\right),\qquad \lambda_{OE}=0.02.
\]

已知能量锚点被`stopgrad`；该辅助项仅经OE forward的`tx_logits`反传。常规ManySig CE、clean→三种`leo_*_weak`一致性和已知source验证保持GeoSat-C配置。它不是VOS：不合成特征外点、不以batch轮换known label伪造unknown；也不是source-Q98后处理：没有Q98扫描、动态阈值或任何部署拒识阈值。

## 冻结矩阵与资源

ManySig TX顺序为`[14-10,14-7,20-15,20-19,6-15,8-20]`。每折5个已知TX训练、1个known-validation，主proxy角色为空且由`phase1_allow_empty_proxy_unknown`仅在此冻结外部OE协议下允许。两臂从头训练120epoch、`final_only`、seed`7281105`、sat seed`9281105`：C为GeoSat-C且`lambda_manytx_real_oe=0`；G仅增加真实OE输入与`lambda_manytx_real_oe=0.02`。

|折|known-validation|known训练TX|C/G物理GPU|
|---|---|---|---|
|F1|14-10|14-7,20-15,20-19,6-15,8-20|C:0；G:1|
|F2|14-7|14-10,20-15,20-19,6-15,8-20|C:2；G:3|
|F3|20-15|14-10,14-7,20-19,6-15,8-20|C:4；G:5|
|F4|20-19|14-10,14-7,20-15,6-15,8-20|C:6；G:7|
|F5|6-15|14-10,14-7,20-15,20-19,8-20|C:1；G:0|
|F6|8-20|14-10,14-7,20-15,20-19,6-15|C:3；G:2|

共12个任务；每张物理GPU最多2个进程，进程内固定`CUDA_VISIBLE_DEVICES=<physical>`与`--device cuda:0`。G在epoch61后每个known batch额外执行一次128样本OE forward，峰值显存与时间开销接近一次额外前向/反向分支；C不构造OE loader。

## 回执、负测与晋级边界

训练回执、epoch telemetry、terminal status和completion receipt记录分区root、80/20/16/20计数、共同physical RX/day标签及其解析index、loader是否构建、以及proxy/locked/reserve均未加载。P0实现门为：分区root与四个精确名单匹配；四角色与ManySig known角色无交集；raw index RX/day被拒绝；已知source、OE source和target RX标签分别精确匹配且无交；OE标签全为`-1`；非OE loss路径为零；真实`lite_d`无query forward/backward可运行；不允许覆盖输出目录。focused负测覆盖锁定`target_new`重定义、stacked proxy/virtual/geometry loss、空main proxy越权、raw index、OE logit维度/非有限值与OE-only梯度。

P1科学门留给独立审查：按每折同一行比较G-C的known clean、三种LEO、min-class与min-receiver floor；任何折的known保护指标降幅超过2pp时该G折拒绝且不得被其他折抵消。proxy不参与epoch、checkpoint、候选或阈值选择；locked`target_new`和reserve不自动成为评估集，更不成为Phase3 confirmed unknown。不得将本开发CV外推为K-shot、独立确认或未知FAR证据。

# Phase1 ADV3B02-BiCAD-XR实现与实验发布报告

## 1.结论先行

本轮已完成报告中Phase1架构级优化的代码落地，并按现行项目协议冻结为`concat_sat_ce_only+LEO_WEAK`版本。当前最高证据状态为`LOCAL_VERIFIED`：方法实现、协议负测、相邻回归、真实历史ADV3B02 checkpoint兼容smoke、24行实验矩阵、非连续源域编号映射和每GPU 3任务调度均已验证；独立P0/P1审查及原问题定点复审已闭合。N607正式性能训练尚未完成，因此本报告不提供准确率提升或“新最佳方法”结论。

冻结的首轮正式矩阵为：

- 候选：`D0`、`D5`、`E1`、`ADV3B02-BiCAD-XDC-V1`；
- source-LORO：fold1、fold8；
- 历史种子：`392001/392002/392003`；
- 每行：5000 optimizer updates；
- 总计：4×2×3=24行；
- 资源计划：GPU0–7各3行；若某卡存在无关进程，则该卡第3行排队，禁止为了“占满”而超过实时安全容量。

## 2.科学场景与协议边界

### 2.1任务定义

本实验是Phase1源域域泛化训练，不是Phase2适配。训练、候选选择、种子选择和重跑决策只能读取Phase1 source数据及source-only验证证据。

明确禁止访问：

- target receiver训练数据；
- Phase2 capsule或split；
- support、query或truth；
- 目标域结果反馈调参、选种或选择性重跑。

### 2.2数据与fold

- 训练天数：day1、day2、day3；
- fold1 source receivers：`[3,4,6,8]`，source-LORO heldout receiver为1；
- fold8 source receivers：`[1,3,4,6]`，source-LORO heldout receiver为8；
- 所有候选使用相同source split、训练天数、更新数、种子和评估场景。

训练完成后的首要评价仍是同row source-only证据。若后续冻结候选与种子，再执行全部接收机、day1–4的严格零适配测试；这些目标结果不能反向改变本轮训练或选种。

### 2.3星地信道协议

报告中原先提出的`70%clean+30%mixed_orbit`单前向与现行项目协议冲突，本实现采用用户确认的现行协议：

- `concat_sat_ce_only=true`；
- 卫星分类权重`lambda_sat_cls=0.68`；
- 卫星一致性权重`lambda_sat_cons=0`；
- 从epoch80开始启用concat卫星视图；
- 训练场景固定为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；
- clean和satellite样本在同一次拼接前向中处理；
- E3配对候选复用该concat输出，不新增整批模型前向；
- V1默认关闭E3配对，不把候选消融偷渡进主方法。

## 3.方法实现

### 3.1保留的ADV3B02基座

没有把ADV3B02替换成单骨干HCF-DG。部署主干继续保留：

- 双骨干身份/域分解；
- 共享Sinc/HF前端；
- RCN；
- CosFace TX分类头；
- 160维`z_id`和`z_dom`身份/域表示；
- 部署推理仍走`return_aux=False`的`z_id→TX`路径。

新模块属于训练期辅助结构，不进入部署推理图。

### 3.2因素化域建模

原单一`rx_day`域头拆为三个因素：

- receiver头：学习接收机相关变化；
- day头：学习采集日变化；
- channel头：区分clean及LEO弱信道因素。

对身份表示采用真实TX one-hot条件CDAN；无标签路径不得使用预测TX代替真实标签。对`z_dom`增加独立GRL的TX adversary，用于抑制域表示中的TX身份泄漏。

WiSig数据集保留全局原始`rx_i/day_i`。正式入口按每个fold冻结的source receiver集合和day1/2/3，把它们严格映射到域分类头使用的连续本地标签：fold1的`3/4/6/8→0/1/2/3`，fold8的`1/3/4/6→0/1/2/3`，day1/2/3对应`0/1/2`。发现集合外编号立即失败，不允许静默截断或越界进入交叉熵。

### 3.3TX条件cross-cov

普通无条件正交项被关闭，`lambda_orth=0`。替代项是在TX条件下计算`z_id/z_dom`cross-cov，使身份与域去相关约束不把不同TX类别间的真实结构误当成泄漏。

### 3.4结构化采样

训练batch冻结为两类：

- 75%普通batch：`batch_size=96`，尽量做TX×receiver平衡，day近似平衡；
- 25%结构化episode：目标形状`6 TX×4 receiver×2样本`。

当某个TX/receiver cell缺失时使用mask并缩小有效episode，不复制样本、不制造placeholder。U_s与结构化有标签episode严格隔离。

### 3.5XDC跨接收机动态分类器

XDC模块每4个optimizer update运行一次：

1.按receiver构造donor支持集；
2.用ridge稳定求解动态线性分类器，不显式求逆；
3.按donor支持质量加权；
4.把donor预测蒸馏到公共CosFace头；
5.donor求解与质量分支停止梯度；
6.缺cell、低质量donor或数值不满足条件时显式mask/skip。

V1启用`sparse_xdc=true`，但不启用后续E2的XDC-KD、E3配对或F1/F2 tangent。

### 3.6margin-tail与接收机切空间

- margin-tail使用三层组EMA风险；
- CVaR组合权重固定为0.6/0.3/0.1；
- 输出Q0.1 margin和最差TX/RX/day/channel组合；
- receiver tangent维护类条件receiver EMA中心，top-K SVD默认K=4；
- F1为factual shift，F2增加最坏方向shift；
- F3使用source-LORO低风险窗口SWAD。

V1只启用margin-tail，不启用receiver tangent和SWAD。

### 3.7梯度控制

- shared Sinc/HF域正向梯度防火墙比例固定为0.05；
- 提供身份对抗和TX-adversary梯度比控制器及artifact字段；首轮冻结V1不启用自适应重标定，避免改变已冻结的损失权重；
- D6每4步执行局部任务保护投影；
- 投影只作用预登记模块，不修改无关参数；
- Stage4进一步降低域项与shared-stem学习强度。

### 3.8训练阶段

5000 updates按进度分为：

|阶段|进度范围|主要作用|
|---|---:|---|
|Stage0|0–10%|建立稳定身份/域基础表示|
|Stage1|10–35%|启用因素化域建模与条件对抗|
|Stage2|35–70%|完整D5主干与稀疏机制|
|Stage3|70–90%|后期tail/tangent类候选开始可用|
|Stage4|90–100%|衰减域压力和shared-stem学习率，F3可做SWAD|

### 3.9默认关闭的旧机制

所有BiCAD-XR候选都fail closed地关闭：FastTrust、pseudo label、CSD、HCF transport、26D content LODO、HDRO、proxy unknown、soft unknown MixUp、open-world feature loss、Fishr、generic MixUp和MixStyle。

只有`lambda_cond_xcov`允许做source-only参数覆盖；候选身份、协议权重和其余开关冻结。

## 4.候选含义

|候选|在D0基础上新增的核心机制|本轮作用|
|---|---|---|
|D0|无BiCAD增强开关|新训练入口和数据协议控制基线|
|D5|因素化域头+条件CDAN+`z_dom` TX adversary+条件cross-cov+梯度防火墙|完整域分解主干|
|E1|D5+sparse XDC|检验跨接收机动态分类器贡献|
|ADV3B02-BiCAD-XDC-V1|D5+sparse XDC+margin-tail|报告推荐的首个冻结主方法|

完整注册表还实现了D1–D6、E0–E4、F0–F3，便于后续单因素消融；首轮只发布上述4个候选，避免一开始扩大矩阵。

## 5.24行实验矩阵与GPU分配

|GPU|行1|行2|行3|
|---:|---|---|---|
|0|D0-F1-S392001|D5-F1-S392003|E1-F8-S392002|
|1|D0-F1-S392002|D5-F8-S392001|E1-F8-S392003|
|2|D0-F1-S392003|D5-F8-S392002|V1-F1-S392001|
|3|D0-F8-S392001|D5-F8-S392003|V1-F1-S392002|
|4|D0-F8-S392002|E1-F1-S392001|V1-F1-S392003|
|5|D0-F8-S392003|E1-F1-S392002|V1-F8-S392001|
|6|D5-F1-S392001|E1-F1-S392003|V1-F8-S392002|
|7|D5-F1-S392002|E1-F8-S392001|V1-F8-S392003|

其中`V1`表示`ADV3B02-BiCAD-XDC-V1`。干跑确认24个`candidate×fold×seed`组合唯一，GPU0–7各3行。

## 6.训练后严格闭合

训练子进程退出码0不等于实验完成。每一行必须依次满足：

1.存在非空final checkpoint；
2.重建模型时`missing_keys=[]`；
3.`unexpected_keys=[]`；
4.`shape_mismatches=[]`；
5.`bicad_xr_runtime`严格恢复；
6.分别生成clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak` JSON；
7.每个场景保存非空日志、accuracy、per-class accuracy和floor；
8.再次调用closure校验；
9.只有上述全部满足，才原子写`ARTIFACTS_COMPLETE.json`。

失败行保留partial artifact并写精确技术失败原因，不把失败伪装成完成。

## 7.验证结果

### 7.1自动测试

- `python -m compileall`：通过；
- `code/tests/phase1_bicad_xr`：240项全部通过；
- 相邻`phase1_hcfdg`和ADV3B03回归：159项全部通过；
- 警告：3条既有`torch.cuda.amp.autocast`弃用提醒，不影响本轮正确性。

### 7.2真实历史checkpoint技术smoke

输入：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，大小8582116字节。

结果：

- 严格重建195个状态张量；
- 域头宽度从checkpoint状态推断为14；
- 输入长度从checkpoint参数读取为256；
- `missing_keys=[]`；
- `unexpected_keys=[]`；
- `shape_mismatches=[]`；
- fresh BiCAD训练器完成一次真实`backward+optimizer.step`；
- 正式入口使用注册的梯度防火墙/投影反向控制，不再绕过为直接`total.backward()`；
- epoch80复用真实concat卫星增强入口；
- clean和三种LEO_WEAK场景均完成模型前向；
- 四场景logits均为finite，形状均为`[8,6]`；
- `source_only=true`，target/Phase2/support/query/truth访问均为false。

边界：历史checkpoint不含`bicad_xr_runtime`，因此该smoke只证明“历史ADV3B02基座可严格重建并与fresh BiCAD训练模块兼容”，不冒充完整BiCAD checkpoint恢复，也不是性能测试。

### 7.3矩阵干跑

- 行数：24；
- 唯一组合：24；
- seeds：392001、392002、392003；
- folds：1、8；
- updates：全部5000；
- train days：全部`[1,2,3]`；
- source-only：全部true；
- GPU计数：GPU0–7全部3行。

### 7.4独立正确性审查

首次Task9审查发现一个直接P0：数据集输出全局原始`rx_i/day_i`，而因素化域头要求连续本地标签，fold1/fold8的非连续receiver编号会导致域交叉熵越界。修复后新增两fold映射、day映射、集合外编号拒绝和D5实际`compute_step`测试；仅针对该原问题的定点复审结果为`CLEAN`。

## 8.N607资源与发布边界

2026-08-31只读预检确认：

- 使用普通账户`szu2070436088`；
- 项目根可见；
- 8张RTX 3090可见；
- 历史checkpoint存在；
- GPU2存在一个无关的Stage2进程，其他GPU空闲。

本任务不得停止、重启或迁移该无关进程。正式发布前再次读取GPU状态：

- 若GPU2已释放，按24行、每卡3个本run worker启动；
- 若GPU2仍占用，则GPU2最多启动2个本run worker，第3行排队；
- 其他GPU仍不得超过用户授权的每卡3个本run worker；
- 不得把GPU2的排队行挤到其他已满3行的GPU。

## 9.停止规则

允许技术停止的情况仅包括：数据越权、错误candidate/fold/receiver/day/seed/update、输出冲突、错误release/CWD、命令无法运行、同一确定性pre-prediction异常重复、无法产生final checkpoint或四场景artifact、进程归属不清。

以下情况不得停止：中间或最终准确率较低、某个seed表现差、缺少额外形式化receipt/hash/seal或报告字段。

## 10.当前结论与下一步

实现层面的发布条件已经满足；当前不能声称BiCAD-XR提高了星地信道性能，因为正式N607矩阵尚未产出性能artifact。

下一步顺序：

1.提交并推送本报告和追踪表，独立核对远端OID；
2.制作单一release归档并做一次本地/远端SHA核对；
3.N607远端编译与真实checkpoint无query smoke；
4.按实时GPU容量启动24行矩阵；
5.启动后一次核对PID/CWD/cmdline/GPU/log；
6.全部闭合后只按source-only证据比较候选和种子；
7.冻结后再做全部接收机、day1–4严格零适配测试，禁止目标结果反馈训练。

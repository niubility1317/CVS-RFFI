# Phase1 MIRAGE-OWDG设计规格

日期：2026-08-17
状态：`DESIGN_APPROVED / WRITTEN_SPEC_REVIEW_PENDING`
研究名称：低标签半监督开放世界射频指纹域泛化与LEO weak目标域确认

## 1.目标与声明边界

本设计从头训练新的Phase1模型，不继承ADV3B02、旧SincNet前端、历史C/G模型、旧loss、旧分类头或旧bundle内部结构。模型只继承CVS的数据场景、source/target接收机互斥、单物理样本单LEO weak观测、目标域盲评以及最小部署接口。

模型同时承担五项任务：

1.在7%有标签、63%无标签的source训练条件下学习发射机身份；
2.对source receiver、日期和LEO weak场景进行域泛化；
3.在未见target receiver上保持已注册TX身份；
4.显式拒绝与source训练及验证TX身份互斥的target unknown TX；
5.导出可供Phase2和Phase3读取的轻量、不可变deployment bundle。

全部Gate通过后，只能声明：新Phase1模型在低标签source训练条件下获得更强的跨接收机与LEO weak域泛化能力，并以source侧冻结的开放集规则完成单节点目标域known/unknown确认。该结果不代表Phase2新类注册、Phase3多节点协同、真实运营确权或真实在轨验证已经完成。

## 2.本设计覆盖的协议差异

用户在设计讨论中明确覆盖了原始目标和当前`项目.md`第4.2节中的两项旧语义：

- `L_s`、`U_s`和source validation允许共享source已知TX身份，但物理样本ID严格互斥；
- source proxy不是固定、身份互斥且禁止训练的真实未知TX集合，而是从source已知类数据中按注册角色生成的代理未知任务。训练proxy可以训练拒识机制，验证proxy可以冻结阈值并参与source-only模型选择。

真正的unknown性能只由最终target测试中的身份互斥TX证明。实施前必须先同步修订根目录`项目.md`和Git公开正文`docs/PROJECT_PROTOCOL.md`；若两者尚未同步，新代码只能标为协议草案实现，不能释放正式实验。

## 3.集合与数据流

### 3.1集合

记source和target接收机集合为`R_s`、`R_t`，source已知TX集合为`Y_s`，最终target unknown TX集合为`Y_t_unknown`：

```text
R_s ∩ R_t = ∅
Y_s ∩ Y_t_unknown = ∅
```

target-known使用`Y_s`中的已注册TX，但来自未见接收机`R_t`。target unknown TX在训练、source validation、proxy生成、阈值冻结和模型选择期间均不可见。

### 3.2统一source划分

对每个`TX×receiver×day`的source物理样本池，先按固定seed和`physical_sample_id`进行不可变划分：

```text
L_s = 7%有TX标签训练样本
U_s = 63%隐藏TX标签训练样本
V_s = 30%source validation样本
```

三部分可以包含相同TX身份，但物理样本ID两两不交。`U_s`的真值不能进入训练进程；真值只供划分构建器一次性检查类别覆盖，之后训练侧只读opaque样本ID、receiver、日期和合法source域信息。

`V_s`在训练开始前继续按物理样本ID等分为`V_cal`和`V_select`：

- `V_cal`用于checkpoint监控、阈值冻结和校准；
- `V_select`用于source Gate和唯一候选选择；
- 两者均不得反向传播或更新EMA、prototype、normalization统计和任何持久状态。

### 3.3角色化source proxy

proxy表示“相对当前episode注册类别表的代理未知角色”，不表示真实未见TX。

训练proxy只由`L_s`生成。每个episode从`Y_s`均匀选择一个`c_proxy`，临时将它从该episode的注册类别表和prototype集合移除；属于`c_proxy`的`L_s`样本以`proxy_unknown`角色进入拒识损失。每个TX在固定周期内等次数充当registered和proxy，生成规则对类别标签置换等价。proxy样本的原始TX标签只用于构造角色，不能作为额外registered logit或专属阈值。

验证proxy分别由`V_cal`和`V_select`采用相同生成器产生，记为`P_cal`和`P_select`。`P_cal`用于阈值冻结，`P_select`用于模型选择和source开放集Gate。验证proxy不更新模型。

`U_s`不生成proxy，因为训练进程没有可靠TX标签。候选B/C可以利用两个registered类的`L_s`表征生成类间边界mixup，但它只是辅助边界样本，不能取代真实IQ proxy episode。

source proxy指标必须写作“代理未知研发性能”。它不能替代target unknown指标，也不能支持“训练阶段未使用任何unknown监督”的声明。

### 3.4目标域盲测

target-known和target-real-unknown采用同一构建与推理规则：

- 每个物理IQ样本只生成一份固定LEO weak接收观测；
- 场景为`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`；
- scene和seed在读取known/unknown角色及truth前冻结；
- 三个scene的物理样本ID集合两两不交；
- known与unknown共用预处理、模型前向、分数和决策规则；
- 零训练、零适配、零状态更新、零阈值调整、零候选重排和零选择性重跑；
- predictor先输出不可变预测artifact，独立scorer之后才能读取truth。

## 4.模型：MIRAGE-OWDG

### 4.1输入与预处理

输入为复数IQ的双通道表示`[I,Q]`。预处理对每个样本执行复均值去除和robust RMS归一化，同时保留归一化前的对数RMS、峰均比和裁剪比例作为质量辅助量。预处理不读取receiver ID或known/unknown角色，训练与部署使用同一实现。

### 4.2轻量表征主干

主干采用复数等价轻量patch encoder：

1.时域patch stem提取短时硬件非理想特征；
2.核宽3、7、15的depthwise多尺度分支分别捕获局部瞬态、中程调制结构和慢变化包络；
3.4层轻量Transformer聚合全局关系；
4.quality-gated fusion按样本质量融合局部与全局表征；
5.输出160维L2归一化身份表征`z_id`和32维域/质量表征`z_dom`。

主干参数预算不超过3M，最终bundle不超过16MiB。新embedding schema冻结为：

```text
mirage_owdg:z_id:l2:160:v1
```

### 4.3开放世界头

每个registered类维护prototype、可学习radius以及正则化的对角/低秩协方差。开放世界头联合输出：

- cosine registered class scores；
- 到每个类别的Mahalanobis距离与radius margin；
- energy score；
- 单调unknown risk；
- quality/reliability；
- `registered/unknown/defer`决策。

unknown risk只依赖已封存模型状态、当前query和注册类别几何。所有类别采用同一公式和超参数；禁止按TX ID设置白名单、专属radius、专属温度或专属阈值。

### 4.4决策规则

冻结三个全局阈值`tau_q`、`tau_reg`、`tau_unk`，且`tau_reg≤tau_unk`：

1.质量低于`tau_q`时输出`defer`；
2.质量合格、最佳类别在radius内且unknown risk≤`tau_reg`时输出registered类别；
3.unknown risk≥`tau_unk`且几何位于registered支持域外时输出`unknown`；
4.其余样本输出`defer`。

registered query的`unknown`或`defer`均按身份识别错误计数。unknown query只有显式`unknown`计入拒识成功，`defer`不进入显式拒识分子。

## 5.训练方法与因果矩阵

### 5.1统一预算

B0、A、B、C共享同一source划分、主干容量、优化器、batch/step预算、200个epoch和评分器。训练分三段：

```text
epoch 1-40：表征warm-up
epoch 41-160：半监督联合训练
epoch 161-200：EMA/SWAD稳定化
```

任何候选不得通过增加epoch、参数量、额外source样本或target访问获得优势。

### 5.2四个arm

| arm | 机制 | 训练目标 |
|---|---|---|
| B0 | 可复现半监督DG基线 | supervised CE、基础EMA伪标签、weak/strong一致性；开放集使用source验证proxy校准的简单energy/prototype scorer |
| A | 高级自监督与半监督DG | B0+masked latent prediction、跨receiver一致性、prototype-aware伪标签 |
| B | 联合开放世界模型 | A+角色化`proxy_train`、prototype/radius、energy separation、类间边界mixup和联合known/unknown校准 |
| C | 最差组域优化 | B+receiver×day×LEO scene Group-CVaR |

该矩阵只归因于相邻机制。B0不是历史ADV3B02，也不读取旧checkpoint、旧feature cache或旧bundle。

### 5.3半监督规则

EMA teacher对`U_s`产生伪标签。无标签样本只有同时满足以下条件才进入分类loss：

- teacher top-1置信度≥0.95；
- top-1与top-2 margin≥0.20；
- weak/strong视图预测一致；
- 表征位于对应`L_s`类别的合法prototype/radius内。

不满足条件的样本只能进入不依赖身份标签的masked/consistency目标。伪标签不得使用真实类别数量、每类配额、receiver配额、target反馈或proxy验证结果进行样本级回填。

### 5.4开放集训练

候选B/C的`proxy_train`只更新共享encoder和开放世界头中的拒识相关目标；同一episode中被选为proxy的TX不能产生registered classification loss。类间mixup使用不同registered类的表征：

```text
z_mix = normalize(lambda*z_a + (1-lambda)*z_b)
lambda ∈ [0.35,0.65], class(a) != class(b)
```

proxy与mixup共同约束energy、radius和unknown risk，但真实target unknown从不进入训练。

### 5.5最差组优化

候选C按`receiver×day×LEO scene`计算组损失，优化最差30%组的CVaR。小组有效样本少于16时，按固定顺序回退：

```text
receiver×day×scene -> receiver×scene -> receiver -> global
```

回退只由样本数触发，不读取性能排名或类别身份。

## 6.source选择、阈值与最终refit

### 6.1六fold研发

运行六个预注册source fold。每个fold使用独立固定seed生成物理样本划分和均衡proxy角色日程。结果按fold和scene等权聚合；不得选择表现最好的fold。六fold用于选择方法配置，而不是选择一个部署checkpoint。

### 6.2阈值冻结

每个训练完成的checkpoint只用`V_cal`与`P_cal`确定`tau_q`、`tau_reg`和`tau_unk`：

1.枚举source分数经验分位点形成候选阈值三元组；
2.要求`V_cal`上known false rejection≤10%；
3.在可行集合中依次最大化`P_cal`显式拒识、registered coverage并最小化defer；
4.阈值冻结后只在`V_select/P_select`评分，不再调整。

若不存在满足known false rejection约束的阈值，记为`NO_DEPLOYABLE_SEPARATION`，该候选停止，不使用target补救。

### 6.3唯一候选

只有通过Gate 1-3的arm可入选。多个arm通过时，先比较最弱Gate余量，再比较source LEO weak macro、proxy AUROC，最后选择bundle更小者。选定方法后，按预注册final seed从头训练一次`M*`，同时训练同预算`B0*`；二者分别使用source validation冻结checkpoint、几何和阈值。target结果不能选择fold、checkpoint、threshold或arm。

## 7.五项不可补偿Gate

### Gate 1：协议与训练闭合

- `0.07/0.63/0.30`物理样本划分正确且ID互斥；
- `R_s∩R_t=∅`，`Y_s∩Y_t_unknown=∅`；
- `proxy_train`只由`L_s`产生，`P_cal/P_select`只由对应validation产生；
- target训练、阈值和选择访问均为0；
- checkpoint、训练日志和真实checkpoint前向完整。

### Gate 2：Source域泛化提升

相对B0，候选必须同时满足：

- 六fold等权source LEO weak macro accuracy提升≥2个百分点；
- 六fold等权minimum-class accuracy提升≥1个百分点；
- worst-scene accuracy不低于B0；
- 至少5/6 folds同时满足：macro退化不超过0.5个百分点、minimum-class退化不超过1个百分点、worst-scene退化不超过0.5个百分点。

### Gate 3：Source代理开放集就绪

在`P_select`上：

- 六fold等权AUROC≥0.85；
- 相对B0提升≥0.05；
- 冻结阈值下`V_select` known false rejection≤10%；
- 验证proxy更新量为0。

该Gate证明source proxy研发性能，不证明真实unknown拒识。

### Gate 4：目标域一次性确认

唯一候选`M*`与`B0*`各运行一次完全相同的target capsule。`M*`必须满足：

- target-known macro accuracy相对B0提升≥2个百分点；
- minimum-class和worst-scene均不低于B0；
- target-real-unknown在global、clear、low-elev、rain的显式unknown rejection均≥70%；
- target-known false rejection≤10%。

target结果不反馈研发。Gate失败是有效研究结果，不调参、不重训、不重跑。

### Gate 5：部署闭合

- 导出单一不可变deployment bundle并由production loader成功reload；
- known与unknown使用同一前向接口；
- 输出有限的class score、distance/radius、unknown score和quality；
- bundle不读取source样本、target truth、外部可替换训练状态或样本级cache；
- 参数量、bundle大小、MAC、单样本延迟和峰值显存完整报告。

五项Gate不可相互补偿。

## 8.软件边界

新实现放入独立命名空间，避免把历史Phase1状态误当成方法依赖：

```text
code/cvsrffi/phase1_mirage/data.py
code/cvsrffi/phase1_mirage/proxy.py
code/cvsrffi/phase1_mirage/model.py
code/cvsrffi/phase1_mirage/losses.py
code/cvsrffi/phase1_mirage/trainer.py
code/cvsrffi/phase1_mirage/calibration.py
code/cvsrffi/phase1_mirage/bundle.py
code/cvsrffi/phase1_mirage/scoring.py
```

入口与配置：

```text
code/scripts/run_phase1_mirage_source_matrix.py
code/scripts/predict_phase1_mirage_target.py
code/scripts/score_phase1_mirage_target.py
configs/phase1_mirage_owdg/
tests/phase1_mirage/
```

`predict_phase1_mirage_target.py`不得接受truth、known/unknown role、类别配额或scorer输出。`score_phase1_mirage_target.py`只读取已封存预测与独立truth表，不能导入trainer或修改bundle。

## 9.错误处理与停止规则

- 数据权限、ID互斥、角色泄漏或target预访问失败：P0，停止当前run，不生成性能结果；
- source Gate失败：停止该候选，不访问target；
- 技术错误：修复一个具体错误，提交新commit并使用新run ID重跑；
- 两个不同row产生相同确定性零预测异常：按系统技术失败停止精确run-owned进程树并保留partial artifacts；
- target Gate失败：记录有效失败，不调参、不重训、不重跑；
- 任何中间性能较差都不是终止健康run的理由。

## 10.验证策略

实现采用测试驱动顺序，至少覆盖：

1.`7/63/30`计数、物理ID互斥和receiver集合互斥；
2.`proxy_train`只能读取`L_s`，验证proxy只能读取对应validation；
3.proxy episode中被移除类别不存在registered logit/prototype；
4.proxy角色轮换对类别标签置换等价；
5.`U_s`训练路径无法读取TX truth且不会生成proxy；
6.伪标签门限、margin、视图一致性和radius四条件均为必需；
7.target predictor无truth/role输入且前向前后模型状态字节一致；
8.known/unknown使用同一预处理和决策函数；
9.独立scorer只能在预测artifact封存后连接truth；
10.所有bundle输出有限，production reload与原模型预测一致；
11.真实checkpoint完成无target输入的一次前向；
12.B0/A/B/C预算、主干容量和数据权限一致。

正式N607释放前另需：协议负测试、真实checkpoint no-target smoke、独立P0/P1审查、Git commit、不可覆盖run ID和本地报告预注册。实现与窄测试通过不构成性能Gate通过；性能只能由完整source矩阵和一次性target artifacts证明。

## 11.交付顺序

1.同步修订`项目.md`与`docs/PROJECT_PROTOCOL.md`；
2.实现数据划分与proxy角色生成器；
3.实现MIRAGE主干、开放世界头和B0/A/B/C loss组合；
4.实现source校准、评分和候选冻结；
5.实现bundle导出、production reload和target truth-blind predictor/scorer；
6.完成本地测试、独立审查、Git版本化和实验报告预注册；
7.由唯一N607 runner运行source矩阵并返回完整artifact；
8.仅在Gate 1-3通过后冻结`M*`与`B0*`；
9.审查B通过后运行一次target确认；
10.更新报告并作五项Gate终裁。

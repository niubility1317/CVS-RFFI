# D41 block-erasure consistency实验报告

## 1.实验身份与状态

- 实验ID：`d41_bec_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`DESIGN_LOCKED_IMPLEMENTATION_PENDING`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；复用同一D18固定received-IQ support，query保持sealed。
- 目标：用同一固定received IQ上的确定性block-erasure一致性训练，同时改善Stage2-B outer-held旧类泛化与Stage2-C old/new联合方向可比性；不引入第二个LEO观测、group bias、radius、HNBR或query gate。

D41是D37–D39三轮回顾后的第二个探索轮。设计、实现、单测、support-view一致性下降或资源通过都不是性能成功。

## 2.D40直接证据与路线选择

D40-HNBR int8在真实90行中得到before-old85.56%、after-old85.00%、seen-new15.33%、H25.16%、遗忘0.56pp、joint floor0、旧→新侵入2/180；150个新类held中127个由旧类取最高分，new-new错序33/150。exact strong B3为87.78%/75.56%/72.67%/73.35%/12.22pp/23.33%、侵入33/180、new-new错序25/150、实际新→旧22/150。

D40不仅有跨阶段new→old翻转，也损伤Stage2-B：before-old比strong B3低2.22pp，90个scene×fold×old-class单元中6个退化；new-new错序也比strong B3多8条。因此只在Stage2-C把old/new联合HNBR重编译，无法修复已知before门，并可能把2/180旧→新侵入重新拉高。joint-HNBR协议上可行，但作为D41首选被拒绝，只保留为因果诊断备选。

D41只检验一个新机制：固定block-erasure consistency（BEC）能否让共享metric和target head不过度依赖某一表征块，从而同时改善old outer泛化、new注册、new-new下尾和old/new尺度。若view一致性下降而真实physical-held指标不升，BEC即被否决。

## 3.锁定表征、数学view与损失

### 3.1基础表征不变

完整view继续使用D40/D38的288维B3几何：`normalized z_id160 + 4×joint-normalized(FFT96,RF32)`再整体L2归一化。D41不改变基础block能量或query特征口径，避免把表示修改与BEC混成两个机制。

定义三个固定索引块：`z=[0,160)`、`fft=[160,256)`、`rf=[256,288)`。对同一个已封存received IQ的完整288维行`x`，构造：

\[
v_{full}=x,\quad
v_{-z}=\operatorname{norm}(x\odot m_{-z}),\quad
v_{-fft}=\operatorname{norm}(x\odot m_{-fft}),\quad
v_{-rf}=\operatorname{norm}(x\odot m_{-rf}).
\]

每个mask只把对应索引块置0，再对剩余向量L2归一化；norm必须finite且大于`1e-12`，否则fail closed。四个view都只读同一固定received IQ，不增加K、不产生额外physical sample或LEO overlay。query只计算`v_full`。

### 3.2共享metric、head与BEC目标

\[
h_\theta(v)=\operatorname{norm}\left(v\odot\exp(\operatorname{clamp}(\ell))\right),
\qquad
s_c(v)=18\,h_\theta(v)^\top\operatorname{norm}(w_c),
\]

其中`\ell`为288维`log_diag`，bounds完全继承D38；`w_c`为当前target注册类方向。令`p_v=softmax(s(v))`，对每个masked view定义：

\[
JS(p_{full},p_v)=\frac12KL(p_{full}\Vert m)+\frac12KL(p_v\Vert m),
\quad m=\frac12(p_{full}+p_v).
\]

实现使用`log_softmax`与`logaddexp-log(2)`计算，不添加可调epsilon。每个view的CE先按类内求均值，再对当前全部注册类等权平均，记为`CE_macro`。锁定总损失：

\[
L_{BEC}=\frac14\sum_{v\in\{full,-z,-fft,-rf\}}CE_{macro}(s(v),y)
+\frac13\sum_{v\in\{-z,-fft,-rf\}}JS(p_{full},p_v).
\]

两个主项系数均固定为1；不加入D38 feature noise、prototype anchor、worst-class surrogate、new anchor、margin、bias、radius、HNBR或mask概率，不扫描temperature、loss权重、step或其他超参数。类宏平均、JS和初始化都对标签置换同式，不是query quota。

## 4.锁定Stage2-B/C生命周期

### 4.1Stage2-B

- 只读取old support。
- `log_diag=0`；target-old权重以完整view类别centroid初始化。
- 使用D38锁定AdamW：learning rate`0.01`、weight decay`0.002`、gradient clip`5.0`、相同`log_diag` bounds，full-batch恰好20步。
- 唯一可训练状态为`log_diag＋全部target-old weights`；当前old6峰值参数`(1+6)×288=2016`。
- 第20步后生成独立不可变before artifact：target-old两级residual-int8＋FP32`log_diag`。它只用于注册前held评分，不得被Stage2-C覆盖或追写。

### 4.2Stage2-C

- 同一原子fit中继续使用Stage2-B最终FP32`log_diag`和old weights；禁止重置。若未来拆成跨调用实现，只能用相同support＋seed确定性重放B，不能保存隐藏FP32 deployment sidecar。
- 在Stage2-B metric下用完整view new support centroid初始化new weights。
- 使用old＋new全部合法support及四个view，对`log_diag＋全部target-old＋全部target-new weights`执行同一`L_BEC`。
- 使用D38锁定SGD：learning rate`0.05`、momentum`0`、gradient clip`5.0`，full-batch恰好10步；不得冻结old或只训练new。
- 第10步后将全部target registry一次性重新编译为两级residual-int8。formal final state不保存FP32 target方向、optimizer或回退副本；matched FP32只作同参考方向精度ablation。
- target-old在Stage2-C允许更新，但Phase1 sealed ground int8组件的code/scale/hash在entry/exit必须逐bit相同。必须在artifact中区分ground old和target-old，不能把target registry重编译误写成ground更新。

K1直接以每类唯一物理support初始化centroid并使用相同四个数学view；无物理LOO、伪样本或其他K统计。K1/K5/K20只在D41方法完全锁定后用于独立确认，不参与本轮选参。

## 5.资源预锁

|资源|current old6/new5|最大old20/new20|硬门|
|---|---:|---:|---:|
|Stage2-B峰值参数|2,016|6,048|≤80,000|
|Stage2-C峰值参数|3,456|11,808|≤80,000|
|epoch/optimizer step|30/30|30/30|≤30/50|
|Stage2-C step|10|10|=10|
|persistent target state|预计<8KB|预计<26KB|≤256KB|
|query view|1个full view|1个full view|无dense query graph|

适配MAC必须计入四个view的transform/classification及三个JS项，不能沿用single-view估算。query MAC只包含full view shared metric、全部注册类cosine和argmax。CUDA peak、完整30步trace、平均/P95 head latency只在真实获选候选的selected-only full-K10中实测；outer未获选时不得虚构。

## 6.固定候选与development矩阵

|候选|角色|
|---|---|
|identity-only single-qKNN|回退/遗忘基线|
|ProtoNet CDA|独立matched基线|
|exact strong B3 FP32|当前最强合法比较器|
|D40-HNBR int8|已证伪旧类主导结构负对照|
|D41-BEC int8|唯一可晋级路线|
|D41-BEC FP32|matched精度ablation，不可晋级|

固定6×3场景×5个outer physical folds=`90`行，每fold8-shot fit、2-shot held。direct ADV3B02只作相同old-held的0-support锚，不进入90行。全部候选共享held ranks、physical-token SHA和source closure。D40负对照必须在D41 Runner中从同fold真实执行，不从旧报告拼接。

## 7.严格晋级门

D41 int8只有全部满足才可进入selected-only full-K10或N607：

1. before-old每scene×fold总体及每旧类不弱于exact strong B3，15fold聚合严格提高。
2. after-old每scene×fold总体及每旧类不弱于D40；forgetting逐row不高于D40。
3. seen-new每scene×fold总体及每新类不弱于exact strong B3，15fold聚合严格提高；最低新类准确率严格高于strong B3最低40%。
4. old→new实际侵入`<33/180`、new→old实际最高分错误`<22/150`、new-new pairwise错序`<25/150`，三项均严格优于exact strong B3；最低new-new和new-old margin都严格提高。
5. 每个matched row的H和joint floor不弱于exact strong B3，15fold两项聚合均严格提高。
6. D41 int8/FP32的before/final outer-held argmax差异为0；formal target state为int8-only，Phase1 ground int8 entry/exit hash一致。
7. 30/30步、Stage2-C=10、当前峰值参数=3456、state≤256KB；四view/BEC MAC为finite且严格大于single-view下界；query/source/clean/role/quota/global assignment闭合。

任一结构、协议或资源门失败即fail closed；任一性能门失败即`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，回退identity，不调整mask、loss、optimizer、step或temperature，不打开query、不访问N607、不扩K/receiver/seed/new-count确认矩阵。

## 8.实现与验证计划

|面|锁定范围|
|---|---|
|D41 core|四view、macro-CE＋JS、B20/C10联合state更新、before/final int8与matched FP32、ground/pairwise/resource audit|
|Runner|`d41_v1`六候选、90行、actual old→new/new→old、new-new pairwise、strict selector、selected-only full-K10与五项artifact SHA|
|测试|view golden、JS golden、B/C参数更新范围、before不可变、ground逐bit、K1/5/10/20、new2/5/10/20、标签置换、row-local query、int8/FP32、90行physical closure与逐门反例|
|Git/N607|本地`ssr-gpu`验证并提交；只有真实90行全部晋级门通过才preflight/SCP/N607|

根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录同名报告仅作非版本化运行镜像。当前goal保持active，D41 development screen不能替代完整确认矩阵。

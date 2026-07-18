# D38 full-batch B3-geometry residual-int8实验报告

## 1.实验身份与当前状态

- 实验ID：`d38_strong_b3_quantized_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`DESIGN_LOCKED_IMPLEMENTATION_PENDING`
- 目标：修复D37的两个直接失败源——弱Fisher旧头和new-new排序错误——在同一合法K10 development cell上检验一个正式资源上限内、target-old/new均为两级residual-int8、逐样本面对全部注册类的轻型Stage2-B/C路线。
- 主要比较：identity-only single-qKNN、ProtoNet CDA、exact legacy strong B3 FP32、D38-A int8、D38-B int8、D38-B FP32 matched ablation；direct ADV3B02另作相同old-held样本的0-support锚。

本报告是D38开始编码前的预注册设计。任何实现完成、单测通过或support screen执行完成都不能自动改写为性能成功。

## 2.D37证据驱动的失败定位

D37真实support-only screen为105/105行，query始终sealed，五项artifact哈希与receipt一致。其两级int8量化均值误差约`0.91e-6–1.01e-6`，内部源旧头决策违规为0，因此量化不是主因。

|问题|D37证据|D38响应|
|---|---|---|
|旧头来源错误|D37保留的是82.22%的D33-FAST/Fisher旧头，而exact legacy strong B3为87.78%|重新实现B3几何中的无bias旧域适应，并把exact legacy strong B3保留为独立matched比较器|
|new-new排序错误|3个D37臂共45/45折以同一原因fail closed；每臂15/15折均有真实新类输给其他新类|取消公共offset主机制；让每个新类权重在统一loss下独立移动|
|旧→新重叠|每臂outer held旧→新侵入33/180；不是offset或旧头量化漂移|Stage2-C loss使用全部old+new support，但梯度只更新新权重|
|类内方差|`09f8`跨场景new margin正确4/30、mean=-0.245；`f608`为10/30、mean=-0.152；full-support自包含均升至20/30|首轮先用CE10做直接判因；若outer-held下尾不改善，再进入类无关whitening/radius，不在本轮混入多原型|

D38必须新增support-only pairwise诊断：`scenario/outer_fold/physical_rank/true_new_handle/top_competing_new_handle/true_new_score/top_competing_new_score/new_new_margin/top_old_score/new_old_margin`。这些字段只来自合法support-held行，不读取query或truth sidecar。

## 3.exact legacy strong B3与D38的声明边界

exact legacy strong B3使用：

- 288D固定received-IQ表征；
- Stage2-B 20epoch、batch size32、AdamW和feature noise；K10 outer train有48个旧support，因此是40 optimizer steps；
- Stage2-C最多20个new-only optimizer steps；
- target-old/new权重为FP32。

D38把Stage2-B改为20个full-batch optimizer steps，再给Stage2-C固定10个full-batch steps；训练动力学已改变。故D38只能称为`full-batch B3-geometry`或`B3-initialized`路线，不能把D38注册前结果直接命名为exact strong B3。exact legacy strong B3必须在完全相同scene、fold和held physical IDs上独立计算。

## 4.预注册数学机制

### 4.1固定288D表征

同一固定received IQ只前向一次得到`z_id160`，FFT/RF均是该received IQ的数学视图：

```text
phi(x)=normalize([normalize(z160); 4*normalize([FFT96; RF32])])
```

FFT96与RF32先拼接后共同归一化，不改成两个独立归一化块，避免把表征变化混入优化机制。

### 4.2Stage2-B：20步full-batch旧域适应

共享正值对角度量：

```text
d=exp(clip(a,lower,upper))
h(x)=normalize(d*phi(x))
s_c(x)=18<h(x),normalize(w_c)>
```

锁定范围：z160与RF32的`a_j∈[-1.5,1.5]`，FFT96的`a_j∈[-log(1.5),log(1.5)]`。旧类权重从各类support均值初始化，使用无class bias的full-batch AdamW：20step、lr`0.01`、weight decay`0.002`、gradient clip`5.0`、feature noise std`0.01`、prototype anchor`0.05`。每步将`a`投影回合法区间。

```text
L_B=mean_old CE + 0.05*mean_c ||normalize(w_c)-mu_c_init||^2
```

Stage2-B结束后，先对旧权重按固定块`(160,96,32)`独立做两级residual-int8编译。Stage2-C看到并冻结的是实际量化旧头的decode值，而不是随后会被替换的FP32旧头。

### 4.3Stage2-C：centroid对照与10步new-only判别训练

每个新类统一初始化：

```text
u_j0=normalize(mean_{y=j}(d*phi(x)))
```

D38-A直接量化`u_j0`，作为0步centroid注册对照。D38-B冻结`d`和已量化旧头，只训练全部新类`u_j`，但loss同时读取合法old与new support：

```text
L_c=mean_{i:y_i=c} CE(all_registered_scores_i,y_i)
L_wc=0.25*(logsumexp(L_c/0.25)-log(C))
L_C=mean_c L_c + 0.20*L_wc + 0.01*mean_j ||normalize(u_j)-u_j0||^2
```

D38-B固定full-batch SGD 10step、lr`0.05`、momentum`0`、gradient clip`5.0`。旧support产生的梯度只能推开新权重，不能更新旧权重或共享度量。最终新权重再独立做两级residual-int8量化并append；旧int8 code、FP16 scale和inverse norm前缀逐bit不变。

D38-B FP32使用完全相同训练轨迹，仅在最终部署权重精度上保留FP32，作为matched量化ablation，不是可晋级路线。正式D38 state只保存共享FP32`log_diag`、old/new int8 code、FP16 block scales/inverse norm和类注册表；不保存FP32 target prototype、optimizer state或FP32回退副本。

## 5.候选、矩阵与选择规则

|候选|机制|角色|
|---|---|---|
|identity-only single-qKNN|现有Z0 support centroid基线|遗忘/资源基线|
|ProtoNet CDA|ADV3B02 z_id160 support均值、最近原型|强制matched基线；若与Z0在本cell数学等价，仍保存equivalence audit|
|exact legacy strong B3 FP32|原20epoch mini-batch旧头＋原20step新注册|同fold性能上界比较器，不可冒充int8正式路线|
|D38-A residual-int8|20步full-batch旧头＋0步new centroid|判定新类判别训练是否必要|
|D38-B residual-int8|A＋10步all-support/new-weight-only训练|唯一promotable主路线|
|D38-B FP32|与B同轨迹、最终权重FP32|matched量化ablation，不可晋级|

最小开发矩阵固定为receiver`20-1`、seed`713101`、K10、new5、3个LEO弱场景、5个outer rank-pair folds，共`6×3×5=90`行。每折8-shot fit、2-shot held；held physical ID不得进入metric、weight、量化scale、checkpoint或candidate选择。A/B只在15fold聚合后全局选择一次，禁止按场景、fold、类或handle路由。

direct ADV3B02只在相同old-held行报告0-support准确率和逐类值，不面对尚未注册的新类，也不参与A/B超参数选择。full-K10 refit只生成锁定候选的部署/资源审计，不得反向更改臂或超参数。

K1/K5/K20固定执行K10锁定配置：A始终20step；B始终20+10=30step。K1不构造self-OOF、不重新选A/B、不early-stop或rollback。K20不用于开发选参。

## 6.资源预核算

当前旧类数6、特征维288。训练阶段FP32权重是瞬态；部署状态按两级int8＋FP16 scale/inverse norm计算。

|new类数|A/B峰值trainable params|B epoch/steps|部署state约值|逐query head MAC|
|---:|---:|---:|---:|---:|
|2|2016|30/30|5856B|4896|
|5|2016|30/30|7620B|6624|
|10|2880|30/30|10560B|9504|
|20|5760|30/30|16440B|15264|

A为20epoch/20step。所有规模均远低于80k参数、30epoch、50optimizer steps和256KB硬上限；无dense query graph或query-dependent batch optimization。实现后仍须现场测量平均/P95时延、峰值显存、适配MAC、backbone/FFT前向次数和实际状态字节，预核算不能替代resource audit。

## 7.可观察晋级门与停止条件

D38-B只有全部满足以下条件才可锁定development query：

1. 注册前D38量化旧头在每个scene×fold×old-class上不弱于exact legacy strong B3；若full-batch20不能保持强旧域结果，立即否证D38旧头路线。
2. B相对A同时改善seen-new总体、最低新类、new-new margin/混淆和H；若10step结束仍有系统性new-new错序，停止调step/lr，下一轮改为类无关whitening/radius。
3. B的after-old、seen-new、H、forgetting、joint floor及全部逐类结果不弱于同row exact legacy strong B3和identity/ProtoNet中更强者。
4. outer held旧→新侵入率不高于exact legacy strong B3；旧prefix逐bit不变只是必要条件，不能替代held安全。
5. D38-B int8相对matched FP32在outer-held的argmax变化数为0；同时报告量化误差，不能只以误差小推断决策不变。
6. target-old/new实际预测均使用int8生命周期；资源、协议、query sealed和逐样本all-registered-class审计全部通过。

任一关键门失败即记`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不打开query、不进入K1/K5/K20、125 screen或确认矩阵。不得继续盲扫epoch、lr、margin或offset。

## 8.协议、版本和运行计划

- 协议：`protocol_schema=p2_min_v1`；直接复用匹配`VALIDATED_ONCE`的D18 cell，不因D38 method变化重验数据。
- 数据：每个physical sample只有一个固定`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`received IQ；support/query及场景physical ID不交。
- 权限：support-only fit/选择；query sealed；无clean/raw、source样本/feature/logit/cache、role Oracle、class quota或global reassignment。
- 代码隔离：新增`stage2_d38_strong_b3_quantized.py`；不编辑或暂存当前有未归属修改的`stage2_diag_cosine_exploration.py`；D38 core不调用run_d19/run_d25私有函数。
- 根目录`E:\type10-7`不是Git仓库。本报告镜像到根目录，Git权威副本位于本文件；开始D38设计时分支ahead origin 1605，其他大量修改/未跟踪文件均不属于D38，后续只暂存D38专属文件和共享runner最小差异。
- N607：尚未触碰。先在`ssr-gpu`完成core/runner窄验证并提交Git；若需N607，按AGENTS.md先做direct preflight、占用审计、本地报告和最小SCP，短连接结束后核验无残留SSH/TCP22。

## 9.实施与实验记录（待回填）

|项目|当前值|
|---|---|
|计划新增core|`code/cvsrffi/stage2_d38_strong_b3_quantized.py`|
|计划runner接线|`code/scripts/run_d25_support_only_concat.py --candidate-set d38_v1`|
|计划测试|core预算/量化/批顺序/标签置换/K规模测试；90行integration、matched gate、receipt/hash闭环；D34–D37聚焦回归|
|Git commit|待实现|
|N607 sync/command/PID/GPU|未启动|
|预期artifact|`training_log.jsonl`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`support_audit.json`、`RECEIPT.json`、完整stdout|

当前goal保持active。D38 development support screen不等于独立确认，更不等于完成5receivers×至少5seeds×3scenes×K×new-count正式矩阵。

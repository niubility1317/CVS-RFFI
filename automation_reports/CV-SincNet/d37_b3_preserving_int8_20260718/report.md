# D37 B3-preserving int8注册实验报告

## 1.实验身份与目标

- 实验ID：`d37_b3_preserving_int8_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`IMPLEMENTED_LOCAL_TESTED_SUPPORT_SCREEN_PENDING`
- 科学目标：在`p2_min_v1`下，以同一target receiver的固定单次`leo_*_weak`接收IQ形成K10旧类与新类support，仅用support内部物理LOSO证据，保留B3的Stage2-B目标域几何并完成target-old/new正式int8注册；逐样本面对全部注册类，不打开development query。
- 比较对象：`identity-only single-qKNN`、`D25-C0-DIM-CONCAT`、`D33-B3-FAST-FISHER-SPHERICAL-BALANCED`以及D37-A/B/C。

## 2.D34–D36三轮强制技术复盘

三轮均为同一合法K10 development cell的完整support-only屏，均未打开query，不能写成正式性能结果。D34与D35各解析105/105个structured rows且receipt五项哈希一致；D36同样完成105/105行并闭合artifact哈希。三轮直接推动旧类适应与新类注册的共同问题，没有新增数据权限、clean/source访问、query真值、角色Oracle、class quota、全局重分配或dense query graph。

|轮次|主要机制|注册前旧类|注册后旧类|seen-new代理|H|遗忘|关键完整日志诊断|结论|
|---|---|---:|---:|---:|---:|---:|---|---|
|D34-C|冻结FAST旧列＋winner-conditioned稀疏碰撞边＋int8新原型|82.22%|71.11%|57.33%|62.23%|11.11pp|180个outer held旧样本有20次旧→新侵入；75个new class-fold中68个不可达；最坏单折old/new/joint floor均为0|拒绝稀疏可见性门；冻结旧列不能阻止新增列越界|
|D35-C|冻结FAST旧列＋所有新类有限score＋fit旧support最大残差安全阈值|82.22%|55.00%|55.33%|53.17%|27.22pp|180个outer held旧样本有49次侵入；68个new class-fold不可达；winner校准cell过半缺证据并回退|拒绝硬winner分桶、fit最大残差阈值和继续调buffer/原型数|
|D36-A|joint compiled int8旧/新头＋无校准|81.11%|65.56%|53.33%|57.80%|15.56pp|注册前量化旧头已低于同折B3；outer held侵入28次、不可达51个class-fold|拒绝重建并联合编译旧头|
|D36-B|同上＋只读ground弱锚＋常数OOF校准|80.56%|62.22%|56.00%|57.91%|18.33pp|outer held侵入32次、不可达49个class-fold|ground弱锚与常数偏置未解除重叠|
|D36-C|同上＋fixed 6D OOF IRLS统一margin校准|80.56%|66.11%|52.00%|56.82%|14.44pp|outer held侵入25次、不可达53个class-fold；loss单调、无NaN/Inf、资源全通过|连续校准训练稳定，但错误的旧类基准几何使其无晋级价值|

### 2.1已拒绝路线

1. 以winner-conditioned edge或`nonedge≈winner_score-2`控制新类可见性。
2. 把旧score prefix逐bit不变或fit support不退化当作最终all-class argmax安全证明。
3. 在硬winner selector上继续增加top-k边、原型数、buffer、floor multiplier或epsilon。
4. 继续从support robust prototype重建、适配并联合编译旧类头；D36证明该步骤在注册前就破坏B3旧域结果。
5. 按历史难类ID设置白名单、专属阈值、权重或定向保护；逐类失败只用于诊断通用floor。
6. 在support-held硬门未过前打开query、进入K1/K5/K20或扩张确认矩阵。

### 2.2保留的证据与剩余假设

- B3的正值对角度量`log_diag`及最终单位类权重是当前最强合法Stage2-B support-only几何；D37直接量化B3最终旧类权重，不再重新估计另一套旧类原型。
- 新类在完全相同的B3变换空间内以同一类无关公式形成单位权重并量化；所有样本对全部新类始终取得有限score。
- 注册后旧类int8权重字节、scale和inverse norm必须append-only、逐bit不变；安全性仍以outer held最终all-class old→new侵入直接判断。
- D36的连续校准不再复用；D37改用可审计的support-OOF硬可行区间，区间为空即否证“单一公共offset足够”，不得用软loss掩盖冲突。

## 3.D37预注册机制

设B3在旧support上得到`d=exp(log_diag)>0`和单位旧类权重`w_i`。D37保留共享`log_diag`算子，并对每个权重按固定特征块`(160,96,32)`做两级残差int8量化：

```text
q1_ib=round(w_ib/scale1_ib)∈int8
e_ib=w_ib-scale1_ib*q1_ib
q2_ib=round(e_ib/scale2_ib)∈int8
u_i=concat_b(scale1_ib*q1_ib+scale2_ib*q2_ib)
z(x)=normalize(x⊙d)
g_i(x)=18<z(x),u_i>
```

每个新类用相同变换后的合法support形成类无关单位权重`v_j`，再使用完全相同的两级int8量化。注册后的基础score为`18<z(x),[u;v]>`。旧类`q1/q2/scale1/scale2`前缀从注册前状态原样append到注册后状态，不能因新类注册或校准而修改；state不保存FP32 target prototype。

对于inner rank-pair OOF基础分数，令公共new-group offset为`b`，预注册margin为`m`。全部旧support-held行给出安全上界：

```text
U=min_x_old(max_old(x)-max_new_raw(x)-m)
```

全部新support-held行给出可达下界：

```text
L=max_x_new(max(max_old(x),second_new_raw(x))-true_new_raw(x)+m)
```

只有`L<=U`时才取`b=(L+U)/2`，否则该候选fail closed。最终score为`[old_raw,new_raw+b]`。OOF可行只用于开发拟合，不能冒充outer held或query安全。

D37只保留三个高信息量臂，主要差异仅是固定margin：

|候选|旧/新int8几何|support-OOF校准|用途|
|---|---|---|---|
|D37-A|B3-preserving residual int8旧/新权重|硬可行区间，`m=0`|检验公共offset是否存在|
|D37-B|同A|硬可行区间，`m=0.05`|检验小正margin鲁棒性|
|D37-C|同A|硬可行区间，`m=0.10`|检验更严格的双侧margin|

所有校准只来自outer-train内部预登记rank-pair OOF行；不读取outer held标签以拟合，更不读取query。量化与区间求解均为闭式，0epoch、0 optimizer step。

声明边界：`B3-preserving`只表示旧权重直接来自B3、旧/新独立量化后旧字节append-only，以及fit-support决策门；任何有限精度量化都可能翻转未见过的近边界样本。因此晋级必须以每个matched scene×outer-fold×old-class对FP32 B3非劣为准，不能声称全输入域数学等价。OOF来源由runner实际构造的rank-pair排除路径和唯一physical ID审计保证，core中的source字符串本身不是安全证明。

## 4.可观察预期、硬门与停止条件

最小矩阵固定为7候选×3场景×5个outer physical folds=105行：`Z0`、`D25-C0`、`B3`、`D33-FAST`、D37-A/B/C。每行同时保存注册前old、注册后old、seen-new代理、H、forgetting、全部逐类、侵入和physical LOSO可达性。

D37候选只有全部满足以下门才可打开锁定development query：

1. 注册前量化旧类总体与每个旧类均不弱于matched FP32 B3。
2. 注册前→注册后旧类int8前缀字节、scale、inverse norm逐bit不变。
3. outer held最终all-class old→new侵入为0，而不是只看fit support。
4. 三个场景全部新类physical LOSO`margin_min>0`，不存在不可达class-fold。
5. 注册后旧类、新类、H、forgetting、逐类old/new floor同时不弱于matched B3/identity中更强者。
6. target-old/new实际预测组件均为int8生命周期，状态<256KB、参数<=50k、epoch<=20、optimizer steps<=20、无dense query graph和query-dependent batch optimization。

任一关键门失败即记`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不打开query、不扩展确认矩阵；若仅注册前旧类不弱于B3而Stage2-C仍呈侵入—不可达重叠，则下一轮必须更换分离机制，不能继续扫offset。

## 5.协议、数据与版本边界

- 协议：`protocol_schema=p2_min_v1`；复用匹配`VALIDATED_ONCE`的现有D18 development cell，不因method变化重验数据。
- 输入：每个physical_sample_id仅一份固定`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`接收IQ；场景之间及support/query物理ID不交。
- 权限：support-only fit/选择；query保持sealed；不访问clean/raw、source样本、query真值/角色/数量或类quota。
- 评分：每个样本独立面对全部已注册类，无跨query联合计算。
- 根目录`E:\type10-7`不是Git仓库；本报告另镜像到Git仓库`E:\type10-7\github_publish\CVS-RFFI-repo`。开始设计时该分支相对origin ahead 1603，存在大量与D37无关的既有修改/未跟踪文件，后续只暂存D37专属文件与对共享runner的最小差异。

## 6.实现、验证与运行记录（待回填）

|项目|当前值|
|---|---|
|本地文件变更|新增`stage2_d37_b3_preserving_int8.py`、两份D37测试、traceability与本报告；最小修改共享`run_d25_support_only_concat.py`接入`d37_v1`|
|本地环境|`ssr-gpu`|
|窄验证命令|`python -m py_compile ...`通过；D37 core+integration 24 passed；D34–D37聚焦回归43 passed；`--help`包含`d37_v1`；`git diff --check`通过|
|Git commit|待回填|
|N607 sync目的地|未同步|
|服务器命令/环境/CWD|未启动；当前N607已有runtime组合不能同时满足模型与NumPy依赖，不修改远端环境|
|run/log目录|待回填|
|PID/GPU|未启动|
|预期artifact|`training_log.jsonl`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`support_audit.json`、`RECEIPT.json`、完整stdout|

实现SHA256：D37 core=`09e89b5d833d246232bb2c7563dac67295618eb1bff5335789c039fc109df110`；runner=`dc2541f0a849893ad53c49a3178d8abfb2b0cb70e8a30429e224975775f5c460`；core test=`330ef3c7b5730feaf85291ace8e92e503d2b5e729119d1d5bf189f1ae999bfbd`；integration test=`f394a077045f4d5b0070ae08e6ab755903f7e373af33f4b935554ed9208c9d83`。

当前已知未闭合项：D37只预注册K10 development cell。K1每类仅一个物理support，无法构造排除自身的新类原型并完成physical OOF校准；在预注册、support-only且不借用K5/K10结果的统一K1规则出现前，D37不得进入目标K1确认行。

## 7.结果表（实验后回填）

|候选|机制|receiver/TX split|K|seed/场景|before-old|after-old|seen-new|H|forgetting|old/new floor|侵入/不可达|资源|最终判定|
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|
|D37-A|B3-weight-source residual int8＋OOF区间`m=0`|20-1/6旧+5新|10|713101/3场景|—|—|—|—|—|—|—|—|SUPPORT_SCREEN_PENDING|
|D37-B|同A，`m=0.05`|20-1/6旧+5新|10|713101/3场景|—|—|—|—|—|—|—|—|SUPPORT_SCREEN_PENDING|
|D37-C|同A，`m=0.10`|20-1/6旧+5新|10|713101/3场景|—|—|—|—|—|—|—|—|SUPPORT_SCREEN_PENDING|

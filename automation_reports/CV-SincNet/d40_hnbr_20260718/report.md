# D40 append-only HNBR实验报告

## 1.实验身份与状态

- 实验ID：`d40_hnbr_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`IMPLEMENTED_LOCAL_VERIFIED_REAL_SCREEN_PENDING`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；复用D18固定received-IQ support，query保持sealed。
- 目标：在保留D38强Stage2-B共享metric和正式int8生命周期的同时，用0步、无可调系数的难负重心方向残差化同时改善old-old、new-new和old-new竞争。

本设计来自D37–D39三轮强制回顾。设计、实现、单测、资源通过或support screen启动都不是性能成功。

## 2.直接证据与单一假设

当前同row最强合法比较器是exact strong B3：before-old87.78%、after-old75.56%、seen-new72.67%、H73.35%、forgetting12.22pp、joint floor23.33%、旧→新侵入33/180；最低旧类60%、最低新类40%。

D38注册前old=87.22%，但D38-A/B在新方向加入后分别发生180/180和179/180旧→新侵入；D38-B虽把seen-new提高到78.67%，仍有32/150条new-new错序。D39只改变angular-radius标尺，侵入仍174/180且32/150错序完全不变。D38/D39的int8与FP32 outer-held argmax差异均为0。

D40只检验一个假设：类别方向与最相似竞争方向的正投影是当前共同混淆源；在每个注册阶段用同一无参数球面投影移除该分量，可提升Stage2-B旧类分离，并让Stage2-C新类方向同时避开冻结旧类和其他新类，而不引入group bias、radius或hard gate。

## 3.锁定数学机制

设D38共享`log_diag`变换后的单位基础方向为`b_c`，固定`T=18`继承D38 scorer。对当前类`c`的难负集合`ℕ_c`，定义：

\[
a_{cd}=\frac{\exp(Tb_c^\top b_d)}{\sum_{j\in\mathcal N_c}\exp(Tb_c^\top b_j)},\quad d\in\mathcal N_c,
\]

\[
n_c=\operatorname{normalize}\left(\sum_{d\in\mathcal N_c}a_{cd}b_d\right),\quad
\rho_c=\max(0,b_c^\top n_c),
\]

\[
w_c=\operatorname{normalize}(b_c-\rho_c n_c).
\]

softmax实现必须先减行最大值；所有norm必须finite且大于`1e-12`，否则fail closed。不增加投影强度、margin、temperature或shrinkage候选。

### 3.1Stage2-B

D40先执行与D38相同的20步full-batch old adaptation，得到old基础方向。随后所有old类同步以其余old基础方向为`ℕ_c`执行HNBR，量化为两级residual-int8 target-old状态。该状态用于注册前评分，也构成Stage2-C冻结old prefix。

### 3.2Stage2-C

每个new基础方向是在同一D38变换空间中对本类合法support求单位均值。所有new类同时计算HNBR；对new类`c`，难负集合由冻结target-old最终方向和其余new基础方向组成。不得把已残差化的某个new方向用于下一个new类，避免注册顺序依赖。

new HNBR方向独立量化后append。target-old code/scale/inverse norm、`log_diag`及密封ground int8组件逐bit不变。正式state不保存FP32 target方向、optimizer或回退副本。

### 3.3推理与置换边界

所有query使用统一`18<h(x),w_c>`面对全部注册类独立argmax；无role分支、batch统计、quota、global reassignment或dense query graph。机制对保持enrollment partition的任意类标签置换严格等变；old/new阶段差异来自合法state provenance，不来自query truth。

K1直接以每类唯一物理support形成new基础方向并执行同一闭式HNBR；0梯度、无伪LOO、不借用其他K统计。new2、只有2个old类、近零重心及近零残差均须单独测试。

## 4.固定候选与development矩阵

|候选|角色|
|---|---|
|identity-only single-qKNN|回退/遗忘基线|
|ProtoNet CDA|独立matched基线|
|exact strong B3 FP32|最强合法比较器|
|D38-B residual-int8|方向/尺度灾难负对照|
|D40-HNBR int8|唯一可晋级路线|
|D40-HNBR FP32|matched精度ablation，不可晋级|

固定6×3场景×5个outer physical folds=`90`行，每fold8-shot fit、2-shot held。direct ADV3B02只作相同old-held的0-support锚，不进入90行。全部候选必须共享相同held ranks、physical-token SHA和源数据闭包。

## 5.严格晋级门

D40 int8只有全部满足才可进入full-K10或N607：

1. before-old总体、每scene×fold×old-class不弱于exact strong B3，聚合严格提高。
2. after-old与每旧类floor不弱于strong B3；forgetting逐row不高于strong B3。
3. old→new侵入≤33/180且相对strong B3严格减少。
4. seen-new逐row不弱于strong B3；new-new错序<32/150；最低新类准确率和最低pairwise margin严格提高。
5. 每个matched row的H和joint floor不弱于strong B3，15fold聚合均严格提高。
6. int8/FP32 outer-held argmax差异为0；old prefix、ground int8和source closure闭合。
7. 0个Stage2-C optimizer step；总epoch/step恰好20/20，trainable parameters≤2016，state≤256KB，HNBR support MAC为finite且严格大于0，无dense query graph或query-dependent batch optimization。

任一关键门失败即`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：回退identity，不扫描投影系数，不叠加bias/radius/gate，不打开query、不访问N607、不扩K或确认矩阵。

## 6.实现与本地验证

|面|锁定范围|
|---|---|
|D38公开接缝|只新增readonly feature transform、state weight decode、int8/FP32 compile和append接口；不让D40调用私有函数|
|D40 core|HNBR公式、同步old/new构造、append-only state、pairwise/geometry/resource audit|
|Runner|`d40_v1`六候选、90行、同physical匹配、selector、selected-only full-K10和五项artifact哈希|
|测试|公式golden、标签置换、同步性、K1/5/10/20、new2/5/10/20、近零fail-close、old prefix、int8/FP32、90行与selector/resource反例|
|Git/N607|本地`ssr-gpu`验证并提交；只有真实K10 outer-held全门通过才preflight/SCP/N607|

### 6.1实现文件

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_d38_strong_b3_quantized.py`|新增readonly transform/decode/compile/append公开接缝|
|`code/cvsrffi/stage2_d40_hnbr.py`|D40-HNBR核心、int8/FP32状态、pairwise及资源/几何审计|
|`code/scripts/run_d25_support_only_concat.py`|`d40_v1`六候选90行Runner、strict selector、selected-only full-K10及artifact closure|
|`tests/test_stage2_d38_strong_b3_quantized.py`|D38公开接缝与append prefix回归|
|`tests/test_stage2_d40_hnbr.py`|公式、同步性、int8 decoded old negative、K/new-count、状态及协议测试|
|`tests/test_run_d40_hnbr_integration.py`|真实fold接线、exact strong B3 pairwise golden、90行/physical closure、selector与full-K10反例|

### 6.2验证证据

- Conda环境：`ssr-gpu`；本地CPU验证，无N607访问。
- `python -m py_compile`覆盖上述6个实现/测试文件：通过。
- `python -m pytest -q`覆盖D38/D39/D40 core及D36–D40 Runner integration：`124 passed`。
- `git diff --check`覆盖上述6个文件：通过；仅有Git的LF→CRLF提示，无whitespace error。
- 实现提交：`bc6c3539 feat(stage2): implement D40 HNBR screen`。
- D40 core不引用D38私有符号；new HNBR实际第二次调用的冻结old negative与`before_int8`解码方向逐元素相等，人为替换matched FP32 old ablation不改变new参考方向。
- 独立只读审查未发现blocker。审查要求的两项medium已修复：资源门改为固定20/20步且HNBR MAC>0；exact strong B3 pairwise补全函数新增独立held行、类别索引、physical token、margin与侵入golden测试。

当前只完成技术实现与本地测试闭环，尚未产生真实90行performance artifact，不能把`124 passed`解释为性能晋级。

根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录同名报告仅作非版本化运行镜像。当前goal保持active，D40 development screen不能替代完整确认矩阵。

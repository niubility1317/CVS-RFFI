# D49严格nested全局余弦原型融合探针报告

## 1.实验身份与状态

- 实验ID：`d49_nested_global_cosine_fusion_probe_20260719`。
- 时间：2026-07-19T01:19:49+08:00。
- 操作者：Codex主代理。
- 状态：`IMPLEMENTED_VERIFIED_NOT_RUN`。
- 目标：在不改变D42固定received-IQ表征、query路径和数据协议的前提下，检验严格nested support-LOO选择的全局cosine prototype head能否补足D45的rain旧类尾部或low-elev新类尾部。
- 比较对象：matched D45、D46当前最强合法development点、D42及已锁基线。

## 2.协议与数据锁

- `protocol_schema=p2_min_v1`，复用`phase2_data_status=VALIDATED_ONCE`的固定capsule/split；本轮仅改变方法，不重验数据。
- development cell固定为receiver`20-1`、seed`713101`、K10/new5、clear/low-elev/rain×5 outer折，共105行，实际每个outer fit为K8。
- before state必须在首次读取new support前物化并保持不可变；query及其view只测试，query读取、更新、真值、角色、配额、全局重分配均为0。
- 不访问clean/source、未批准derived state、receiver/scene/handle、class ID分支；old/new prototype按完全同式分别由本类support生成。
- 本轮不访问N607，不运行125，不新增data admission/hash/governance机制。

## 3.唯一方法定义

D42已产生全局单位球特征`x=normalize(exp(logdiag)⊙f)∈R^288`。D49的cosine head为：

`p_c=normalize(mean_{i∈S_c}x_i)`，`z_cos,c(x)=x^T p_c`。

因此`W_cos,c=p_c,b_cos,c=0`，无需新增非线性query view，仍可编译为一个`C×288`系数和`C`个intercept。

当K>1时，顶层每个held rank仅用其余`K−1`个rank：

1. 完整重拟合D45 base learner，包括其自身full/block inner-LOO可靠性权重；
2. 由同一inner-train生成global cosine prototypes；
3. 两head各自用inner-train class-centered logit RMS归一；
4. 每个held support恰评分一次，计算两head的macro class-balanced CE；
5. 以FP64稳定`softmax(-C×CE)`给出严格正且和为1的单一global权重。

完整support再独立拟合D45和cosine、重算各自RMS，并直接从量化前FP32 affine合成：

`W=w_D×W_D/s_D+w_C×P/s_C`，`b=w_D×b_D/s_D`。

合成后只做一次class-common canonical centering，再进入既有int8 coefficient/FP16 intercept lifecycle。K1不建立顶层cosine权重，系数、intercept及audit必须逐位回退D45。K2的nested D45 inner-train为K1，但D45/cosine顶层权重不强制1:1。CE完全相等时权重必须精确0.5/0.5。

## 4.禁止项与fail-close

- 不得再次做三块query L2归一化；不增加temperature、block-cosine、固定1:1 arm、一维stacking、clip、median、阈值或权重扫描。
- `C<2`、prototype resultant norm≤既有`ENERGY_EPSILON`、任一head RMS退化、非有限CE、权重为0/1或和偏离`1±1e-12`均停止。
- support rank顺序/class label置换必须等价；FP32或int8出现精确top tie必须停止，不能借registry顺序宣称对称。
- 完整support只融合和量化一次；不得先量化D45再解码融合。

## 5.资源预注册

一次D45 before或final fit的LDA次数为`F(1)=2`，`F(n>1)=2+2n`。D49 before+final严格nested总数为`2[F(K)+K×F(K−1)]`；K≥3时为`4+4K+4K²`，当前K8必须为292次，而不是D45的36次。LDA MAC必须按before/final及outer-main/nested-inner分组逐项闭合。

cosine额外适配成本必须分别审计support类均值、prototype归一、support RMS和FP32融合；query仍只有一套affine state，参数量、state字节和query MAC按实际artifact报告。CUDA峰值、host FP64峰值若未测量必须明确标为未测，不能用估算替代实测。

## 6.首次outer评分前验证门

- global-cosine reference与单affine score逐元素等价；query view仍为`full_288d_only`。
- 每折D45 state、prototype和RMS只读inner-train，held/train无交集且所有support恰held一次。
- K1逐位D45回退；K2、CE tie、零范数、RMS退化、权重端点及FP32/int8 top-tie覆盖测试。
- rank顺序和class-label置换测试；完整support只融合/量化一次。
- nested LDA fit/MAC闭合，K8精确292次；最终只有一套`C×288+C`state。
- D42–D48回归通过；独立代码复审P0=0、P1=0。

## 7.开发晋级与停止门

相对matched D45必须至少改变1条final outer prediction，否则立即拒绝。aggregate和clear/low-elev/rain各自的after-old、seen-new、H、joint、min-old/min-new不得退化，forgetting不得增加；至少严格改善一个D45已知失败项，优先rain after-old或rain forgetting。before/final/margin的int8相对FP32翻转必须为`0/0/0`。

任一门失败即记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，仍完整报告每候选、每场景、每类、15个outer行、混淆、训练轨迹、量化、资源、artifact和相对D45变化；不追加温度、权重变体、第二development seed或125。只有全部门通过才另行formalize，之后才可讨论125 screen。

## 8.版本、文件、命令与输出占位

- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`；当前分支`codex/cvs-rffi-release-20260626`。工作树存在用户的无关改动，本轮只stage本报告、D48复盘修正、D49 trace/script/tests。
- 根目录`E:\type10-7`含不可用`.git`目录但`git status`判定不是Git仓库；根报告仅为运行报告镜像，不宣称根目录已版本化。
- 已新增：`code/scripts/probe_d49_nested_global_cosine_fusion.py`、`tests/test_probe_d49_nested_global_cosine_fusion.py`、`analysis/d49_nested_global_cosine_fusion_traceability_20260719.md`。
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- 计划输出：`E:\type10-7\automation_reports\CV-SincNet\d49_nested_global_cosine_fusion_probe_20260719\nested_global_cosine_fusion`。
- exact launch command、commit/SHA、PID/GPU、log和最终artifact在首次运行前补锁；当前没有PID、GPU分配或结果。

## 9.完成后详细性能表占位

|Candidate|机制|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|待运行|strict nested D45/global-cosine fusion|—|—|—|—|—|—|—|—|—|—|—|

完成后还必须写入：三场景表、11类before/after/new表、15个outer行、相对D45 correct→wrong/wrong→correct变化、两head CE/RMS/权重分布、prototype resultant norm、B20每epoch/step轨迹、FP32/int8误差与top-tie、292次LDA资源闭包、state/MAC/显存、全部artifact大小/SHA和缺陷机制解释。

## 10.运行前实现与审计闭包

D49实现没有改动D42/D45文件。每个顶层held fold确实调用完整D45 fit；K1逐位直返D45，K2真实locked路径和非强制1:1权重已测试。cosine从D42全局单位球support按实际target向量重算；support transformed rows和targets在top-level wrapper中分别与正式support输入＋frozen log-diag、正式labels/classes逐元素绑定。verifier从绑定support重算full/nested prototype、resultant norm、两head RMS、held logits、macro/per-class CE和global权重，并将一次FP32融合绑定D42 matched state，再独立重编译核对双残差int8系数、FP16尺度及intercept。runner全局D42 score入口在所有outer FP32/int8 `argmax`前执行exact top-tie fail-close。

K8 before/final的8组LDA库存闭合为292次。D49额外适配MAC预注册为prototype`445,536`、两head RMS`11,635,584`、nested held scoring`1,446,912`、一次FP32融合`19,618`，合计`13,547,650`；最终总MAC以真实artifact为准。CPU不得误报CUDA峰值已测；host FP64峰值未测必须继续标为未测。

测试演化：初版D49为11项；独立复审先后发现outer tie、held证据、资源账、before生命周期、显存语义和自报字段闭环等P1，均在运行前修复。最终D49`13 passed`，D45＋D48＋D49`51 passed`，D43–D49`100 passed`，D42–D49全链`144 passed`，`py_compile`和`git diff --check`通过。pytest退出码为0；结束后的`pytest-current`清理出现Windows`WinError 5`，属于临时目录清理噪声，不是测试断言失败。

最终独立设计复核为P0=0、P1=0。仅保留声明边界：strict nested只描述冻结outer-B20后的head层，B20本身仍由outer-fit support训练，不能宣称全链路nested或无泄漏泛化；292次是与锁定调用结构逐项一致的精确理论库存，不是独立函数调用计数器；当前development为K8，K1正式证据应使用明确D45 fallback artifact。

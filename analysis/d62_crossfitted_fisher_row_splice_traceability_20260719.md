# D62交叉拟合Fisher行级Pareto拼接追溯与预注册

## 1.问题与单一机制

D61把D46的after-old从81.67%提高到83.33%、forgetting从10.56pp降到6.67%、min-after从53.33%提高到60.00%，但seen-new从84.67%降到76.00%、min-new从73.33%降到43.33%。这说明共享Fisher残差包含旧类保护信号，却不能整体替换D46。D62不扫描D61强度，也不使用old/new角色；它只问：support inner-held是否能证明某个匿名类的D61仿射行相对D46在真阳性与假阳性上Pareto安全？

对当前fit的每个physical-rank leave-one-out折，仅用其余support分别构造：

- `S0`：D46 full/block classwise likelihood融合分数；
- `S1`：同一fold train support闭式计算D61 `A=I+Udiag(b/(b+w))U^T`，把full/block系数编译为`W1=W0A^T`，并用各自train-support RMS和inner-held CE重新计算classwise full/block权重后得到的Fisher残差分数。

对每个匿名类`c`，构造只把`S0[:,c]`替换为`S1[:,c]`的坐标候选。仅当全部inner-held上`positive_correct_c`不降、`false_positive_c`不增且至少一项严格改善时，初始接受该行。然后同时替换全部初始接受行；若任一类positive下降或false-positive增加，则全部原子回退D46。通过时，在full support上按同一D61公式生成完整Fisher仿射头，只替换最终mask对应类行，删除类公共仿射项后编译为单一state。

## 2.与历史路线的非重复边界

- 不同于D57：D57门控的是D56混淆图截距流，30/30 fit无坐标通过；D62门控的是D46→D61整行系数与截距替换，候选信号来自闭式Fisher几何。
- 不同于D61：D61整体替换全部类行；D62无残差倍数、rank、gain、左右乘或场景扫描，只做预注册的双向Pareto行门与联合原子门。
- 不同于D46：D46只在full/block之间做逐类likelihood融合；D62保留该机制，并在其后检验基础行与Fisher行。
- 不同于role-aware head：类序号只作匿名坐标；公式对类标签置换等变，before/final使用同一流程，无old/new分支。

## 3.协议与证据闭包

- 固定receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold，实际outer fit K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule。
- 每个inner held fold的D61变换、组件RMS、CE与权重只用该折train support；held只用于预注册positive/false-positive计数；outer-held/query完全不可达。
- 不访问clean/source、receiver/scene handle、old/new角色、query truth、真实batch类数、quota或global assignment。
- K1/K2精确D46回退；K≥3激活。无alpha、temperature、threshold、rank、gain指数、类顺序、贪心、第二arm或事后门宽松化。
- 最终只有一套int8系数＋FP16截距仿射state，逐query独立全注册类argmax；dense query graph和query额外MAC为0。

## 4.预注册判门与停止条件

必须至少保持D46：before92.22%、after81.67%、new84.67%、H82.33%、forgetting≤10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%、混淆≤25/8/15；三场景不得交换伤害，量化翻转0/0/0，并至少严格改善after、forgetting或任一floor且改变≥1/15 prediction。

必须完成105/105行、query0、每个inner partition exact-once、逐类单坐标与联合门计数可重放、source/artifact/resource闭包。即使全部通过也只进入下一独立开发验证，不直接运行125。

失败即停止D62，不放宽positive/FP门，不改为accuracy-only，不扫描残差权重、按场景mask、按角色mask或行替换顺序。结果报告必须覆盖7候选、3场景、11类、15fold、D46/D61/D62同折变化、accept mask、atomic fallback、Fisher rank/gain、量化、训练、资源和artifact。

## 5.实现与资源计划

- 复用D46外层fit；额外为full/block各执行一次完整support fit，并对每个physical rank执行一对inner component fit，同时生成D46与D61 held分数，不嵌套完整D46 runner。
- 资源据实记录额外LDA fit、Fisher SVD/变换MAC、标量比较和编译开销；D62本身0训练参数、0optimizer step、0query state。
- 实现：`code/scripts/probe_d62_crossfitted_fisher_row_splice.py`；测试：`tests/test_probe_d62_crossfitted_fisher_row_splice.py`。
- 输出：`automation_reports/CV-SincNet/d62_crossfitted_fisher_row_splice_probe_20260719/crossfitted_fisher_row_splice`。
- 本地`ssr-gpu`串行验证并使用detached clean worktree；本轮不访问N607。

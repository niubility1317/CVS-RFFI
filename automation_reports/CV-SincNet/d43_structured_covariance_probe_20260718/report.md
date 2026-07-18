# D43结构化协方差与量化稳定探针报告

## 1.身份与状态

- 实验ID：`d43_structured_covariance_probe_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`PREREGISTERED_PENDING_LOCAL_15_FOLD_PROBE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、clear/low-elev/rain；复用D42同一`p2_min_v1/VALIDATED_ONCE`固定received-IQ enrollment capsule与5个physical-rank held折。
- query：sealed；本探针无query/truth/scorer输入，不产生正式指标声明。

## 2.三轮复盘给出的单一问题

D40把旧类标尺推高而压死新类；D41反向压死旧类；D42统一等先验LDA首次同时提高聚合before-old、after-old、seen-new和H，但最低after-old从B3的60%降到50%，joint floor只与B3持平23.33%，且int8/FP32出现before/final argmax变化1/3与margin翻转3。D42量化误差主要来自幅值达0.999的FP16 intercept误差。

D43不再改变旧类适应器、特征、支持集、loss或注册规则，只检验两个可解释假设：full shared covariance的跨模态协方差使小样本下尾不稳；LDA的类公共仿射项不影响argmax/margin，却放大量化动态范围。

## 3.预锁arm与等价score

|arm|协方差结构|score编译|角色|
|---|---|---|---|
|`full_centered_control`|D42完整auto-shrinkage covariance|`w_c←w_c−mean_c(w_c)`、`b_c←b_c−mean_c(b_c)`|只隔离公共项去除效果，不参与结构选择|
|`block3_centered`|保留z160、FFT96、RF32三个对角块，跨块元素置零|同上|候选结构1|
|`diagonal_centered`|只保留auto-shrinkage covariance对角|同上|候选结构2|

在实数代数中，对任意样本，所有类同时减去`x^T mean_c(w_c)+mean_c(b_c)`，argmax和任意两类margin严格不变。实现会在转FP32后再次断言support argmax不变，并报告FP32 pairwise drift；outer-held上的FP32/int8变化继续由真实矩阵审计，不能用代数等价代替。formal int8仍由D42现有3-block two-level residual量化器与FP16 intercept编译；因此full-centered对照只测量公共项去除能否消除量化边界翻转。K1/rank0继续使用真实support均值的单位协方差fallback，不构造伪物理样本。

不允许增加第四个结构，不扫描shrinkage、threshold、rank、lr、epoch或类专属参数，不根据匿名handle设置分支。

## 4.预注册判定

三个arm均运行同一15折development support-held代理，并保留D42 Runner的七候选矩阵以固定B3/D40/D41/D42历史比较面。探针只决定是否值得实现正式D43候选，不允许直接晋级full-K10或N607。

判门基准不是下列显示值，而是SHA256=`4ee51dd3d21ae8751bfaa64eb82d2a5a5371728fc7c1502bdb3af221d349614a`的D42`training_log.jsonl`中`D42-USLDA-INT8`的15条原始全精度同row字段。所有均值按15条等权算术平均；逐场景均值按该场景5折等权；最低类为先对同一匿名类跨15折求均值再取类间最小值。非退化使用`candidate>=reference−1e-12`，遗忘使用`candidate<=reference+1e-12`，严格改善使用`candidate>reference+1e-12`；报告四舍五入值不参与判门。

|D42原始基准|全精度值|
|---|---:|
|聚合before-old|0.9055555555555554|
|聚合after-old|0.8166666666666667|
|聚合seen-new|0.8133333333333336|
|聚合同rowH|0.8063144081331686|
|average forgetting|0.08888888888888886|
|mean joint floor|0.23333333333333334|
|最低before-old类|0.7666666666666667|
|最低after-old类|0.5|
|最低seen-new类|0.7|

逐场景全精度基准为：clear的before/after/new/H/forgetting/joint=`0.9833333333333332/0.9/0.9400000000000001/0.9152815783250565/0.08333333333333333/0.4`；low-elev=`0.85/0.7666666666666667/0.74/0.7373028949766562/0.0833333333333333/0.2`；rain=`0.8833333333333332/0.7833333333333334/0.76/0.7663587510977931/0.09999999999999998/0.1`。

结构进入正式实现必须同时满足：

1. lifecycle、ground、source、query、registry与资源闭包全部通过；
2. int8/FP32的before/final argmax变化均为0，三类pairwise margin符号翻转为0；
3. 聚合before-old、after-old、seen-new、同rowH均不低于D42，average forgetting不高于D42；
4. 最低before-old类不低于0.7666666666666667、最低after-old类不低于0.5且最低seen-new类不低于0.7；mean joint floor不低于D42；最低after-old、最低seen-new与mean joint floor三者中至少一项严格改善；
5. clear/low-elev/rain每个场景的before-old、after-old、seen-new、H和joint floor均不低于D42，forgetting均不高于D42；
6. 若两个结构均通过，先最大化`min(最低after-old,最低seen-new)`，再最大化mean joint floor、聚合H，最后选择更低状态/MAC者；不得查看query打破并列。

`full_centered_control`不参与D43结构选择，即使单独消除量化翻转也不能在本轮晋级或正式化；它只能成为下一轮重新预注册的机制证据。若两个结构均未通过上述全部门，则拒绝本轮结构并进入新的类对称机制，不访问N607。

## 5.实现与执行计划

- 探针脚本：`code/scripts/probe_d43_structured_covariance.py`。
- 单测：`tests/test_probe_d43_structured_covariance.py`。
- 基础Runner：D42已提交版本`55a76bc1`及其后续纯报告提交；执行前创建隔离worktree并记录Git head。
- 运行时：只读预加载D41已验证的三个封存运行时文件，逐文件断言完整SHA；随后加载D43 worktree中的D42 core和Runner。
- 每个arm写独立输出，并附加不改写基础`RECEIPT.json`的`D43_PROBE_METADATA.json`，明确`formal_candidate=false`、脚本SHA、基础receipt SHA、所有基础artifact SHA和运行时SHA。D41`run_d19`会预加载12个`cvsrffi`模块；脚本在导入前后逐个锁定实际path+SHA，并把完整预加载闭包、legacy SHA、探针脚本SHA和arm写入patched candidate lock。基础Runner随后把该复合lock SHA写进receipt，避免实际执行D41代码却声明D43 worktree同名文件。包装器还逐条核对receipt的`candidate_set/mode/query/formal/status/selected`、105行hash和30条D43 fit audit，把selector强制改为identity并禁用selected-only full-K10 refit。
- 结果报告必须保留全部匿名类的before-old、after-old和seen-new逐类准确率及三类最低值；不得只报聚合或单独极值。
- 本地验证：`ssr-gpu`环境串行运行单测、`py_compile`与`git diff --check`；不得并发调用Conda。

当前预注册与30/30项目测试已通过；pytest结束后的Windows临时目录清理出现一次`WinError 5` atexit噪声，但测试进程exit code为0且30项全部通过。尚未运行真实arm，不构成机制有效、实验完成或性能晋级。

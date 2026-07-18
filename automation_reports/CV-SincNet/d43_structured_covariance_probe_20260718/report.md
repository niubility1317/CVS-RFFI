# D43结构化协方差与量化稳定探针报告

## 1.身份与状态

- 实验ID：`d43_structured_covariance_probe_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
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

## 6.真实执行与启动证据

- 实现提交与隔离worktree：Git`2a1206b71f18145c18abf781358363b5aed68f81`，`E:\type10-7\code\snapshots\d43wt`，执行前`git status -sb`为detached clean。
- 运行时：本地`ssr-gpu`绝对解释器、`device=auto`，实际metric fit落在`cuda:0`；D41 legacy worktree为`E:\type10-7\code\snapshots\d41wt`。
- 输入：D18 K10/new5同一before/after capsule、两份seal、同一formal policy及before/after v2 authorization/envelope；D22 ground int8 component；D19 class binding。所有输入SHA与D42相同。
- 输出：本报告目录下`full_centered_control`、`block3_centered`、`diagonal_centered`。
- 首次full arm命令沿用D42报告的“关键CLI”摘录，遗漏8个必需policy/envelope参数，argparse在support打开前fail closed且未创建输出。补回D41成功命令中的完整8项后原arm重试成功；这属于本地启动参数不完整，不是机制或数据失败。
- 三个arm随后串行完成，各105/105行；receipt elapsed分别`32.640s/33.314s/32.823s`。三个selector均被探针强制为identity，`selected_positive_route=false`、`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`，selected-only full-K10均未执行。

完整CLI以D41报告第9节成功命令为底，只把入口换为：

```powershell
python E:\type10-7\code\snapshots\d43wt\code\scripts\probe_d43_structured_covariance.py `
  --d43-arm <full_centered_control|block3_centered|diagonal_centered> `
  --runtime-root E:\type10-7\code\snapshots\d41wt `
  --probe-root E:\type10-7\code\snapshots\d43wt `
  <D41第9节完整before/after policy+authorization+envelope、component、class-binding参数> `
  --output E:\type10-7\automation_reports\CV-SincNet\d43_structured_covariance_probe_20260718\<arm> `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 7.完整同row结果

|实验/候选|机制/精度|before-old|after-old|seen-new|同rowH|遗忘|joint floor|最低after-old|最低seen-new|old→new/new→old/new-new|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|B3 reference|exact strong B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|60.00%|40.00%|33/22/19|合法比较器|
|D42 original|full covariance int8|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|D43全精度基准|
|full-centered|int8|91.11%|81.11%|82.67%|80.97%|10.00pp|23.33%|50.00%|73.33%|27/9/17|量化对照；按锁不参选|
|full-centered|FP32 matched|91.11%|81.11%|82.67%|80.97%|10.00pp|23.33%|50.00%|73.33%|27/9/17|与D42 original FP32逐折prediction SHA相同|
|3-block centered|int8|92.22%|83.89%|82.67%|82.30%|8.33pp|30.00%|56.67%|66.67%|19/10/16|聚合正信号，最低新类/场景门失败|
|3-block centered|FP32 matched|92.22%|83.89%|82.67%|82.30%|8.33pp|30.00%|56.67%|66.67%|19/10/16|与int8 outer完全一致|
|diagonal centered|int8|76.67%|68.33%|58.67%|61.41%|8.33pp|3.33%|40.00%|46.67%|29/23/39|全面退化|
|diagonal centered|FP32 matched|76.11%|68.33%|58.67%|61.41%|7.78pp|3.33%|40.00%|46.67%|29/23/39|before与int8差1个argmax|

3-block相对D42 original聚合改善before`+1.67pp`、after-old`+2.22pp`、seen-new`+1.33pp`、H`+1.66pp`、joint floor`+6.67pp`，遗忘下降`0.56pp`，并把final错误从26/10/18降到19/10/16；但最低seen-new从70%降到66.67%，因此不能用聚合改善掩盖新类下尾。

## 8.逐场景结果

|场景|方法|before-old|after-old|seen-new|同rowH|遗忘|joint floor|
|---|---|---:|---:|---:|---:|---:|---:|
|clear|D42|98.33%|90.00%|94.00%|91.53%|8.33pp|40.00%|
|clear|full-centered|98.33%|90.00%|96.00%|92.70%|8.33pp|40.00%|
|clear|3-block|98.33%|93.33%|96.00%|94.52%|5.00pp|50.00%|
|clear|diagonal|83.33%|75.00%|70.00%|70.88%|8.33pp|0|
|low-elev|D42|85.00%|76.67%|74.00%|73.73%|8.33pp|20.00%|
|low-elev|full-centered|86.67%|75.00%|74.00%|72.79%|11.67pp|20.00%|
|low-elev|3-block|88.33%|81.67%|72.00%|74.72%|6.67pp|20.00%|
|low-elev|diagonal|70.00%|63.33%|46.00%|51.24%|6.67pp|0|
|rain|D42|88.33%|78.33%|76.00%|76.64%|10.00pp|10.00%|
|rain|full-centered|88.33%|78.33%|78.00%|77.41%|10.00pp|10.00%|
|rain|3-block|90.00%|76.67%|80.00%|77.65%|13.33pp|20.00%|
|rain|diagonal|76.67%|66.67%|60.00%|62.10%|10.00pp|10.00%|

3-block的阻断单元是low-elev seen-new`72%<74%`，以及rain after-old`76.67%<78.33%`和forgetting`13.33pp>10pp`。full-centered的阻断单元是聚合after-old/forgetting，以及low-elev after-old`75%<76.67%`、H`72.79%<73.73%`、forgetting`11.67pp>8.33pp`。

## 9.全部匿名类逐类结果

|角色|handle前缀|D42 original|full-centered|3-block|diagonal|
|---|---|---:|---:|---:|---:|
|before-old|`1f33`|90.00%|90.00%|90.00%|73.33%|
|before-old|`33bb`|93.33%|93.33%|96.67%|90.00%|
|before-old|`75aa`|93.33%|93.33%|96.67%|90.00%|
|before-old|`8b02`|76.67%|80.00%|80.00%|60.00%|
|before-old|`a53c`|100.00%|100.00%|100.00%|76.67%|
|before-old|`f8df`|90.00%|90.00%|90.00%|70.00%|
|after-old|`1f33`|86.67%|86.67%|90.00%|73.33%|
|after-old|`33bb`|93.33%|90.00%|93.33%|83.33%|
|after-old|`75aa`|90.00%|90.00%|93.33%|80.00%|
|after-old|`8b02`|50.00%|50.00%|56.67%|40.00%|
|after-old|`a53c`|76.67%|76.67%|80.00%|63.33%|
|after-old|`f8df`|93.33%|93.33%|90.00%|70.00%|
|seen-new|`09f8`|70.00%|73.33%|66.67%|46.67%|
|seen-new|`1c2a`|90.00%|90.00%|93.33%|66.67%|
|seen-new|`b8fb`|70.00%|73.33%|76.67%|63.33%|
|seen-new|`d3af`|86.67%|86.67%|90.00%|63.33%|
|seen-new|`f608`|90.00%|90.00%|86.67%|53.33%|

3-block改善了最低旧类`8b02`，但把新类`09f8`从70%降到66.67%；这是预注册最低新类门的直接失败，不允许为该handle增设专属分支。

## 10.预注册门逐项

|门|full-centered|3-block|diagonal|
|---|---|---|---|
|协议/lifecycle/ground/source/state/resource闭包|PASS|PASS|PASS|
|before/final int8-FP32 argmax0变化且margin0翻转|PASS|PASS|FAIL（before1）|
|聚合before/after/new/H不退化|FAIL|PASS|FAIL|
|聚合forgetting不增加|FAIL|PASS|PASS|
|最低before/after/new与joint floor不退化|PASS|FAIL（new）|FAIL|
|final三项floor至少一项严格提高|PASS|PASS|FAIL|
|三场景before/after/new/H/joint不退化|FAIL|FAIL|FAIL|
|三场景forgetting不增加|FAIL|FAIL|PASS|
|全部预注册门|FAIL|FAIL|FAIL|

因此两个正式结构都被拒绝，full-centered按预注册只保留为下一轮机制证据。D43不实现正式Runner候选、不运行selected-only full-K10、不访问N607。

## 11.量化、pairwise、完整训练日志与资源

|方法|before/final argmax变化|margin翻转|max outer score误差|final coefficient误差|max intercept误差|final support score误差|
|---|---|---:|---:|---:|---:|---:|
|D42 original|1/3|3|1.0283|0.0460|0.9990|1.0273|
|full-centered|0/0|0|0.0968|0.0152|0.0567|0.0555|
|3-block|0/0|0|0.0654|0.0097|0.0598|0.0665|
|diagonal|1/0|0|0.1245|0.0082|0.1199|0.1252|

full-centered的FP32逐折before/final prediction SHA与D42 original FP32完全相同，证明在真实outer面公共项去除没有改变FP32决策，却把量化翻转降为0；这是D43最清晰的机制结论。FP32 support两两margin的有限精度最大漂移为full`0.00587/0.00512`、block`0.00473/0.00464`、diagonal`0.00375/0.00536`（before/final），均另行记录，没有误写为浮点逐bit等价。

|方法|pairwise错序old→new/new→old/new-new|原始最低margin old→new/new→old/new-new|
|---|---|---|
|D42 original|31/20/19|−69.19/−139.95/−39.13|
|full-centered|32/19/18|−69.89/−140.80/−39.08|
|3-block|27/17/19|−87.97/−152.81/−37.26|
|diagonal|39/41/51|−131.33/−144.17/−47.79|

三个arm的int8路线各15条fit×20步=300条完整trace，全部finite且逐条与D42 B20 trace相同；FP32 matched也复用同一轨迹。每条真实outer资源仍为2016参数、20 epoch/20 step、8583B state、65,442,816 adaptation MAC、6624 MAC/query、CUDA peak22,886,912B；host FP64 covariance peak仍未实测。三种结构的最大condition number（before/final）为full`204648/132824`、block`167398/112223`、diagonal`156195/101176`。结构越简单虽降低条件数和量化误差，但diagonal丢失判别相关性并造成性能崩塌。

## 12.Artifact闭包

|arm|candidate lock|training log SHA/大小|support audit SHA/大小|selection SHA/大小|receipt SHA/大小|metadata SHA/大小|
|---|---|---|---|---|---|---|
|full-centered|`eb2f23b4…61c7af`|`46039246…8eb7b`/3,827,188B|`78814f36…d0bb7`/313,173B|`93a702ba…376a0`/2990B|`47627892…4f992`/4561B|`180eb53c…3f4e9`/2737B|
|3-block|`62622e54…40720`|`e1a66ed1…a2872`/3,826,539B|`64d5d33b…0c8c5`/313,168B|`12e6af66…46a3a`/2991B|`9639337f…3946d`/4560B|`1cfffdf7…2795f`/2731B|
|diagonal|`dafedbf8…fdc98`|`c520e80d…84549`/3,826,722B|`53dd0478…5f413`/313,168B|`1e75615e…e0c02`/2993B|`3730da79…78cf`/4560B|`888b3f53…54f0`/2738B|

三个arm的`geometry_audit.json`均为`ae4b735a…300dc`/5132B，`resource_audit.json`均为`00f364e5…6e2b`/6498B；它们只描述未执行full-K10的共同部署面。每个metadata均反向核对base receipt和五个基础artifact，并记录探针脚本`50b2d476…8f7e5`、D41 legacy及12模块完整SHA闭包。

## 13.解释与下一轮

D43给出两个可复用但必须分开的事实：

1. 去除类公共仿射score是有效的量化稳定机制：full FP32 outer决策与D42逐折相同，int8却从1/3个argmax变化和3个margin翻转降为全0，intercept最大误差从0.999降到0.057。
2. 3-block提供最强聚合/旧类/混淆正信号，但硬置零全部跨块协方差会牺牲`09f8`与low-elev新类、rain旧类稳健性；纯diagonal则证明跨维相关性不可完全丢弃。

下一轮最高价值机制是类对称的full-centered＋3-block-centered双几何融合：保留两套已验证0翻转state，各自用support-only、全类对称的pairwise-logit RMS做单标量归一化，再固定1:1平均；不扫描融合权重、不设置类/场景分支、不读取query。它直接利用full对新类/rain floor的保护与block对旧类/聚合的改善。该建议必须在D44另行预注册并先跑同一15折，D43本身仍是完整执行的诊断性负结果。

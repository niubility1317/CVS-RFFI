# D44 full/block RMS固定融合探针报告

## 1.身份与状态

- 实验ID：`d44_full_block_rms_fusion_probe_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、clear/low-elev/rain、5个physical-rank held折。
- query sealed；复用D42/D43同一`p2_min_v1/VALIDATED_ONCE`固定received-IQ capsule、old-only B20、D22 ground int8与D19 class binding。

## 2.单一机制

D43证明full-centered保护新类/rain并实现0量化翻转；3-block-centered同时提高聚合before/after/new/H、旧类floor、joint floor和三类混淆，但硬置零跨块协方差使最低新类从70%降到66.67%，low-elev new和rain old退化。D44只验证两者能否以类对称方式互补，不再改变特征、B20、协方差估计器、support split或量化器。

对每套组件`g∈{full,block}`，在同一fit support上计算所有类logit并做行内类中心化：

`s_g=sqrt(mean_{i,c}((δ_g(x_i,c)−mean_j δ_g(x_i,j))^2))`。

融合score固定为`δ_D44=0.5δ_full/s_full+0.5δ_block/s_block`。`s_g`只读取support feature和两套全registry score，不读取support label、old/new role、匿名handle或场景身份；标签只用于两套原始等先验LDA拟合。两套affine可在编译前线性合成为一套coefficient/intercept，因此query仍是单行单state全registry argmax，不产生双模型选择、quota或global assignment。资源审计把before与final各两次covariance fit计为4次closed-form LDA，不能沿用D42的2次计数；query MAC/state仍只按融合后的单state计算。

固定权重只有`0.5/0.5`，不扫描融合权重、threshold、rank、lr、epoch、shrinkage、类/场景参数。若RMS非finite或≤`1e-12`则fail closed。

## 3.预注册判定

基准仍是D42 SHA256=`4ee51dd3d21ae8751bfaa64eb82d2a5a5371728fc7c1502bdb3af221d349614a`的15条原始全精度int8行；定义、全精度基准和`1e-12`比较规则完全沿用D43第4节，不根据D44结果更改。

D44进入正式实现必须同时满足：

1. lifecycle、ground、source、state、resource、query与复合source lock全部通过；
2. before/final int8-FP32 argmax变化均为0，三类margin符号翻转为0；
3. 聚合before-old、after-old、seen-new、H不低于D42，average forgetting不高于D42；
4. 最低before-old≥0.7666666666666667、最低after-old≥0.5、最低seen-new≥0.7、mean joint floor≥0.23333333333333334；后三者至少一项严格改善；
5. clear/low-elev/rain每个场景的before/after/new/H/joint均不退化，forgetting均不增加；
6. final old→new/new→old/new-new不超过D42的26/10/18；
7. 全部匿名类before/after/new、三类最低值、pairwise、量化、完整trace与资源均保存并报告。

探针复用D43已审计的D41 legacy＋12模块精确闭包、patched candidate lock、support/selection/receipt三方SHA重算和identity/full-K10 guard。即使D44全部门通过，本探针仍不自我晋级；只能据此另行实现正式候选。

## 4.文件与验证计划

- 探针：`code/scripts/probe_d44_full_block_rms_fusion.py`。
- 公共已审计helper：`code/scripts/probe_d43_structured_covariance.py`，仅把arm→structure表抽成可扩展常量，不改变D43三arm算法。
- 单测：`tests/test_probe_d44_full_block_rms_fusion.py`及D43/D42回归。
- 真实输出：`E:\type10-7\automation_reports\CV-SincNet\d44_full_block_rms_fusion_probe_20260718\full_block_rms_equal`。
- 环境：本地`ssr-gpu`，串行执行；只有全部本地门通过才考虑正式实现，当前不访问N607。

本地实现验证已完成：`py_compile`通过；D42–D44定向回归`56 passed`。验证覆盖固定RMS等权融合、公共score不变性、类别置换等变、K1 unit-covariance fallback、四次LDA拟合/MAC口径、D44专属30行artifact字段、资源闭包及篡改失败；D43的额外source closure同时改为拒绝覆盖保留键。pytest退出后的Windows临时目录清理出现既知`WinError 5`噪声，但进程退出码为0，不是测试失败。

## 5.真实执行

- Git提交：`34764bd3cda82870ca4b4a4319a94430e0232844`；隔离worktree：`E:\type10-7\code\snapshots\d44wt`，detached clean。
- 环境：`ssr-gpu`绝对解释器；`device=auto`，实际metric fit为`cuda:0`。
- 输入：D42/D43同一D18 K10/new5 before/after capsule、两份seal、formal policy、v2 authorization/envelope、D22 ground int8组件和D19 class binding；未重建或重验数据。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d44_full_block_rms_fusion_probe_20260718\full_block_rms_equal`。
- 结果：105/105行，7候选×3场景×5fold；elapsed`37.529s`。`selected_candidate_id=Z0_SUPPORT_ONLY`、`query_opened=false`、`formal_metric_claim_allowed=false`、`selected_positive_route=false`，未执行selected-only full-K10，未访问N607。
- D44 metadata专属verifier核验30条D44 int8/FP32行、before/final共60份融合audit、4次LDA资源口径、单融合query state及D43 helper SHA，全部通过。

## 6.完整同row结果

|候选|精度|before-old|after-old|seen-new|同rowH|遗忘|joint floor|最低before|最低after|最低new|old→new/new→old/new-new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|D42 original|int8|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|76.67%|50.00%|70.00%|26/10/18|预锁基准|
|D43 3-block|int8|92.22%|83.89%|82.67%|82.30%|8.33pp|30.00%|80.00%|56.67%|66.67%|19/10/16|聚合最强旧正信号，最低new失败|
|D44 fixed RMS 1:1|int8|92.22%|82.22%|84.00%|83.10%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|遗忘、rain与量化门失败|
|D44 fixed RMS 1:1|FP32 matched|92.22%|82.78%|84.00%|83.38%|9.44pp|26.67%|80.00%|53.33%|70.00%|23/8/16|仍未通过遗忘/rain门|

D44相对D42 original提高before`+1.67pp`、after-old`+0.56pp`、seen-new`+2.67pp`和H`+2.47pp`，最低after-old提高`+3.33pp`，最低seen-new持平，三类final混淆均下降；但聚合遗忘增加`1.11pp`。FP32只比int8多保住1个旧类样本，恰好使joint从23.33%升到26.67%，说明当前int8量化仍会在低margin单元改变结论。

## 7.逐场景结果

|场景|方法|before-old|after-old|seen-new|同rowH|遗忘|joint floor|
|---|---|---:|---:|---:|---:|---:|---:|
|clear|D42|98.33%|90.00%|94.00%|91.53%|8.33pp|40.00%|
|clear|D44 int8|98.33%|90.00%|98.00%|93.83%|8.33pp|40.00%|
|low-elev|D42|85.00%|76.67%|74.00%|73.73%|8.33pp|20.00%|
|low-elev|D44 int8|88.33%|80.00%|74.00%|76.88%|8.33pp|20.00%|
|rain|D42|88.33%|78.33%|76.00%|76.64%|10.00pp|10.00%|
|rain|D44 int8|90.00%|76.67%|80.00%|78.30%|13.33pp|10.00%|

clear与low-elev的old/new/H均不退化，rain的新类与H提高，但rain after-old退化`1.67pp`且遗忘增加`3.33pp`，直接触发逐场景硬门。

## 8.全部匿名类

|角色|handle前缀|D42 original|D44 int8|D44 FP32|
|---|---|---:|---:|---:|
|before-old|`1f33`|90.00%|90.00%|90.00%|
|before-old|`33bb`|93.33%|96.67%|96.67%|
|before-old|`75aa`|93.33%|96.67%|96.67%|
|before-old|`8b02`|76.67%|80.00%|80.00%|
|before-old|`a53c`|100.00%|100.00%|100.00%|
|before-old|`f8df`|90.00%|90.00%|90.00%|
|after-old|`1f33`|86.67%|90.00%|90.00%|
|after-old|`33bb`|93.33%|93.33%|93.33%|
|after-old|`75aa`|90.00%|90.00%|93.33%|
|after-old|`8b02`|50.00%|53.33%|53.33%|
|after-old|`a53c`|76.67%|73.33%|73.33%|
|after-old|`f8df`|93.33%|93.33%|93.33%|
|seen-new|`09f8`|70.00%|70.00%|70.00%|
|seen-new|`1c2a`|90.00%|93.33%|93.33%|
|seen-new|`b8fb`|70.00%|76.67%|76.67%|
|seen-new|`d3af`|86.67%|90.00%|90.00%|
|seen-new|`f608`|90.00%|90.00%|90.00%|

D44恢复了D43 3-block丢失的`09f8`最低new，并改善`8b02` after-old，但`a53c`从76.67%降到73.33%；通用下尾仍未形成可晋级改善。

## 9.量化、资源与完整日志

|项目|D44 int8结果|门|
|---|---:|---|
|before int8/FP32 argmax变化|0|PASS|
|final int8/FP32 argmax变化|1|FAIL（要求0）|
|margin符号翻转|1|FAIL（要求0）|
|最大score绝对误差|0.001220703125|记录|
|final old→new/new→old/new-new|24/8/16|PASS（不高于26/10/18）|
|pairwise old→new/new→old/new-new|30/16/19|记录|

D44 int8与FP32 matched各15行×20 epoch/step=`300`条完整训练trace，全部finite，epoch/step完整覆盖1–20。每条int8资源为2016个trainable参数、20 epoch、20 optimizer steps、8583B持久state、125,908,992 adaptation MAC，其中两阶段四次LDA为120,932,352 MAC、metric为4,976,640 MAC；query为6624 MAC，CUDA peak22,886,912B。FP32 matched state为14,809B。host FP64 covariance peak未实测。full/block support logit RMS范围分别为`26.5145–76.5324`和`25.2803–68.9226`，均正且finite。

## 10.门与artifact闭包

|门|结果|
|---|---|
|协议/lifecycle/ground/source/state/resource闭包|PASS|
|聚合before/after/new/H均不退化|PASS|
|聚合forgetting不增加|FAIL|
|最低before/after/new与joint不退化|PASS|
|至少一项final floor严格改善|PASS（最低after）|
|三场景before/after/new/H/joint不退化|FAIL（rain after）|
|三场景forgetting不增加|FAIL（rain）|
|量化0/0/0|FAIL（0/1/1）|
|三类final混淆不增加|PASS|

|artifact|SHA256|大小|
|---|---|---:|
|`training_log.jsonl`|`b01e2a7e…e1cb1`|3,873,061B|
|`support_audit.json`|`c6218dbe…3b1c`|313,268B|
|`selection.json`|`07e5e751…ea01`|2991B|
|`RECEIPT.json`|`16d23376…fb5`|4655B|
|`D44_PROBE_METADATA.json`|`d7eea800…d6d8`|3195B|
|`resource_audit.json`|`00f364e5…6e2b`|6498B|
|`geometry_audit.json`|`ae4b735a…300dc`|5132B|

## 11.结论与下一轮

D44证明“full保护新类＋block提高旧类”的score级融合方向有效：它取得本轮最高seen-new和H，并同时改善最低after-old、保持最低new、降低三类混淆。失败集中在两个相互关联的单元：全局固定1:1在rain仍继承block的旧类损失，同时融合后的近零margin造成1个int8翻转。下一轮不应扫描固定权重；更有信息量的D45是使用完全相同、类对称的support内部可靠度公式，对每个fit自适应估计full/block全局权重，并对低margin量化稳定性加入support-only连续惩罚。公式不得读取outer-held/query、old/new角色、handle或场景ID，权重仍对当前fit所有类共享。

D44本身为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不实现正式candidate、不生成125 capsule、不消耗confirmation query、不访问N607。当前goal仍active。

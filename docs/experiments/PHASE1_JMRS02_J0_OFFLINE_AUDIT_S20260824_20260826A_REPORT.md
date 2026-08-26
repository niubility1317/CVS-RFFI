# PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A预登记与追踪报告

## 一、目标与边界

本轮只执行JMRS02的J0离线联合审计。输入为已闭合的JMRS01 428064条prediction和对应truth；不训练模型、不调用GPU、不改写旧run、不重新生成prediction。J0只能判断旧分支错误集合是否存在独有rescue与组合协同，不能证明角色重构后的RC-X、RC-Z、稳健谱残差或phase nuisance有效，也不能声明target receiver DG。

## 二、不可覆盖运行定义

- run ID：`PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A`
- 本地代码分支：`codex/phase1-jmrs02-j0-20260826`
- 实验代码commit：`ad2e756b803849315a77785e7d8a7b86462c92f6`
- 本地验证环境：`ssr-gpu`
- N607运行解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；服务器无`ssr-gpu`，沿用原JMRS01已验证环境，不安装新环境
- release Git状态：`3f73806b5fbc51131f68910eb6424e210e2633d6`
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_jmrs02_j0_20260826/3f73806b5fbc51131f68910eb6424e210e2633d6`
- release归档：本地`E:\type10-7\release_archives\PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A_3f73806b.zip`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_jmrs02_j0_20260826/PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A_3f73806b.zip`
- release SHA-256：`8605265bdc83a762187212ad07b939de31fd9653d17cc35cb6321a4d0b5f34d5`
- 输入prediction：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/predictions.jsonl`
- 输入truth：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/truth.jsonl`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j0_20260826/PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A`
- GPU：不使用
- bootstrap：按`receiver×day×scenario`分组，2000次，seed=20260826
- 系统技术停止：输入闭合失败、缺少预登记row/scenario、输出根已存在、非有限结果或无法产生全部J0 JSON时保留现场并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

精确运行命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/score_phase1_jmrs02_j0.py --predictions /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/predictions.jsonl --truth /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/truth.jsonl --output_dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j0_20260826/PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A --bootstrap_resamples 2000 --seed 20260826
```

## 三、预登记组合与晋级规则

组合固定为`R1+D1`、`R1+P2`、`R2+D1`、`D1+P2`、`R1+D1+P2`。对每个组合计算oracle gain、相对最佳单机制的synergy、分组bootstrap 95%CI、rescue Jaccard、相对S1的独有rescue及成员独有rescue。

只有`synergy>0`且95%CI下界>0的组合可形成J0协同信号。J0通过只允许进入角色正确的J1单模块设计，不直接授权联合训练；若没有组合通过，则停止JMRS02联合路线，不启动J1/J2。

## 四、设计追踪

|ID|原报告要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|
|P0-1|`nondegraded`同时相对M0和family sham|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|S1明确标为历史共享容量控制，不伪装成family-specific sham|
|P0-2|breadth相对Core90 rescue|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|只统计M0错误且候选正确|
|P0-3|safe gate排除`alpha=0`虚假通过|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|零采用记为不可评价且不通过|
|P0-4|J0 pairwise/triple unique rescue|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|固定五个组合和分组bootstrap|
|P0-5|统一身份margin|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|使用fold-local Fisher ratio|
|P0-6|成本改名为incremental runtime|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|显式标注不含Core90|
|P0-7|核对day3唯一日期|协议artifact与报告|verified|`protocol_and_smoke.json`为`2021_03_23`|修复旧报告的`2021_03_22`|
|P1|角色正确的RZ0/RZ1/RX1/D1′/P0|未来J1|deferred|等待J0|不得先训练联合|
|P2|只训练有J0证据的组合|未来J1后|deferred|等待J0/J1|J0不直接授权联合|
|P3|正式target DG一次性确认|未来J2|deferred|等待J1|只允许1—2个冻结候选|

## 五、当前状态

聚焦测试：JMRS02-J0 6项通过；JMRS01 scorer回归3项通过。`py_compile`通过。一次P0/P1审查发现bootstrap逐样本展开会造成约5亿次索引操作，已定点改为`receiver×day×scenario`组计数预聚合并通过RED→GREEN复测。

N607 preflight：prediction/truth各428064行，大小约460MB/29MB；输出根不存在；可用内存493GiB，`/home`可用7.3TiB；没有既有JMRS02 scorer进程。J0为CPU离线审计，不占用GPU。

`ANALYZED / J0_SIGNAL / PROCEED_TO_ROLE_CORRECT_J1_SINGLE_MODULES`

## 六、运行闭合与数据健康

- N607仅启动一次该run；旧JMRS01 prediction、truth和run目录均保持只读，未重复训练、未重生成prediction、未覆盖旧产物。
- 输入prediction与truth各428064条，sample ID一一闭合且无重复；预登记row、scenario及所需字段完整。
- 独立scorer自然退出并生成5个预期JSON：语义审计、联合rescue、身份几何、成本口径和决策。日志无`Traceback`、OOM、NaN或非有限值。
- J0组合分析使用50400条`V_select`主审计样本；分组bootstrap按56个`receiver×day×scenario`组进行2000次重采样。这里的50400不是新数据，而是428064条封闭prediction流中用于同row组合判断的主审计子集。
- 本地回收目录：`E:\type10-7\local_artifacts\PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A`。

## 七、语义修复结果

### 7.1 `nondegraded`双基准

|Row|相对S1不劣化receiver|相对M0不劣化receiver|解释|
|---|---:|---:|---|
|M0|7/7|7/7|Core90基线|
|S1|7/7|0/7|历史共享容量控制，不是各family专属sham|
|R1|7/7|1/7|相对弱sham安全不等于相对Core90安全|
|R2|7/7|0/7|旧“7/7通过”不能解释为系统不劣化|
|D1|6/7|0/7|相对Core90仍无receiver达到不劣化|
|P2|1/7|0/7|独立分类角色尤其不合适|

该结果确认复盘报告对旧scorer的批评成立。之后必须把“family机制有效性”和“完整系统安全性”分开报告。

### 7.2 Core90错误救回广度

|Row|rescue数|receiver广度|day广度|LEO场景广度|
|---|---:|---:|---:|---:|
|S1|1004|7|2|3|
|R1|476|7|2|3|
|R2|1202|7|2|3|
|D1|1259|7|2|3|
|P2|648|7|2|3|

广度现已严格定义为“Core90错误而候选正确”，不再统计候选仅优于S1的普通正确样本。所有机制都覆盖7个receiver、2个day和3个LEO场景，但数量只能证明存在错误互补，不能证明这些救回可以由truth-blind gate识别。

### 7.3 safe gate去除`alpha=0`虚假通过

- M0、R1、D1、P2均没有非零alpha fold，因此记为`not evaluable / fail`，不再把回退Core90写成安全通过。
- S1仅1个fold采用非零alpha，选中效用为0.0001389；R2仅2个fold采用非零alpha，选中效用为0.00006944。两者虽按旧约束通过，但覆盖极低，不构成可部署门控证据。

## 八、J0联合互补性结果

|组合|oracle gain|最佳单机制gain|synergy|95%CI|rescue Jaccard|相对S1独有rescue|结论|
|---|---:|---:|---:|---|---:|---:|---|
|R1+D1|3.123pp|2.498pp|0.625pp|[0.472,0.776]pp|0.1023|1070|通过J0|
|R1+P2|2.050pp|1.286pp|0.764pp|[0.609,0.917]pp|0.0881|660|通过J0|
|R2+D1|3.893pp|2.498pp|1.395pp|[1.111,1.639]pp|0.2543|1255|通过J0，协同最大|
|D1+P2|3.258pp|2.498pp|0.760pp|[0.603,0.921]pp|0.1614|1098|通过J0|
|R1+D1+P2|3.778pp|2.498pp|1.280pp|[1.056,1.532]pp|0.0200|1296|通过J0|

五个预注册组合均满足`synergy>0`且分组bootstrap 95%CI下界>0，因此状态为`J0_SIGNAL`。`R2+D1`的oracle accuracy为95.933%，相对Core90 92.040%提高3.893个百分点；但这是使用真值定义“任一分支正确”的不可部署oracle上限，不是联合模型accuracy，也不是域泛化结果。

### 8.1 场景分解

|组合|clean synergy|clear synergy|low-elevation synergy|rain synergy|
|---|---:|---:|---:|---:|
|R1+D1|0.032pp|0.746pp|0.873pp|0.849pp|
|R1+P2|0.040pp|0.841pp|1.079pp|1.095pp|
|R2+D1|0.238pp|1.262pp|1.857pp|2.063pp|
|D1+P2|0.238pp|0.548pp|1.230pp|1.000pp|
|R1+D1+P2|0.270pp|1.183pp|1.929pp|1.714pp|

协同主要来自LEO扰动场景，clean协同很小。这与“机制可能是Core90的条件辅助信息”一致，但尚不能区分物理互补与旧分支各自的source receiver捷径。

## 九、统一身份几何

使用fold-local Fisher ratio `trace(S_B)/(trace(S_W)+eps)`替代跨维欧氏margin。M0为21.084；R1为52.597（相对M0为249.46%）；S1、R2、D1、P2分别为1.148、1.396、1.499、0.649（仅为M0的5.44%、6.62%、7.11%、3.08%）。

R1的高Fisher ratio说明其32维空间在source-LORO审计中具有较强类间/类内比，但不能据此认定receiver信息被移除；JMRS01既有receiver probe表明R1仍可能重排receiver局部结构。其余机制作为独立身份空间明显弱，进一步支持把R2重定义为canonicalizer、D1重定义为残差专家、P2重定义为nuisance/质量估计器。

## 十、参数量与增量运行成本

|Row|可训练参数|缓存Core90后增量时延(ms/sample)|口径|
|---|---:|---:|---|
|M0|0|0.00173|仅缓存Core90访问|
|S1|6438|0.00468|新增分支|
|R1|10711|0.01858|新增分支|
|R2|4454|0.00843|新增分支|
|D1|6438|0.01627|新增分支|
|P2|3878|0.06173|新增分支|

这些数值不包含Core90前向，`full_system_runtime_ms_per_sample`明确为`null`；因此不得再称为端到端推理时延。P2在旧分支中是增量成本最高者。

## 十一、结论、暴露问题与后续边界

### 11.1 已证实

1. 复盘报告指出的三项scorer语义漏洞均真实存在并已修复：sham/M0混淆、错误breadth基准、`alpha=0`虚假安全通过。
2. 旧机制的Core90 rescue集合存在统计稳定的互补性，五个组合全部形成J0信号，最佳为`R2+D1`。
3. 单机制身份几何高度不均衡；除R1外，其余旧分支不适合继续承担独立六分类职责。
4. day3由数据元数据唯一确认为`2021_03_23`，旧报告中的`2021_03_22`已纠正。

### 11.2 尚未证实

1. oracle互补无法证明truth-blind rescue-harm gate能提取收益。
2. J0没有训练RC-X、IQ条件RC-Z、稳健谱残差D1′或circular phase nuisance P0，不能把旧R1/R2/D1/P2的互补直接归因于这些新物理角色。
3. 本轮是source receiver增量审计，不是receiver7—11、day2—3或strict UDU目标域泛化实验。
4. 没有端到端系统时延、显存和FLOPs结果。

### 11.3 决策

J0只授权进入J1角色正确的单模块实现与source-LORO验证。下一轮优先顺序为`RZ0→IQ条件RC-Z→identity-init RC-X→稳健谱残差D1′→phase nuisance P0`；仍禁止把旧R1、D1、P2直接拼接训练。只有J1证明单模块在相对Core90的rescue、harm、receiver floor和非零gate上有效，才允许基于J0优先验证receiver+spectral组合。正式target receiver DG声明必须留到冻结候选后的J2一次性确认。

## 十二、设计吸收、修正与延期统计

- 已落地并验证：7项P0要求。
- 由J0证据授权但尚未实现：5项P1单模块。
- 延期：P2联合训练、P3目标DG确认，共2个阶段。
- 明确否定：把S1当family-specific sham、把相对S1不劣化写成相对Core90安全、把`alpha=0`写成gate通过、把缓存分支时延写成端到端时延、把J0 oracle accuracy写成部署accuracy或DG accuracy，共5类错误解释。
- 当前最高风险：能否在完全truth-blind、held-receiver条件下学习到正的rescue-harm效用；若不能，J0互补只能作为不可提取的oracle上限。

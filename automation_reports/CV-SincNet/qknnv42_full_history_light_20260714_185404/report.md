# 完整历史qKNN星上轻量化报告

## 1.实验定义

|字段|内容|
|---|---|
|实验ID|`qknnv42_full_history_light_20260714_185404`|
|目标|优化完整历史qKNN链路，而不是只优化非参数head；最终候选相对严格完整历史基线的`old_acc`、`seen_new_acc`和`H_old_new`矩阵均值均不得下降超过3pp|
|严格历史链路|`ADV3B02+60epoch id_norm_late_feature+5-view TTA+FFT96+三LEO场景+old/new角色Oracle+类别配额Hungarian+dense LP`|
|历史基线|严格加载125行：`old_acc=84.07%`、`seen_new_acc=93.24%`、`H_old_new=88.23%`|
|声明边界|角色与类别配额使用Oracle，因此历史基线和保留该约束的压缩候选均为`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`，不能写成卫星自主部署性能|
|特征适配原则|ADV3B02完全冻结且梯度更新为0；适配发生在qKNN enrollment阶段，由K-shot support拟合`support_diag_whiten_fisher`，不读取query，不更新ADV3B02|

严格历史基线证据来自`runs/cvs_qknnv42_full_legacy_oracle_strict125_20260714_183556`，完整125行及审计见`automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/report.md`。

## 2.完整链路计算热点

使用严格ADV3B02 checkpoint、输入`2×256`和THOP进行静态MAC审计；qKNN head资源由严格完整125行的375个场景row汇总。

|排序|组件|历史资源|星上重复方式|结论|
|---:|---|---:|---|---|
|1|5-view下的完整dual ADV3B02辅助前向|`37.014M MAC/view×5=185.070M MAC/physical sample`|每条support/query重复5次|最大推理项；其中qKNN只消费`z_id`，domain分支结果未使用|
|2|qKNN dense LP+support/prototype+FFT双head|均值`22.725M MAC/query`，最大`46.203M`|每批query执行|移除前端冗余后成为下一重项|
|3|dense query graph|均值`1.659MB`，最大`3.277MB`|随query batch临时分配|不是最大算力，但影响峰值内存和逐样本流式部署|
|4|96维FFT sketch|每个TTA view执行一次256点FFT并压缩到96维|历史为5次/sample|明显轻于ADV3B02卷积主干，但随view数线性重复|
|5|场景选择、old/new角色筛选、类别配额Hungarian|类别数8、query数160量级|每批一次|计算量小；主要问题是使用Oracle，而不是资源占用|

60epoch`id_norm_late_feature`另有289,685个可训练参数，需要反复执行ADV3B02前向/反向。它属于适配阶段而非单次推理，但若放到星上仍是不可接受的最大训练成本。因此本轮不再训练ADV3B02，而是把适配下沉到qKNN support enrollment。

## 3.第一项无损压缩：只执行identity分支

历史导出调用dual模型的`return_aux=True`，同时执行`id_backbone`、`dom_backbone`、domain enhancer、domain head和GRL相关head；后续qKNN只读取`z_id`与TX logits，domain输出全部未使用。

本轮新增identity-only严格前向：直接调用同一checkpoint中的`id_backbone`并使用原模型`_pick_z_id`，跳过整个domain路径。随机3样本的完整dual与identity-only逐元素复核结果：

|输出|max absolute difference|
|---|---:|
|`z_id`|0.0|
|TX logits|0.0|

资源变化：

|前端|MAC/view|5-view MAC/sample|相对历史dual 5-view|
|---|---:|---:|---:|
|历史dual auxiliary前向|37.014M|185.070M|基线|
|identity-only前向|8.912M|44.561M|-75.92%|
|identity-only 1-view候选|8.912M|8.912M|-95.18%|

因此第一项压缩在保持5-view时已经将ADV3B02前端MAC下降75.92%，且特征与logits bit-exact，理论性能下降为0pp。按历史平均head MAC合并估算，完整推理从约`207.794M`降至`67.285M MAC/sample`，整体下降约67.62%；此处尚未计算FFT小项。

## 4.qKNN侧特征适配

新增显式配置`qknnv42_feature_adapter_mode=support_diag_whiten_fisher`。适配器由当前Stage2-C row的K-shot old+seen-new support拟合中心、对角scale和类Fisher权重；只在support注册/更新时计算一次，query仅应用固定变换。记录字段包括：

- `feature_adapter_gradient_updates=0`；
- `feature_adapter_uses_query=false`；
- `feature_adapter_updates_adv3b02=false`；
- ADV3B02 exporter的`skip_adapter_training=true`与`adv3b02_gradient_updates=0`。

这满足“特征适配重点在qKNN，而不是再次训练ADV3B02”的要求。60epoch历史适配器仍保留为原始比较基线，不进入新候选。

## 5.待执行完整历史对照矩阵

候选保持FFT96、三个`leo_*_weak`场景、dense LP、old/new角色和类别配额Oracle不变，只改变：

1. ADV3B02始终严格加载、冻结、identity-only；
2. 不进行60epoch模型适配；
3. 特征适配由qKNN support完成；
4. 比较`none/rx_shift3/rx_cfo3/rx_light5`的1/3/3/5-view。

矩阵为`4 policy×5 receiver×5 seed×5 K=500`行。每个candidate row必须与严格历史125行相同receiver/seed/K的support/query split SHA256一致。晋升条件是候选的`old_acc`、`seen_new_acc`和`H_old_new`三个矩阵均值相对严格历史完整体分别下降均不超过3pp；不是只与候选自身5-view比较。

## 6.实现、验证与远端边界

本地变更：

- `code/cvsrffi/identity_only_forward.py`：qKNN identity-only bit-exact前向；
- `code/scripts/train_apply_phase1_iq_preadapter_20260703.py`：新增ADV3B02零训练导出路径；
- `code/export_spaceborne_features.py`：普通严格导出同样跳过未使用domain分支；
- `paper_reproduction/cvs_aligned/cvs_method_runner.py`：显式qKNN support特征适配及无query/无ADV3更新审计；
- `paper_reproduction/scripts/benchmark_qknnv42_tta_policies.py`：增加完整历史profile和严格历史≤3pp门槛；
- `paper_reproduction/scripts/run_cvs_qknnv42_tta_ablation_20260714.sh`：改为冻结ADV3B02、零epoch、完整历史head的500行启动器。

本地验证：Python编译通过；首轮相关测试为`26 passed`；审查修复后为`29 passed`；两个CLI`--help`和Bash`-n`通过；真实strict checkpoint的identity-only等价性为`z_id/logits max_abs_diff=0/0`。新增真实`DualCVSincNetDisentangle`回归测试同时验证完整`return_aux=True`与identity-only的`z_id/TX logits`逐元素一致，并确认轻型路径不调用`dom_backbone`。

2026-07-14 19:13直连N607预检再次PASS。19:15只读进程/GPU审计显示6个活动GPU训练进程及其调度器仍在运行，属于`phase1_dgleo_p0factorial8_20260714`。按活动任务monitor-only规则，本轮尚未同步或启动500行矩阵；没有干预现有任务，SSH/TCP22连接已在每次检查后归零。

## 7.独立代码审查与修复

独立审查首轮结论为`Request changes`，包含1项Critical、3项Important和2项Minor。现已全部处理：

|审查问题|修复|验证|
|---|---|---|
|历史门槛目录只校验125行和split，未锁定`84.07/93.24/88.23`|新增历史125行三指标矩阵均值锁定，容差只允许原始精度向两位小数取整；重复run key直接失败|错误历史目录无法成为promotion reference|
|候选继承历史`ID_NORM_LATE_FEATURE_E60`模型名|候选显式覆盖为`...FROZEN_ZID_QKNN_SUPPORT_DIAG_WHITEN_FISHER`|`resolved_config/split_manifest`不再冒充60epoch适配模型|
|summary无条件声明零训练，但未校验feature cache|逐cache校验`payload_source`、checkpoint SHA256、`skip_adapter_training=true`、`adv3b02_gradient_updates=0`、identity-only和domain未执行|旧60epoch cache或错误checkpoint将fail closed|
|直接调用exporter可覆盖已有artifact|每个cell的输出目录在数据加载和导出前拒绝非空目录|不会静默覆盖历史feature cache|
|真实dual bit-exact缺少持续回归|新增真实dual逐元素等价与domain调用计数测试|纳入`29 passed`|
|零训练顶层manifest仍称`phase1_iq_frontend`|改为`qknnv42_frozen_adv3b02_identity_only_features_v1`和`frozen_adv3b02_identity_only_z_id`|来源口径与执行路径一致|

第二轮复审另发现1项Critical和2项Important，均已闭合：`full_legacy_oracle`现在强制提供历史目录且固定三指标不可由CLI覆盖；feature cache额外强制`checkpoint_load_strict=true`且load audit三类异常计数均为0；identity独立导出目录也在任何计算前拒绝覆盖。复审修复后的针对性测试为`15 passed`。

待同步文件SHA256：

|文件|SHA256|
|---|---|
|`code/cvsrffi/identity_only_forward.py`|`8E262522BCFDC956A68835BCDD7AF1E33345B3F88CACDC1AE4BAC0D9F3DCB247`|
|`code/export_spaceborne_features.py`|`70941AED6C9FE90F398096162613A1C613A88F57FBFBDECA80C82624A95D04B2`|
|`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`|`DA2092D0A5FECCBD1481EA023F8AB0E9941E38840BDD15C7285F09DA154F1CFC`|
|`paper_reproduction/cvs_aligned/cvs_method_runner.py`|`89ED64745AEEF4CDA53584FC9F67AF2FB98A3EC9C2AEA7B54452B2B1BE033C80`|
|`paper_reproduction/scripts/benchmark_qknnv42_tta_policies.py`|`7B46EC735F359309A38A67D849D28C978FFE54C86A0614367F211DEEE6E668FF`|
|`paper_reproduction/scripts/run_cvs_qknnv42_tta_ablation_20260714.sh`|`2026F46863C27B2F00C53B8E7001FB8F59BB6739120B726630327A2ADD5354B8`|

## 8.完成判据

- 新导出manifest必须为`checkpoint_load_strict=true`、`identity_only_forward=true`、`domain_branch_executed_for_qknn=false`、`adv3b02_gradient_updates=0`；
- 4组policy均为125/125完成，无support/query重叠、无错误日志；
- 所有candidate split hash与严格历史reference逐行一致；
- 最终选择view数最少且三个矩阵均值相对历史完整体均下降不超过3pp的候选；
- 若冻结ADV3B02的5-view候选已经下降超过3pp，则先优化qKNN feature adapter，不得用重新训练ADV3B02绕过要求；
- 角色/配额Oracle未移除前，结论始终保持`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`。

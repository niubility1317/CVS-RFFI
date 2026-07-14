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
|3|dense query graph|主特征与FFT顺序执行时峰值均值`0.829MB`，两路累计分配均值`1.659MB`|随query batch临时分配|不是最大算力，但影响峰值内存和逐样本流式部署|
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

候选保持FFT96、三个`leo_*_weak`场景、old/new角色和类别配额Oracle不变。前端矩阵同时用两种head复核：严格历史dense LP/all-support，以及本地875行筛选出的streaming residual/prototype-only。改变项为：

1. ADV3B02始终严格加载、冻结、identity-only；
2. 不进行60epoch模型适配；
3. 特征适配由qKNN support完成；
4. 比较`none/rx_shift3/rx_cfo3/rx_light5`的1/3/3/5-view；
5. 对每个view策略分别运行dense历史head和prototype-only推荐head。

矩阵为`2 head×4 policy×5 receiver×5 seed×5 K=1000`行。每个candidate row必须与严格历史125行相同receiver/seed/K的support/query split SHA256一致。晋升条件是候选的`old_acc`、`seen_new_acc`和`H_old_new`三个矩阵均值相对严格历史完整体分别下降均不超过3pp；不是只与候选自身5-view比较。

## 6.实现、验证与远端边界

本地变更：

- `code/cvsrffi/identity_only_forward.py`：qKNN identity-only bit-exact前向；
- `code/scripts/train_apply_phase1_iq_preadapter_20260703.py`：新增ADV3B02零训练导出路径；
- `code/export_spaceborne_features.py`：普通严格导出同样跳过未使用domain分支；
- `paper_reproduction/cvs_aligned/cvs_method_runner.py`：显式qKNN support特征适配及无query/无ADV3更新审计；
- `paper_reproduction/scripts/benchmark_qknnv42_tta_policies.py`：增加完整历史profile和严格历史≤3pp门槛；
- `paper_reproduction/scripts/run_cvs_qknnv42_tta_ablation_20260714.sh`：改为冻结ADV3B02、零epoch、完整历史head的500行启动器。

本地验证：Python编译通过；完整相关测试最新为`31 passed`；两个CLI`--help`和Bash`-n`通过；真实strict checkpoint的identity-only等价性为`z_id/logits max_abs_diff=0/0`。新增真实`DualCVSincNetDisentangle`回归测试同时验证完整`return_aux=True`与identity-only的`z_id/TX logits`逐元素一致，并确认轻型路径不调用`dom_backbone`。

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

第二轮复审另发现1项Critical和2项Important，均已闭合：`full_legacy_oracle`现在强制提供历史目录且固定三指标不可由CLI覆盖；feature cache额外强制`checkpoint_load_strict=true`且load audit三类异常计数均为0；identity独立导出目录也在任何计算前拒绝覆盖。最终独立复审结论为通过，未发现剩余Critical或Important；针对性测试为`15 passed`。

head压缩与双head矩阵扩展再次独立复审：首轮指出历史split配对、Oracle transductive标记和Hungarian资源边界三项问题；修复后最终复审通过，当前diff无剩余Critical或Important，复审针对性测试`19 passed`。

待同步文件SHA256：

|文件|SHA256|
|---|---|
|`code/cvsrffi/identity_only_forward.py`|`8E262522BCFDC956A68835BCDD7AF1E33345B3F88CACDC1AE4BAC0D9F3DCB247`|
|`code/export_spaceborne_features.py`|`70941AED6C9FE90F398096162613A1C613A88F57FBFBDECA80C82624A95D04B2`|
|`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`|`DA2092D0A5FECCBD1481EA023F8AB0E9941E38840BDD15C7285F09DA154F1CFC`|
|`paper_reproduction/cvs_aligned/cvs_method_runner.py`|`DBEC74464BC09F1845B12F7BB9C131E314C68EBE2F93850C12310700B0EB7A98`|
|`paper_reproduction/scripts/benchmark_qknnv42_support_compression.py`|`848B9C543663BDA47B4A947D38E84543A84FF9B1C9B2699446FE0FCF661CBE97`|
|`paper_reproduction/scripts/benchmark_qknnv42_tta_policies.py`|`49D13FE2EB9EEDD58D6670DE5504A12D047BCB1431B429FB174580C639549F59`|
|`paper_reproduction/scripts/run_cvs_qknnv42_tta_ablation_20260714.sh`|`55A91CC0A66A08AB1CB650446F0F451A89EED33AA73628C11EBB7059A8A82976`|

## 8.第二大热点压缩：qKNN head

N607仍有活动训练时，本轮只读拉取既有严格历史5-view+adapter60+FFT96特征cache到本地，不改变服务器状态。5个接收机cache共约225MB，SHA256已逐文件核验，SSH进程与TCP22连接随后均归零。基于这些严格历史特征执行`7模式×5 receiver×5 seed×5 K=875`行本地Oracle head矩阵，全部完成；代码同时校验模式间split和候选对严格历史125行split逐行一致。结果artifact位于`local_artifacts/qknnv42_full_history_head_compression_20260714_1941/`。

所有模式均保留old/new角色Oracle与类别配额Hungarian，因此仍为`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`。变化仅限dense LP、support表示和流式prototype residual：

|模式|old/new/H|相对严格历史下降pp|评分MAC/query|评分MAC下降|持久状态|状态下降|dense graph峰值|判定|
|---|---|---|---:|---:|---:|---:|---:|---|
|dense all-support|84.07/93.24/88.23|0/0/0|22.725M|0%|36.62KB|0%|0.829MB|历史基线|
|stream all-support|84.05/92.81/88.02|0.02/0.43/0.21|3.146M|86.16%|36.62KB|0%|0|通过|
|disabled all-support|83.89/92.89/87.96|0.18/0.35/0.27|2.818M|87.60%|36.62KB|0%|0|通过|
|stream diverse-4|83.71/92.85/87.84|0.35/0.39/0.38|1.638M|92.79%|26.90KB|26.53%|0|通过|
|stream diverse-2|83.25/92.76/87.55|0.81/0.48/0.68|1.245M|94.52%|24.37KB|33.45%|0|通过|
|stream medoid|82.58/92.11/86.87|1.49/1.13/1.36|0.983M|95.67%|22.68KB|38.07%|0|通过|
|stream prototype-only|83.59/92.76/87.74|0.47/0.48/0.49|0.655M|97.12%|20.57KB|43.84%|0|通过，推荐|

推荐`stream prototype-only`：它不保存support code、不构建query-query图，只保存每类prototype与qKNN support拟合的中心/scale；相对严格历史三个矩阵均值下降均小于0.5pp，明显低于3pp门槛。它在精度上也优于medoid，同时评分MAC和持久状态更低。若保持5-view identity-only前端，ADV3B02前端与qKNN评分合计由约207.794M降至`44.561M+0.655M=45.216M MAC/sample`，该MAC路径下降约78.24%，FFT96小项未计入；FFT仍随view线性变化，但远小于前端卷积。

必须保留一个重要边界：角色/类别配额Hungarian没有被压缩。每个场景仍需完整old/new query block，`query_used_for_transductive_inference=true`、`decision_batch_state_required=true`；同row下所有模式的assignment下界均为`1.792M cubic work units`，score-slot工作区下界为`115,200B`。这类work unit不能直接等同于MAC，因此97.12%和78.24%均明确只表示神经前端与qKNN评分MAC，不把Hungarian复杂度藏入MAC。prototype-only移除了dense graph，但完整历史Oracle仍不是逐样本星上部署算法。

已将500行冻结前端矩阵扩展为双head复核：同一组导出cache分别运行完整dense Oracle和推荐prototype-only Oracle，总计1000个qKNN row。这样最终结果可同时分离“前端压缩损失”和“联合轻量化损失”。

## 9.完成判据

- 新导出manifest必须为`checkpoint_load_strict=true`、`identity_only_forward=true`、`domain_branch_executed_for_qknn=false`、`adv3b02_gradient_updates=0`；
- 2种head下的4组policy均为125/125完成，总计1000/1000，无support/query重叠、无错误日志；
- 所有candidate split hash与严格历史reference逐行一致；
- 最终选择view数最少且三个矩阵均值相对历史完整体均下降不超过3pp的候选；
- 若冻结ADV3B02的5-view候选已经下降超过3pp，则先优化qKNN feature adapter，不得用重新训练ADV3B02绕过要求；
- 角色/配额Oracle未移除前，结论始终保持`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`。

## 10.冻结ADV3B02后的qKNN侧特征适配搜索

本轮在严格冻结、单视图target cache上完成三组support-only搜索。所有候选均不更新ADV3B02，不读取query标签，且逐行校验与严格历史125行split一致。

|搜索族|候选数/运行数|最佳候选|old/new/H|相对严格历史下降pp|结论|
|---|---:|---|---|---|---|
|基础中心/对角白化/Fisher×FFT权重|16/2000|`support_center+FFT w=0.70`|81.80%/83.45%/82.23%|-2.26/-9.79/-5.99|失败；new和H未过3pp|
|old/new角色分支适配×FFT权重|15/1875|`support_role_center+FFT w=0.70`|81.84%/85.05%/83.00%|-2.22/-8.19/-5.23|当前最佳冻结qKNN替代，但仍失败|
|类均值子空间×FFT权重|12/1500|`support_mean_subspace1+FFT w=0.65`|81.74%/83.41%/82.19%|-2.32/-9.83/-6.04|失败|
|源域教师线性ridge|4 policy/500|最佳H约81.94%|未达到门槛|源域holdout cosine约0.90仍不能恢复target几何|

结论：仅靠support中心化、对角Fisher、role拆分或低秩类均值子空间，能够把old差距压进3pp，但无法恢复adapter60对seen-new几何的贡献。当前最佳`81.84/85.05/83.00`不能晋升为“完整历史≤3pp轻量替代”。问题不在qKNN评分MAC，而在移除60epoch identity内部适配后，冻结`z_id`对两类seen-new的类间结构不足。

## 11.单qKNN性能标签与独立确认

单qKNN路径与完整历史Oracle严格分开：单视图、逐样本argmax、dense LP关闭、无角色Oracle、无类别配额、无decision workspace。seed713101-713105用于adapter/FFT权重/old-bias选择；seed713106-713110用于独立125行确认。

|性能标签|输入与配置|选择集old/new/H|独立确认old/new/H|head MAC|持久状态|判定|
|---|---|---|---|---:|---:|---|
|`STRICT_SINGLE_QKNN_NOFFT_CONFIRMED`|`z_id160`；Fisher；FFT关闭；bias=-0.10|56.23%/56.83%/55.44%|56.47%/58.74%/56.83%|1.761M|22.81KB|严格单qKNN无FFT确认结果|
|`STRICT_SINGLE_QKNN_FFT96_CONFIRMED`|`z_id160+FFT96`；Fisher；w=0.70；bias=-0.08|70.50%/72.83%/71.18%|70.98%/74.69%/72.33%|2.818M|36.62KB|当前最佳可部署单qKNN标签|

`STRICT_SINGLE_QKNN_FFT96_CONFIRMED`相对无FFT确认行提高old 14.51pp、new 15.95pp、H 15.51pp；两行使用同一确认seed网格和同一严格冻结cache，但bias分别由各自选择集确定，因此该差值表示完整配置差异，不应写成纯FFT单因素效应。两行均没有Oracle和batch决策状态，计算量远低于完整历史链路，但都不满足相对88.23%H下降不超过3pp的完整历史门槛。

对应artifact：

- `local_artifacts/qknnv42_single_qknn_fft_holdout_20260714_2200/`；
- `local_artifacts/qknnv42_single_qknn_nofft_holdout_20260714_2200/`；
- `local_artifacts/qknnv42_single_qknn_adapter_sweep_none_20260714_2120/`；
- `local_artifacts/qknnv42_single_qknn_bias_ext_sweep_none_20260714_2140/`。

## 12.下一步qKNN轻量学习适配

为避免重新训练ADV3B02，新增`LayerNorm+160→rank→160`的qKNN后置低秩残差MLP。它只使用1440个源域冻结/adapter60教师特征对进行蒸馏，target row和target query均不参与拟合；计划在N607物理GPU6搜索`rank={32,64,128}`、`alpha={0.25,0.5,1.0}`并训练200epoch。独立实验报告为`automation_reports/CV-SincNet/qknnv42_source_mlp_n607_20260714_2045/report.md`。该路线只有在映射cache回到严格125行并同时满足old/new/H三项3pp门槛后才能晋升。

## 13.源域教师MLP最终结果

N607物理GPU6完成200epoch搜索，选中`rank32/alpha0.25`：源域physical-key holdout cosine=0.918858、MSE=0.001014。适配器为10,560参数、10,240MAC/sample和42,244B状态；ADV3B02梯度更新为0，target row不参与拟合。v2实测显存约355MiB，完整日志无Traceback、NaN或OOM；日志未写逐epoch loss，因此该实验没有可重建的epoch收敛曲线。

严格历史Oracle下，最佳为`support_role_center+FFT0.65`：old=82.19%、new=85.23%、H=83.25%，相对84.07/93.24/88.23为-1.87/-8.01/-4.97pp。它相对无MLP最佳角色分支只提高约0.35/0.17/0.25pp，仍因seen-new和H未通过3pp门槛而失败。该路径含MLP后的head计算为15.220M MAC/场景、状态81.01KB，并继续依赖角色/类别配额Oracle，属于`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`。

独立seed713106-713110单qKNN确认得到71.06/74.00/72.01，head+MLP为5.079M MAC/场景、状态77.01KB、decision workspace=0。相同确认网格的无MLP结果为70.98/74.69/72.33，MLP使H下降0.32pp。最终判定：`NEGATIVE_DIAGNOSTIC_NOT_PROMOTABLE`。计算瓶颈已不在这个后置MLP，而在冻结`z_id`缺少能泛化到target seen-new的几何结构；仅提高源域教师特征相似度不足以恢复历史adapter60。

完整训练、20个Oracle候选、单qKNN确认和逐候选资源表见`automation_reports/CV-SincNet/qknnv42_source_mlp_n607_20260714_2045/report.md`。对应本地artifact为：

- `local_artifacts/qknnv42_source_mlp_n607_20260714_v2/`；
- `local_artifacts/qknnv42_source_mlp_oracle_sweep_none_20260714_2135/`；
- `local_artifacts/qknnv42_source_mlp_role_oracle_sweep_none_20260714_2140/`；
- `local_artifacts/qknnv42_source_mlp_single_sweep_none_20260714_2145/`；
- `local_artifacts/qknnv42_source_mlp_single_holdout_none_20260714_2150/`。

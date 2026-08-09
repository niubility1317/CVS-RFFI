# Phase1单读出local4控制bundle v1设计卡（Revision9）

状态：`LOCAL_VERIFIED_PENDING_REAL_BUILD`；Revision9仅修复部署纵切的可复现性与fail-closed合同，已完成本地验证与独立复核`P0=0／P1=0／ALLOW`，真实F1C＋ManySig构建仍待执行，不产生性能结论。

日期：2026-08-09

## FEASIBILITY（20行内）

1.本纵切只构建Phase3的A臂技术控制，不选择或晋级新Phase1候选。
2.控制固定为C臂中字典序首折`F1C_CP_SFCE12/final_ssdg.pth`，SHA256=`0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8`。
3.状态必须写`TECHNICAL_LOCAL4_CONTROL_BUNDLE`；四类、单折、proxy诊断均不得外推为六类部署或unknown性能。
4.所有已拒绝G、角度B、双读出JS和历史阈值均不进入控制bundle。
5.`z_id`和`tx_logits`只来自该checkpoint的冻结identity分支；class order必须与local4 head逐项一致。
6.该模型模式中的`z_dom=z_id`属于别名，必须拒绝；v1改用固定IQ统计`domain_descriptor`并诚实标注其非学习来源。
7.`domain_descriptor`由I/Q功率不平衡、crest factor、I/Q相关、相位增量圆周离散度和谱平坦度构成，经source-only稳健归一化；当前模型输入已做RMS归一化，故不伪造幅度维。
8.`d_class`为归一化`z_id`到local4类中心的角距，再除以类内稳健半径。
9.`e_unknown`只融合冻结C的距离非一致度和energy非一致度；不读取proxy、held或query进行拟合、调权或选阈值。
10.类几何严格只读有标签L片；U片不得读取TX标签，view／descriptor只依赖label-blind source index；checkpoint未见的V校准片与L／U physical互斥。
11.两个非一致度只转为固定大小的技术stress-tail rank；取`known_consistency=max(rank_distance,rank_energy)`，只在两项都异常时给unknown证据，不声称p-value、conformal覆盖或跨域校准。
12.`e_unknown=1-known_consistency`不是unknown后验；固定`alpha=0.01`只激活`p_local`的技术unknown质量，精确决策仍由C＋1 argmax给出。
13.`q`只称`model_reliability`，由域一致性与注册类熵置信度组成，不宣称真实SNR、链路质量或轨道质量。
14.`p_local`严格为local4＋unknown simplex；N=1按其argmax，平局归unknown，数据不完整或超时才defer。
15.bundle不得含raw IQ、checkpoint、physical ID、样本级feature cache、role、truth或scorer输出。
16.bundle封存checkpoint／config／class order／校准physical集合／公式／成员allowlist／content root。
17.任何missing、extra、nonfinite、别名、类序或SHA漂移均fail-closed。
18.发布前必须通过真实IQ eager↔runtime六字段与决策parity、状态零更新、硬资源上限及CARE N=1规范化恒等测试。
19.proxy只能在bundle root冻结后作source-held诊断，不能反向修改公式、统计、`alpha`或成员。
20.完成后只关闭P1-03／P1-06／L-01的local4技术接口缺口；P1晋级、六类最终bundle和Phase3性能仍保持pending。

## 冻结数学定义

对一条接收IQ `x`，冻结identity runtime输出`z_id(x)`与`l(x)`。令`p_reg=softmax(l)`，energy为`E=-logsumexp(l)`。类中心`mu_c`与类内角距半径`r_c`严格只由有标签L片生成；U片的TX标签不可读取。

每个L physical固定生成且只生成`clean、leo_clear_weak、leo_low_elev_weak、leo_rain_weak`四个view，场景实现、seed与顺序进入receipt；它们仍只计一个physical。令`v_{i,k}=normalize(z_id(x_{i,k}))`，先形成physical方向`v_i=normalize(sum_k v_{i,k})`，再按类形成：

```text
mu_c=normalize(sum_{i in L:y_i=c} v_i)
a_i=max_k acos(clamp(<v_{i,k},mu_c>,-1,1))
r_c=quantile_higher({a_i:i in L,y_i=c},0.95)
d_c(x)=acos(clamp(<normalize(z_id(x)),mu_c>,-1,1))/r_c
s_D(x)=min_c d_c(x)
```

所有方向和聚合以physical key字典序、float64累加；每类至少32个L physical，任一输入方向范数`<=1e-8`、中心和向量和范数`<=1e-8`、`r_c<=1e-6`或非有限均fail-closed。主模型输入view仍遵循checkpoint的固定RMS归一化；类几何view policy不得在构建时改变。

L／U／V physical-ID互斥。canonical physical key固定为`(tx_label,rx_label,day_label,sig_i)`；前三项必须是Unicode NFC后的非空string，`sig_i`必须是非bool、非负int，任何隐式数字／字符串转换均禁止；`eq_label`是同一physical的view lineage，不得制造第二个physical。本bundle只接受checkpoint／dataset receipt共同证明的单一`equalized=1`，出现`both`、多eq或eq漂移立即失败。physical key只在独立partition-audit路径生成集合SHA和互斥证明，绝不进入view RNG、U descriptor的opaque输入、排序或数值。构建时以训练receipt的seed、split mode和冻结source dataset index重建L／U／V，逐项验证三份indices SHA；class geometry只来自L，U只可在完全不读取`y／tx_label`的路径用`opaque_index`和原始IQ进入全局descriptor位置尺度，V只形成技术tail summary且不得进入optimizer、EMA、prototype或训练loss。每个V physical固定生成四view并以最大非一致度形成一个技术stress-tail原子，不能把四view当四个独立样本。

对V校准physical原子的`s_D`和`E`分别形成技术stress-tail rank。V原子取固定四view中的最大非一致度，而runtime输入是一条单view观测；二者不满足exchangeability，因此以下量只用于技术控制，不称p-value或coverage：

```text
r_D(x)=(1+#{s_D(cal)>=s_D(x)})/(n_cal+1)
r_E(x)=(1+#{E(cal)>=E(x)})/(n_cal+1)
known_consistency(x)=max(r_D(x),r_E(x))
e_unknown(x)=1-known_consistency(x)
u(x)=clip((alpha-known_consistency(x))/alpha,0,1), alpha=0.01
p_local(x)=[(1-u(x))*p_reg(x),u(x)]
```

`max(r_D,r_E)`表示“距离与energy同时异常才给强unknown证据”的保守合取诊断，不依赖两个证据独立；`e_unknown`不是unknown概率。实现不得保存逐physical的`s_D／E`数组；只允许保存固定129点单调quantile aggregate、计数和聚合尺度。level与value均为float64：`q_k=1-10^(-k/32),k=0..128`，`Q_k=np.quantile(scores,q_k,method="higher")`；`Q_64`就是唯一q99一致性锚点，不另存第二份q99。对有限query score `s`，重复值安全反演唯一固定为：

```text
floor=max(1e-4,1/(n_cal+1))
if s<=Q[0]: cdf_lt=0
elif s>Q[128]: return floor
else:
    j=searchsorted(Q,s,side="left")
    if Q[j]==s: cdf_lt=q[j-1] if j>0 else 0
    else:  # Q[j-1] < s < Q[j]
        cdf_lt=q[j-1]+(q[j]-q[j-1])*(s-Q[j-1])/(Q[j]-Q[j-1])
return clip(max(floor,(1+n_cal*(1-cdf_lt))/(n_cal+1)),floor,1)
```

`searchsorted`、比较和插值全部在float64 score值域执行；非有限score／Q、`Q[j]<=Q[j-1]`落入非等值插值支路、level／value非单调或端点漂移均fail-closed。由aggregate得到的rank是固定大小聚合状态上的技术近似，manifest必须写`finite_sample_exact_conformal=false`和`source_exchangeable_calibration=false`。若未来要求严格有限样本conformal，须另获样本级状态协议授权，不能暗中扩大bundle。

设`m=max_c p_reg,c`。若完整性与截止时间有效，则local decision精确定义为：

```text
unknown  iff u >= (1-u)*m   （平局归unknown）
registered otherwise，label=argmax_c p_reg,c
defer only for incomplete/deadline/integrity failure
```

等价地，unknown需要`known_consistency<=alpha/(1+m)`；因此校准原子数还必须使最小rank分辨率`1/(n_cal+1)`不大于`alpha/2`，即`n_cal>=199`，否则构建失败。

`domain_descriptor`记为`s(x)`，只读取与identity checkpoint逐字节相同的model-input tensor。runtime外部输入固定为finite contiguous float32 `[B,2,256]`，channel 0为I、channel 1为Q；dataset构建先把原始`[T,2]`转为`[2,T]`，`T>256`时从`start=(T-256)//2`取中心256点，`T<256`时在左侧填`(256-T)//2`个零、余数填右侧。随后按现有`dataset_wisig._rms_normalize_iq`定义计算`scale=sqrt(mean_t(I^2+Q^2)+1e-12)`并除以scale，不做center。预处理operator ID固定为`wisig_center256_rms_iq_v1`，构建时封存`dataset_wisig.py`字节SHA和operator formula SHA；descriptor hook与identity runtime必须接收同一Tensor对象且其CPU contiguous bytes SHA一致。合成LEO校准view则严格复用训练receipt绑定的scenario实现：clean model-input tensor进入`apply_sat_channel_for_scenario`后，其输出同时送descriptor与identity runtime，不再二次RMS归一化。

令该model-input tensor的复序列`a=I+jQ`、`eps=1e-12`，五维统计严格为：

```text
s1=(mean(I^2)-mean(Q^2))/(mean(I^2)+mean(Q^2)+eps)
s2=max(|a|)/(sqrt(mean(|a|^2))+eps)
s3=mean(IQ)/(sqrt(mean(I^2)*mean(Q^2))+eps)
s4=1-|mean(exp(j*angle(a[t]*conj(a[t-1]))))|，仅保留相邻两点幅度均>=0.1*RMS者
s5=exp(mean(log(P+eps)))/(mean(P)+eps)，P=|FFT(Hann(L)*a)|^2
```

当前数据与checkpoint在模型前做逐样本RMS归一化，故`log RMS`几乎为常数；v1明确删除该维，不从预归一化幅度另造与训练runtime不一致的支路。Hann固定为periodic形式`w[t]=0.5-0.5*cos(2*pi*t/L),t=0..L-1`（等价`symmetric=false`）；FFT长度固定`n=L=256`、complex forward DFT、`norm="backward"`，`P=abs(FFT(w*a))^2`。所有descriptor中间量以float64计算，最终`z_dom`写float32；有效相位增量少于16个、`L!=256`或任一统计非有限均fail-closed。L与U的descriptor以完全无标签的累加器给出每维median和`1.4826*MAD`；任一尺度`<=1e-8`即失败，不以epsilon掩盖退化。归一后的`z_dom`定义域非一致度`s_dom=||z_dom||_2/sqrt(5)`；V physical同样取四view最大`s_dom`并形成独立129点quantile aggregate。其技术上尾source一致性`r_dom`只参与：

```text
q(x)=sqrt(clip(r_dom(x),0,1)*clip(1-H(p_reg(x))/log(4),0,1))
```

字段名仍按CARE schema输出为`z_dom`；固定语义只写入bundle manifest：`z_dom_provenance=fixed_iq_statistical_domain_descriptor_v1`、`learned_domain_representation=false`和`q_semantics=model_reliability_not_physical_quality`，禁止在论文中简称为“学习域表征”或“SNR／链路质量”。CARE v3行不得增加旁路provenance字段；每行`bundle_id`必须严格等于该manifest content root，而`bundle_id`本身进入`evidence_hash`，由此唯一绑定逐行语义。

## 数据与身份边界

|集合|用途|允许写入bundle|禁止用途|
|---|---|---|---|
|source registered labeled geometry physical（L）|类中心、类内半径、无标签descriptor位置尺度的一部分|仅类级聚合、descriptor聚合和集合SHA|保存IQ、逐样本feature或physical ID|
|source registered unlabeled physical（U）|只进入无标签descriptor位置／尺度累加器；view与累加器输入只含`opaque_index→SHA256 opaque_hash`和IQ views|仅descriptor聚合和独立集合SHA|读取或推断U的TX标签、将physical key／其hash带入view seed或descriptor排序、进入类中心／半径、保存样本级状态|
|source registered calibration physical（V）|rank sketch与runtime校准|仅129点quantile sketch、order statistic、计数和集合SHA|模型训练、逐样本数组、proxy调参|
|source proxy unknown TX|bundle冻结后的诊断|否|训练、聚合、校准、选择|
|source held／target／query|无|否|任何fit、update、threshold、selection|

构建器必须验证L／U／V的canonical physical集合两两互斥，L与V的TX集合严格等于local4 registry，并验证三者与proxy／held／target集合不交；还须重建并匹配`labeled_indices_sha256／unlabeled_indices_sha256／source_validation_indices_sha256／split_manifest_sha256`，绑定dataset SHA、seed、RX／day、view/scenario实现SHA，且静态＋动态receipt证明V未进入optimizer。partition validator可读取dataset结构以生成L／U／V canonical token和集合SHA；进入U descriptor累加器的接口只允许`(opaque_hash,IQ_views)`，其中`opaque_hash=SHA256(b"SCB1-OPAQUE-SAMPLE\\0"+str(opaque_index))`，不得携带`y／tx_label／class_id`、physical key或其hash，并以label置换／抹除下view bytes和最终stats字节不变的负测证明U标签路径不可达。若任一证据缺失，本纵切停止，不能退回训练重放冒充正式校准。

真实构建输入固定为F1C checkpoint、`phase1_training_completion_receipt.json`、`phase1_terminal_status.json`、`phase1_cp_sfce_terminal_receipt.json`、ManySig数据集及其预注册SHA。三类receipt的schema与只读JSON pointer固定如下；其他字段可以存在但不得影响输出，完整输入文件SHA仍进入source-partition receipt：

|输入|schema|唯一允许影响构建的字段|
|---|---|---|
|training completion|`cvs.phase1.training_completion_receipt.v1`|`run_id、phase1_training_complete、terminal_status、exit_code、technical_only、formal_performance_claim、selected_checkpoint_sha256、source_split_receipt.{schema,seed,split_mode,source_days,target_days,source_receivers,target_receivers,source_target_receiver_overlap_count,labeled_indices_sha256,unlabeled_indices_sha256,source_validation_indices_sha256,split_manifest_sha256,labeled_size,unlabeled_size,source_validation_size,source_pool_size,requested_labeled_ratio,requested_unlabeled_ratio,requested_source_val_ratio,requested_rho_label,realized_rho_label,realized_rho_tolerance,realized_rho_within_tolerance,realized_source_val_fraction,realized_source_val_tolerance,realized_source_val_within_tolerance}`|
|terminal status|`phase1_terminal_status_v2`|`run_id、candidate_id、status、exit_code、selection_source、selected_checkpoint、selected_checkpoint_exists、selected_checkpoint_sha256、technical_only、promotion_ready、performance_result_available`；其`source_split_receipt`只作与training completion上述精确projection的等值复核；`satellite_protocol.{schema,train_scenarios,eval_scenarios,train_families,eval_families,scenario_config_sha256,train_config_sha256,eval_config_sha256,channel_implementation,registry_version,registry_sha256,disjoint,require_disjoint,evaluation_claim}`|
|CP terminal|`cvs.phase1.cp_sfce_receipt.v2`（`phase1_cp_sfce_terminal_receipt.json`自身的顶层receipt，不存在`cp_sfce_receipt`嵌套层）|`enabled、checkpoint_role、source_train_tx、source_known_validation_tx、source_proxy_unknown_tx、local_tx_class_order、checkpoint_train_tx_class_order、local_to_head_class_ids、checkpoint_head_class_count、live_head_class_count、class_order_binding_sha256、class_order_contract、selected_checkpoint、selected_checkpoint_sha256、terminal_status、terminal_exit_code、technical_only、promotion_ready、performance_result_available`|

ManySig dataset manifest root固定为实际PKL字节SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；receipt中的空dataset SHA不能覆盖该外部预注册root。三份本地小receipt的冻结字节SHA256为：`training_completion=c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c`、`terminal_status=0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2`、`cp_terminal=5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0`。构建器必须同时确认三者组合后唯一指向`phase1_cp_sfce12_20260809_v2`的`F1C_CP_SFCE12`、同一checkpoint SHA、`NON_PROMOTABLE_P0_DISABLED`／exit 8；CP顶层receipt还必须为`enabled=false`、`checkpoint_role=training_final_only`、类序合同精确匹配，且三者的promotion／performance声明均不得为true。原训练收据的`technical_only=false`和`phase1_training_complete=false`是该formal C臂在P0 final gate被禁用后的真实原值，不得篡改成bundle结论；bundle必须在自己的manifest中另行标记`TECHNICAL_LOCAL4_CONTROL_BUNDLE`。构建器按receipt中的seed、split mode、ratio、RX／day和equalized设置机械重建索引，并逐项比对上述四个split hash；不要求把原始indices复制进bundle。checkpoint或数据只存在N607不阻塞本地纯函数实现，但在SHA匹配的真实构建与smoke完成前，状态不得超过`LOCAL_VERIFIED_PENDING_REAL_BUILD`。

`resolved_config_sha256`不读取模糊的“架构相关键”，而是对以下唯一projection哈希：上述receipt允许projection、三份receipt字节SHA、checkpoint字节SHA、dataset root、下表的resolved model config、strict-load state tensor schema SHA（对字典序`[name,dtype,shape]`列表哈希）、`input_len=256／equalized=1`、local4 class order、预处理operator ID／代码SHA，场景配置／代码SHA／seed规则、公式ID、`alpha`、129个level和资源门。resolved model config正好是`build_baseline_model`真正读取的全部键；除三个派生值外，每个键从`/checkpoint/args/<key>`读取，缺失时严格使用表内默认值：

|key|类型|缺省／派生规则|
|---|---|---|
|`num_classes`|int|无缺省，必须存在且等于4，head行数亦为4|
|`num_domains`|int|不读args；只由strict loader按domain-head state首维推导且`>=1`|
|`input_len`|int|不读args；固定256|
|`model_size`|str|`M`|
|`dataset`|str|`wisig`，解析后仍必须为`wisig`|
|`sample_rate_hz`|float64|缺失或`<=0`时`25000000.0`|
|`id_feature_key`|str|`feat_joint`|
|`dom_feature_key`|str|`feat_imp`|
|`model_variant`|str|`lite_d`|
|`branch_ablation`|str|`no_dac`|
|`use_mixstyle`|bool|`true`|
|`mixstyle_p`|float64|`0.18`|
|`mixstyle_alpha`|float64|`0.10`|
|`mixstyle_eps`|float64|`1e-6`|
|`mixstyle_layers`|str|`time_down,t1`|
|`mixstyle_use_domain_label`|bool|`true`|
|`mixstyle_mix`|str|`same_tx_crossdomain`|
|`mixstyle_strength`|float64|`0.70`|
|`mixstyle_fallback`|str|`skip`|
|`domain_branch_ablation`|str|`no_stats`|
|`domain_enhancer`|str|`rcn_stats`|
|`domain_enhancer_strength`|float64|`0.35`|
|`id_time_stability_mode`|str|`off`|
|`id_freq_stability_mode`|str|`off`|
|`domain_time_stability_mode`|str|`off`|
|`domain_freq_stability_mode`|str|`off`|
|`time_stability_channels`|int|`8`|
|`freq_stability_channels`|int|`4`|
|`fast_infer_when_no_aux`|bool|`true`|
|`arch_family`|str|`cvsincnet`|
|`representation_mode`|str|`dual`|

上表值必须与`build_exact_ssdg_model_from_checkpoint→merge_checkpoint_args→_apply_model_cli_args`得到的最终namespace逐键相等；任一表内键类型不符或strict state load不闭合即失败。checkpoint中的表外training args可以存在，但不得进入config projection或改变runtime；完整checkpoint字节SHA仍绑定它们。projection同时封存`code/cvsrffi/checkpoint_loading.py`、`code/post_stage_common.py`、`code/SSDG/train_ssdg.py`和`code/model_dual_cvsincnet.py`的字节SHA，防止表外键在实现漂移后悄然进入架构。hash专用canonicalizer递归只允许string-key mapping、list、Unicode NFC string、bool、非bool int和有限binary64；float先转为`"f64:"+float.hex(value)`字符串，所有正负零统一为`f64:0x0.0p+0`。随后固定`json.dumps(ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)`的UTF-8字节并取SHA256；list保序，mapping以Unicode code point键序列化，禁止`null`、NaN、Inf和其他类型。

场景顺序唯一为`[leo_clear_weak,leo_low_elev_weak,leo_rain_weak]`，`fs_hz=25000000.0`、`fc_hz=2462000000.0`；三个config SHA必须分别等于terminal receipt中的`c046cdfbb48d8a0a6b011418374939e86f2a4ff450ab40a3f3ed4a333a53f159`、`323aa6613292049605e04eb6be63c9754acb0655176a52c5d11501e2a1ae7e87`、`66e72208dc21c4dea80130435eec50afd03d24fc3f014d0f8a73d720a14ead2b`，registry SHA必须等于`d38c3bcc85699c97c9bca53a84a5268b51db140a6767c28fae06cf65cc5db215`。构建器必须用当前`training_controls.satellite_protocol_manifest`重算完整三场景projection并同时匹配冻结值、terminal receipt与registry SHA。projection还必须封存`code/training_controls.py`、`code/cvsrffi/eval.py`、`code/sat_channel.py`和`code/cvsrffi/tensors.py`的字节SHA，任一变化都必须产生新root。clean view不用RNG；每个source sample的每个LEO view以batch1、独立CPU `torch.Generator`执行，且唯一seed定义为：

```text
seed_material = b"SCB1-SOURCE-VIEW-SEED\0" + str(split_seed).encode("ascii") + b"\0" +
                str(opaque_source_dataset_index).encode("ascii") + b"\0" +
                scenario.encode("ascii")
seed = int.from_bytes(sha256(seed_material).digest()[0:8],"big",signed=False) & 0x7fffffffffffffff
```

其中`opaque_source_dataset_index`是source_view建立后冻结的非bool、非负global/local dataset index；它不是TX／class label、physical key或其hash，且L／U／V三集合必须各自去重并两两不交。`split_seed`必须是非bool int且精确等于冻结receipt的7281105；`scenario`必须是上述固定列表中的ASCII string，不得用Python `ascii()`或`repr()`。每个场景都重建generator并`manual_seed(seed)`，不得依赖遍历次序、batch尺寸或全局RNG。所有view先在CPU生成，再按需迁移到runtime device；seed规则ID`SCB1-SOURCE-VIEW-SEED-v2`、场景代码SHA和实际每场景config projection均进入`resolved_config_sha256`。

CARE桥接字段分为两类：bundle固定输出`bundle_id、class_handles=[20-15,20-19,6-15,8-20]、z_id、z_dom、q、d_class、e_unknown、p_local`及local decision；采集系统在推理前提供truth-free context：`linkage_mode、emission_event_id或proxy_group_id、satellite_reception_id、node_id、base_manifest_id、correlation_group_id、delay_ms、deadline_ms、sealed_at_ms`以及verified physical模式所需binding字段。runtime不得从IQ、分类结果、role或truth推断event／reception／node；`bundle_id`必须等于已外部锚定的content root，`p_local`第5位固定为unknown，只有`registered`可写local4中的`local_label`，`unknown／defer`必须为null。完整、无冲突且可验证的context才可seal并输出本地行；缺失、冲突、hash／字段完整性失败均是不可封存的fail-closed异常，由上层采集／传输层映射defer，bundle不得伪造一行；仅已完整绑定的context在`delay_ms>deadline_ms`时输出`defer／SCB_CONTEXT_DEFER`。

## 不可变bundle成员

```text
runtime/local_evidence.ts
state/class_geometry.npz
state/domain_descriptor_stats.npz
state/rank_tail_summary.npz
locks/checkpoint_binding.json
locks/class_binding.json
locks/source_partition_receipt.json
locks/runtime_parity_receipt.json
locks/resource_receipt.json
```

上述9个是payload members；`manifest.json`是第10个顶层root envelope，不得把自己放入`members`。manifest的`members`必须恰好列出上述9个相对路径及各自字节SHA256／字节数，并封存checkpoint SHA、`resolved_config_sha256`、dataset／preprocessing／scenario registry SHA、local4 class order、公式ID、`alpha=0.01`、校准集合SHA以及以下字段：`raw_iq=false`、`source_checkpoint_container=false`、`runtime_embeds_frozen_weights=true`、`sample_feature_cache=false`、`physical_ids=false`、`role_or_truth=false`、`performance_promoted=false`、`finite_sample_exact_conformal=false`、`source_exchangeable_calibration=false`。任一payload member都不得含最终`content_root`或manifest hash，避免间接循环。

content root唯一定义为`SHA256(canonical_json(manifest_without_content_root))`；计算时manifest已含上述全部语义字段和9个member descriptor，但键`content_root`完全不存在。随后只把该64位hex写入最终`manifest.json` 1次。loader必须接收非空的外部`expected_content_root`，验证输出根恰好只含9个member加manifest、逐项复算字节数／SHA、删除manifest中的`content_root`后按上述canonicalizer复算root，并要求复算值同时等于manifest内值和external expected root；任一不符立即失败。CARE行中的`bundle_id`只能取已通过该外部锚定加载的root。构建输出根必须预先不存在，任何成员覆盖均拒绝。

硬资源门冻结为：bundle总字节`<=32MiB`、单条canonical local evidence JSON`<=64KiB`、CPU batch1峰值RSS增量`<=512MiB`、可用CUDA时batch1峰值VRAM`<=256MiB`；在N607冻结Python／Torch环境以`torch.set_num_threads(1)`、20次warm-up＋100次计时测得的CPU batch1 p99必须`<=250ms`。batch1输入固定为finite contiguous float32 `[1,2,256]`，由receipt固定seed生成且输入SHA封存。资源receipt必须在真正新Python子进程中产生：模块导入完成、任何TorchScript／三份state payload读取前采CPU RSS基线；随后从payload bytes重建state、独立load CPU runtime并用与部署相同的完整local-evidence函数计时。峰值范围包含load、20次warm-up和100次推理，并在load后及每次推理后以同一进程RSS采样；时延采用`perf_counter_ns`，p99固定为对100个值执行`np.quantile(q=.99,method="higher")`，即排序后索引`ceil((N-1)*.99)`。CUDA可用时同一新进程另行load CUDA runtime，在load前无参`empty_cache()`与`reset_peak_memory_stats`，每次计时前后`synchronize`，峰值取`max_memory_allocated`，CUDA时延只记录不设门。state digest只在load／测量边界前后封存，不在每条query重算；runtime input必须自动迁移到其唯一parameter／buffer device。`resource_receipt`在manifest前生成，只记录单条evidence字节、CPU／CUDA内存、warm-up／计时参数、state边界digest与时延统计；严禁写入payload总字节、自身或其他member字节、final root、manifest hash或预测manifest字节。manifest写完后由builder与loader独立对实际10文件求和并执行32MiB门，不回写payload，从而无size／hash循环。上述只是该代理硬件上的技术可运行门，不是星上处理器时延声明。

## 追溯与验收

|ID|设计要求|实现证据|验收|
|---|---|---|---|
|SCB-01|字典序F1C local4控制及所有G排除|CLI常量、checkpoint／class binding负测|精确SHA与路径|
|SCB-02|geometry／calibration按canonical physical key互斥且无proxy／held／target|index重建、partition receipt与view跨片负测|不交集合、indices／集合SHA|
|SCB-03|诚实五维statistical `z_dom`与model reliability `q`|RMS退化维删除、provenance字段、alias／命名负测|不得声明学习域、幅度质量或物理质量|
|SCB-04|距离＋energy连续诊断及固定stress-tail rank映射|纯函数、手算边界、有限n、重复值、proxy零影响测试|非p-value语义、决策不等式与`alpha`不可漂移|
|SCB-05|严格bundle allowlist与content root|build/load/tamper测试|missing／extra／hash漂移拒绝|
|SCB-06|真实IQ六字段／决策parity和零状态更新|real-checkpoint no-query smoke|逐字段容差、前后state SHA一致|
|SCB-07|CARE N=1规范化恒等|固定local4 handles＋truth-free context，seal→validate→fuse测试|validated `p_local`、决策、label、reason、evidence hash一致|
|SCB-08|资源／非覆盖／声明边界|resource receipt、existing-root负测、technical terminal|四项固定硬门＋可用CUDA时一项附加门；无性能／六类／协同／注册声明|
|SCB-09|Revision9 label-blind view、fresh资源与context合同|opaque-index seed／streaming descriptor／fresh subprocess／CUDA device／context负测|标签置换不改view／stats；不完整context不出行；资源与state边界完整闭合|

### Revision9实现追溯

|ID|来源段落|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|SCB-R9-01|数据与身份边界、场景seed|L／U／V view RNG和U descriptor opaque输入只依赖冻结source dataset index，不含label或physical key；逐行释放IQ，仅紧凑累计全量N×5 float64以精确求median／1.4826MAD|`phase1_single_control_bundle_v1.py`、`test_phase1_single_control_bundle_v1.py`|verified|label置换／抹除下view bytes和descriptor stats不变；>257行全量reference等值；重复／交集index拒绝|物理key仅保留在partition-audit局部变量；不使用sampling／sketch近似|
|SCB-R9-02|资源门|fresh Python子进程在payload读取前采基线；CPU／CUDA独立load并走完整local-evidence路径|`phase1_single_control_bundle_v1.py`、测试|verified|fresh worker实测、无`psutil`fallback、实际CUDA parameter runtime|state digest只在测量边界检查|
|SCB-R9-03|runtime／CARE桥接|runtime input跟随唯一parameter／buffer device；完整context才可seal；仅已绑定超时生成`SCB_CONTEXT_DEFER`|core、测试|verified|CUDA参数runtime、缺context拒绝、有效超时defer|上层负责把不可封存错误映射为传输defer|
|SCB-R9-04|场景与输出根|live三scenario／registry projection闭合；既有output或staging在任何昂贵读取前拒绝|core、测试|verified|真实F1C receipt projection＋模拟live drift；missing-input前existing-root负测|不读取proxy／held／query|
|SCB-R9-05|真实输入纵切|用真实F1C checkpoint和ManySig完成build、resource、full six-field parity smoke|CLI、core|pending|尚未执行；必须在本地可用的冻结输入上运行|当前状态仍为`LOCAL_VERIFIED_PENDING_REAL_BUILD`，不构成部署或性能声明|

Revision9首轮实际diff复核确认CUDA、label-blind、早碰撞、live场景、fresh subprocess资源与context等工程闭合，但发现未经科学冻结批准的257槽descriptor priority sketch会改变`z_dom→r_dom／q→C+1决策`，因此返回`P0=1、P1=0、REVISE`。最终实现删除sketch、容量常量和配置字段，保持IQ逐行流式，并以紧凑`array('d')`保存全量N×5 float64 descriptor后精确计算median与`1.4826MAD`。300行测试及独立1000行reference重放均严格相等；冻结字节最终复审为`P0=0、P1=0、ALLOW`。`ssr-gpu`下`py_compile`、16项focused tests、CLI fixture build＋外部root verify和`git diff --check`均通过；这些仅是本地技术证据，不替代SCB-R9-05真实输入构建。

## 文献定位

本设计只吸收三条可复核原则：稳定闭集分类器是开放集基线的重要前提；单前向不确定性应保持距离感知；后训练feature density可在不使用OOD微调的条件下提供连续证据。它们只说明为何保留C身份头并输出连续距离证据，不证明本数据上的unknown性能。最坏组训练、SNGP／DDU或新Phase1候选另立设计卡，不塞入本控制bundle。

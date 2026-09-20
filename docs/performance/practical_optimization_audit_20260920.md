# CVS Practical LEO执行优化与重复计算审查

日期：2026-09-20。基线：064aae4097b306cf65a162b0b4836d279773f6ea。状态：本地代码优化和验证完成；未发布N607，未修改/重启活动实验。固定评估缓存默认关闭，需新运行显式指定目录。不是新的性能实验结果。

## 覆盖范围与证据边界

对当前Phase1训练入口、L/U采样、普通增强→星地增强顺序、Practical适配器与全部Practical模块、双骨干模型、DAOT/RC4、损失与梯度诊断、源域验证、周期测试、checkpoint和日志路径做结构扫描及逐项调用/消费检查。完整文件清单和静态调用计数见optimization_scope_20260920.json。静态调用数量不等于运行次数。没有宣称扫描CVS仓库全部历史方法或所有未启用分支。

使用原有完整日志耗时审计：full_noeq E1–142、ZF E1–129、MMSE E1–152、residual E1–200、原LEO E1–200。结构化记录全部解析；活动日志范围截至该快照。未重新访问目标truth用于调参，以下判断依据代码和耗时。算子级N607训练profiling、优化后正式epoch吞吐仍未测量。

## 关键纠正：重源域验证里存在未消费的星地前向

四组resolved_config均为direct_metric_multiview_separate=false、source_val_heavy_eval_interval=1、eval_max_batches=0。原_evaluate_source_val_tail_geometry每批clean前向后仍生成一个按batch轮换的practical场景并进行sat前向；最终却因multiview=false只对clean特征调用direct_metric_acceptance_loss。sat_features只贡献multiview_sample_count日志，不参与几何结果。

这是明确的可删除工作，不是泛泛地把“验证”称为无用。当前V=27000，旧路径每轮额外27000次星地样本生成及对应前向；200轮约540万次，另有最终检查调用。**不是V每轮三个场景各27000条**。修复按消费开关生成sat_features；多视图开启时仍完整执行。multiview_sample_count现在为0，并记录satellite_geometry_requested=false；配置场景列表保留。几何结果、真实CVS模型state_dict和全局RNG轨迹已在有界合成数据上对照冻结旧函数验证一致。

旧重验证总耗时通常80–162秒/epoch，还包括clean前向和几何计算，不能全部当作节省量。具体秒数须优化后N607验证。

## 搜索发现与处理清单

|编号|发现与代码位置|实际状态/判断|本次处理|
|---|---|---|---|
|01|SSDG/train_ssdg.py:_evaluate_source_val_tail_geometry生成未消费sat_features|四组当前multiview=false，明确无效|已按消费开关跳过；保留多视图路径|
|02|practical_adapter.py固定评估IQ反复生成|目标各checkpoint相同；源验证输入/seed须匹配|新增严格键的opt-in磁盘IQ缓存|
|03|practical_adapter.py:return_meta=False仍构造state/SNR/CFO/角度等GPU张量|返回值被丢弃，_last_meta仍需维护|已提前返回，保留原诊断records|
|04|leo_practical/batch.py每条样本receiver_for_session重复初始化|Receiver为frozen dataclass，同cfg/seed/session恒定|同batch缓存不可变硬件，ChannelStream仍每样本独立|
|05|train_ssdg.py梯度finite+总范数+子模块范数多次扫描与item同步|a1_runtime_fast=false，当前路径存在|报告现有fast路径可用；未修改活动配置|
|06|fast路径裁剪前后重复capture，即使未裁剪|当前max_grad_norm=5且fast=false，因此不是本组已获得收益|无裁剪时复用snapshot；裁剪后仍重取|
|07|RC4弱视图teacher执行全双骨干前向|当前fast=false；identity-only helper已存在|待独立GPU等价/吞吐验证，不直接改配置|
|08|目标predictor完整model前向，最终只取tx_logits|domain支路可能属于未使用计算|列为下一项；需验证完整模型各配置eval等价，未直接替换|
|09|普通source验证与几何验证重新读取同一V并做clean前向|同轮存在重复，但输出/状态消费不同|可汇集一次eval输出后复用；当前未改接口|
|10|训练末尾再次调用source几何验证|同epoch重复可能存在；通用流程要考虑状态/回滚变化|未按epoch数字盲目复用，需绑定模型状态|
|11|zid_leakage_probe_required=false仍执行最终probe|required控制是否阻断，不等于不需要报告|非纯无用，保留；建议未来单独设诊断执行开关|
|12|每epoch构造checkpoint payload/state_dict/RNG，即使非保存点|部分为后续判断/最终导出/恢复服务|不能整体删；未来延迟序列化须检查消费者|
|13|dataset_wisig.py NumPy→Torch兼容回退|N607 probe: from_numpy失败，as_tensor(arr.copy())成功|排除“每样本反复异常”的错误诊断，不盲换ABI桥接|
|14|num_workers=0，主线程读取数据及生成信道|目前增加loader worker不会自动搬走GPU之后的信道调用|异步生产需要重构依赖，未仅改worker数|
|15|L_s在一个U长度epoch中反复遍历|222步由56700个U、batch256确定，L重启约4.5次|这是预算/标签利用设计，不当作重复样本错误删除|
|16|E80前仍生成拼接sat并执行fused前向|虽然sat CE尚未启用，但共享BN等状态受其影响|禁止称为无损可删；需要科学消融而非执行优化|
|17|clean_duplicate仍参与拼接|即使输入相同，训练态dropout/批统计/梯度语义需保留|不改成单份前向|
|18|同为high的拼接/DAOT视图似乎重复|输入变换和seed不同；不能按场景名合并|保留独立视图身份|
|19|DAOT教师跨step输出似乎可缓存|EMA持续更新，静态logit缓存会过期|禁止跨更新缓存；只复用相同step已有输出|
|20|lambda_tx_proto/rx_proto/mask等为0，use_*为true|桥接代码将它们列为未接线审计项，非零会拒绝|不能凭开关推断存在耗时损失；已排除|
|21|Fishr/orth/开放集loss看起来复杂|当前lambda_fishr=.04、lambda_orth=.05等非零且有消费|属于训练方法，不能为加速静默删除|
|22|DAOT的两个教师视图与clean教师重复问题|identity_sequential已启用；U clean教师已复用|已有优化，不重复记功|
|23|clean+sat与MUSE student多次前向疑似重复|真实fused single forward已启用|保留，不按旧日志横幅推断“两次前向”|
|24|周期目标测试看不见成绩|主循环只读record_count，详细分数在独立score/scorer.log|是日志分流，不是没有测试；保留truth-last边界|
|25|CPU逐记录信道和tolist中转|真实主开销候选，已知兼容路径|缓存/硬件复用先落地；异步流水线、内核向量化另需profiling|
|26|超大每batch诊断映射与CPU标量读回|存在多类统计和同步；不能把全部字段都判无用|按是否控制训练逐项拆分后才可延迟汇总，未盲降日志频率|

## 已实现缓存的准确含义

新cvsrffi/practical_view_cache.py只接受target_fixed_v3/source_validation_fixed_v3。所有source_dynamic_E*_L/U始终bypass，不缓存训练视图、teacher logits、梯度或模型特征。先按原实现抽seed，再查缓存，因此source generator前进次数不变。

缓存键包含实际float64输入字节（前置增强后的实际输入）、完整cfg、场景、ordered sample/session IDs、namespace、实际seed、receiver_seed、NumPy版本及Practical实现文件版本。不同batch划分可造成额外miss，但不会命中错误的其他输入。cfg含采样率、载频、route、均衡开关/方法、正则化和增益限制，四配置三场景隔离；不通过truth生成键。缓存输出包含float32 IQ、states及完整JSON元数据，以uint8 UTF-8存储，不使用pickle。原子写入；损坏或不可写时重新计算，不让缓存替代正确性。

这是按batch的按需缓存，不是完整离线视图库，也不是CPU异步训练生产线。首次仍要完整计算，并有写盘成本；第二次及之后可复用。同一缓存目录可以跨checkpoint使用，不跨不同输入/配置误用。缓存不会自动删除，也不设置自动容量清理；需要显式选择足够空间的专用目录。训练每轮多样性保持原规则。

启用方式（仅新运行/独立评估使用）：训练参数`--practical_eval_cache_dir /专用缓存路径`；独立predictor覆盖参数`--practical-eval-cache-dir /专用缓存路径`。默认空字符串关闭。周期predictor从checkpoint args继承该路径。N607现有release未同步，因此给旧release传新参数不会使优化生效，必须走新release验证发布。

## 验证与性能边界

25项本地聚焦测试通过：12个四配置×三场景精确IQ/元数据命中；实际输入、ID、配置和source seed失效；动态训练bypass；命中与不命中generator推进一致；损坏bytes/null/missing-fields缓存重算；真实CVS模型source几何旧/新结果、state_dict与RNG一致；多视图开启的source最后小batch、实际checkpoint周期predictor和独立scorer链路。2项审查P2健壮性建议已修复并回归，独立审查无P0/P1。残留AMP API弃用提示属于旧接口警告，不是本次功能失败。

基准为本机CPU、合成B16×2×256、每case重复3次、Torch2.10.0；覆盖12case。未使用数据集或checkpoint，不能替代N607 Torch2.1环境的全流程验证。下表为重复读取固定评估视图的中位数，单位毫秒。

|route/均衡|场景|重新生成|缓存命中|局部倍数|
|---|---|---:|---:|---:|
|full/noeq|practical_high|71.01|5.36|13.2|
|full/noeq|practical_mid|75.77|4.58|16.5|
|full/noeq|practical_low_urban|72.61|3.04|23.9|
|full/zf|practical_high|89.66|4.17|21.5|
|full/zf|practical_mid|92.03|4.53|20.3|
|full/zf|practical_low_urban|86.67|2.76|31.4|
|full/mmse|practical_high|89.72|3.05|29.4|
|full/mmse|practical_mid|77.35|4.53|17.1|
|full/mmse|practical_low_urban|82.02|4.07|20.1|
|residual/noeq|practical_high|40.96|4.01|10.2|
|residual/noeq|practical_mid|42.90|3.50|12.3|
|residual/noeq|practical_low_urban|39.32|3.69|10.6|

局部范围10.2–31.4倍，仅表示CPU视图获取，不是整轮/整实验加速。12批共192条视图的缓存实测1199923bytes；包含metadata，因此不能把“纯IQ约3.85GiB”当成全量缓存容量。文件系统、batch大小和元数据分布都会影响占用。没有据此给出固定完工时间承诺。

## 下一阶段优先级

1. 在独立新release做N607保真检查及一段固定源域训练/验证的分段profiling，记录GPU等待、数据搬运、CPU信道、teacher/student和反向耗时。只做合成benchmark不够支撑整轮加速结论。
2. 评估只消费tx_logits/z_id的只读入口使用identity-only，合并同模型同轮clean source eval输出；先核对所有消费键及状态。
3. 把视图计划显式化，按原RNG计划将CPU信道生产与GPU训练重叠。L先常规增强再Practical的顺序不能改变，不能把所有信道挪到原始IQ之前。
4. 对信道逐记录循环分块并行/向量化，限制每进程CPU线程数，保留各record独立stream。正式科学改动（困难场景权重、DAOT视图数量、验证频率、训练步预算）另立对照，不混入执行优化。

本次交付是明确无用路径修复、固定评估缓存、局部冗余消除与完整审查清单；CPU/GPU异步生产线、信道GPU内核、全量离线视图库尚未实现。现有运行、数据、checkpoint和测试产物均未改写。

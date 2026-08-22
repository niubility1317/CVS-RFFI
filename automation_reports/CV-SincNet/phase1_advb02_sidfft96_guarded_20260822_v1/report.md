# Phase1 ADV3B02 SID-FFT96受控修复最小验证

## 当前状态

- run ID：`phase1_advb02_sidfft96_guarded_20260822_v1`
- 终态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- N607训练终态：`ARTIFACTS_COMPLETE`，退出码0，200/200个epoch完成。
- 目的：验证首轮坍缩是否由SID适配器承受冻结Core90辅助损失并形成无界残差所致。
- 核心结论：本轮消除了首轮随机级坍缩，并在三种LEO弱场景取得一致的小幅提升；但预登记的10%残差上界没有进入实际模型，科学增益也未达到晋级门槛。该结果只能作为“损失隔离有效、受控上界实现缺失”的诊断证据，不得晋级或替换CORE90。

## 结论摘要

1. S3G最终clean准确率为90.0093%，相对S0下降0.1309pp；三场景LEO均值为76.6587%，提升0.1899pp；三场景Strict UDU均值为70.9733%，提升0.4106pp。
2. clean保护门槛通过，但LEO均值、Strict UDU和LEO floor增益均低于预登记门槛；`rescued>harmed`及匹配的RX/TX probe条件因缺少相应同样本证据而不可判定。
3. 受控残差配置存在直接实现偏差：`post_stage_common.build_baseline_model()`没有把`sid_max_residual_ratio`传给`build_dual_model()`。launcher与checkpoint记录值为0.1，但运行时`model.sid_fft96.max_residual_ratio=0.0`。
4. 非SID的195个checkpoint张量相对ADV3B02 CORE90最大绝对漂移为0；S3G只学习了41,280个SID参数。坍缩消失主要归因于冻结成熟基座、清除SID路径上的域对抗及其他open损失，并保留clean/satellite TX CE和轻量身份锚定，不能归因于未生效的残差上界。

## 首轮故障定位

- S1/S2/S3均完成200个epoch和独立clean/三种LEO_WEAK评测，进程、数据协议和评测链路无技术异常。
- 三行非SID参数相对基线checkpoint的最大绝对差均为0，排除成熟基座意外更新。
- `train_sid_delta_norm`从E1约0.04增长到E200的21,261.57、9,982.57和16,122.02；raw/SID预测一致率降到0.40%、4.05%和0.38%。
- 加权域对抗损失与SID残差范数的相关系数分别为0.99994、0.99995和0.99995；验证精度跌破50%的epoch分别为45、90和88。
- clean最终准确率仅16.66%、18.40%和16.65%，接近六分类随机水平。首轮矩阵判定为`SCIENTIFIC_FAILURE_NO_PROMOTION`。

## 修复候选与矩阵

|row|候选|作用|
|---|---|---|
|S0|`S0_FROZEN_CORE90`|同checkpoint、同协议冻结基线回读|
|S3G|`S3G_SIDFFT96_GUARDED`|仅训练SID投影，诊断首轮坍缩根因|

预登记的S3G包含三项约束：逐样本残差范数不超过原始身份嵌入的10%；SID梯度只来自clean TX CE、Core90既定satellite TX CE和轻量身份锚定；checkpoint只按source-only `V_select`的`source_val_sat_hmean`选择。实际运行仅后两项生效，10%残差上界因参数传递缺失而未生效。

## 协议与同row条件

- Phase1 source-only：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- `R_s∩R_t=∅`；source split receipt记录接收机重叠数为0，发布前真实checkpoint smoke记录`query_input_count=0`、`target_input_count=0`。
- 基座：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- seed：`392002`；S0与S3G使用相同test split、204,000个clean样本、相同LEO场景seed及每场景204,000个样本。
- 必评场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 基座checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- S3G最终checkpoint epoch：178；独立评估checkpoint SHA-256：`c47573cc9c85bad4615e230a73d41a35ce18041f99f0c0bd5fffbcb22027027c`。
- 独立评估加载缺失键与意外键均为0。

## 最终同row结果

以下结果以`independent_final_eval/final_eval.json`为正式读数，所有差值单位为百分点。

|指标|S0 CORE90|S3G|S3G-S0|预登记要求|判定|
|---|---:|---:|---:|---:|---|
|clean overall|90.1402|90.0093|-0.1309|下降不超过0.3pp|PASS|
|clean Strict UDU|86.0900|86.3300|+0.2400|诊断项|小幅改善|
|LEO overall均值|76.4688|76.6587|+0.1899|至少+1.0pp|FAIL|
|LEO overall floor|75.2912|75.4505|+0.1593|至少+0.5pp|FAIL|
|LEO Strict UDU均值|70.5628|70.9733|+0.4106|至少+1.0pp|FAIL|
|LEO Strict UDU floor|69.2717|69.6767|+0.4050|诊断项|低于0.5pp|

### 分场景结果

|场景|S0 overall|S3G overall|差值|S0 Strict UDU|S3G Strict UDU|差值|
|---|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|78.4691|78.6353|+0.1662|72.5533|72.9817|+0.4283|
|`leo_low_elev_weak`|75.6461|75.8902|+0.2441|69.8633|70.2617|+0.3983|
|`leo_rain_weak`|75.2912|75.4505|+0.1593|69.2717|69.6767|+0.4050|

对应净正确数变化为：clean减少267/204,000；clear增加339/204,000；low-elevation增加498/204,000；rain增加325/204,000。没有保存逐样本S0/S3G配对prediction，因此这些净变化不能替代`rescued`与`harmed`计数。

## 域切片分析

S3G的收益不是均匀域泛化。三种LEO场景中，`test_unseen_day_seen_rx`分别提升0.4893、0.5190和0.4655pp，Strict UDU分别提升0.4283、0.3983和0.4050pp；但`test_seen_day_unseen_rx`分别下降0.5483、0.2950和0.5150pp。该模式说明SID残差对未见日期和日期×接收机复合偏移有小幅正效应，却损害单独的未见接收机迁移，不能解释为普遍接收机不变性增强。

clean切片也呈现接收机重分配：`test_unseen_day_rx_8`提升1.6333pp，RX10提升0.2583pp，RX9提升0.0167pp；RX7下降0.6083pp，RX11下降0.1000pp。seen-day unseen-RX总指标下降0.6367pp。聚合clean仅下降0.1309pp掩盖了接收机间的异质变化。

## 完整训练曲线

- `metrics_epoch.csv`和`metrics_epoch.jsonl`均包含连续的200条epoch记录，JSONL解析错误数为0；stdout包含200个`EPOCH-BEGIN`与200个`EPOCH-END`。
- source validation TX在98.8095%–98.8571%之间，成熟基座没有崩坏。source-only选择分数从E1的92.1071上升到E44的92.8969，最终在E178以92.9068刷新；E44至E178仅增加0.0099，后半程基本处于平台。
- E178的source satellite mean/floor为89.2407%/87.6508%；E200为89.1958%/87.5873%，与按source-only选择E178一致。
- satellite CE在E80计入总目标后，训练loss由约1.1跳至约3.1，但validation保持稳定。全程`train_loss_open_group=0`，证明域对抗、正交和其他open分量虽保留遥测，但没有进入SID优化目标。
- SID绝对残差范数从E1的0.00367增长到E178的0.25982和E200的0.29056；raw/SID预测一致率从99.9826%缓慢下降，最低97.2743%，E178为97.6736%，E200为98.2813%。这与首轮残差上万、预测一致率低于5%的坍缩形成明确区分。
- 全程没有非有限loss，但E1、E10、E74、E100和E190各有1/45个batch因非有限梯度被跳过，合计5/9,000个训练batch。该比例很低，但不满足预登记的“全程有限gradient”技术条件。
- stdout全量扫描未发现Traceback、RuntimeError、CUDA OOM或Killed；日志中的NaN均来自按设计未执行的训练期heldout test及未启用遥测占位，不能当作最终评估NaN。

## 10%残差上界未生效的根因

代码路径已定位到`code/post_stage_common.py`的`build_baseline_model()`。该函数向`build_dual_model()`传递了`sid_fft96_mode`、`sid_mask_path`和`sid_residual_scale`，但漏传`sid_max_residual_ratio`。因此：

- launcher和checkpoint args均记录`sid_max_residual_ratio=0.1`；
- checkpoint重建后运行时`model.sid_fft96.max_residual_ratio=0.0`；
- 训练CSV有10个epoch的均值比值超过0.1，最大值因小分母放大到40,853,819.79；
- 对最终checkpoint的5880个`L_s`样本进行只读前向，eval模式有126个样本超过0.1，最大比值2.1035；train模式有302个样本超过0.1，最大比值1.7394。

这不是报告字段缺失，而是候选实现偏离预登记配置。S3G不能标记为“10%受控残差验证通过”。

## checkpoint与资源证据

- S3G选择checkpoint与`final_ssdg.pth`包含200个相同张量，最大绝对差为0；最终独立评估确实连接到E178选择状态。
- S3G与CORE90共有的195个非SID张量全部完全一致，最大绝对差为0；只有4个SID projector张量被训练，另有1个固定mask张量。
- 总参数量1,090,945，可训练SID参数41,280，占3.78%。
- 训练墙钟时间8,685.59秒，即2小时24分45.59秒；峰值CUDA allocated/reserved约545.66/746.00MiB。该资源记录是本次共享服务器上的训练观测，不是隔离延迟基准。
- 同一E178状态的frozen heldout eval与独立final eval之间，核心指标最大差0.0033pp；该微小重复评估差异不改变任何门槛判定，但说明当前GPU评估链未达到逐样本完全确定性。

## 预登记门槛闭合

|门槛|状态|证据|
|---|---|---|
|clean下降不超过0.3pp|PASS|-0.1309pp|
|LEO均值至少提升1pp|FAIL|+0.1899pp|
|Strict UDU至少提升1pp|FAIL|+0.4106pp|
|LEO floor至少提升0.5pp|FAIL|+0.1593pp|
|`rescued>harmed`|UNKNOWN|缺少逐样本配对prediction|
|RX probe相对下降至少20%|UNKNOWN|仅有S3G source probe，无匹配S0 probe|
|TX probe不下降|UNKNOWN|无预登记的匹配TX probe artifact|
|全程有限loss/gradient|FAIL|5个batch跳过非有限gradient|
|有效SID残差比例不超过10%|FAIL|运行时上界为0.0，最终前向确认越界|
|非SID参数零漂移|PASS|195个共有非SID张量最大绝对差0|

## 最终判定与下一步

最终判定为`SCIENTIFIC_FAILURE_NO_PROMOTION / IMPLEMENTATION_DEVIATION_GUARD_NOT_ACTIVE`。该候选不替换CORE90，不进入多seed或完整确认矩阵，也不能用于Phase2注册、Phase3未知拒识或真实卫星性能声明。

本轮仍提供两条可复用的机制证据。第一，首轮坍缩的主要驱动确实来自把冻结Core90的域对抗及其他辅助目标压到唯一可训练的SID投影上；移除这些梯度后，clean从约16%恢复到90%，残差从上万降到0.29量级。第二，SID方向在三种LEO场景和Strict UDU上均呈同号小增益，但会损伤seen-day unseen-RX，因此下一轮应控制残差并直接约束接收机迁移，而不是扩大SID容量。

若继续该路线，最小下一步是：在`build_baseline_model()`显式传递`sid_max_residual_ratio`；增加“构建后运行时属性等于0.1”和“逐样本残差比不超过0.1”的聚焦测试；记录非有限梯度的参数名与batch条件；复用已完成的S0，仅重跑单seed修正S3G。修正S3G仍需先通过同一门槛，再决定是否进入多seed，不能把本轮未受控结果作为晋级依据。

## 发布与路径

- 实现提交：`633da733b9c849592b9f90eeaf11f031095b949e`。
- N607 release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_633da733`。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_guarded_20260822_v1`。
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_guarded_20260822_v1`。
- release归档本地与远端SHA-256均为`0023abf9a98c6344d7204361d0b92b297b12ac74d62e653e9c5c0a19c1f36de0`，状态`VERIFIED`。

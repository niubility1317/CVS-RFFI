# ADV3B02→FCR R1-R8重新发布v4完整实验报告

## 最终摘要

本批`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`已经完成ADV3B02基线训练、R1-R8八行FCR训练、四场景无标签prediction和独立truth-last评分，最终状态为`ANALYZED`。所有候选统一使用seed392005、同一ManySig equalized划分和E200最后一个checkpoint；target测试集没有进入训练、选模或候选筛选。

最可靠的性能结论是：R3在本批单seed上取得最佳LEO鲁棒性，clean/clear/low-elev/rain分别为78.4435%/64.6369%/62.8964%/62.6756%，LEO均值63.4030%、LEO下界62.6756%、四场景均值67.1631%；相对ADV3B02分别提高2.2167pp/3.2633pp/3.2119pp/3.0016pp。R6取得最高clean=78.4601%，R8取得次高LEO均值63.2895%，但都没有超过R3的综合结果。

全量训练日志同时限定了机制结论：R1-R5中self、swap、shared、latent-cycle和basic drop-f necessity按预定阶段产生了非零信号；R6虽配置了targeted transplant，但200轮`active_fingerprint_pairs`始终为0，因此R6不能被解释为“定向移植有效”；所有行的`eta`项均为0，R7/R8的physical与R8的factor项确有非零运行信号。由此，本批支持的是“FCR训练族相对ADV3B02有收益、R3组合在单seed最优”，不支持“全部后续机制逐项有效”或“完整三因子可辨识已经得到验证”。

当前最突出的类别瓶颈仍是TX1：ADV3B02的LEO均值为33.9155%，R3为33.4214%，反而下降0.4940pp；R3总体增益主要来自TX0（+8.5298pp）和TX2（+7.7512pp）。因此下一步不应直接把R8作为默认模型，应先修复干预配对覆盖和`eta`监督，再用新的单seed同协议最小矩阵验证；R3可作为当前研究候选，但尚不足以跨seed晋级。

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`
- 替代原因：v3首轮遥测在非MUSE路径访问未初始化`rc4_route`；v4仅初始化可选遥测变量。
- 启动阶段先运行seed392005的`ADV3B02_CORE90_SOFT_E200`；R1-R8等待其E200 `final_ssdg.pth`后再启动。
- source：ManySig equalized；receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；`L_s/U_s/V=6300/56700/27000`。
- target训练边界：`test_eval_policy=never`；不训练、不筛选、不选择checkpoint；最后epoch冻结后才独立预测与评分。
- checkpoint：final-only；`best_metric=clean_val_tx`仅满足source-only兼容检查，held-out joint guard关闭。
- GPU：ADV3B02使用GPU0；不干预既有任务。该条记录对应R1-R8尚未启动的发布时点。
- Git固定版本：`8f1de7971853aa9650e4f83d6ad979f359c434c2`。
- release：本地`E:\type10-7\releases\phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_8f1de797.zip`；远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_8f1de797`；归档SHA256本地/远端一致：`6D8484618441CE2E8470556864C82D8DF64A3BD8E8D826A5D5713428CFE40B10`。
- 发布前验证：远端编译通过；真实checkpoint无query smoke通过；真实训练入口dry-run通过。
- 启动命令：`bash code/scripts/launch_phase1_adv3b02_fcr_r1r8_s392005_20260903.sh phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`。
- 远端CWD：`/home/szu2070436088`；launcher PID=`381957`；ADV3B02训练PID=`382450`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200.out`。
- 启动健康检查：PID/CWD/cmdline/run-root绑定正确；PID`382450`绑定GPU0，显存约2802MiB；日志已增长至17991字节并完成E002/200，未发现Traceback。中间checkpoint标记为`NOT_SAVED_FINAL_ONLY`，R1-R8仍未启动，等待训练结束后写入的E200 `final_ssdg.pth`。
- 该发布时点状态：`RUNNING`；最终状态见后文`ANALYZED`。

## ADV3B02 E200基线独立测试

- eval_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_ADV3B02_baseline_eval_v1`。
- checkpoint：`ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`，仅使用E200最终checkpoint，不进行选模。
- 数据：ManySig equalized；target receiver=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`、TX=`[0,1,2,3,4,5]`；每场景168000条。
- 场景：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 隔离：prepare生成独立无标签IQ package和truth sidecar；predictor仅读取IQ package及稳定opaque `sample_id`；prediction完成后由独立scorer连接truth。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/baseline_eval_v1`。
- GPU：GPU0；用户已明确允许在现有进程数上继续增加实验，不停止或干预现有任务。
- 预期artifact：`truth/truth_sidecar.json`、`inputs/iq.npy`、`inputs/manifest.json`、`prediction/predictions.json`、`prediction/score.json`及各阶段日志。
- 技术停止规则：仅在路径覆盖风险、checkpoint/数据绑定错误、prediction不完整、scorer连接错误、确定性异常、OOM或非有限输出时停止本评估；不因性能高低停止。
- v1结果：prepare完成并生成168000条无标签输入及独立truth sidecar；predict在首批DataLoader读取时确定性失败，错误为`TypeError: expected np.ndarray (got numpy.ndarray)`；未生成prediction，未执行scorer，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。v1全部现场保留。
- 根因：N607的NumPy2.2.5与PyTorch2.1.0组合无法通过`torch.from_numpy`转换普通连续`float32 ndarray`；predictor绕过了项目已有兼容转换入口。
- 修复：`_OpaqueTargetDataset`复用`dataset_wisig._safe_to_torch_float_tensor`，当`torch.from_numpy`和`torch.as_tensor`均失败时以plain-list构造tensor；不改变数据、模型、场景或truth-last边界。
- 验证：新增ABI回归测试先RED后GREEN；聚焦测试20/20通过；Python编译与diff检查通过；原问题定点P0/P1复审结论为FIXED。
- v2 eval_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_ADV3B02_baseline_eval_v2`；使用全新`baseline_eval_v2`输出根，重新执行prepare→predict→独立scorer，不覆盖或复用v1输出根。
- v2修复提交：`0cc19956a817c14d8f1155114a714a206aed2b21`；本地与远端分支OID一致。
- v2 release：`phase1_adv3b02_baseline_eval_s392005_20260903_v2_0cc19956.zip`；本地/远端归档SHA256一致为`502ED1AFD930451B652817693D3F05A7660E2E9BD105704DCAEECE6F9C17E4A0`；远端编译和真实无标签IQ单样本转换smoke通过。

### v2测试结果

|场景|正确数/总数|Accuracy|相对clean下降|
|---|---:|---:|---:|
|clean|128061/168000|76.2268%|0.0000pp|
|leo_clear_weak|103098/168000|61.3679%|14.8589pp|
|leo_low_elev_weak|100107/168000|59.5875%|16.6393pp|
|leo_rain_weak|99899/168000|59.4637%|16.7631pp|

- LEO三场景均值：60.1397%；LEO最差场景：59.4637%（`leo_rain_weak`）；clean到LEO均值下降16.0871pp；四场景宏平均64.1615%。

|TX类别|clean|clear|low-elev|rain|四场景均值|
|---:|---:|---:|---:|---:|---:|
|0|90.8536%|56.1464%|55.7179%|57.1607%|64.9696%|
|1|51.2964%|35.5321%|33.3321%|32.8821%|38.2607%|
|2|62.0179%|46.5679%|44.2321%|43.4036%|49.0554%|
|3|64.7286%|61.2750%|58.6964%|57.1179%|60.4545%|
|4|99.8893%|83.4464%|80.3250%|79.9214%|85.8955%|
|5|88.5750%|85.2393%|85.2214%|86.2964%|86.3330%|

- 闭环：无标签predictor package为168000条，`contains_labels=false`；独立truth sidecar为168000条并绑定`ManySig|tx_rx_day_1_7_2|392005`；prediction为672000条，四场景各168000条；独立score schema为`cvs.phase1.truth_last_score.v1`。
- checkpoint未被测试改写：测试后仍为15039647字节，mtime=`2026-09-03 08:00:52 +0800`。prediction为197064257字节；全量prepare/predict/score日志未发现异常指纹。
- 机器可读结果：[baseline_eval_v2_score.json](baseline_eval_v2_score.json)。
- 解释：该单seed基线在clean上达到76.2268%，但LEO三场景均值下降至60.1397%；类别1是所有场景的最低类，rain下仅32.8821%。这表明当前E200最终checkpoint的主要短板是跨LEO弱信道鲁棒性和类别1稳定性。本结果仅是指定split/seed/checkpoint的Phase1闭集六类基线，不代表R1-R8结果，也不构成真实在轨性能声明。
- v2状态：`ANALYZED`。

## R1-R8 E200最终checkpoint独立测试v2预登记

- eval_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_R1R8_target_eval_v2`。
- 触发原因：R1-R8均完成200轮训练并生成`final.pth`，但原release中的predictor在首批DataLoader读取时均触发`TypeError: expected np.ndarray (got numpy.ndarray)`；8行均未生成prediction，旧`predict.log`与空输出目录原样保留。
- 修复来源：使用已完成TDD、20/20聚焦测试、真实IQ smoke和定点P0/P1复审的提交`0cc19956a817c14d8f1155114a714a206aed2b21`及release`phase1_adv3b02_baseline_eval_s392005_20260903_v2_0cc19956`。
- checkpoint：仅使用R1-R8各自E200最终`final.pth`；禁止target测试集选模、筛选、重排或反馈训练。
- 数据：复用已生成的同一无标签输入包`target_inputs`与独立truth sidecar`target_truth/truth_sidecar.json`；不重新划分数据，不改变split seed、receiver、day、TX或场景。
- 场景：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，每场景168000条。
- 输出：每行写入全新不可覆盖目录`FCR_Rn/Rn/target_prediction_v2`；日志为`predict_v2.log`和`score_v2.log`。
- GPU：R1-R8分别使用GPU0-7；用户已明确允许在现有进程数上继续增加实验。
- 执行顺序：8行并行prediction全部完成后，独立scorer逐行连接truth；任何行prediction失败时不产生该行性能结论。
- 技术停止规则：仅因路径覆盖、checkpoint/数据绑定错误、prediction不完整、scorer连接错误、确定性异常、OOM或非有限输出停止；不因性能高低停止。
- 启动前状态：`LOCAL_VERIFIED`。

### R1-R8 v2测试闭环与最终结果

- 完成状态：R1-R8训练均完成E200；修复后8行prediction均为672000条，四场景各168000条；8份独立score均为`cvs.phase1.truth_last_score.v1`且`record_count=672000`。目标测试集没有参与训练、checkpoint选择、候选筛选或重跑选择。
- checkpoint：每行只测试E200最后一个`final.pth`，训练日志均明确记录`selection=final_epoch_only target_metrics_consumed=0`。
- 数据绑定：ManySig equalized，`split_mode=tx_rx_day_1_7_2`，`split_seed=392005`；source receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；target receiver=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`、TX=`[0,1,2,3,4,5]`。
- truth-last：predictor只读取既有无标签`target_inputs`；全部prediction固定后，独立scorer才读取`target_truth/truth_sidecar.json`。
- 训练日志完整性：每行CSV与JSONL均为200条且epoch连续E1-E200；每行均有200个`[EPOCH-END]`和E200最终checkpoint标记；结构化数值未出现NaN/Inf。日志中的未启用诊断字段会打印`nan`占位，不进入训练数值或最终评分。

|行|新增机制（相对上一行）|clean|clear|low-elev|rain|LEO均值|LEO下界|四场景均值|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|R1|self reconstruction（eta配置但无有效监督）|78.1315%|64.1256%|62.3679%|62.1202%|62.8712%|62.1202%|66.6863%|
|R2|+swap|77.9190%|63.8333%|62.1863%|62.0560%|62.6919%|62.0560%|66.4987%|
|R3|+shared|78.4435%|64.6369%|62.8964%|62.6756%|63.4030%|62.6756%|67.1631%|
|R4|+latent cycle|77.9542%|64.2768%|62.4488%|62.2012%|62.9756%|62.2012%|66.7202%|
|R5|+basic drop-f necessity|77.9929%|64.4536%|62.7238%|62.5280%|63.2351%|62.5280%|66.9246%|
|R6|+targeted transplant开关（有效pair=0）|78.4601%|64.4024%|62.6607%|62.4554%|63.1728%|62.4554%|66.9946%|
|R7|+physics-ordered decoder/phys|78.3542%|64.4476%|62.6518%|62.4833%|63.1942%|62.4833%|66.9842%|
|R8|+three-axis factor intervention|78.2369%|64.5875%|62.7304%|62.5506%|63.2895%|62.5506%|67.0263%|

|行|clean正确数|clear正确数|low-elev正确数|rain正确数|
|---|---:|---:|---:|---:|
|R1|131261/168000|107731/168000|104778/168000|104362/168000|
|R2|130904/168000|107240/168000|104473/168000|104254/168000|
|R3|131785/168000|108590/168000|105666/168000|105295/168000|
|R4|130963/168000|107985/168000|104914/168000|104498/168000|
|R5|131028/168000|108282/168000|105376/168000|105047/168000|
|R6|131813/168000|108196/168000|105270/168000|104925/168000|
|R7|131635/168000|108272/168000|105255/168000|104972/168000|
|R8|131438/168000|108507/168000|105387/168000|105085/168000|

#### 相对ADV3B02 E200基线

ADV3B02基线为clean=76.2268%、LEO均值=60.1397%、LEO下界=59.4637%、四场景均值=64.1615%。

|行|Δclean|ΔLEO均值|Δ四场景均值|clean→LEO均值下降|
|---|---:|---:|---:|---:|
|R1|+1.9048pp|+2.7315pp|+2.5249pp|15.2603pp|
|R2|+1.6923pp|+2.5522pp|+2.3372pp|15.2271pp|
|R3|+2.2167pp|+3.2633pp|+3.0016pp|15.0405pp|
|R4|+1.7274pp|+2.8359pp|+2.5588pp|14.9786pp|
|R5|+1.7661pp|+3.0954pp|+2.7631pp|14.7578pp|
|R6|+2.2333pp|+3.0331pp|+2.8332pp|15.2873pp|
|R7|+2.1274pp|+3.0546pp|+2.8228pp|15.1600pp|
|R8|+2.0101pp|+3.1498pp|+2.8649pp|14.9474pp|

#### 逐TX准确率

|行/场景|TX0|TX1|TX2|TX3|TX4|TX5|
|---|---:|---:|---:|---:|---:|---:|
|R1 clean|90.8500%|50.7571%|71.1214%|67.6464%|99.7964%|88.6179%|
|R2 clean|91.5250%|50.9000%|70.7143%|66.9750%|99.8036%|87.5964%|
|R3 clean|91.8107%|50.4179%|71.8929%|67.8643%|99.7929%|88.8821%|
|R4 clean|91.8321%|50.9429%|70.0643%|67.4964%|99.7643%|87.6250%|
|R5 clean|91.8786%|49.9107%|70.6250%|67.1464%|99.7893%|88.6071%|
|R6 clean|92.0393%|49.8286%|72.5536%|67.6357%|99.7893%|88.9143%|
|R7 clean|92.0286%|51.7500%|70.5143%|67.0536%|99.7679%|89.0107%|
|R8 clean|91.6107%|48.8607%|70.2000%|68.5643%|99.7607%|90.4250%|
|R1 clear|63.1679%|34.9750%|51.9786%|62.6679%|85.3464%|86.6179%|
|R2 clear|62.9500%|33.5107%|51.4107%|63.5643%|85.6321%|85.9321%|
|R3 clear|65.1607%|35.0643%|53.6286%|61.9857%|85.0571%|86.9250%|
|R4 clear|63.5143%|34.8250%|51.9571%|62.8000%|84.8286%|87.7357%|
|R5 clear|65.4786%|34.2357%|52.1964%|62.1607%|85.6500%|87.0000%|
|R6 clear|64.3964%|33.4607%|52.8714%|62.8107%|84.9786%|87.8964%|
|R7 clear|65.7929%|34.4250%|52.1179%|62.3429%|85.0536%|86.9536%|
|R8 clear|65.5214%|33.1393%|53.6500%|62.4536%|84.5286%|88.2321%|
|R1 low-elev|62.5821%|32.8250%|50.3393%|59.5571%|82.4071%|86.4964%|
|R2 low-elev|62.5250%|31.2250%|50.2464%|60.5143%|82.9536%|85.6536%|
|R3 low-elev|64.2286%|32.9607%|52.1357%|58.8536%|82.6357%|86.5643%|
|R4 low-elev|62.3750%|32.7786%|50.1607%|59.7214%|82.2179%|87.4393%|
|R5 low-elev|64.7821%|32.3000%|50.5821%|59.0643%|83.1714%|86.4429%|
|R6 low-elev|63.6607%|31.5321%|51.3821%|59.7464%|82.4429%|87.2000%|
|R7 low-elev|64.9893%|32.1500%|50.5750%|59.1321%|82.5500%|86.5143%|
|R8 low-elev|64.6821%|31.1321%|51.7643%|59.4250%|81.9107%|87.4679%|
|R1 rain|63.6643%|32.1750%|49.8750%|58.2429%|81.9107%|86.8536%|
|R2 rain|63.8179%|30.5429%|49.8857%|59.3000%|82.2929%|86.4964%|
|R3 rain|65.2250%|32.2393%|51.6929%|57.6929%|82.0321%|87.1714%|
|R4 rain|63.2286%|32.1107%|49.6036%|58.6536%|81.5536%|88.0571%|
|R5 rain|65.9964%|31.4571%|50.3714%|57.6571%|82.5821%|87.1036%|
|R6 rain|64.9107%|30.9536%|50.8286%|58.4393%|81.7357%|87.8643%|
|R7 rain|66.2179%|31.7036%|50.3964%|57.6929%|81.8214%|87.0679%|
|R8 rain|65.8036%|30.6607%|51.3750%|57.9786%|81.2357%|88.2500%|

#### 训练末态与资源

|行|E200 train acc|E200 source-val acc|最佳source-val（epoch）|累计epoch时间|
|---|---:|---:|---:|---:|
|R1|94.7226%|98.6778%|98.7000%（E154）|2.467h|
|R2|94.3878%|98.6556%|98.6815%（E161）|2.069h|
|R3|94.1486%|98.6889%|98.6963%（E190）|2.079h|
|R4|93.2876%|98.6815%|98.6963%（E179）|2.094h|
|R5|94.4675%|98.6556%|98.6852%（E166）|2.243h|
|R6|93.5906%|98.6741%|98.7037%（E173）|2.230h|
|R7|93.8138%|98.7000%|98.7185%（E154）|2.768h|
|R8|94.1805%|98.6667%|98.6963%（E122）|2.281h|

#### 结果解释与结论

- R3是本批次鲁棒性主结果：LEO均值63.4030%、LEO下界62.6756%、四场景均值67.1631%，分别比ADV3B02提高3.2633pp、3.2119pp和3.0016pp；它同时取得clear、low-elev和rain三项行级最高值。
- R6取得最高clean准确率78.4601%，比ADV3B02提高2.2333pp，但其LEO均值比R3低0.2302pp。
- R8没有超过R3：相对R7，factor/three-axis使LEO均值提高0.0953pp、四场景均值提高0.0421pp，但clean下降0.1173pp；该增益小且仅有单seed证据。
- R2相对R1全面小幅退化，说明单独加入swap在这一固定seed和最终checkpoint协议下没有正收益。R3加入shared后，相对R2的clean、LEO均值和四场景均值分别回升0.5244pp、0.7111pp和0.6644pp。
- 所有R行都高于ADV3B02的四场景均值，但类别瓶颈没有解决：TX1在LEO场景仅30.5429%至35.0643%，仍是每行最弱类别；TX4/TX5保持最高。
- 本批次只覆盖单seed392005和指定ManySig划分。R3可作为下一步确认候选，但当前结果不足以声明跨seed稳定胜出，也不代表真实在轨性能。
- 之前显示的R7=`92.90%/82.22%/79.22%/79.03%`不是本次truth-last测试结果。本次可追溯的R7最终结果是`78.3542%/64.4476%/62.6518%/62.4833%`。
- 机器可读完整结果：[r1r8_target_eval_v2_summary.json](r1r8_target_eval_v2_summary.json)。
- 最终状态：`ANALYZED`。

## 历史高分截图完整溯源

### 结论

截图中的R1-R8数值可逐项追溯到`phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`报告第52-59行。该批与本run不是同一checkpoint、同一source/target划分、同一测试集合或同一checkpoint选择规则，不能直接比较。旧表高出约12.78-17.77pp的主要原因是旧“overall test”混合了训练已见接收机、训练已见日期和双未见子集；它不是本次规定的7个接收机全部作为receiver-disjoint target的168000条测试。

### 两批实验的关键差异

|项目|历史截图v6|当前v4|
|---|---|---|
|FCR run|`phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`|`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`|
|初始化模型|`ADV3B03_MU10_ALPHA20_E200`|新训练`ADV3B02_CORE90_SOFT_E200`|
|初始化seed|392002|392005|
|初始化checkpoint|`.../phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1/S392002_ADV3B03_MU10_ALPHA20_E200/final_ssdg.pth`|`.../phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`|
|checkpoint大小/时间|15035743字节；2026-08-30 06:15|15039647字节；2026-09-03 08:00|
|source receiver|`[0,1,2,3,4,5,6]`|`[1,3,4,6,8]`|
|source day|`[0,1]`|`[1,2,3]`|
|source池及角色|84000；L/U/V=`5880/52920/25200`|90000；L/U/V=`6300/56700/27000`|
|旧overall组成|84000条未见日×已见RX+60000条已见日×未见RX+60000条未见日×未见RX|7个receiver-disjoint target×4天×6类×1000条|
|每场景样本数|204000|168000|
|测试接收机口径|未见RX仅`[7,8,9,10,11]`，overall还包含已见RX`[0..6]`|固定target=`[0,2,5,7,9,10,11]`|
|训练期间target可见性|每个epoch运行有标签`[TEST]`和`[SAT-TEST]`并保存test-best checkpoint|`test_eval_policy=never`；E200冻结后才生成prediction|
|正式报告checkpoint|`best_joint.pth`，按source-val选择；R7为E146|只使用E200最后一个`final.pth`|
|评分闭环|内置有标签评测；prediction随机HMAC ID且无可连接truth sidecar，独立评分未闭合|无标签prediction先固定，之后独立scorer连接truth sidecar，闭环完成|

旧v6把当前target中的receiver0、2、5当作训练已见receiver；这3个receiver占当前目标集合3/7。旧overall的204000条中，84000条（41.18%）直接来自训练已见receiver，只是日期未见；另有60000条（29.41%）来自训练已见日期，只是receiver未见；真正receiver和day同时未见的只有60000条（29.41%）。因此旧overall天然比当前全receiver-disjoint target容易。

旧v6的双未见子集也不能与当前target等同：它只含receiver`[7,8,9,10,11]`，其中receiver8在当前协议是source；当前target额外包含receiver`[0,2,5]`，其中`14-7`是已知最困难接收机。因此即使比较旧表的`strict_udu`，也仍然不是同一测试集。

### 截图数值与当前truth-last结果差值

|行|Δclean|Δclear|Δlow-elev|Δrain|
|---|---:|---:|---:|---:|
|R1|+14.9785pp|+17.3944pp|+16.3121pp|+16.4398pp|
|R2|+15.0810pp|+17.6067pp|+16.2237pp|+16.3340pp|
|R3|+14.5765pp|+16.4831pp|+14.9836pp|+15.2844pp|
|R4|+15.1758pp|+16.6332pp|+15.5912pp|+15.7888pp|
|R5|+14.6871pp|+14.2764pp|+12.7762pp|+12.8020pp|
|R6|+14.5399pp|+17.0476pp|+15.7293pp|+15.8046pp|
|R7|+14.5458pp|+17.7724pp|+16.5682pp|+16.5467pp|
|R8|+14.7931pp|+17.0525pp|+15.7996pp|+15.8694pp|

### 历史初始化checkpoint的真实身份与配置

历史截图使用的初始化checkpoint并不是ADV3B02。其真实身份为：

- 生成run：`phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1`。
- candidate：`S392002_ADV3B03_MU10_ALPHA20_E200`。
- checkpoint：`final_ssdg.pth`，epoch200，15035743字节。
- 代码提交：`42df44e70f79e76072b4a98a568870c460cc35d6`。
- 训练方式：`from_scratch=true`，没有加载ADV3B02权重；仅复用了历史ADV3B02 CORE90的seed中心392002。
- 数据：ManySig equalized；`split_mode=tx_rx_day_1_7_2`；source receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；L/U/V=`0.07/0.63/0.30`；batch128；E200。
- 训练阶段：label130epoch+pseudo70epoch；EMA teacher；`lambda_u=0.16`、`lambda_ent=0.01`、`lambda_domain=1`、`lambda_adv=0.35`、`lambda_group_ce=0.16`、`lambda_fishr=0.04`。
- ADV3B03新增损失：`lambda_proto=0.0032`、`lambda_open_world_feat=0.0024`、`lambda_zid_compact=0.032`、`lambda_proxy_unknown=0.0050`、`lambda_soft_unknown_mixup=0.0045`、`lambda_source_episode=0.0035`。
- 星地增强：concat masked/CE-only，权重1.0；E1-40 clear概率0.30，E41-90 low/rain概率0.60，E91-200三场景概率0.80；`lambda_sat_cls=0.68`、`lambda_sat_cons=0`，sat start E80。
- source侧E200评测：clean97.8741%、clear88.4963%、low84.7185%、rain85.7926%；这是source留出结果，不是目标receiver结果。
- 该ADV3B03实验的冻结冠军其实是seed392005，不是392002；seed392002排名第2。

### 判定主要问题

1. **首要问题是测试定义改变**：旧overall把已见receiver/已见day混合进总分；当前是固定7个receiver-disjoint target。它解释了绝大部分15-18pp差距。
2. **其次是训练/测试隔离不合格**：旧训练日志每个epoch读取有标签target并保存多种test-best checkpoint。虽然截图报告加载的是source-val选择的`best_joint.pth`，没有证据表明target标签直接反向传播，但存在确定的test peeking和跨R1-R8人工筛选风险，不能作为无偏最终测试。
3. **checkpoint身份误标**：旧FCR批次名含`adv3b02`，但实际初始化来自从头训练的ADV3B03 seed392002；把它称为“ADV3B02 checkpoint”不准确。
4. **checkpoint选择规则不同但不是主因**：旧R7报告值来自E146 source-val最优checkpoint；同一日志E200约为clean92.99%、clear81.97%、low79.07%、rain78.98%，与截图仅差-0.09/+0.25/+0.15/+0.05pp，无法解释15-18pp主差距。
5. **旧prediction不可独立复分**：随机HMAC `sample_id`没有同步truth sidecar，所以截图只可视为可追溯的旧内置评测，不是当前truth-last正式测试。

### 交叉验证

最直接的反证是历史ADV3B03冻结seed392005已按当前同一7个target receiver×4天×每场景168000条做过零适配测试，结果为clean78.4363%、clear62.8625%、low60.9440%、rain60.8655%。它与当前R1-R8的约78%/64%/63%/62%处于同一量级，而不是旧截图的约93%/82%/79%/79%。因此当前低分不是一次异常崩塌；旧表与当前表的评测协议不一致才是主因。

### 证据状态

- 历史截图数字来源：`VERIFIED`，逐项匹配v6报告和R7完整日志。
- 历史初始化checkpoint身份、路径、大小和config：`VERIFIED`，已在N607只读回查。
- 当前v4结果：`VERIFIED`，8行均有672000条prediction及独立truth-last score。
- 旧v6独立truth-last评分：`FAILED/未闭合`，不能补算或伪造。
- 是否存在target梯度更新：未发现证据；是否存在训练期间test peeking：`VERIFIED`。

## 全量机制、训练过程与证据边界复核

### 分析范围

本节重新读取了ADV3B02的200条`metrics_epoch.csv`和200条`metrics_epoch.jsonl`，以及R1-R8每行全部200条结构化epoch记录、全部stdout行和全部truth-last score，而不是抽查日志尾部。每行CSV/JSONL的epoch均严格连续E1-E200，stdout均含200个`[FCR]`和200个`[EPOCH-END]`记录；8行均无Traceback、RuntimeError、CUDA error、OOM或Killed指纹。

ADV3B02训练期`test_eval_ran`合计为0；R1-R8每个epoch的`[TEST]`均明确为training-time gate跳过。最终测试统一在E200冻结后完成，每行prediction为672000条，四场景各168000条，再由独立scorer连接truth sidecar。因而当前性能数据属于正式测试结果，而不是source-val或训练期测试日志。

### 数据、初始化与共同训练配置

|项目|固定值|
|---|---|
|数据|ManySig，equalized=true，`split_mode=tx_rx_day_1_7_2`，seed392005|
|source|receiver=`[1,3,4,6,8]`，day=`[1,2,3]`，pool=90000|
|source角色|`L_s=6300`、`U_s=56700`、source validation=`27000`|
|target|receiver=`[0,2,5,7,9,10,11]`，day=`[0,1,2,3]`，TX=`[0..5]`|
|target规模|每个receiver×day为6000条；每TX×receiver×day为1000条；每场景168000条|
|初始化|本run新训练`ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`，seed392005，epoch200|
|FCR加载|每行`loaded=195`、`missing=36`、`skipped=0`、`unexpected=0`；36个缺失键为新FCR模块|
|优化|AdamW，lr=`2e-4`，min lr=`1e-6`，weight decay=`1e-4`，batch128，E200，AMP|
|模型|ADV3B02为1049827参数；FCR为1061996参数，新增12169参数（+1.16%），全部可训练|
|主干损失|身份CE、domain loss=1.0、GRL identity去域=0.5、orth=0.05、cons=0.10；PA辅助保留|
|source星地增强|E1-40 clear概率0.30；E41-90 low/rain概率0.60；E91-200三场景概率0.80；`lambda_sat_cls=0.68`、`lambda_sat_cons=0`|
|测试规则|`test_eval_policy=never`；只测试E200最后一个checkpoint；禁止target选模|

`use_meta_ssl_cvs`在数据角色和loader层面启用，但其伪标签/原型/域分支损失权重均为0；U_s的作用来自FCR允许的无标签重构项，而不是硬伪标签CE。`L_s`可使用身份和全部合法FCR目标；`U_s`只允许self、swap、shared、latent-cycle、eta和phys，不允许身份、factor或transplant；source validation只读。

### 网络结构落地

FCR不是普通三向量拼接。实际前向链为：保守canonicalizer先解析并有界移除公共gain、phase和CFO；内容分支生成低采样率序列`z_s`并重建`ŝ`；指纹分支把ADV3B02身份特征、canonical IQ、残差IQ和激励特征编码为160维单位球`z_f_id`与16维`z_tx_state`；激励条件化算子生成小残差`δ_f`；结构化nuisance分支只读取11个统计量，输出`z_ch(16)+z_rx(8)+z_sync(6)+z_gain(3)`；Decoder输出有界方差的复IQ条件均值。

R1-R6使用`control` Decoder，即`μ=ŝ+δ_f`，结构化nuisance只控制方差而不进入波形链路；R7-R8切换为`full_physics` Decoder，严格按短信道→RX残差→同步/增益的顺序把nuisance施加到`ŝ+δ_f`。指纹算子接收的内容激励采用detach，避免`G_f→E_s`反向捷径。该结构忠实落地了设计报告中的“内容生成→发射机响应→信道/接收机”主线，但是否真正得到物理可辨识因子仍需诊断证据，不能仅由结构命名推出。

### 四阶段训练日程

|阶段|epoch|训练行为|
|---|---:|---|
|基本重构|E1-40|身份主损失+self+eta；其余FCR项关闭|
|交叉重构渐入|E41-90|self/eta持续，swap/shared/latent-cycle从0线性增至1|
|干预阶段|E91-150|在optimizer step层面交替完整组合更新与Decoder冻结的necessity更新|
|身份精修|E151-200|完整合法目标持续；self和swap缩放到0.25，其余保持1|

E91的训练准确率在各行从E90约92.3%短暂降至约85.5%-86.5%，与交替干预阶段切换同步，但source-val始终约98.6%，没有形成持续崩塌。E151降低重构权重后总loss出现预期台阶下降，source-val继续稳定。

### 消融路由与实际非零信号

下表中的“配置”来自冻结消融路由，“实际信号”来自200条`[FCR]`全量记录。非零epoch数按日志中的epoch均值统计；E41的ramp恰为0，因此交叉项最多为159个非零epoch。

|行|相对上一行的配置变化|实际非零信号|判定|
|---|---|---|---|
|R1|self+eta|self 200/200；eta 0/200|实际是self基线；eta没有监督信号|
|R2|+swap|swap 159/200|真实启用，但相对R1负收益|
|R3|+shared|shared 159/200|真实启用，并取得最大正增量|
|R4|+latent-cycle|latent-cycle 159/200|真实启用，但相对R3退化|
|R5|+basic drop-f necessity|transplant汇总项110/200，均值约0.05|basic necessity真实启用，小幅恢复LEO|
|R6|+targeted transplant能力|`active_fingerprint_pairs=0`持续200/200|定向移植未获得合法跨TX配对，不能归因|
|R7|+full-physics Decoder+phys|phys 110/200；后50轮均值0.0339|物理解码及约束真实启用，净增益近零|
|R8|+factor+three-axis|factor 110/200；后50轮均值0.0734|三轴/去相关汇总信号真实启用，LEO小幅增加|

另有三个重要解释边界：

1. `[FCR] id=0`不等于没有身份监督。身份CE在主`core.loss_cls`中通过`fcr_identity_head(z_f_id)`计算，因此未重复计入FCR分量。
2. eta持续为0，是因为当前pair中的nuisance监督有效掩码没有提供可用值；所以不能把R1收益解释为eta回归收益。
3. R6虽然打开targeted transplant开关，但当前批次缺少公共preamble/content-window元数据，严格fingerprint pair没有形成。日志累计记录每行`missing_common_preamble_metadata=19600`、`missing_content_window_metadata=19600`，`active_fingerprint_pairs=0`。这正是设计报告要求“同内容、不同TX”配对尚未闭合的直接证据。

### 逐行增量效果

|新增步骤|Δclean|ΔLEO均值|解释|
|---|---:|---:|---|
|R1-ADV3B02|+1.9048pp|+2.7315pp|FCR重构族整体起点有明显收益|
|R2-R1|-0.2125pp|-0.1794pp|单独加入swap无正收益|
|R3-R2|+0.5244pp|+0.7111pp|shared是本批最大、最一致的正向增量|
|R4-R3|-0.4893pp|-0.4274pp|latent-cycle在当前权重下过约束或扰动分类表征|
|R5-R4|+0.0387pp|+0.2595pp|basic drop-f necessity恢复部分LEO性能|
|R6-R5|+0.4673pp|-0.0623pp|clean升、LEO降；targeted pair为0，不能归因于定向移植|
|R7-R6|-0.1060pp|+0.0214pp|full-physics/phys基本持平|
|R8-R7|-0.1173pp|+0.0952pp|factor/three-axis轻微换取LEO、牺牲clean|

消融不是独立重复试验而是单seed累加路线，因此小于约0.1pp的差异应视为方向提示，不应作稳定机制结论。R3的+0.7111pp LEO增量较清晰，但仍需多seed确认。

### 类别层面：总体提升掩盖了TX1退化

|TX|ADV3B02 LEO均值|R3 LEO均值|R3-基线|R8 LEO均值|R8-基线|
|---:|---:|---:|---:|---:|---:|
|0|56.3417%|64.8714%|+8.5298pp|65.3357%|+8.9940pp|
|1|33.9155%|33.4214%|-0.4940pp|31.6440%|-2.2714pp|
|2|44.7345%|52.4857%|+7.7512pp|52.2631%|+7.5286pp|
|3|59.0298%|59.5107%|+0.4810pp|59.9524%|+0.9226pp|
|4|81.2310%|83.2417%|+2.0107pp|82.5583%|+1.3274pp|
|5|85.5857%|86.8869%|+1.3012pp|87.9833%|+2.3976pp|

R3的LEO总体提升主要由TX0和TX2贡献，TX1并未改善。R8进一步提高TX0/TX5，却把TX1降至31.6440%。因此“LEO均值提高”不能等价为“所有发射机鲁棒性提高”；若部署关注最弱设备，现有R3/R8都没有解决关键短板。

### 收敛、稳定性与资源

|行|最佳source-val（epoch）|E200 source-val|末20轮source-val均值±标准差|总epoch时间|unsafe skip|
|---|---:|---:|---:|---:|---:|
|R1|98.7000%（E154）|98.6778%|98.6731%±0.0094pp|2.467h|9|
|R2|98.6815%（E161）|98.6556%|98.6500%±0.0054pp|2.069h|9|
|R3|98.6963%（E190）|98.6889%|98.6841%±0.0094pp|2.079h|9|
|R4|98.6963%（E179）|98.6815%|98.6744%±0.0074pp|2.094h|9|
|R5|98.6852%（E166）|98.6556%|98.6469%±0.0070pp|2.243h|9|
|R6|98.7037%（E173）|98.6741%|98.6802%±0.0069pp|2.230h|10|
|R7|98.7185%（E154）|98.7000%|98.6961%±0.0096pp|2.768h|8|
|R8|98.6963%（E122）|98.6667%|98.6728%±0.0095pp|2.281h|8|

各行仅有8-10个unsafe backward/step被保护性跳过，约占9800个主训练batch的0.08%-0.10%；4个集中在E1，其余离散分布，未形成连续故障，最终结构化指标均有限。它们不是技术失败，但说明训练并非完全无数值保护触发。日志中显示的`nan`主要是关闭模块的余弦/测试占位值，不进入总loss；不能把这些占位符误报为训练NaN。

ADV3B02观测wall time约6.26h、峰值CUDA allocated约9.51GiB、reserved约9.79GiB；FCR各行总epoch时间为2.069-2.768h。由于ADV3B02与FCR的loader路径、并发GPU占用和计时口径不同，这些时间只能作为本次资源记录，不能据此声称FCR比ADV3B02更快。FCR行没有生成独立resource summary，因此其峰值显存为`UNKNOWN`；不能从启动时显存快照替代峰值统计。

### 设计报告对应关系与未闭合项

|设计要求|代码实现|本批运行证据|状态|
|---|---|---|---|
|同物理片段clean/LEO配对|同步pair builder与严格mask|self/swap/shared/cycle有非零信号|已实现并启用|
|物理顺序Decoder|content→fingerprint→channel/RX/sync/gain|R7/R8 phys非零|已实现并启用|
|激励条件化指纹算子|固定响应基+小残差，content excitation detach|R1-R8均经FCR前向|已实现并启用|
|结构化低容量nuisance|33维分块统计编码，无逐采样skip|代码和checkpoint存在|已实现并启用|
|异方差噪声建模|有界variance head+complex NLL|self/swap重构使用|已实现并启用|
|latent cross-cycle|交叉生成后重新编码|R4-R8非零|已实现并启用，但本批负增量|
|basic necessity|drop-f margin，干预阶段交替冻结Decoder|R5-R8汇总项非零|已实现并启用|
|定向跨TX移植|target-id/preserve-s/preserve-n/same-f/drop-f|严格pair为0|已实现但本批未实际激活|
|三轴干预立方体|nuisance/content/fingerprint严格索引|R8 factor汇总非零，但fingerprint轴缺失|部分激活|
|nuisance参数监督|eta head与有效mask MSE|eta 0/200|配置存在但本批未激活|
|Fisher门控物理特征|冻结特征bank+Fisher gate|R7/R8 phys非零|已实现并启用|
|独立latent/probe诊断|`phase1_fcr_diagnostics.py`已实现|run目录没有`fcr_diagnostics.json`|无运行证据|

最后一项很关键：当前run没有生成任何R1-R8的`fcr_diagnostics.json`，因此无法报告`z_f`的TX/receiver probe、`z_n`的TX泄漏、clean/LEO latent距离、谱条件数或移植目标身份成功率。最终分类性能可以确认，但“纯内容/纯指纹/纯nuisance”不可辨识性声明仍然没有被实验验证。

### 最终科学结论与后续优先级

1. **性能层面**：R1-R8全部高于同协议ADV3B02的四场景均值；R3是当前单seed综合候选，R6只适合作为clean最高记录，R8未取得综合冠军。
2. **机制层面**：最强正证据来自R3的shared一致性；swap单独、latent-cycle、full-physics和factor的边际效果分别为负、负、近零和小正。不能用R6宣称targeted transplant成功。
3. **类别层面**：FCR改善的是部分TX，尤其TX0/TX2；最弱TX1没有改善，R8甚至进一步退化。
4. **协议层面**：本批truth-last闭环合格，当前结果比历史高分截图更可信；历史v6数据只能作为不同协议的旧内置评测。
5. **晋级边界**：R3可进入同协议多seed确认候选，但当前只有seed392005，不能声明稳定胜出或升级默认ADV3B02。
6. **修复优先级**：先补公共preamble/content-window元数据，使fingerprint pair真正非零；再补可用eta监督；随后输出独立FCR diagnostics并针对TX1加类别级保护。完成这些之前，不应扩大R6-R8的物理可解释性表述。

本次全面复核使用的机器数据仍以[baseline_eval_v2_score.json](baseline_eval_v2_score.json)和[r1r8_target_eval_v2_summary.json](r1r8_target_eval_v2_summary.json)为准；上述机制激活统计来自R1-R8全部训练stdout，未用target结果反向选择checkpoint。

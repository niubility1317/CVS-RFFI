# ADV3B02→FCR R1-R8重新发布v4预登记

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`
- 替代原因：v3首轮遥测在非MUSE路径访问未初始化`rc4_route`；v4仅初始化可选遥测变量。
- 当前只启动seed392005的`ADV3B02_CORE90_SOFT_E200`；R1-R8等待其E200 `final_ssdg.pth`。
- source：ManySig equalized；receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；`L_s/U_s/V=6300/56700/27000`。
- target训练边界：`test_eval_policy=never`；不训练、不筛选、不选择checkpoint；最后epoch冻结后才独立预测与评分。
- checkpoint：final-only；`best_metric=clean_val_tx`仅满足source-only兼容检查，held-out joint guard关闭。
- GPU：ADV3B02使用GPU0；不干预既有任务。R1-R8当前未启动。
- Git固定版本：`8f1de7971853aa9650e4f83d6ad979f359c434c2`。
- release：本地`E:\type10-7\releases\phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_8f1de797.zip`；远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_8f1de797`；归档SHA256本地/远端一致：`6D8484618441CE2E8470556864C82D8DF64A3BD8E8D826A5D5713428CFE40B10`。
- 发布前验证：远端编译通过；真实checkpoint无query smoke通过；真实训练入口dry-run通过。
- 启动命令：`bash code/scripts/launch_phase1_adv3b02_fcr_r1r8_s392005_20260903.sh phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`。
- 远端CWD：`/home/szu2070436088`；launcher PID=`381957`；ADV3B02训练PID=`382450`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200.out`。
- 启动健康检查：PID/CWD/cmdline/run-root绑定正确；PID`382450`绑定GPU0，显存约2802MiB；日志已增长至17991字节并完成E002/200，未发现Traceback。中间checkpoint标记为`NOT_SAVED_FINAL_ONLY`，R1-R8仍未启动，等待训练结束后写入的E200 `final_ssdg.pth`。
- 当前状态：`RUNNING`。

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
|R1|self reconstruction+eta|78.1315%|64.1256%|62.3679%|62.1202%|62.8712%|62.1202%|66.6863%|
|R2|+swap|77.9190%|63.8333%|62.1863%|62.0560%|62.6919%|62.0560%|66.4987%|
|R3|+shared|78.4435%|64.6369%|62.8964%|62.6756%|63.4030%|62.6756%|67.1631%|
|R4|+latent cycle|77.9542%|64.2768%|62.4488%|62.2012%|62.9756%|62.2012%|66.7202%|
|R5|+basic need diagnostic|77.9929%|64.4536%|62.7238%|62.5280%|63.2351%|62.5280%|66.9246%|
|R6|+targeted transplant|78.4601%|64.4024%|62.6607%|62.4554%|63.1728%|62.4554%|66.9946%|
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

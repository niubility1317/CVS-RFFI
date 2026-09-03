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

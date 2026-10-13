# 频域上下文：原头 clean 三种子完整结果

全部9模型和1512000条预测、162项指标已完成并独立核验。主候选GLOBAL_FREQ均值79.4298%，低于历史旧伪标签门控＋EMA＋cos基线80.6454%；84%目标未达成。

| 方法 | seed01 | seed02 | seed03 | clean均值±SD | Macro-F1均值 | 最差RX均值 |
|---|---:|---:|---:|---:|---:|---:|
| LEGACY50 | 80.7417% | 78.5071% | 82.6875% | 80.6454% ± 2.0918 | 80.1381% | 67.8611% |
| LOCAL_FREQ | 79.3173% | 77.1274% | 82.1613% | 79.5353% ± 2.5240 | 78.9676% | 66.4861% |
| GLOBAL_FREQ | 78.9226% | 76.9786% | 82.3881% | 79.4298% ± 2.7402 | 78.7944% | 66.4486% |

LOCAL_FREQ−LEGACY50配对均值−1.1101个百分点，3/3种子负；GLOBAL_FREQ−LEGACY50为−1.2157个百分点，3/3负。GLOBAL_FREQ−LOCAL_FREQ为−0.1056个百分点，1/3正。固定矩阵全部保留，不以测试结果重排、改参数或选择性重跑。

实际数据：ManySig raw/post_sync/noeq，源RX索引1/3/4/6/8、day1/2/3、全部6TX；split392005，L6300/U56700/V27000，7/63/30。训练U真值隐藏。固定clean capsule168000条/row，物理ID与源角色不交；本轮仅clean，无K/support，不声明LEO部署表现。

实际配方：9模型从零50轮×222=11100更新，前32轮label、后18轮pseudo；RX/day quantile .86、tau .92至.97、batch_neighbor窗口2/minconf.8、strong agreement；EMA .999 versioned，无startup average；lambdaU .16、entropy .01、label smoothing .01、ordinary augmentation开启、无卫星训练；AdamW LR.0002→.000001 cosine50、WD.0001。LEGACY50身份177025参数，局部/全局198273；分类器仍6×160普通cosine/scale30。

完整核验：450轮、99900条逐step记录、9185400次U包前向、148082个接受伪标签（1.6121%），EMA更新连续，全部记录有限且无梯度/损失跳步，U真值可用0。每checkpoint50/11100、scratch祖先为空、终轮SOURCE_FROZEN固定。逐行168000唯一opaque ID预测、6类整数、18项混淆矩阵准确率/Macro-F1重算一致。audit仅重算已独立评分矩阵，不读truth。

训练总进程时间6.1491小时，各模型39.45至42.64分钟。并行进程时间之和不是wall时间/GPU独占时间。修复恢复追加训练0。

训练run：20261013-phase1-frequency-context-legacycos-clean-manysig-m9-r01；执行代码43bd20494aa470568a68553ce664dc76811b9707。9行训练结束后预测统一遇到baseline_origin_sat_view反序列化依赖错误，无原预测/评分。原失败目录、权重、日志保留，不覆盖、不停止其他owner。

评估恢复run：20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01；执行代码9ea84c4d22076532dd613a28d1b2da3988591feb。修复native_modules先于torch.load；真实终轮逐rowCPU无query smoke PASS。一archive本地/远端SHA对比、compile及普通用户preflight PASS，dispatcher1570590的CWD/argv和完整预测/评分独立读回VERIFIED。原发布inspector的字符串替换/launch额外字段格式故障不影响任务，已修本地读取且未重复发布。恢复输出新目录，与原训练source_proof严格同config。

证据位置：N607 runs/20261013-phase1-frequency-context-legacycos-clean-manysig-m9-r01/<row>/source/{final_ssdg.pth,epoch_metrics.jsonl,step_metrics.jsonl.gz,source_frozen.json}；runs/20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01/<row>/{prediction,scores,completion.json}与analysis.json/completion.json。本地 local_artifacts/frequency_context_recovery_delivery/{readback.json,post_launch_independent.json,full_audit.json}。原始IQ、权重、预测和日志不进入Git。

声明边界：以上为已暴露基准上的Phase1原头零适配clean性能，不是新独立盲测，也不能证明丢头后的特征可迁移性。后续按用户新说明固定joint160的类别无关边界，另做源域重头与合法Phase2 support-only/旧新类统一验证。

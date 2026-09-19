# practical四组最新保存权重测试r02

固定2026-09-19 20:05快照：full_noeq E100、MMSE E100、residual E160；ZF尚无checkpoint，等待首次E100。复用既有自动测试预测并独立CPU评分。

r01评分程序遗漏periodic row ID的epoch后缀，技术失败；r02仅修复精确身份校验，不重新推理。原训练健康运行。等待full_noeq E100测试完成及ZF首次E100，已保存MMSE E100和residual E160优先复核。

## 接续测试与评分状态：RUNNING／VERIFIED

CPU评分PID3932760，独立读回argv/CWD及CUDA_VISIBLE_DEVICES为空；不增加GPU任务。固定快照后不追逐新epoch。full_noeq E100测试中，ZF尚无checkpoint待首次E100；MMSE E100与residual E160已有预测复核完成。此次没有重新推理；剩余预测由原训练已配置的E100测试产生。评分脚本24小时内每60秒检查一次，全部完成或出现终止条件后退出。

|组|epoch|状态|clean Acc/F1|high Acc/F1|mid Acc/F1|low urban Acc/F1|
|---|---:|---|---|---|---|---|
|DAOT_RC4_PRACTICAL_FULL_NOEQ_s392005|100|WAITING_FOR_SCHEDULED_TEST|待完成|待完成|待完成|待完成|
|DAOT_RC4_PRACTICAL_FULL_ZF_s392005|100|WAITING_FOR_SCHEDULED_TEST|待完成|待完成|待完成|待完成|
|DAOT_RC4_PRACTICAL_RESIDUAL_NOEQ_s392005|160|VERIFIED|79.6185%/79.2561%|80.1798%/79.9528%|77.9500%/77.7206%|53.3946%/52.7912%|
|DAOT_RC4_PRACTICAL_FULL_MMSE_s392005|100|VERIFIED|81.0488%/80.7245%|81.1071%/80.9066%|78.9190%/78.5735%|53.3512%/52.0895%|

每组每场景168000条物理样本；已完成两组各672000条冻结预测，逐ID无重复/遗漏，6类预测合法，checkpoint路径、source-only scratch来源及信道配置匹配，复算正确数与原score.json完全一致。全部混淆矩阵与每类F1见[evidence/latest_readback.json](evidence/latest_readback.json)。各组epoch及增强模型不同，不能据此作均衡优劣或同预算的严格因果对照。中间目标测试已曝光，结果属于探索性测试而非盲测确认。

服务器输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20260919-phase1-daot-rc4-practical4-test-s392005-r02`；每组`<row_id>.json`，汇总`state.json`，终态`completion.json`。服务器日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/20260919-phase1-daot-rc4-practical4-test-s392005-r02/recount.log`。训练r03继续，不改变现有200epoch预算及训练状态。

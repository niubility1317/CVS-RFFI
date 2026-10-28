# Phase3 多设备源辅助暴露数据

2026-10-29。数据候选 `20261029-phase3-source-oe-diverse-data-r01`。前三个完整六类源开发候选尚未达到95%拒识；本次先准备更多源辅助设备，模型训练和性能另行登记。本候选不改原L/U/V、源身份模型、三个既有源novel验证设备或正式目标池。

既有元数据已确认Full174TX比四compact并集150TX多24台。保留19-15、19-16、7-1只做源novel验证；剩余21台作为新辅助训练角色，包含原4台暴露设备和另外17台。都在原source5RX、D1–D3内，排除旧六类和正式四个未知TX。选择只依赖已验证元数据与固定seed2026102906，不按IQ质量、模型置信或评分挑选。

每TX/RX/day单元最多32条，少于32全取，零容量为空；不复制稀少记录、不筛掉4–7条小单元。预登记6134唯一物理记录，315单元中220非空／95空，原4台贡献1920、新17台贡献4214。按RX原顺序1-19/18-2/19-2/2-19/3-19共588/1936/983/630/1997条；D1/D2/D3为2217/1889/2028。RX18-2和3-19合占64.1180%，不能把这一不平衡隐去。

后续训练拟按每个role的RX/day域均匀1/15加权，OE每条质量为`1/(15*n_domain)`，known L对应域也是1/15；它不需要辅助TX逐条身份。此处只保存真实记录与元数据，不复制或预加权IQ，不因容量计算就声明该训练已实现。成员“非旧六类”仍是新增源监督，目标端TX标签仍为0。

复用已有158PKL原始Full验证，不重扫20ZIP或新增成员hash。只读取所需15个source容器并选21台的合法行；容器中的未选TX不进入输出、拟合或源选择。采用原`iq_record`的256点FP32、2×256转置和一次unit RMS，不重裁剪，不生成channel，不运行F/G，不读取正式RX24 U/Q或三个novel设备的IQ。

新不可覆盖输出只有`exposure_train`角色：public精确`iq.npy`、`ids.npy`、`metadata.npz`（rx_i/day_i/domain）；TX只放private truth/builder report。complete使用独立`SOURCE_OE_DIVERSE_DATA_VALIDATED`与`VALIDATED_ONCE`、capsule/split/count最小句柄，不能静默冒充旧4200数据。原4台跨实验可能物理复用，按各独立角色实际唯一6134计数，不宣称新增6134条全部未使用历史样本。

本机CPU数据准备，新增GPU工作进程为0，无模型/训练/预测/评分；保留所有其他用户任务。一次新builder P0/P1、相关合成测试、实际代码提交、精确进程与产物读回留在同名最小登记。实际数据已VERIFIED/VALIDATED_ONCE；源95%拒识和完整Phase3目标均未完成，本数据不提供性能结果。

发布前验证：新builder与CLI已完成，5项测试和5个subtests通过（原4项数据测试＋新增CLI观测测试）；一次独立P0/P1审查PASS并独立运行当前5项全部通过。CLI首行只输出实际PID/CWD/argv的运行JSON，供原生进程信息与日志读回核对，不安装新依赖。真实构建已完成，模型性能尚无新证据。

## 实际构建结果（2026-10-29）

run `20261029-phase3-source-oe-diverse-data-r01` 已ANALYZED，实际代码`2f41fc2835798f583c54900ab4c791d762d0101c`。6134条唯一记录、220非空／95空cell，原4台1920＋新17台4214；15个source容器实际读取完成。capsule=`source-oe-diverse-9dd73fa533b19a64fd4db8d9`，split=`5e0f620d5c7c72512e9e9edd`。完整18行／1467B日志无异常，独立poststate为VERIFIED、进程已退出；只复用既有Full验证，无全量重扫／原始hash。原L/U/V、novel验证IQ、模型、channel、正式target读取／调用为0，新增GPU及foreignstop为0。额外source成员监督明确保留，未训练／预测，不构成拒识或Phase3达标。

[完成标记](E:/type10-7/local_artifacts/pamf_phase3_source_oe_diverse_data_20261029_r01/data/complete.json)；[实际builder report](E:/type10-7/local_artifacts/pamf_phase3_source_oe_diverse_data_20261029_r01/data/builder_report.json)；[完整日志](E:/type10-7/local_artifacts/pamf_phase3_source_oe_diverse_data_20261029_r01/build.log)；[独立poststate](E:/type10-7/local_artifacts/pamf_phase3_source_oe_diverse_data_20261029_r01/build_post_state.json)；[摘要](E:/type10-7/local_artifacts/pamf_phase3_source_oe_diverse_data_20261029_r01/analysis_summary.json)。

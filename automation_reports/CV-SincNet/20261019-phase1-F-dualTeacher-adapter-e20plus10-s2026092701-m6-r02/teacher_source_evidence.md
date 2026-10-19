# 既有同 IQ 源教师证据

本轮复用已完成的固定源观测，不新增模型前向、checkpoint/IQ 读取或 target 访问。比较学生是纯 F 普通 As 的 B1 E20；教师是固定 CSIL last200。用于确认教师互补性，不把该优势作为学生可达上界。

固定 360 个源 V 物理样本，90 个 TX×RX×day 单元各按原顺序取前 4 个；每场景 2 次信道实现，每场景 720 条观测，共 1440 条。物理 ID 与 TX/RX/day 精确连接，模型种子数为 1。

|场景|分层|n|教师准确率 %|B1 准确率 %|修复|新增错|净差（百分点）|
|---|---|---:|---:|---:|---:|---:|---:|
|practical_mid_urban|overall:ALL|720|86.2500|75.1389|109|29|11.1111|
|practical_mid_urban|state:LOS|376|98.4043|95.4787|12|1|2.9255|
|practical_mid_urban|state:PARTIAL|73|98.6301|93.1507|4|0|5.4795|
|practical_mid_urban|state:BLOCKED|271|66.0517|42.0664|93|28|23.9852|
|practical_mid_urban|lock:locked|439|98.4055|95.2164|15|1|3.1891|
|practical_mid_urban|lock:degraded|140|80.0000|57.8571|42|11|22.1429|
|practical_mid_urban|lock:unlocked|141|54.6099|29.7872|52|17|24.8227|
|practical_mid_urban|quality_snr_db:<8|225|61.3333|37.7778|78|25|23.5556|
|practical_mid_urban|quality_snr_db:8_to_<16|63|90.4762|68.2540|17|3|22.2222|
|practical_mid_urban|quality_snr_db:>=16|432|98.6111|95.6019|14|1|3.0093|
|practical_mid_urban|RX:1|144|88.1944|72.2222|25|2|15.9722|
|practical_mid_urban|RX:3|144|81.9444|70.8333|22|6|11.1111|
|practical_mid_urban|RX:4|144|84.7222|76.3889|21|9|8.3333|
|practical_mid_urban|RX:6|144|88.1944|81.2500|19|9|6.9444|
|practical_mid_urban|RX:8|144|88.1944|75.0000|22|3|13.1944|
|practical_mid_urban|day:1|240|85.0000|74.1667|35|9|10.8333|
|practical_mid_urban|day:2|240|84.5833|75.4167|37|15|9.1667|
|practical_mid_urban|day:3|240|89.1667|75.8333|37|5|13.3333|
|practical_low_urban|overall:ALL|720|78.1944|62.7778|148|37|15.4167|
|practical_low_urban|state:LOS|211|97.6303|96.6825|3|1|0.9479|
|practical_low_urban|state:PARTIAL|112|98.2143|89.2857|11|1|8.9286|
|practical_low_urban|state:BLOCKED|397|62.2166|37.2796|134|35|24.9370|
|practical_low_urban|lock:locked|269|97.3978|95.1673|8|2|2.2305|
|practical_low_urban|lock:degraded|123|88.6179|67.4797|26|0|21.1382|
|practical_low_urban|lock:unlocked|328|58.5366|34.4512|114|35|24.0854|
|practical_low_urban|quality_snr_db:<8|389|62.2108|36.5039|135|35|25.7069|
|practical_low_urban|quality_snr_db:8_to_<16|70|95.7143|85.7143|7|0|10.0000|
|practical_low_urban|quality_snr_db:>=16|261|97.3180|95.7854|6|2|1.5326|
|practical_low_urban|RX:1|144|78.4722|61.1111|30|5|17.3611|
|practical_low_urban|RX:3|144|76.3889|57.6389|32|5|18.7500|
|practical_low_urban|RX:4|144|78.4722|67.3611|29|13|11.1111|
|practical_low_urban|RX:6|144|77.0833|65.9722|22|6|11.1111|
|practical_low_urban|RX:8|144|80.5556|61.8056|35|8|18.7500|
|practical_low_urban|day:1|240|76.6667|62.9167|46|13|13.7500|
|practical_low_urban|day:2|240|78.7500|63.3333|52|15|15.4167|
|practical_low_urban|day:3|240|79.1667|62.0833|50|9|17.0833|

quality 使用 `quality_snr_db`，为白输入传播质量代理，单位 dB，固定分箱 <8、8 至 <16、≥16；不冒充实测总 SNR。所有分层完整保留，不择箱、筛样或拟合温度。

来源：既有 `20261019-diagnostic-F-CSIL-inputJacobian-s2026092701-m1-r01` 的 source_derivatives 与既有噪声诊断 panel_metadata。原 inputJacobian 损失路线已经关闭，不据此重新开放；本轮差异是分视图教师监督与新增浅中层特征容量。完整无损统计位于本轮 local_artifacts/teacher_existing_source_strata.json。历史 V1800 对另一 F+CROSS_RX 学生的教师资格结果与本次 V360 证据分别命名，不拼接。

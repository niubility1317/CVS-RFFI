# SF-TAPFT H6部署化与HardPair设计追踪

来源：用户提供的《H6 Fast-Strong V3星上轻型适配优化报告》，2026-08-27；用户于2026-08-28确认实施方案。状态：`LOCAL_VERIFIED_EXPERIMENT_PENDING`。

|ID|设计要求|状态|计划落地与证据|
|---|---|---|---|
|H6D-01|新增固定full-support部署入口，不执行4-fold或研究validation|local_verified|runner行为测试证明folds=0且无query能力|
|H6D-02|H6固定head+t3.norm(w+b)、300+150+70日程|local_verified|冻结matrix及config解析测试|
|H6D-03|FP32前缀缓存logit与许可参数梯度误差均小于1e-5|local_verified|Toy和真实checkpoint均为logit差0、梯度差0|
|H6D-04|FP16缓存与严格等价分开判定|local_verified|真实checkpoint FP16输出有限；性能待R0B16|
|H6D-05|严格delta-only FP16部署包不超过10KB|local_verified|runner测试delta小于10KB；真实字节待矩阵|
|H6D-06|M02使用新引擎、all-time norm和历史固定步数公平复跑|configured|历史artifact恢复327步，R1待运行|
|H6D-07|HardPair完全由support自动发现困难类别对|local_verified|类别置换不变和缺类fail-closed测试通过|
|H6D-08|只测试lambda 0.03和0.05，其他H6设置不变|configured|R2A/R2B仅覆盖hard_pair_weight|
|H6D-09|最大Q180在prediction闭合后truth-last评分|pending|独立prediction/scoring artifact|
|H6D-10|报告BA、floor、逐类准确率、NLL、ECE和配对翻转|pending|最终报告|
|H6D-11|报告trainable/changed、forward、delta、wall、RSS和GPU峰值|pending|selection、GNU time和GPU采样|
|H6D-12|仅同时满足性能与资源门槛的最小候选晋级|pending|同row判定表|
|H6D-13|不把单receiver/scene/K/seed结果宣称为泛化完成|pending|结论边界|

## 已解决歧义

- 报告同时提出FP16前缀缓存与`<1e-5`严格等价。设计把FP32缓存作为数学等价层，把FP16缓存作为资源候选层，分别验收。
- 报告要求M02固定约300～350步，但不给出唯一整数。本轮从既有M02 selection artifact读取历史选中步数并冻结；若artifact无法给出唯一值，则R1在启动前标记技术阻塞，不用query或新CV选择步数。
- 本轮只执行R0/R1/R2最小因果矩阵。CVaR、vector scaling和完整泛化矩阵仅在R2通过后另行预登记。

# CVS与D92实验完成情况（2026-09-28）

核实时间：2026-09-28T10:36:11.004811+08:00。VERIFIED：5个Phase1均完成200轮；D92三组完成、两组技术失败，原自动评分未执行。

完整解析五个训练stdout及1000条epoch结构化记录；预测文件逐行统计。原始证据：`E:\type10-7\local_artifacts\cvs_matched_20260927\completion_full_20260928.json`。

|seed|轮数|训练耗时/h|第200轮源域V准确率/%|D92预测记录|D92状态|
|---|---:|---:|---:|---:|---|
|392005|200|20.25|98.6593|830/2121|技术失败|
|2026092701|200|19.09|98.3556|2121/2121|完成预测|
|2026092702|200|20.02|98.6741|2121/2121|完成预测|
|2026092703|200|19.51|98.5407|1968/2121|技术失败|
|2026092704|200|19.34|98.4815|2121/2121|完成预测|

四个新seed源域V准确率均值±样本标准差：98.5130%±0.1323个百分点。392005为历史参考单列。该数值不是clean目标测试成绩，也不是星地域泛化成绩。

最终checkpoint均为约15.99MB的final_ssdg.pth，第200轮；source-only、scratch、物理角色EXACT_MATCH与此前核验一致。没有用历史最佳轮替换。1000条记录中的train_loss、train_tx_acc、val_tx_acc均有限；inactive指标中的NaN/空值不等于训练损失发散。

每个seed已完成168000条clean及同ID星地预测、36036条固定received特征和新生成的v2 ground。D92版本为E0去RF32，P2-256-FULL、identity160＋FFT96，runtime配置已核实。

D92每seed应输出2100个注册划分及21个frozen DG记录，共2121条。当前9161/10605条；这些是预测记录数，不是准确率。未完成seed不能剔除后当作五seed结果。

两个失败均为 `D42 sklearn coefficient deployment prediction drift`。调用栈定位于support内LOO组件拟合进入D42 LDA，显式求解的系数转为float32后，其预测与sklearn estimator.predict不一致而抛异常。已确认触发位置；精度舍入、近似平局或求解器差异的具体根因尚未复现，不能断言。

原队列在全部五lane完成之前不打开truth，因此phase1_final_results.json、scored_results.json及总completion.json均不存在。当前没有正式clean/星地准确率、D92旧新类准确率或相对对比方法的优势结论。所有checkpoint、完整/partial预测和日志保留。

下一步应使用失败split的support复现数值一致性问题，不读取query truth，不通过简单关闭检查来掩盖差异。模型训练已完成，恢复评估通常不需要重训，但具体修复与重跑范围须由复现决定。此次结果检查未启动新实验。

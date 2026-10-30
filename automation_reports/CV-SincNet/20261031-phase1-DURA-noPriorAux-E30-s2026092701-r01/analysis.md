# 取消中间 prior IQ 拟合：30轮完整结果

本版未改善星地识别。clean比raw仅多1条正确，七景均低于同预算E30 physical读出；六星地均低于默认DURA，逐景3pp与原R17每景1pp容忍均未通过。默认保持。

## 正式七景与分组

固定原F50/defaulthead，168000物理样本×7景，rawclean84.6761904762%。每个角色全七景预测固定后独立truth-last；额外独立复算126个总体/TX/RX/day记录、MacroF1、纠错/损伤计数一致。完整分组、混淆矩阵与IQ误差见results.json。单seed、既有暴露工程基准，不作新盲测或多seed稳定收益声明。

|场景|本版准确率%|相对E30 physical（百分点）|相对默认（百分点）|距raw clean（百分点）|
|---|---:|---:|---:|---:|
|clean|84.676786|-0.165476|-0.001190|-0.000595|
|practical_high|79.888690|-0.129167|-2.936310|4.787500|
|practical_mid|78.726190|-0.014286|-2.834524|5.950000|
|practical_low_suburban|77.264881|-0.136310|-2.793452|7.411310|
|practical_high_urban|78.910119|-0.051190|-2.835714|5.766071|
|practical_mid_urban|72.676786|-0.174405|-2.809524|11.999405|
|practical_low_urban|66.020238|-0.686905|-2.869643|18.655952|

clean纠回386条、损伤385条，净+1。lowurban相对默认修复12980条、损伤17801条，净-4821条。clean保持=True，六景3pp=False，R17每景最多下降1pp=False，三个条件分别判断。

## 已实际执行的消融

仅在noVar基础上将priorIQ辅助拟合权重0.1设0。两个prior网络、四CG和全部最终IQ梯度路径保留；最终IQ、STFT0.05、sourceaction0.05不变，variance辅助权重延续0。prior误差仍计算并记录，不能将其增长误判为训练故障。该版不是删去最终求解的历史prior-only运行路径。

从零单阶段30轮/7410步、L189000/U1701000曝光，20份原腐蚀缓存后复用01至10；没有学习教师、G权重继承、F/TX/V/target训练或另训头。完整30条加权loss复算等于IQ+.05STFT+.05action，未隐含prior/variance项。

最终IQNMSE 0.02289425，priorIQNMSE 0.25067545，action误差 0.01885538。中间prior不必长得像clean的源观察成立，但直接删去这一辅助约束并未在本次从零训练中带来正式收益；旧默认跨模型/预算比较不能提供该损失单项的因果结论。

![完整30轮与正式七景](E:/type10-7/automation_reports/CV-SincNet/20261031-phase1-DURA-noPriorAux-E30-s2026092701-r01/training_and_formal.png)

## 实际效率与部署成本

训练循环和保存 1001.619769秒（16.694分钟）；控制器到完整分析 1183.545899秒（19.726分钟），其中七景target预测 155.837987秒。source/model准备没有独立计时，不伪填0，已包含全流程。

按实际前序worker/scorer句柄退出放行，检查 0.055396秒；含清理的评估队列阶段 0.211000秒，marker-only=False，没有停止或热改前序。训练峰值allocated 1086765056字节。固定原模拟缓存与F既有训练属于共同上游成本，未在本次重训。

图仍266951参数、1067804参数字节、checkpoint 1305444字节，Conv/Linear MAC25068072、2prior/4CG/27FFT-IFFT。GPU6同卡交替G+F B1中位默认 5.235480ms、本版 5.145547ms；完整P95/吞吐/B128见profile。共享负载不作跨run因果加速比较，显存profile包含共同模型和图，不当独立部署显存。

E30 physical是同预算优化参照，其物理参数没有F梯度并独立裁剪，但整个原进程含F指导增强分支与成本，不冒充一场独立F-free训练。源机制研究读过F，实际本版训练没有F访问，两者分别披露。源V固定1800面板，不扩为整个V。

## 验证与边界

执行 eb46cab8ef9c253c193a10db1724ed9835ed2cf9；6项聚焦检查、唯一P0/P1、真实2Ls/source梯度/严格重载/原头no-query smoke已有PASS记录。完整日志无Traceback/OOM/非有限数，controller1104507/worker1104599自然退出。其他owner未干预。此次完整负结果保留，不回流更改轮次、prior权重或选择性重跑；默认与六景3pp/clean/效率目标保持ACTIVE。

# 固定 SIG_TIME：接收机 CE 风险方差源端证伪

状态 PLANNED，代码 `a9b04efa1441fce2f0a2810e4e51c88b41532dde`。网络和现有默认分类头不变，0模型/头/optimizer更新。完整配置与唯一row `SOURCE_VREX-s2026092701` 见 experiment.json。

## 机制与边界

[REx 原文](https://proceedings.mlr.press/v139/krueger21a.html)提出通过训练域风险差异控制考察分布外泛化；本轮只检验其简化风险方差目标的源端骨干导数，不移植论文收益或因果保证。既有GroupDRO改变RX风险权重，Fishr匹配参数梯度统计；此处用同一次默认前向计算五个源RX的身份CE均值R_e，再计算 mean_e(R_e−mean(R))²（population variance），系数固定1。匹配CE为五域CE均值，smoothing=.01，不另训分类器或引入第二网络。

既有风险加权没有建立稳定目标收益，不代表风险方差一定有效。方差梯度也可能通过提高简单RX损失来缩小差异；因此先检查其与身份CE的实际导数兼容性，不能把风险变齐称为特征改善。另一对话已推进E20 R-Drop，本对话保留自身三模型源结果、不重复正式R-Drop训练；历史VAT矩阵复用不重跑。

## 固定数据与预算

父固定原DRO首SIG_TIME seed2026092701 final50，原执行 `977d4fb609436bac23621ef8e32c2017495acef7`；原scratch50/33L+17SSL/11100updates/旧门控+ownEMA+cos。运行加载前核对既有数据契约和完整来源，不用Sinc变体或按目标挑父。复用VALIDATED_ONCE源契约：L6300/U56700隐藏TX/V27000、标注率10%、split392005、源RX1/3/4/6/8、day1/2/3。固定既有前五L180cohort，每TX/RX/day每批2包，共900唯一物理L。U张量和真值不读，V缓存返回即丢弃，无V或目标前向。

一次原生train-mode strong .015 IQ前向、原dropout有效；strong/dropout RNG2026101529+cohort。预算900包前向/5梯度批/20参数层记录/40保存向量；测t1/t2/SIG投影和现有W的CE/方差导数。真实checkpoint no-query smoke另用2合成零IQ。0训练轮、0optimizer/头拟合更新。普通N607既有CVS-RFFI，FP32 CPU2线程、CUDA_VISIBLE_DEVICES为空，不分配GPU。

源资格提前固定：5批方差各>0，三骨干层导数各5/5非零，五批平均方差导数对平均CE导数余弦均≥0，参数与buffers不变。PASS只允许另登记同公式02/03源检查；FAIL完整保留，不扫系数、分组、父权重或阈值，不自动发布训练。此末轮L代理不能证明早期训练、U正确、DG或84%。

## 执行和证据

CWD `PROJECT/releases/cvs_source_risk_variance_20261015_r01`；命令 `python -u -m experiments.cvs_source_risk_variance.audit`。独占输出 `PROJECT/runs/20261015-phase1-source-risk-variance-manysig-s2026092701-r01`。本地发布 `python -m experiments.cvs_source_risk_variance.publish --output LOCAL_ARTIFACTS/source_risk_variance_delivery_r01`；自动preflight/真实checkpoint no-query smoke通过后直接继续。单release包校验、远端编译、PID/CWD/argv及日志增长核实沿用既有流程。

预期resolved_config、实际源导数、逐批五RX风险/方差与默认准确率、原joint160、analysis、source_qualification、独立保存产物复算、completion和日志。技术停止限本run权限/来源/输入不符、非有限值、cohort缺项、状态变化、独立复算失败或输出冲突；保留产物，不自动重试、不操作无关任务。14针对性负例、风险公式、常量零方差/零导数和实际原生网络四参数导数/RNG/状态检查PASS；一次独立P0/P1审查PASS。真实远端源资格尚未知。

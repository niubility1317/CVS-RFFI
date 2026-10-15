# 固定 SIG_TIME：风险方差剩余两模型源复核

状态 ANALYZED / VERIFIED，三模型源规则PASS，详见[完整结果](final_analysis.md)。原预登记：PLANNED，实现 `c514c20a11c88f35b0154eb2a15ede7a4bc992fe`。首源PASS及不变扩展已在 `f989c97284bbd428b7ad07029050faf7ff5b24f6` 上传。这里只增加seed2026092702/03源端零更新检查，首模型复用原产物，不重新前向。完整两行配置以 experiment.json 为准。

沿用固定V-REx源公式：同一次原native train-mode strong IQ（噪声.015、原dropout）默认前向，五源RX各计算身份CE均值R_e（smoothing.01），CE参考为mean(R_e)，方差为mean((R_e−mean(R))²)，系数1。测原t1/t2/SIG投影及现有W的真实导数；0模型/buffer/头/optimizer更新，无新分类器。首源一般依据和机制局限继续适用，不把风险变齐等同泛化提升。

父固定原DRO剩余运行 `20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01` 的各自SIG_TIME final50，执行a283291279cb01b939325cf5ccba1d0a671b28c4；原scratch50/33L+17SSL/11100updates/旧门控+ownEMA+cos，不用Sinc或目标最佳权重。实际来源核验在加载前执行。复用VALIDATED_ONCE源契约：L6300/U56700隐藏TX/V27000、标注率10%、split392005、源RX1/3/4/6/8/day1/2/3；相同有序900物理L、5批180、每TX/RX/day每批2包；strong/dropout RNG2026101529+cohort与首模型相同。U张量/真值不读，V缓存返回即丢弃，无V或目标前向。

新增预算2模型/1800包前向/10梯度批/40参数层记录/80向量；加保留首模型共3模型/2700前向/60层120向量，实际唯一物理样本仍900。0训练/optimizer/头拟合更新；真实checkpoint no-query smoke另用2合成零IQ。普通N607既有CVS-RFFI，FP32 CPU2线程、CUDA_VISIBLE_DEVICES为空，不分配GPU。

每模型保持原规则：5批方差均>0；t1/t2/SIG方差导数各5/5非零；五批平均方差导数对平均CE余弦各≥0；参数和buffers不变。三模型均PASS才支持另登记正式训练设计，不自动发布训练。FAIL保留，不扫系数、分组、父模型或阈值。末轮L代理不证明早期训练、U正确、DG或84%。

CWD `PROJECT/releases/cvs_source_risk_variance_seeds_20261015_r01`；命令 `python -u -m experiments.cvs_source_risk_variance_seeds.audit`。不可覆盖root `PROJECT/runs/20261015-phase1-source-risk-variance-manysig-s0203-m2-r01`，两row与owner见experiment.json。本地发布 `python -m experiments.cvs_source_risk_variance_seeds.publish --output LOCAL_ARTIFACTS/source_risk_variance_seeds_delivery_r01`，自动preflight/真实checkpoint无query smoke通过即继续。单发布包校验、远端编译及PID/CWD/argv/loggrowth核实沿用既有流程。技术停止仅本run输入/来源/非有限值、固定cohort缺失、模型状态变化、复算失败或输出冲突；保留产物，不自动重试、不操作无关任务。

预期保存逐row resolved_config、实际导数、analysis、独立保存产物复算、combined_three_seed_source_analysis、source_qualification、completion和日志。13个适配边界负例、首默认接口和两个原生网络真实导数/RNG/状态检查PASS；唯一独立adapter P0/P1审查PASS，未重审首公式。实际远端源资格待运行核实。

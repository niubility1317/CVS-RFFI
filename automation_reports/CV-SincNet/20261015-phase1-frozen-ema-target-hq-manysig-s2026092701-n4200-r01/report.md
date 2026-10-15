# 冻结 EMA：目标域 low urban HQ 配对验证

本次由侧对话用户明确要求：先查已有目标HQ实验，没有则启动验证。2026-10-15登记搜索HQ仅有源域L_dev720诊断；远端未发现target HQ run。本次独立工作树/branch为target_hq_diagnostic_20261015，不改动主线已有cvs_urban_bridge50代码或训练。完整配置、数据及逐row/seed/命令/输出见[experiment.json](experiment.json)。

## 固定模型与对比

使用源诊断64.86%→70.97%的同一A0 final50 checkpoint内`ema_model`和EMA自身默认cosine头。原训练为相同5580/720源内部折、旧门控33+17、E50/11100；本次0新增epoch、0 optimizer/EMA/head-fit更新。own_model严格核对实际source契约、source_frozen/completion/initialization/inner折及CP，再严格加载同CP EMA；不按目标成绩选择模型。

从原168000个固定target匿名ID以seed2026101513的固定随机排序选4200，完全不读TX/质量/损失来采样。覆盖范围由评分后实际RX/day/class计数披露；这是4200小试，非全168000结论。原full received缓存逐值保持只读；根据原manifest的low urban config、seed392005、receiver_seed2027、namespace/session重放full并逐样本比较（rtol/atol2e-6），不一致即技术失败。

四个输入条件预先固定：原full、同传播/同接收状态下独立同强度噪声、该同一独立噪声降低6dB的HQ、原共享噪声降低6dB的HQ对照。只改变模拟器新增噪声；原录音噪声保留、总SNR未知。传播/残余/EQ核、RX硬件、lock、quality代理、补偿参数、AGC增益/限幅、24点FIR及reflect边界都冻结为full动作。HQ不再做输出RMS，不增加可训练模块。EQ仍关闭G=1、refSNR30、post_sync残差路径不变。

## 预测、评分与解释

CPU2/CUDA_VISIBLE_DEVICES空；4200×4+真实CP smoke2，共16802前向。launcher先做真实EMA/原头无query smoke，通过立即继续；所有模型eval、requires_grad=False、no_grad，逐值确认state_dict不变。每个ID的四视图logits全部固定及complete后，以exec启动独立新进程scorer连接原truth；缺失/错序/非有限预测或更新标记会在truth打开前拒绝。

报告accuracy、Macro-F1、配对修复/新增错误，以及全部实际RX、day、LOS/PARTIAL/BLOCKED、lock和full新增噪声区间。同强度独立噪声对照用于区分换噪声与降噪效应；独立HQ与共享HQ分开报告。不按结果扩样、换seed或选择性重跑，不把本结果回流主线训练、候选或参数。

范围明确为`TARGET_HQ_COUNTERFACTUAL_NOT_FORMAL`。即使HQ目标准确率更高，也只证明较好输入下冻结EMA的表现，不是原full practical成绩或增强训练收益。正式clean＋六practical固定IQ不修改、旧产物不覆盖。单模型seed/单独立噪声实现不构成跨seed重现证据。

## 本地验证与发布

5项聚焦检查实际PASS（0.336秒）：标签盲稳定抽样、canonical full保持/独立噪声剂量、重放不匹配/弱化config拒绝、四预测完整后才允许评分及非有限/错序/更新负例、准确率与Macro-F1计数。独立scorer模块导入不加载torch/model/teacher。原coupled物理函数复用已完成源诊断的检查/审查结果。侧对话明确禁止subagents，因此本次不调用或声称完成独立Agent审查；新接线作一次手工P0/P1范围检查，并以协议负测和真实CP烟测验证。不启动额外审批或其他聊天。

代码与config Git提交后，一次release archive/传输SHA比较、远端compile与路径/普通账号/资源preflight；随后独立读回PID/CWD/argv/CPU/GPU/日志增长。精确命令为既有N607 Python `-u -m experiments.cvs_target_hq.predict`，CWD为新release。唯一owner为codex/side/target-hq-20261015，输出根/日志/源CP路径详见experiment.json。任何来源、handles、replay、非有限、预测、目录冲突或技术失败仅退出本owned CPU进程，保留产物，无自动重试，不动其他任务。

当前尚未发布；不存在目标域HQ性能结论。实际发布commit、独立运行读回与完整结果将在同一报告补齐。原A0 E50成本与本次0训练/评价墙钟分开报告。

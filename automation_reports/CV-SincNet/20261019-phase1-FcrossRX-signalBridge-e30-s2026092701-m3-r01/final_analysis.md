# HQ 标签监督完整结果

三条新增E10尾段、累计E30均训练并完成clean及六种practical独立评分；技术缺项0。固定+6dB/.25配方源条件FAIL，关闭、不扩E50或种子。

|场景|HQ准确率|相对等成本As+raw|
|---|---:|---:|
|clean|84.6768%|−0.6042pp|
|high|82.0155%|+0.5929pp|
|mid|79.3988%|+0.8006pp|
|low suburban|76.6851%|+0.7881pp|
|high urban|78.4381%|+0.6667pp|
|mid urban|63.8369%|+1.0994pp|
|low urban|52.2661%|+1.0804pp|

相对原As，low+1.0220pp、mid+1.0304pp、urban均值+0.9077pp，clean−0.6643pp。相对匹配NoAs+raw，clean−0.4982pp，不满足0.1pp阶段限制。较清晰训练观测带来小幅urban改善，但未实现clean保持，也未达low60%。仅一个模型seed，无跨seed复现证据。

源资格先于本会话读取正式指标保存并观察：midlow相对As+raw+1.1250pp，fullVclean相对NoAs+raw−0.1963pp，clean边界失败。正式结果不能推翻该决定。

完整宏F1、逐RX/day/TX、配对修复/新增错误、每行成本见results_summary.json。完整2220更新×3结构化日志已读，见source_log_summary.json；actual物理序列相同，全部流齐全。新增30模型轮/6660更新，每行L1989120前向，原U学生/teacher各567000；45k源比较另属新研究，不归本实验。历史父E20成本与新增E10分列。原基准已暴露，不称独立盲测。

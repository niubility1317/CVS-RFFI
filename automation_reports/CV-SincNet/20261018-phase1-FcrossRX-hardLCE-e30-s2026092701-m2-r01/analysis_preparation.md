# P45 完整分析准备

状态：IMPLEMENTED_LOCAL_VERIFIED；训练仍健康运行，尚无最终源或正式成绩。新增分析脚本固定四格 B00/H00/B01/H01、五个配对比较及 loss定义×As 交互，不增加模型前向、拟合、IQ或checkpoint读取。

source 模式只读原源角色/physics、冻结源logits、全30轮/6660步、L/U完整物理序列、实际前向计数及硬标签CE记录。独立重算源条件，逐步恢复 CE0、同一H更新权重下的CE01、原KL与SupCon总值；不把反事实CE01称为旧对照训练轨迹。只比较 U opaque ID，不恢复U标签。原native CE01诊断列不当作实际CE0目标。

formal 模式须已有完成的 source-only分析以及全部四格七视图固定prediction/独立scoring。随后独立重算504个CM、accuracy/macroF1/最差RX、固定B00 FINAL30 clean正确集合保持率、同ID修复/新增错误、cosine间隔÷30、urban均值、各suburban绝对值与clean差距及交互。2新模型与2复用控制的成本分列，既有P43共享烟测不冒充本轮新增成本；本轮新增烟测1538NN。

已有分支/门控逐包训练轨迹没有保存，本分析不补造；源state/lock/quality不是正式目标状态或因果控制标签。每seed结果仍只有2026092701，信道draw不当模型seed。统计、曲线或导数不替代真实身份准确率，也不把本损失消融称为IQAg。

7项纯合成检查PASS：固定源条件及缺失/非有限反例、跨run解析作用域恢复、因子交互单位、六类CM计算，以及source完成/全部冻结检查先于formal collect的程序路径。0真实源/target、IQ、CP、NN或更新；两脚本AST PASS。分析准备中修正了函数局部变量遮蔽模块的缺陷，并加上述路径测试；没有运行失败的真实分析或改动健康训练。

预期source图：完整120 epoch记录（60新＋60复用）、CE值重算、全部非空state/quality层、源margin/固定集合保持率。预期formal图：全七视图、全部效应、mid/low逐RX、margin。仅在实际产物生成并目视检查后交付，不预告图已存在。原始输出保留local_artifacts/远端run，公开必要聚合。

一次独立候选P0/P1审查已在训练发布完成，不对未改训练作重复审查。分析代码先Git提交，零NN阶段只发布实际消费代码；一次新分析归档传输比较与编译，不再加载无关checkpoint或要求额外许可。source waiter只跟踪已独立证实live的所属dispatcher/workers，源文件未齐且所属handles全消失时保留并报错，不自动重启。source完成后先固定并提交源决定，再手动启动已授权formal分析；自动训练后predict/scoring保持原队列，不因source负值停健康任务。

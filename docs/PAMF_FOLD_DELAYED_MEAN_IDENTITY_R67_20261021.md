# R67：近期裁剪与轨迹平均版的固定无 F 身份诊断

状态：**VERIFIED / WAITING_PARENT_FORMAL**，已发布。33 项聚焦检查、CLI 和唯一一次独立 P0/P1 审查通过。R66正在源训练，尚未读取任何R66识别率。全部已有方法、权重、正式预测与评分保持；本诊断不训练或选择恢复器。

固定纳入R64整组折叠full/half、R65延迟裁剪5+15/10+10、R66同一源轨迹的final/late_mean/step_ema七个新读出，另列指定M35和R53静态同图控制。负结果同样纳入，不按源域或测试表现筛选。R66三个读出共享一次源E20，不能当作三次独立训练或三个seed；本诊断源训练成本为零。

等待R64、R65、R66全部七景正式评分完成，随后读取已保存的同168000条IQ×7视图；不重跑任何神经预测、不加载checkpoint/F权重/embedding/logits，不拟合分类头或新校准。CPU2、无GPU，等待父时不占R41评分锁，不延迟原训练测试及交付。

复用R56的raw/M35/R53三读出84个解析单元，按既有R34源LU8192固定坐标计算七新读出196单元，共280单元。mirror64、spectral4、reference24、combined92与std下限0.05完全保持。所有单元固定、有限且物理ID一致后，独立truth侧才加载本诊断标签。

clean参考每TX/RX/day最多2条，按opaque physicalID排序，参考RX和day均与query不同，完整六TX单元等权。报告4536条分组、756条199次RX/day内query标签置换阴性对照及504条固定配对比较。七新版本逐一对M35、R53；另外保留R64full对half、R65两时机和R66均值/EMA对同轨迹final。独立NumPy复算验证全部数值，缓存M35/R53的1008组记录与R56复用一致。

这是已暴露工程基准的事后描述性同TX关联，使用target clean参考及truth作评价。不是新增识别率、未知TX泛化、真实在轨验证或逐细节TX来源证明。尤其原clean描述符在TX2已有不足；所有弱关联/负margin、逐TX/RX/day、有效及零向量/缺参考计数完整保留，不能以某一组提高代替全部性能或真实性。用户lowurban70%、其他五景距rawclean≤5pp及更强输入TX细节目标继续ACTIVE。

停止规则：父技术失败、方法/路径/物理ID/旧源校准不符、缺失或非有限输入/特征、输出冲突时保留产物并记录技术失败，不自动重试，不停止/修改健康父任务。代码Git固定、最小聚焦否定检查、一次限P0/P1审查、普通账户一次预检/单归档SHA/compile及真实两条Ls解析no-query smoke后自动继续。


实际启动状态：VERIFIED / WAITING_PARENT_FORMAL：普通账户、单归档SHA、compile、真实2Ls无参数解析no-query smoke、实际PID/CWD/argv/空CUDA/CPU2及日志增长已核实。等待R64/R65/R66全部七景正式评分；复用R56raw/M35/R53共84单元，新算七读出196单元，总280先于truth固定。0F/checkpoint/新神经预测/训练/新校准/头拟合/GPU，不改变父实验或选择；无新统计结论，目标ACTIVE。

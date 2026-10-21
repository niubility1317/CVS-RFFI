# R33：响应、统计与梯度协调八版的固定无 F 身份关联

状态：STOPPED_BY_USER（2026-10-21 15:59，UTC+8）。用户明确停止本诊断以及 R32 两行训练，不影响 432 行域泛化对比。

停止结果 VERIFIED：本诊断调度器已独立核实退出，未继续等待或启动提取、评分；既有配置、smoke 与日志全部保留。R32 两行及其调度器同时按精确归属停止，432 行对比继续正常运行。远端停止证据为本 run 的 `user_stop_20261021.json`；未删除产物、未自动重启。

固定纳入已发布的 R26 两版扰动输出/响应、R28 两版冻结 prior 合并、R30 两版四阶统计参考、R32 两版梯度合成。所有八版均在原168000×七景正式预测与独立评分完成后才进入解析评价，保留负结果，不按分数选择、调整或重跑。固定原350521父与R22两版历史控制，不使用这些分数选择新训练父；所有新训练父与IQ教师仍只有原350521。

六个新训练版是E20/19700；R28是冻结重参数化、299297部署参数、0新增训练，原R22源权重为278180/E20/19700。R33不要求R28不存在的training_complete，逐行检查checkpoint_source/runtime_ready/collapse及原R22完整训练权限；没有伪造新训练或混同来源。原350521父source-finalE20/4940另核对。所有行原F50 seed2026092701/defaulthead、TX/F/V/target训练权限保持，父任务不因本评价等待或受干预。

评价完整沿用R24/R34固定mirror64、spectral4、reference24、combined92和原sourceLU8192均值/尺度/std下限0.05。复用R24 raw、原350521及native R22两版112特征单元，八新版新增224单元，总336；新增9408000次参数自由解析输入，新单元约13848576000字节，其余为只读symlink。0F权重、embedding/logits/输出、神经checkpoint、新神经预测、模型训练、头拟合或源拟合。

四个父run全部八行完成原168000物理样本×clean/三郊区/三urban七景独立评分后才读保存restoredIQ。全部336特征及原ID固定、有限后，独立truth-side统计6TX/7RX/4day；不改变预测。clean参照只用于事后评价：每TX/RX/day按opaqueID取至多2条，RX和day同时不同，完整6TX单元等权。固定199次RX/day内query标签置换，参照标签不动，非独立p值。

总11候选（原父、nativeR22两版、新八版），5544组同/异TX余弦、margin、正margin比例（不是分类准确率）、有效/缺失/零参照、配对描述误差；924条件控制；固定19对比×七景×四解析组＝532配对差。对比包括八版各对原父，R26组内、R28对对应native两版、R30组内、R32组内及对历史R30teacher两版、R26/R30各对历史R22恒等。完整矩阵见design配置，无候选重排。

已有clean TX2描述符区分不足，逐TX保留。解析同TX关联改善不能替代分类收益或证明每个细节的TX来源，负margin也不单独判为错误TX。七景accuracy/MacroF1、父逐景0.8pp、lowurban70%、其余五景距rawclean5pp、clean保持及实际部署成本仍由各原run单列。既有已暴露工程基准，不宣称新增独立盲测。

N607普通账户、CPU2、空CUDA/0GPU；等四父全8行评分完成后在R06共享score.lock内串行extract/独立score。复用已验证源数据句柄，0新增数据重验证。一个方法commit、一次预检/归档两端SHA/compile；launcher首步真实2Ls参数自由解析noquery smoke，无神经checkpoint被该阶段消费。实际PID/CWD/完整命令/空CUDA/日志增长后核实。父技术失败、错误路径/ID/源统计、缺失或非有限输入/输出冲突保留产物，不自动重试或干预健康父任务。

43项合成检查覆盖R24精确数学别名、全矩阵、已训练/冻结两类预算与权限负例、四父未完成时不打开array、全部cells固定前truth关闭、cache与新矩阵计数区分、原源坐标和ID、不变配置。唯一独立P0/P1审查PASS，无P0/P1。精确命令、方法commit、环境/CWD和路径见experiment.json；整体联合优化目标仍ACTIVE。


## 真实发布与自动收尾

方法commit 19e91827a4e8555eaf3cdc29a3f8d4137ea04800 已提交、推送并独立匹配远端OID。普通账户一次预检核实源READY/EXACT_MATCH、R34原8192统计、已完成R24缓存和四父实际控制器；归档两端一次SHA一致，远端compile/module入口通过。

真实2Ls参数自由解析no-query smoke PASS，0神经checkpoint/F/TX/V/target/更新。实际CPU控制器PID/CWD/完整命令/空CUDA/2线程和日志390→1170字节增长均已核实。当前等待R26/R28/R30/R32全部八行七景正式评分，extract/score均未启动，0GPU占用，无身份统计结果；父任务保持。

43项执行检查及唯一独立P0/P1审查PASS，独立复跑43项；结果工具21项检查PASS，核对336cells/5544组/924控制/532比较、原坐标、权限、完整分组、非有限/缺项/重复及margin算术。以上合成检查不是实际身份保真收益。

单一本地完成owner只读等待既有controller/extract/score全部实际退出，取回已有完整summary与配置/时序，独立核对全部聚合算术，自动报告、提交、推送并核验远端OID。它不启动、停止、重试或修改远端任务，不重新读取IQ/truth或重复评分；其他既有完成owner保留。联合目标仍ACTIVE。

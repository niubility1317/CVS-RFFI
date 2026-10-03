# CVS补偿与顺序差：完整源V冻结归因

状态PLANNED。本轮推进纯架构研究，补齐[源机制报告](../20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/mechanism_report.md)明确缺失的D读出/投影及完整V上的支路贡献证据。8份已验证scratch E200 checkpoint仅用于冻结诊断，不继续训练，不选择或输出新部署模型。

对dual四seed各测5条件，order四seed各测6条件：all_on、auxiliary_off、u_off、v_off、g_identity；order另含d_off。全开须复现原源V/最差RX，所有干预面对同27000条合法V记录、同6类；完整模型状态保持不变。保存44份预测以及每包标量，独立重新计算全部准确率、逐RX混淆与帮助/损害计数、标量汇总。

只测源V；目标数据、truth、query、已有测试评分均不作为本程序输入。原选中模型已完成clean确认，当前诊断不追溯重测。终止条件仅技术异常、来源/物理角色不符、非有限值或原全开无法复现；保留所有产物，不干预健康旧任务。

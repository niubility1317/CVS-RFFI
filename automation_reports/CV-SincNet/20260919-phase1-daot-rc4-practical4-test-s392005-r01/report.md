# practical四组最新保存权重测试

固定2026-09-19 20:05快照：full_noeq E100、MMSE E100、residual E160；ZF尚无checkpoint，等待首次E100。复用既有自动测试预测并独立CPU评分。

状态：PLANNED。所有结果不反馈训练，不进行checkpoint选优；本次是探索性测试复核。唯一launch owner为当前/root。CPU接续读取训练原有固定预测，最长等待24小时；缺失或失败保留并明确报告。

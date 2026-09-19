# practical四组最新保存权重测试r02

固定2026-09-19 20:05快照：full_noeq E100、MMSE E100、residual E160；ZF尚无checkpoint，等待首次E100。复用既有自动测试预测并独立CPU评分。

r01评分程序遗漏periodic row ID的epoch后缀，技术失败；r02仅修复精确身份校验，不重新推理。原训练健康运行。等待full_noeq E100测试完成及ZF首次E100，已保存MMSE E100和residual E160优先复核。

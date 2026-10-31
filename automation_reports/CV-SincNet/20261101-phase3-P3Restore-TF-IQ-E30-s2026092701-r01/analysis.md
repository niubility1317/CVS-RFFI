# 两臂源训练及辅助测试归档

VERIFIED / ANALYZED。本版完整从零单阶段30轮/7410，原L_s/U_s曝光189000/1701000，0TX/OE/V/目标训练。全部30行结构化日志及完整stdout核对，源顺序、clean masks、初始参数与另一臂逐项一致。完整原定Phase3四角色及旧F50七景预测/独立评分已通过20261101-evaluation-P3Restore-TF-frozen-E30-s2026092701-r02补齐，未重训。

原训练控制器在30轮后因发布包漏掉既有synthetic配置而测试失败，原failure/controller/log保持；新r02 release增加同一b338已跟踪配置，首个真实输入前smoke通过后完成原定矩阵。不是性能原因重跑或选模。

用户最新明确性能为主、效率为支，基座为地面Tri-SIG-TIME SSL。上述Phase3 RX34/旧F50为辅助归档，主结果见[正确Main split主测试](../20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03/analysis.md)。在主测试中，本权重clean保持、六景3pp通过0/6，性能不通过，不替换默认DURA。

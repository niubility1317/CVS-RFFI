# 唯一独立 P0/P1 审查

结论：PASS，P0/P1 为 0。独立 reviewer：`/root/wave_geometry_p0p1_review`。

范围限下一次真实 source-only run 正确性：原 L/V 物理 ID 复用和互斥、U 未加载、整 RX 完全排除 fit、固定投影无数据拟合、逐包归一化、四头同种子同顺序、每头 800 更新与合计 16000、数值退化保留、不可覆盖输出、总 GPU 容量、完整导入依赖和实际 publish 替换。没有新增数据重验、报告 SHA 或审批。

已核实 `preflight/run/dispatch/summarize/publish` 导入成功，发布闭包包含已加载源码依赖。CUDA 聚焦检查 PASS：常复增益最大误差 2.3841858e-7，GL2 最大误差 1.1920929e-7；零包保留、TX GL2 消失反例、5 个非法 config 和源混淆计数检查通过。远端真实 checkpoint 无 query smoke 随发布第一步执行，PASS 后继续。

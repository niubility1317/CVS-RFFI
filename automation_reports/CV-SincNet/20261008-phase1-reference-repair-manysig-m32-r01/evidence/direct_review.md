# 控制器接管独立复核

Agent repair_review：P0/P1 PASS。此前预约竞争P1已关闭：同PID预CUDA预约、连续3次容量检查、超额仅CPU等待，SIGTERM前核对start_ticks。32行原配置和ae783c81 worker保持不变；旧R5/R6不改不signal，完整冻结、预测和独立评分路径保留。

发布后独立读回见direct_start_readback.json：新owner及4个worker存活，原等待owner退出，4行完成至少2个epoch，旧3个owner及4个R5 worker存活，每GPU实际1个CUDA计算进程。3项控制器容量测试通过。

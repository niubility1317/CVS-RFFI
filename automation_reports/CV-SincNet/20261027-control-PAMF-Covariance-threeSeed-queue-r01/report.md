# 三种子协方差原矩阵共享 GPU 队列接管

状态：PLANNED。原科学 run `20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01` 的4个seed02模型已训练到E50；全空闲GPU准入在其他任务进入同卡后，阻塞原定预测与seed03。独立观察GPU4/6/7各只有一个外部CUDA进程、剩余约20763MiB，原队列仍不准入。本控制只调整排队资源，不新增或重选方法。

本run最多4条GPU作业，每卡最多1条own、最多2条总CUDA／pending作业，空余至少10000MiB；CPU scorer最多1条且计入其他owner。保持当前已经实际使用的own并发上限及同卡总负载上限。原28job、3seed、12模型、原冻结release/config、默认头、最终E50预算、所有完成模型/读出均保持；执行科学代码仍是原commit68b6338d20423498c110284c0c40b65a50d37de7。

只对PID/CWD/argv/uid均核实的原本run controller发单次SIGTERM；不终止其训练、预测、评分子进程。停止后按实际进程及完成artifact接管；健康own worker继续，完成结果复用，未知partial/重复worker不覆盖。其他owner无停止、修改、消息或文件写入。新控制状态/快照在独占root，原科学root只继续写既定可变队列状态和未完成输出，不覆写模型和已有日志。

3项针对性检查PASS，覆盖总pending/CUDA计数、每卡1own/总并发4、精确控制器身份、完成复用/健康接管和partial保护。原真实checkpoint无query smoke与科学候选P0/P1继续有效；不新增runner复审或数据重验。实际发布、单archive比较/compile、接管后PID/GPU/log增长将独立读回。科学测试和truth-last原样继续，不能以控制器启动或部分模型完成宣布全部目标完成。

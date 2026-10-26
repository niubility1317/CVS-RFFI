# 三种子协方差原矩阵共享 GPU 队列接管

状态：ANALYZED_CONTROL_COMPLETE。原科学 run `20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01` 的4个seed02模型已训练到E50；全空闲GPU准入在其他任务进入同卡后，阻塞原定预测与seed03。独立观察GPU4/6/7各只有一个外部CUDA进程、剩余约20763MiB，原队列仍不准入。本控制只调整排队资源，不新增或重选方法。

本run最多4条GPU作业，每卡最多1条own、最多2条总CUDA／pending作业，空余至少10000MiB；CPU scorer最多1条且计入其他owner。保持当前已经实际使用的own并发上限及同卡总负载上限。原28job、3seed、12模型、原冻结release/config、默认头、最终E50预算、所有完成模型/读出均保持；执行科学代码仍是原commit68b6338d20423498c110284c0c40b65a50d37de7。

只对PID/CWD/argv/uid均核实的原本run controller发单次SIGTERM；不终止其训练、预测、评分子进程。停止后按实际进程及完成artifact接管；健康own worker继续，完成结果复用，未知partial/重复worker不覆盖。其他owner无停止、修改、消息或文件写入。新控制状态/快照在独占root，原科学root只继续写既定可变队列状态和未完成输出，不覆写模型和已有日志。

3项针对性检查PASS，覆盖总pending/CUDA计数、每卡1own/总并发4、精确控制器身份、完成复用/健康接管和partial保护。原真实checkpoint无query smoke与科学候选P0/P1继续有效；不新增runner复审或数据重验。实际发布、单archive比较/compile、接管后PID/GPU/log增长将独立读回。科学测试和truth-last原样继续，不能以控制器启动或部分模型完成宣布全部目标完成。


实际队列执行commit `285fd8c2bd485fe7043a788f39e94b72214d5d05`，archive比较VERIFIED、远端compilePASS。原controller精确身份核实后退出；接管时所有原GPU worker已自然完成，因此实际adopt数0，没有终止训练／预测／评分进程。新controller的实际PID/CWD/argv已独立核实，原科学job继续使用原release与原config。

接管后4份seed02七景raw预测和独立CPU评分均已完整结束；4份seed03训练已真正启动并推进至E21/4662，真实源样本smoke均PASS，日志增长已核实。原28job目前16完成、4训练运行、8预测/评分排队，0失败。未读取新分数调整seed03，系数／掩码／预算此前已固定。原PL/Sup不重训。该控制run仍RUNNING，不因排队问题解除而标整个三seed矩阵完成。[接管和运行核实](launch_summary.json)。

原28job全部成功，control complete与自然退出已独立读回；没有新增科学行或训练、没有改动其他owner。恢复排队已完成，其本身不表示科学目标完成。完整科学结果见原run的results.md。

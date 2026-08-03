# D131 D92-Lite160 Target125 r2实验报告

- 实验ID：`d131_d92_lite160_target125_20260804_r2`
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 协议：`p2_min_v1`
- 覆盖：125 outer、375 scene、375单candidate arm pair、750 before/after surface、8固定shard。
- r1终态：method lock LF/CRLF归档SHA漂移，在prepare前停止，`NO_PERFORMANCE_RESULT`，不续跑不覆盖。
- r2唯一变更：`.gitattributes`将D131 method lock标记为`-text`，确保Git archive保留冻结LF原始字节；科学方法、模型代码、方法锁内容与SHA、矩阵和参数不变。
- method lock SHA256：`6cfe8659390bf887bf1689edd24a17b6bed9ef103ccf6f5bfde4d36574725e15`。
- 远端新run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_target125_20260804_r2`。
- 修复commit：`680e016835390492fcdc6d3d42e8e2e7848e13bb`；26项回归通过。
- r2 runtime archive：`d131_d92_lite160_target125_20260804_r2_runtime_680e0168_v3.tar`，SHA256=`7c8b20911e2a914bcca7912b665992061d8ae3693dc971d1650187dc1d34e406`；以固定Git pathspec命令生成两次，大小与SHA一致；实际解包method lock/core/adapter/CLI均匹配冻结SHA。
- 独立复审：`P0=0,P1=0,RELEASE_READY=yes`。
- N607 prepare与真实checkpoint no-query smoke通过；8个shard启动后，5个不同shard出现同一确定性异常`D108 exact top tie must fail closed`，触发预注册技术停止。
- 其余run-owned PID已精确终止，GPU释放，SSH连接清零；保留281个partial prediction作为执行证据，未merge/validate/truth/score，不产生性能结果。
- 后续以新方法锁和新run ID运行D131-D92-LITE160-QTIE/r2；r2不续跑、不覆盖、不读取partial性能。

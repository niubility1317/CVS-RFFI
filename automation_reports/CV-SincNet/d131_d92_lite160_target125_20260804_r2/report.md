# D131 D92-Lite160 Target125 r2实验报告

- 实验ID：`d131_d92_lite160_target125_20260804_r2`
- 当前状态：`LOCAL_PACKAGING_REPAIR_VERIFYING`
- 协议：`p2_min_v1`
- 覆盖：125 outer、375 scene、375单candidate arm pair、750 before/after surface、8固定shard。
- r1终态：method lock LF/CRLF归档SHA漂移，在prepare前停止，`NO_PERFORMANCE_RESULT`，不续跑不覆盖。
- r2唯一变更：`.gitattributes`将D131 method lock标记为`-text`，确保Git archive保留冻结LF原始字节；科学方法、模型代码、方法锁内容与SHA、矩阵和参数不变。
- method lock SHA256：`6cfe8659390bf887bf1689edd24a17b6bed9ef103ccf6f5bfde4d36574725e15`。
- 远端新run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_target125_20260804_r2`。
- 待完成：26项回归、独立P0/P1复审、commit、正式runtime archive、N607预检、真实smoke与完整矩阵。

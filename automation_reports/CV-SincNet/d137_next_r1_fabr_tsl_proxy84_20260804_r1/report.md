# D137 NEXT-R1 FABR-TSL proxy84实验报告

本文件为根目录D137报告的Git承载镜像。

|字段|值|
|---|---|
|状态|`LOCAL_VERIFIED / CODE_REVIEW_PASS_P0_0_P1_0 / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|目标|以新run ID修复D136正有限rho_h的FP16表示上溢，执行完全不变的84行六臂|
|实现提交|`87fd85daa1b221b3f09f56c3d252603557d96c81`|
|方法语义|定向向下`FP16 mantissa×2^exp`，保证`0<effective≤raw`；Type-7 q05、rho_cell及`eta=min(1,rho_h/D)`不变|
|矩阵|42 folds×K1/K5=84行；`R0Q/R0F/R0L/R1Q/R1F/R1L`|
|判定|K5三项比较均需`ΔH>0`、总正确数增加且retained、held、floor不下降|
|验证|58项通过；独立复核`P0=0、P1=0、CODE_REVIEW_PASS`|
|TSL SHA256|`3abe3dfaf87021aea4719f8871ced8ee26ac9d98efbf12837d9001d7f2caacc0`|
|远端run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d137_next_r1_fabr_tsl_proxy84_20260804_r1`，发布前必须ABSENT|
|停止/重试|仅协议、安全或确定性技术故障停止；不按性能停止；`retry authority=false`|

完整负结果关闭，不调参、不复跑；完整84行封存后才独立score。

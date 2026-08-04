# D136 NEXT-R1 FABR-TSL proxy84实验报告

本文件是根目录报告`E:\type10-7\automation_reports\CV-SincNet\d136_next_r1_fabr_tsl_proxy84_20260804_r1\report.md`的Git承载镜像。实验ID、冻结84行六臂矩阵、输入hash、N607路径、命令、停止规则和结果表均以该根报告的同版本内容为准。

## 发布摘要

|字段|值|
|---|---|
|状态|`LOCAL_VERIFIED / CODE_REVIEW_PASS_P0_0_P1_0 / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|科学实现提交|`d2d8f7b3`|
|候选|唯一`NEXT-R1 FABR-TSL/r1`|
|矩阵|42 folds×K1/K5=84行；六臂`R0Q/R0F/R0L/R1Q/R1F/R1L`|
|关键变更|Phase1总正确数、floor和LOO cosine仅审计，不再阻止首个真实performance；所有数值、绑定、Fisher、tie和functional正确性门保留|
|验证|`ssr-gpu`联合55项通过；`py_compile`、`git diff --check`通过；独立Terra复核`P0=0、P1=0`|
|assets SHA256|`cad04253436421c0721c36b9618fc13a7031dd47ff9a409bfbb721591158ae6d`|
|real SHA256|`532cf86b0276de2d992051a9154bc0b48c9c7c5419d44c1cbac7894dbbe6fbf3`|
|远端run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d136_next_r1_fabr_tsl_proxy84_20260804_r1`；发布前必须ABSENT|
|停止/重试|仅协议、安全或确定性技术故障停止；不按性能停止；`retry authority=false`|

## 同行判定

K5的DA=`R1Q-R0Q`、Lite D92=`R0L-R0F`、联合=`R1L-R1F`均需`ΔH>0`、总正确数增加且A_retained、A_held_proxy、floor不下降；完整负结果立即关闭，不调参、不复跑。

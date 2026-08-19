# ERBT-IDR M2.3／D92 E1-RFGuard首轮同row实验报告

正式报告的Git镜像。当前状态：`LOCAL_VERIFIED / PRE_RELEASE / NO_PERFORMANCE_RESULT`。

完整启动前登记与后续实验回填同步自：`E:/type10-7/automation_reports/CV-SincNet/erbt_idr_m23_rfguard_targetscreen_20260820_v1/report.md`。

冻结矩阵：receiver=`3-19`、seed=`7282101`、`K1/new20`与`K10/new5`、F0–F5、三个`leo_*_weak`场景；协议为`p2_min_v1`和匹配的`VALIDATED_ONCE` capsule/split。真实scorer返回前不得形成性能声明。

本地实现已闭合RF-lite、RF-quality、显式中心／不确定性、实际参与旧类先验传输的类LOO交互偏移、PSD nuisance covariance、K感知门控、无FP32旁路的256/266维F3头、F0–F5执行器、四状态评分和truth-last配对翻转诊断。F0使用当前`P2-FULL`，F1使用原生`P2-A1`并保留历史288维实现以确保严格对照，F2固定`β_rf/β_fft=1/8`；只有M2.3紧凑头进入降维资源声明。聚焦测试26项、相邻回归48项和完整编译检查均通过；独立审查首次发现5个P1，定点修复后复审为`PASS`且无P0。真实checkpoint smoke、commit与N607结果待回填。

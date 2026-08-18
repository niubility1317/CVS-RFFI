# Phase1与Phase2论文方法对比限定报告（Git镜像）

完整中文报告位于工作区：

`E:\type10-7\automation_reports\CV-SincNet\phase1_phase2_paper_method_comparison_20260818\report.md`

本镜像固定本次报告的范围与核心结论，便于代码发布树追踪：

- Phase1只纳入ADV3B02（用户消息中的ADVB02）、RIEI、DRIFT及ADV3B02相关注册头消融；CEN_A31的RIEI/DRIFT同协议结果作为CVS-RFFI论文范围参考，不冒充ADV3B02。
- Phase2只纳入ERTB-IDR与MRIOR-SDA等域适应方法、CSIL/MoPC-HR/Orthogonal Incremental等类增量方法。
- 当前正式MRIOR-SDA基座上的平均结果为：CSIL `old/seen-new/H/F=52.592/19.246/23.626/32.870`，MoPC-HR `5.010/34.006/4.165/80.452`，ERTB-IDR `73.165/72.468/72.409/12.100`（百分比；F为百分点）。
- ERTB-IDR在同一300行交集上的域适应增益为：old `+7.567pp`、seen-new `+15.073pp`、H `+11.833pp`、F `−3.150pp`。
- 不把Oracle、clean-access、开发期诊断或跨协议结果用于正式排名；REG0的新类与H记为`N/A`。

原始实验报告和逐slice表格仍以工作区完整报告及其链接为准。

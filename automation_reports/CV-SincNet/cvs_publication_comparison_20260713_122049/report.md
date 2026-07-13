# CVS论文级方法对比实验Git交接

- 根报告：`E:\type10-7\automation_reports\CV-SincNet\cvs_publication_comparison_20260713_122049\report.md`
- 追踪表：`analysis/cvs_publication_comparison_traceability_20260713.md`
- 统一协议：`docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md`
- 当前阶段：Phase1 seed713101三个baseline已启动并通过Epoch1健康检查；MRIOR-SDA/DADDA-SDA监督目标函数与support/query防泄漏validator已实现并通过6项聚焦测试。CIL适配和监督DA完整runner尚未完成。
- N607当前GPU0-7各有1个既有Phase1训练进程；本次只在GPU0-2各追加1个baseline，符合每GPU最多2个训练实验的默认容量。
- 本地回归：`18 passed,12 subtests passed`。
- Phase2监督DA聚焦测试：`6 passed`。既有CVS-aligned扩展回归另有2项因历史配置文件缺失而失败，与本次新增路径无关。
- 声明边界：尚无本次完整结果，不构成论文性能结论或部署成功证据。

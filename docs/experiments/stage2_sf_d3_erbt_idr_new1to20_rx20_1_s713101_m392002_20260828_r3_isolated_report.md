# D3+ERBT-IDR嵌套新类矩阵实验报告（r3私有代码闭包）

- 状态：STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT
- run ID：`stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r3_isolated`
- fresh run原因：r1/r2均在共享N607代码树导入阶段暴露依赖版本不完整，均无performance result
- Git commit：`5a677938ae1b8261180ef5408845a54516f27049`
- 发布策略：完整Git版`code/cvsrffi`部署到run私有`release/`，不覆盖共享项目代码
- 科学声明：`DIAGNOSTIC_NON_FORMAL`
- 数据：复用r1已闭合`p2_min_v1/VALIDATED_ONCE`嵌套切片
- checkpoint：ADV3B02 CORE90
- receiver/data seed/method seed/K：`20-1/713101/392002/10`
- 场景：3个`leo_*_weak`
- 新类数：`N={1,2,3,5,10,15,20}`
- D3：327步+4-fold support-only OOF温度；每场景一次
- ERBT-IDR：`M29-FFT96-A4/D92-E0-NORF32`
- 四状态：`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`

## 资源、路径与停止规则

输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r3_isolated`。D3使用GPU0/1/2；预测每GPU最多两个。启动前必须在私有release根真实导入D3与ERBT入口。只在协议/query泄漏、错误row、输出覆盖、错误代码根、进程归属不清、无prediction闭合或重复确定性技术异常时停止；低性能不得停止。

## 运行结论

3个D3 delta完成；21格预测首波中13格在support状态拟合前以同一`build_d62_fit`参数不兼容异常退出，3格因SSH并发握手未landed，0个prediction闭合。根因是release含完整`cvsrffi`但未含配套`code/scripts`，回退加载共享旧probe脚本导致版本混用。r3无performance result；r4复用有效D3 delta并部署完整Git`code/`。

# D3+ERBT-IDR嵌套新类矩阵实验报告（r2技术修复）

- 状态：STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT
- run ID：`stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r2_techfix1`
- fresh run原因：r1 release归档遗漏既有模块`stage2_sf_erbt_oldonly.py`，导入阶段失败，无performance result
- Git commit：`afdc007893eedeed99df2aaa581e1e5a0d368805`
- 环境/CWD：N607，`/home/szu2070436088/2510044040/CV-SincNet`，Conda环境`CVS-RFFI`
- 科学声明：`DIAGNOSTIC_NON_FORMAL`
- 数据复用：r1中已一次闭合的`p2_min_v1/VALIDATED_ONCE`嵌套切片；capsule=`cvs-full-ablation-phase2c-t1-rx20-1-method7282101-new20-nested-v1`
- checkpoint：ADV3B02 CORE90
- receiver/seed/K：`20-1/713101/K10`；D3 method seed=`392002`
- 场景：`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`
- 新类数：`N={1,2,3,5,10,15,20}`
- D3：327步、4-fold support-only OOF温度、R16、单A段
- ERBT-IDR：`M29-FFT96-A4/D92-E0-NORF32`
- 四状态：`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`

## 资源与输出

D3三行分配GPU0/1/2；D3闭合后21个预测格按实时资源每GPU最多两个。输出根为`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r2_techfix1`，预期3个delta、21个prediction闭合和21个truth-last score。输入数据只读复用r1的`input/<scene>/nested`。

## 技术停止规则

只在协议/query泄漏、错误receiver/seed/K/scene/split、输出覆盖、错误checkout、进程归属不清、无prediction闭合、scorer连接错误或同一确定性预prediction异常至少出现两次时停止run-owned进程。低性能不得停止。

## 运行结论

clear smoke在导入阶段再次失败：共享N607项目树中的`target_only_progressive_runner.py`缺少`run_sf_tapft_deploy_no_query`。r2未训练、未打开query、无performance result。连续两次依赖缺失表明共享代码基线整体落后，故不再逐文件补漏；r3改用run私有完整Git代码闭包。

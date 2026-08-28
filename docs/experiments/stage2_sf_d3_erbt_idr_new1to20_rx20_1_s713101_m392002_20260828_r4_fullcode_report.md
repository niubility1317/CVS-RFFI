# D3+ERBT-IDR嵌套新类矩阵实验报告（r4完整code闭包）

- 状态：STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT
- run ID：`stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r4_fullcode`
- Git基线：`7c4a650dc5f05840c600a2b99dcfafaf65de40e8`
- 科学声明：`DIAGNOSTIC_NON_FORMAL`
- D3输入：复用r3已完成的3个support-only delta；不重复训练
- 数据：复用r1已闭合`p2_min_v1/VALIDATED_ONCE`嵌套切片
- 矩阵：receiver`20-1`、K10、三场景、`N={1,2,3,5,10,15,20}`，共21个四状态prediction
- release：run私有完整Git`code/`，共享`code/baseline_origin_sat_view.py`仅用于checkpoint反序列化
- 资源：GPU0–7，每GPU最多2格；首波最多16格，第二波5格
- 输出：21个support状态receipt、prediction、receipt及truth-last score
- 停止：协议/query泄漏、错误row、覆盖、错误代码根、无prediction闭合或重复确定性异常；低性能不停止

## 最终状态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。r4完整代码闭包已通过，但首个`leo_clear_weak/new1`在REG1 support拟合阶段触发`D92RegistrationBalancedCovarianceError: D92 requires a locked finite symmetric support registry`。异常发生在query加载前：support receipt=0、prediction receipt=0、truth读取=0。由于用户冻结矩阵包含new1，不能跳过该格或把部分规模冒充完整结果。

## D3域适应已完成数据

|场景|support|训练步数|可训练/实际变化元素|OOF温度|support OOF NLL前|support OOF NLL后|NLL变化|argmax|墙钟|最大RSS|delta|
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
|`leo_clear_weak`|60（6×10）|327|1584/1584|1.387174|1.051367|0.991862|-0.059505|保持|64.74s|1,886,748KiB|8,275B|
|`leo_low_elev_weak`|60（6×10）|327|1584/1584|1.535270|1.177293|1.071981|-0.105312|保持|57.66s|1,888,380KiB|8,275B|
|`leo_rain_weak`|60（6×10）|327|1584/1584|1.280733|0.883969|0.849675|-0.034294|保持|56.07s|1,885,884KiB|8,275B|

三行共同资源：backbone forward/optimizer step均为327；cache forward=0；resident model tensor=4,199,312B；snapshot tensor=6,336B；`query_opened=false`、`query_truth_opened=false`、`source_opened=false`。这里的NLL仅是support-only 4-fold OOF校准数据，不是query性能。

## 嵌套矩阵数据量

|新类数N|注册类总数|support总量|单场景最大query|三场景query|
|---:|---:|---:|---:|---:|
|1|7|70|140|420|
|2|8|80|160|480|
|3|9|90|180|540|
|5|11|110|220|660|
|10|16|160|320|960|
|15|21|210|420|1,260|
|20|26|260|520|1,560|

每个注册类均为K=10 support和20条独立query；全部21格合计5,880条query/状态，四状态原计划产生23,520次独立决策。三场景builder audit均为`VALIDATED_ONCE`且`predictor_truth_isolated=true`。

## 发布与故障链

- r1：release缺`stage2_sf_erbt_oldonly.py`，导入失败。
- r2：共享runner版本落后，缺D3部署入口。
- r3：3个D3完成；prediction因私有包未含配套`code/scripts`而回退到共享旧probe，13格同一异常退出、3格未landed。
- r4：完整Git`code/`闭包、远端编译、D92/D62路径与签名均通过；new1暴露D92方法本身的非对称注册几何限制。

因此目前只有D3 support-only适配结果，不能给出DA0/DA1、REG0/REG1的query准确率、类别准确率、NLL、ECE或交互效应；这些指标全部为`NO_PERFORMANCE_RESULT`，不是0。


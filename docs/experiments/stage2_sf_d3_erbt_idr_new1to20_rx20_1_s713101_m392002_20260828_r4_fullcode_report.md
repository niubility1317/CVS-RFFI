# D3+ERBT-IDR嵌套新类矩阵实验报告（r4完整code闭包）

- 状态：LOCAL_VERIFIED
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


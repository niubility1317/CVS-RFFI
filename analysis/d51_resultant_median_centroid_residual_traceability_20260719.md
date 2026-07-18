# D51 resultant缩放中位数centroid残差追踪

|需求|设计位置|验证计划|状态|
|---|---|---|---|
|D45稳定底座|先完整拟合D45，再加系数残差|audit持久化base/actual state，verifier再调用D45闭包|VERIFIED_PRE_RUN|
|稳健类几何|mean prototype到coordinate-median prototype方向|support逐元素重算＋outlier测试|VERIFIED_PRE_RUN|
|连续类无关缩放|`gamma_c=1-rho_c`|范围、类置换、support rank置换|VERIFIED_PRE_RUN|
|K1/K2回退|median=mean时D45 bitwise fallback|K1/K2合成测试|VERIFIED_PRE_RUN|
|单affine int8|只改FP32 coefficient，随后一次canonical/编译|retry1量化argmax/margin变化0/0/0|VERIFIED|
|无query/role/scan|固定support-only同式公式|105行query0，role/quota/count/global assignment均false|VERIFIED|
|资源闭包|D45 36 fit＋新增support几何账|831,296 MAC-equivalent和117,504比较上界测试|VERIFIED_PRE_RUN|
|详细性能|D51报告第11–20节|完整读取D51/D45/D46各105行|VERIFIED|
|三轮回顾|D51报告第21节|重读目标/项目协议、重建1008条会话索引、联合复核D49–D51|VERIFIED|
|D52预注册|D51报告第21.3节|base-relative有界残差唯一公式，无尺度扫描|LOCKED_BEFORE_IMPLEMENTATION|

声明边界：coordinate median依赖冻结特征坐标，不宣称旋转等变；resultant只作连续离散度缩放，不是置信概率或场景标签。

运行前：D51定向9项、D45＋D51联合20项、D42–D51全链161项均exit0；代码复核P0=0、P1=0。实际outer、量化和artifact状态仍待运行。

attempt0在首fold资源wrapper读取不存在的`resource.coefficient_dimension`而退出，无性能行。修复改为从实际formal state的`log_diag_fp32`一维数组取288，新增直接回归；修复后D51＋D45联合21项、D42–D51全链162项通过。失败目录保留，retry1使用新目录。

retry1完成105/105行、query0、exit0；相对D45改变11/15行，rain after/forget改善至80.00%/11.67pp，但总体new/H降至82.00%/81.16%，min-after46.67%，new→old增至12。最终`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不进入125。

D49–D51三轮回顾确认：D49的全局support CE代理失真，D50未跨越D45决策边界，D51的稳健方向有效但全局小RMS尺度不安全。D52只允许测试报告第21.3节预注册的`gamma_c * base_discriminant_norm * unit_median_direction`，继续同时审查注册前/后old、seen-new和逐类floor。

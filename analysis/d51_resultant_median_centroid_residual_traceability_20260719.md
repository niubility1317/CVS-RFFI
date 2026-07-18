# D51 resultant缩放中位数centroid残差追踪

|需求|设计位置|验证计划|状态|
|---|---|---|---|
|D45稳定底座|先完整拟合D45，再加系数残差|audit持久化base/actual state，verifier再调用D45闭包|VERIFIED_PRE_RUN|
|稳健类几何|mean prototype到coordinate-median prototype方向|support逐元素重算＋outlier测试|VERIFIED_PRE_RUN|
|连续类无关缩放|`gamma_c=1-rho_c`|范围、类置换、support rank置换|VERIFIED_PRE_RUN|
|K1/K2回退|median=mean时D45 bitwise fallback|K1/K2合成测试|VERIFIED_PRE_RUN|
|单affine int8|只改FP32 coefficient，随后一次canonical/编译|D42–D51全链161项；outer待运行|IMPLEMENTED|
|无query/role/scan|固定support-only同式公式|静态字段通过；artifact待运行|IMPLEMENTED|
|资源闭包|D45 36 fit＋新增support几何账|831,296 MAC-equivalent和117,504比较上界测试|VERIFIED_PRE_RUN|
|详细性能|D51报告第5节|完整105行解析|DESIGNED|

声明边界：coordinate median依赖冻结特征坐标，不宣称旋转等变；resultant只作连续离散度缩放，不是置信概率或场景标签。

运行前：D51定向9项、D45＋D51联合20项、D42–D51全链161项均exit0；代码复核P0=0、P1=0。实际outer、量化和artifact状态仍待运行。

# D92 E0当前256维模块消融最小同row实验

## 当前交付状态

- `run_id`：`d92_e0_256_module_ablation_hard11_20260826_v1`
- 状态：`RUNNING`
- 固定代码提交：`03d00050d5eb7998afd53cd2a02515d4b9996d56`
- 这是一个最小同rowscreening，不是完整125矩阵或跨seed确认；尚未产生性能结论。

## 冻结设计

所有7个臂共享接收机`3-19`、`K10/new5`、method seed`7282101`、support seed`7282201`、query seed`7282301`和新类抽样seed`7282401`。每个row内覆盖`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。query只用于预测和独立truth-last评分，绝不进入拟合。

|逻辑臂|比较因素|相对当前256维参考的唯一差异|
|---|---|---|
|`P2-256-FULL`|当前256维参考|无|
|`P2-256-A0`|联合特征|`feature_profile`|
|`P2-256-B0`|稳健中心|`center_profile`|
|`P2-256-S0`|自动收缩协方差|`covariance_profile`；经验协方差加固定`10^-6`ridge|
|`P2-256-C3`|旧/新任务均衡协方差|`covariance_profile`|
|`P2-256-D0`|双几何控制|`geometry_profile`|
|`P2-256-D2`|交叉拟合融合控制|`geometry_profile`|

`F0`被明确排除：本run不创建或运行FP32状态对照。模块六仅记录F3编译状态的字节数和数值闭合，不从本run推导量化性能差异。

## 已核验的启动证据

- 相关Phase2本地回归：`107 passed`；新增目录与计划测试先失败、实现后通过。
- `S0`在活动注册路径中使用经验类内协方差加固定`10^-6`ridge，随后与旧/新任务等权平均；不对不可逆的原始经验协方差直接求逆。
- release bundle本地/远端SHA256一致：`dff455e597b206f789cf0c7936ab7c9d758060b8571f1123008a611ca50d8b35`。
- 远端release目录读回HEAD为固定提交，远端编译通过且检出洁净。
- 7个逻辑臂绑定到同一`VALIDATED_ONCE`feature cache、同一capsule和split；没有received-IQ重建或数据重验。
- sealed plan包含7个独立物理执行、无alias、无`F0`。正式runner PID=`3651047`，其CWD和命令行与冻结release/plan一致。
- GPU4、GPU5的外部任务保持不变；本run只在GPU0–GPU3的7个预登记slot运行。首次健康检查已观察到7个predictor子进程，未见重复的预预测技术异常。

## 结果边界

只有在每个row完成prediction closure并由独立scorer连接truth后，才会追加同row旧类准确率、已见新类准确率、`H`、`F`、`min-old`和`min-new`。中间准确率不触发停止或性能结论。

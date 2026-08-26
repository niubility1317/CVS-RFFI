# D92 E0当前256维模块消融最小同row实验

## 当前交付状态

- `run_id`：`d92_e0_256_module_ablation_hard11_20260826_v1`
- 状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`（针对真实256维D92 E0性能主张）
- 固定代码提交：`03d00050d5eb7998afd53cd2a02515d4b9996d56`
- 这是一个最小同rowscreening，不是完整125矩阵或跨seed确认。7个物理执行与7个独立评分已闭合；但随后发现运行时仍实际使用288维零填充状态，故该批结果不得作为256维性能结论。

## 256维一致性纠正

评分资源收据记录`query_head_mac=3168`。本row共有11个注册类，故实际判别头维数为`3168/11=288`；真实256维状态应为`11×256=2816`MAC。回溯确认根因是：特征投影保留被移除的32维为零填充，D42运行时及row资源收据仍固定使用288维。原始产物完整保留，但标为`NO_PERFORMANCE_RESULT`，不进入本文的256维模块结论。

纠正要求如下：

|ID|要求|状态|验证|
|---|---|---|---|
|R1|FULL使用160+96=256维已编译状态|已验证|修复后FULL/B0/S0/C3/D0/D2均为256维，块边界为`(0,160,256)`|
|R2|A0使用160维已编译状态|已验证|修复后A0为160维，块边界为`(0,160)`|
|R3|资源MAC由已编译状态维数导出|已验证|端到端row测试验证11类FULL为`11×256=2816`MAC|
|R4|相同同row矩阵纠正重跑，且不含F0|待执行|原run不会被覆盖|

修复后的本地验证覆盖了完整7臂目录：FULL、B0、S0、C3、D0、D2均生成真实256维F3已编译状态；A0仅使用160维身份坐标。65项聚焦回归全部通过。一次独立P0/P1复核未发现代码级阻塞项；复核确认新紧凑运行时文件必须随提交和release显式发布，随后才可启动纠正run。

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

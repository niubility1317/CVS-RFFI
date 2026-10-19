# 四步路径与实际推理终点训练 R21

状态：LOCAL_VERIFIED，尚未发布。方法 commit `130e9f594fc9da0244b3ad22fb2ca2021485a6bb` 已推送并独立核对远端 OID。30 项聚焦检查、compile/help、一次独立 P0/P1 PASS。真实源 checkpoint 入口在发布后首步执行，PASS 立即继续。


状态：LOCAL_VERIFIED，已实现并完成本地检查，尚未发布。总目标保持原定义：冻结原 F50/defaulthead 后，正式 low urban≥70%，其余五受损场景各距原始 raw clean≤5 pp；一份恢复 IQ 一致供给原 PAMF 的 FFT/PA/统计/时间分支。恢复训练完整合法 L/U、0 TX标签、0 F权重/输出、0 target；模拟配对 clean 监督须明确披露。

R20 源域负结果及同批128条源V诊断支持检验训练／推理路径差异：含clean端点的路径点误差很小，而自由运行16步没有改善粗恢复。这不是所有流匹配方法无效的证明，也尚未确定唯一失败原因。R21只使用这些源证据提出假设，不根据R19正式分数调整方案。

两个新row为 `path_four` 和 `endpoint_four`。两者均为零起点、相同Encoder5/Core2/time64完整初始权重、四次Euler速度场调用和末端6步软MAP，只有训练目标不同。只继承原R07冻结coarse/作用/白输入方差代理，新速度场scratch，不继承R20或R19训练后prior。输入只含当前received和其估计上下文；不输入clean、真实作用或噪声、TX、场景或F信息。

`path_four` 保持R20的直线路径速度回归：e=(clean−base)/SCALE，z_t=t e，target=e，t均匀。`endpoint_four` 从z0=0真正自由运行四步，再执行与部署相同的6CG，以最终输出对clean的逐包IQ NMSE训练，梯度穿过全部四步和CG，不detach学习路径。二者都消耗相同epsilon/t随机流，即使确定性终点训练不使用它们；不把这些未使用随机数写成额外观测。模型参数、数据曝光、更新预算及完整部署路径匹配；训练Core调用与计算成本不同，实测披露。

四步是本轮新的匹配计算路径，不能把R20的16步旧结果当作仅改变损失的严格控制，也不把步数减少本身当作性能提升。两行均实际推理四步；当前初步选择为零起点确定性恢复，不声称后验生成。仍保留固定128源样本三draw的完整性诊断，但确定性分歧0是构造属性，不是身份真实性证据。

每行完整6300L/56700U，scratch E20/4940、batch256、R07原20轮缓存15%clean/85%residual、data2001/augmentation1912/model2701，固定最后一轮。AdamW2e-4到2e-5、warm247、WD1e-4、clip1、FP32。组件成本coarse20+action20+prior20=E60；历史F50监督E50单列。只改变对应row目标，无附加FFT、解析统计、身份、特征教师或其他损失。

源资格保护原native/R15FFT/R16IQ/R17，并新增已合格R19的源summary作为固定第五参考，不加载其权重。对每个参考均要求mean6NMSE≤1.05倍、cleanNMSE≤.02、low/midurban各≥+.5pp、clean准确率下降≤.1pp、最差lowurban逐TX下降≥−1pp、其他五场景最差变化≥−.5pp。R19源low/mid为88.2778%/92.5556%，对应新行最低88.7778%/93.0556%；目标分数不参与资格。合格者按low/mid均值、最差低urbanTX、最差RX、名称排序冻结一个；没有合格者0target。先做单seed falsification，不擅自扩种子。

只对源合格选定行自动原168000条×七视图正式预测、独立truth-last评分；比较缓存R17/R19正式结果只在新评分完成后进行，不重复旧控制预测。补充同物理clean配对诊断使用已固定的CPU工具，不改变部署决定。逐TX/RX/day、repair/damage、完整参数/曝光/成本及负结果均报告，不能把源提升代替正式跨RX效果。

N607普通账户、新不可覆盖root/release、Git实际代码commit、一次聚焦负测与真实2Ls/noquery入口、一次P0/P1、最小登记、单归档SHA比对/编译、PID/CWD/argv/GPU/log增长。池0/1/5/7、最多两条本任务GPUworker、每卡一条、全owner cap2/free12GB，共享capacity4锁到CUDA PID可见。R20原控制器/worker已退出后才能发布；技术失败保留输出、停止失败行/pending推进，不停止其他健康任务，不按低分重启。

|ID|要求|当前状态|
|---|---|---|
|A21-01|匹配完整初值、四步部署路径与零起点|verified：本地检查|
|A21-02|仅训练目标不同，终点梯度经过四步和6CG|verified：本地梯度及 clean 隔离检查|
|A21-03|全L/U、scratch、原R07来源、0TX/F/target|pending|
|A21-04|五个固定源参考、排序、选定行target隔离|verified：正负路径检查|
|A21-05|真实checkpoint/默认F入口、发布与资源保护|pending|
|A21-06|E20/4940、完整源七视图及固定三draw|pending|
|A21-07|有条件正式七视图、配对身份与完整交付|pending|
|A21-08|多seed、独立物理族、新oracle及逐细节真实性|deferred|

当前 3 verified、4 pending、1 deferred。30 项聚焦检查、编译及入口帮助通过；一次独立 P0/P1 审查 PASS，无阻断问题。真实 checkpoint 入口在远端发布后的首步执行，PASS 后立即继续。训练随机流为 202610202100，主预测为 202610202101，三 draw 为 202610202102/103/104。本地验证不代表源资格或正式目标已经达成。


完整矩阵、命令、环境/CWD、路径、GPU 与停止规则见[登记](experiment.json)。

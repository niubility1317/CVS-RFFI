# PAMF严格L系统：复用CE及T1监督权重的DURA-L读出

当前：**QUEUED（发布已核实，尚无新增性能结果）**。执行代码提交 `da8c217ba2701923255b7782bfc514f8d1591bc9`，Git push与独立remoteOID、一次release归档传输校验、远端compile均已完成。

实际源checkpoint无query检查、八结构L_s输入／梯度与DefaultConcat首末段调用PASS。16项聚焦回归、限定P0/P1独立审查及本地SHUFFLE_TIME CUDA图捕获PASS。

## 矩阵与科学权限

完整固定设置见[监督与物理消融设计](../../../docs/PAMF_SUPERVISED_PHYSICS_ABLATION_20261022.md)；逐行配置与六类seed以本目录experiment.json为准。新增8结构×3seed＝24训练；复用F15/F20各3权重加DURA-L读出，共30预测／评分行；9结构专用时延18case×5batch×5轮＝450case-round。追加85阶段任务，不重训G，不新增m5源数量扫点。

F0固定5源RX、3源日、6TX，L_s6300、V27000；U_s IQ使用0。目标7RX四日168000包，7视图全部raw/G-L预测后独立truth-last；不混入6目标RX的源数量曲线。新F纯CE＋默认sat拼接.68、LS0，无CRIC/RDrop/U/EMA；完整X3 CE复用修复F15，保留T1监督目标的F20单列。G-L仅用同契约L_s及完整L祖先，仍为TX-free/F-free恢复训练。每F从零20+own10+own20，共E50/11100，默认头联合训练。

## 实际运行证据

原supervised-control-r01先完整接管25健康worker并加入全部85任务；随后由`20261022-phase1-T1-DURA-stop-invalid-F-control-r01`正常接管，当前PID1219178/CWD/argv及每worker配置均已核实。17条相同PID日志在约495秒观察中增长；新增24条F仍QUEUED，不能写作已GPU训练。18条尚未启动的DefaultConcat依赖消费者只改变执行release，原预算、配置和输出路径保持。未重复启动已退休控制器。

公开核验：[launch_verification.json](../20261022-phase1-PAMF-supervised-control-r01/launch_verification.json)。完整私有PID/命令/日志元数据保留在local_artifacts/PAMF_supervised_physics_20261022，不提交IQ、权重、原日志和私有runtime。原旧timing失败仍由已经完成的r02替代，不作为新失败重测。

## 剩余工作与交付

截至2026-10-22 21:52香港时间的只读耗时汇总：剩余F126、完整G谱系11、预测172／评分172、专用时延2。约181.7共享worker小时的F训练在假设14至18个有效F槽位下为10.1至13.0小时；含G依赖、测试／CPU评分、独占时延与最终论文／导出的整体暂估16至24小时（香港10月23日约13:52至21:52）。新CE结构速度仍为同路径历史耗时代理；不得将共享worker小时称为独占GPU小时或将估计每小时重置。

IEEE实验章节已补严格监督／物理控制设置，9页PDF编译并目检无溢出；当前只公布设置，不填造新结果。完整源几何与11条件物理敏感性诊断、逐RX/TX/day、Accuracy/MacroF1、净纠错、三Fseed配对与成本在冻结评分后汇入。后续完成全部新旧科学矩阵和导出包后才结束automation1-6。

2026-10-23 完成：严格 L-only 六条读出（F15/F20 各三 F seed）已 ANALYZED。raw 精确复用，4032 CM 与2016净纠错分层复算；实际U曝光0，同一G-L全八阶段只用L。F15纯CE+sat.68/LS0与F20监督正则配方分别报告，384/768前向量不同，不作单组件归因。10份CSV63934数值跨Python最大差4.62e-14，解包再现最大差0。

[严格监督完整六行结果](Lonly_results.md)。更新16页IEEE和不可覆盖r08包；结构24、细F、外部DURA、公平性和两专用时延继续。

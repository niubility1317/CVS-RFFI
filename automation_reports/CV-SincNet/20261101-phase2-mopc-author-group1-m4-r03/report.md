# MoPC-HR 原生 Group1 补全

补齐此前明确列入 Group1 而因原生encoder未绑定的 MoPC-HR。网络采用固定作者代码 ae6554316ad1a2175920e330133a2f103408bf78 的 attention ResNet，保留十个残差注意力块及作者池化；明确适配 WiSig 单记录256点的 I/Q/幅度输入，不拼接多记录、不重采样，不使用旧generic MoPC权重。六类网络999339参数、embedding256；该身份为“MoPC-HR作者网络 WiSig256 输入适配”，不声称原论文数据逐字复现。

四原模型种子各从零训练源A：原Group1 L_s/U_s契约，E200、Adam0.001、L128/U256，clean+satellite L拼接CE0.68，101至200轮普通PL阈值0.95权重1。U TX真值隐藏；固定最终A后，仅clean L_s导出旧六类原型6×256，再执行真实checkpoint无query smoke。没有目标权重继承。

C直接从A开始，无B10，不读取目标旧标签。当前NK新类SUP按每5类一session、每session20epoch、SGD0.01/momentum0.9/wd0.0002，使用已实现paper语义原生CE/原型增强/平方HR/动量原型校正；全注册类logits在query eval。原Group1矩阵：3池、7RX、4seed、MIX01–08主点K5N20、MIX01全16KN、三单场景各5episode主点；共1933预测job与4源训练，逐job全部prediction固定后独立truth-last。

GPU1/5每卡最多一条新增本run训练/预测，启动前实际free>=6000MiB且总GPU进程<3，R42 GPU0/2/3/4/6和其他owner不改。独立source增强缓存/服务和不可变root。任一技术/权限/输出错误停止本profile新入队，排空健康owned进程，无盲重试；固定预算不按target分数改。精确环境、命令和逐行矩阵见experiment.json。

24项定向本地检查通过；一次独立P0/P1未发现阻断。当前PREPARED，尚未发布，无正式性能证据。

本r02仅修复r01发布包漏带源增强cache模块的技术错误，明确包含 practical_full_2x2 cache/plan 与 leo_practical。隔离Python先复现旧边界缺依赖、补齐后通过；一次直接问题定向复查。科学网络、源E200/PL/增强、种子、directC/session预算与目标矩阵保持；新run从零训练，不加载失败产物，全部旧产物保留。

本r03补带完整既有practical_full_2x2依赖包，防止部分包遮蔽固定Phase1中的compat入口。隔离导入测试覆盖 numerical_policy 和 Reader/service/Config；远端真实activate检查通过后才提交controller。方法、矩阵和预算不变，不加载失败run状态。

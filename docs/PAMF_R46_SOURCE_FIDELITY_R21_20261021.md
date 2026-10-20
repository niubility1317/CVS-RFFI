# R21：原基底与压缩恢复器的源 IQ 保真诊断

状态：LOCAL_VERIFIED，尚未发布，无新训练或识别成绩。

本轮从用户原350521基底路线继续，固定诊断原父及R18两版278180完整恢复器，不根据已暴露的目标结果挑选模型。目的在合法源IQ上找出clean恒等、复频谱及相关残差的保真缺口，支持后续F-free性能研发；主对话进行通道/相位/锚定约束研发，本组检查源复频谱和解析保真。

输入仅原7/63/30划分中的训练角色：128条Ls、128条Us，分别在各角色不透明位置均匀取样，不读取TX标签。视图固定为原train clean、原R07 received epoch01及20；既有READY/EXACT_MATCH复用。不读取V、target、F权重或输出，不进行TX评分。每份真实冻结权重首步执行2Ls no-query smoke，然后立即诊断；状态前后逐位检查，不更新权重或持久状态。

记录pairedclean IQ相对误差、prefix232/tail24、按每例clean功率划分的四组复FFT误差、lag1/2/4/8/16归一化相关误差、按clean幅度四分位分组的残差和复增益误差，分别给all/Ls/Us的mean/median/p90/p95/max。额外记录coarse及三stage IQ误差。指标仅说明源IQ拟合，不证明TX细节真实。四分位由每例clean自身计算，不涉及TX、RX或目标评分。

使用N607原普通账户、CPU2、GPU0条worker、一条诊断进程；不占用或修改其他owner健康任务。原完整coarse/action/prior/物理解算全部执行。无新增训练、网络、F或头拟合，因此本诊断不产生一个待七景测试的新模型；后续每个新训练版本仍自动完成168000×7正式测试。

代码与配置由一个Git方法commit固定，25项聚焦检查已通过；唯一独立P0/P1审查发现的输出保护问题已修复，两个定向检查和原问题一次定向复核PASS。发布使用新不可覆盖run/release/output，一次预检、一次归档两端SHA、一次compile；启动后核对实际PID/CWD/argv/log增长。技术失败保留、不自动重启或回退。精确矩阵、环境、命令与路径见experiment.json。

R18/R19及R20正式结果已暴露；此研究不称独立盲测，也不从目标成绩另选父模型、教师或rank。后续训练仍只从原R46 source-final E20/4940、方法bbb8daf817de050b958d217b70d4c33f4aefe341出发。

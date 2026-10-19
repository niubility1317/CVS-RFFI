# 无TX/F训练的观测不确定性恢复：三网络匹配对照

状态：LOCAL_VERIFIED，尚未发布。方法 `b57eca9946b2a912a88870e24826fe3e2f70985d` 已提交推送并核实远端OID；10项聚焦检查及一次独立P0/P1通过。N607普通账户预检已完成，原source句柄、四类父权重及R07完整20轮缓存可用，新run/release不存在。

本轮检验R07源域结果引出的假设：估计接收作用存在误差时，只按原生噪声求逆是否过度信任该作用。直接残差先验、原生噪声软MAP、额外作用误差软MAP采用同一新增先验、初始化、数据顺序和20轮/4940更新预算。完整6300L/56700U训练不打开TX标签，不读F输出；损失为配对IQ NMSE+0.1FFT项，置信度另用0.01残差高斯NLL。恢复重建不更新方差头，NLL不更新先验，所有父权重冻结。

额外方差基准只用8192条L/U拟合一次标量。它表示包含偏差的有效残差二阶矩，不是校准后的TX置信度或精确贝叶斯后验。参考组固定identity/coarse/原R03/原R07/全局floor。三个新网络均固定最后20轮，原1800条sourceV的七视图预测全部固定后独立CPU评分，V不更新模型。

仅uncertain主候选可按预登记标准晋级：六场景平均NMSE至少优于匹配direct 2%、不差于原R03、clean NMSE≤0.02；原F50源低城区accuracy比R03至少+0.5个百分点、clean下降≤0.2个百分点、最差TX低城区变化≥−1个百分点，direct clean NMSE≤0.02。波形A不读取F/TX；身份B使用源V标签单独披露。全部通过后自动对direct与uncertain执行原168000条×七视图正式确认，使用原F50/defaulthead和独立truth-last；失败保留全部负结果并维持0target。

单一输出IQ，FFT/PA/time全部由原F图计算。三档郊区、三档城区及clean均单列accuracy、MacroF1、repair/damage、波形与特征指标，并保留逐TX/RX/day。新增训练预算相同，继承成本不同：direct累计E40、物理分支累计E60；原F50历史E50/11100单列，不重训头。科学选择不使用R09目标分数。

运行池GPU0/1/5/7，优先空卡，最多3条本任务worker，每卡1条；全owner cap2且空闲≥12GB。共享capacity4锁从检查保持到子CUDA PID可见，子进程不重复加锁。技术故障只停失败行，保留健康行、暂停未启动行，无自动重试。完整命令、路径、矩阵、来源、阈值、停止规则和预期产物见[experiment.json](experiment.json)，设计追踪见[R10追踪](../../../docs/PAMF_UNCERTAINTY_RESTORE_R10_20261019.md)。

发布状态：RUNNING / VERIFIED。单归档SHA比较、远端compile/help、真实2条L_s smoke均通过；8192LU标量校准完成（825L/7367U，0V/TX/F/target）。参考组七视图预测和独立CPU评分完成。三个训练worker实际位于GPU0/1/5，每条PID/CWD/argv/NVML已核实；训练及最终验收仍待完成，不能宣称有效。

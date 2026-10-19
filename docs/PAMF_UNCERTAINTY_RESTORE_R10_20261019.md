# R10：计入作用估计不确定性的恢复

本轮从 R07 源证据出发：单独监督作用和附加噪声的训练收敛，但平均源恢复 NMSE 增加1.5535%。待检验的假设是，软逆过程把估计作用当成精确作用、只计附加噪声，从而过度相信观测。该解释尚未证明；本轮不依据 R09 等目标结果改网络或阈值。

新增三组均从同一个 R02 frozen coarse 开始，残差 prior 网络相同且从零训练，新增 E20/4940，原6300L/56700U，每轮相同物理顺序与缓存观测。复用 R07 的完整20轮fresh cache（15%clean/85%residual），不重合成信道，不读取任何TX标签或F训练信号。direct_control只执行coarse和残差prior，避免无用物理分支。native_map_prior、uncertain_map_prior额外复用冻结R07的source-only作用估计器。

三组比较：

|组|输出|训练信号|
|---|---|---|
|direct_control|b(r)+s C(r,b(r))|配对IQ NMSE＋0.1 FFT logpower L1|
|native_map_prior|同一learned prior＋6步软CG，使用冻结R07的附加噪声方差|同一波形损失|
|uncertain_map_prior|同一learned prior＋6步加权软CG，方差为native＋学习的excess variance|同一波形损失；另用proper Gaussian NLL单独训练方差|

作用A和coarse始终冻结。uncertainty网络只读当前r、b(r)、A(b(r))，输出每个时刻的正实额外方差。其NLL目标使用训练期clean x形成e=r−A(x)，因而同时包含附加噪声和估计作用误差；clean不输入推理网络。重建损失对方差分支停止梯度，NLL不更新prior，避免把自由调大方差当成优化重建的捷径。标量各向同性/逐时对角噪声均是近似，并不建模全部相关误差。

该有效残差二阶矩还包含作用偏差的平方，并不等于无偏物理噪声方差；也不是经过校准的TX正确概率。这里检验的是软正则更新的置信权重，不宣称严格独立的Bayes后验。

先用固定8192条L/U第一轮观测估计一个共享excess variance初值：max(mean(real residual²)−mean(native variance),1e-6)。这是1个source统计参数拟合，0 optimizer/标签/F/V/target读取，额外曝光单列。另保留该全局floor＋冻结coarse的零新NN更新控制，和原R03确定性恢复器作为源参考。它们不参与新增训练候选排序。

direct_control继承coarse E20＋新增E20，组件累计E40。物理组还继承R07估计器E20，组件累计E60；不能声称累计训练成本完全相同。共同增量预算为E20/4940、batch256、AdamW2e-4、WD1e-4、247步warmup后cosine至2e-5、clip1、FP32。方差NLL权重0.01，仅控制方差网络更新。

源验收固定后不扫阈值：uncertain_map_prior六practical平均NMSE比匹配direct_control至少低2%，且不高于原R03；clean NMSE≤0.02；原F50源低城区accuracy比R03至少高0.5pp，clean相对R03下降≤0.2pp，低城区任一TX下降≤1pp。波形规则不访问F/TX；身份资格另行明确使用源V标签，恢复训练仍是0TX/F。只有预登记的uncertain候选满足全部源条件，才自动对uncertain及匹配direct_control执行原完整正式七视图、原F50/defaulthead的独立truth-last；否则保持0target并保留负结果。原F50/R03参考先固定，不按目标回流选模。

每row完整预测后独立CPU scorer才读truth。报告source与条件性target的clean、三郊区、三城区、TX/RX/day、accuracy/Macro-F1/repair/damage、配对波形与频谱指标、方差校准/成本。只输出一份IQ，所有PAMF分支从它导出，无场景真值路由或输出裁剪。

N607使用允许卡池0/1/5/7，最多3个本run GPU worker，每卡最多1个本run worker，计入其他owner且cap2/free12GB，优先空闲卡。共享capacity4锁覆盖容量检查与CUDA进程可见；不停止其他健康任务。输出root不可覆盖，唯一launch owner；非有限/越权/错来源/碰撞/无法合法预测按预登记停止本row、保留其他健康row，无性能止损或自动重试。

|ID|要求|目标文件|状态|验证|
|---|---|---|---|---|
|U10-01|正确逐时加权实线性软解|pamf_uncertainty_restore/models.py|verified|稠密公式、标量/列向量等价、CG目标下降与有限梯度测试通过|
|U10-02|方差NLL只更新方差，重建只更新prior|models.py、training.py|verified|梯度隔离、冻结父、direct无A前向和推理接口负测通过|
|U10-03|原L/U、固定增量、原cache、0TX/F训练|runtime.py、training.py|verified|3x20/4940，L378000/U3402000，0TX/F/target训练，完整60epoch解析|
|U10-04|8192训练源标量校准和全局floor控制|runtime.py|verified|8192LU(825L/7367U)单标量，0V/TX/F/target；globalfloor七视图完成|
|U10-05|源阈值固定，合格后自动原F50七视图truth-last|evaluation.py、controller.py|verified|源固定阈值实际冻结，未通过全部源标准，0本轮target；合格target分支未执行|
|U10-06|三档郊区和类/RX损伤分别报告|report/results|verified|source七视图和全部TX/RX/day、三郊区accuracy/F1/repair/damage完整|
|U10-07|独占root、共享锁、单owner、健康任务保护|controller.py|verified|实际GPU0/1/5 PID/CWD/argv/NVML/log增长与自然退出已核实|
|U10-08|多seed、独立物理族、逐细节真实性|后续验证|deferred|本轮单seed机制检验，不提前声称最终目标完成|

执行结论：SOURCE_NOT_QUALIFIED。三网络完整20轮、全量L/U和七源视图完成，全部原进程自然退出；实现验证不等于最终逐细节身份目标已完成。

初始化偏差：direct/physical先验70/128状态张量不同，native/uncertain完全一致；不能把本轮称为三组共同初值严格消融。后续种子重置已移到父加载后，11项检查通过及一次定点复审无P0/P1；原R10 release不变。uncertain相对原R03最差TX失败不依赖direct初值。

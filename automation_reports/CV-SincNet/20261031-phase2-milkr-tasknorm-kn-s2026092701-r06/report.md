# MILKR 描述符任务归一化

预登记，未启动。唯一矩阵/seed/命令/环境/owner/路径/stop rule在experiment.json。基座固定完整Tri-SIG-TIME SSL，DURA v1.0及原默认头保持；F/G新增训练与更新0。

## 固定方法与源选择

原R9对三112维signature逐块单位化。R10仅对这三块再做当前support坐标中心化与可选标准化：std=sqrt(每坐标支持方差+fraction×块内平均方差)，最后逐块单位化。原五个特征块与IQ不变。所有location/scale来自当前row支持特征，query只使用冻结参数。K1也有多类支持，统计不使用query；不把support自拟合当正式性能。

源模式固定为none、center、std0.01、std0.1、std1、std10。none精确恢复R9。五源RX×两个既有episode，每类最多20 support与互斥32 held。每模式用K1源held选择原24核×5ridge规则，K5/10/20按当前support条件LOO选核；模式目标是四K的源H均值等权平均。精确平局按预定顺序优先none/简单模式。一套模式共享全部目标K/N/RX，不根据query或某一目标格调fraction。

源选中模式须非none，四K平均H较none至少+0.5pp，且每K不低超过0.5pp，才进入预定目标；否则记SOURCE_ONLY_SCIENTIFIC_STOP、目标预测/评分0。源六类已被F见过，源门槛只是科学筛选，不是真新类性能证据。

## 条件目标与判定

源PASS后运行P01/MIX01、seed2026092701、全部七RX、K1/5/10/20×N2/5/10/20，全补集query和四严格S/Q，560行指标。全部预测固定后独立scorer接truth，匹配原MILKR同split预测复用。主S0Q0平均宏H至少+5pp才达到主性能目标；全部16格均+5pp才称各K/N均达标。P01已曝光，不声称盲测或跨池R9单项因果增益。

两项定向测试PASS：none精确R9，支持统计冻结/查询批次独立/核PSD/schema；一次独立P0/P1审查PASS。真实source-checkpoint无query smoke继续核对实际336及原z/logits不变。每卡GPU0/2/4/6最多2新worker，启动前检查显存，不修改其他owner。独立不可覆盖run root，源科学失败正常停止，技术错误保留产物不原地盲重启。

## 已核实源端结果

**SOURCE_ONLY_SCIENTIFIC_STOP / VERIFIED**。全部六模式完成，选择none（精确R9）。最佳非空模式std0.01相对none平均H变化-0.2285pp。源筛选用时81.340秒，目标IQ/预测/评分0。实际commit `e5ea21ef47427ead33f3965f68827f9eef81234d`，归档匹配与远端compile PASS。

真实F/G checkpoint无query smoke PASS，336捕获及F原z/logits一致。PID1406029曾独立核实存活/CWD/argv/GPU；日志由128增至254字节，后续完整source_selection和正常终态确认结束。无技术故障，不降低原门槛。

对源K5/10/20按路径分解也未发现被混合路径掩盖的正均值：最佳std0.01在S0Q0/S1Q1平均分别-0.3947/-0.3059pp。其他模式同路径平均亦负。该结论限当前源代理与模式集合，不声称所有特征归一化无效；没有目标性能结果。见[source_summary.json](source_summary.json)与[source_path_analysis.json](source_path_analysis.json)。

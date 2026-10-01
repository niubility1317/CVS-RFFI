# RFF 源域已知激励诊断：预登记

状态 PLANNED，尚未发布。研究问题：256 点、25 Msps、equalized1/RMS 接收 IQ 中的 20 点重复段，是否对应已知 20 MHz Wi-Fi L-STF 的系数关系？既定 L6300/V27000 全量，固定 0:80、80:160、160:240 窗口；不训练、不读取 checkpoint、不迭代 U 或 target。测试集评分为 N/A，本诊断没有分类器，不将其完成宣称为性能目标完成。

作者实现的 shifted64 点系数经独立解析核实：tone ±4、±8、±12、±16、±20、±24，间距312.5kHz。连续式在25Msps采样为 s[n]=Σ S_k exp(j2πkn/80)，单位平均功率；20点周期。实现固定320个时移（步长1/16 sample），逐包由lag20估计wrapped CFO后，对四周期平均量求最大归一化相关。记录模板相关、有效tone能量比例、DC、重复相干性、相干周期能量比例、相对频偏与时移；输出角色总体分位数及完整180个TX×RX×day×role单元均值/总体SD。label仅用于分层，不参与任何匹配。

模板残差包含 TX、RX、剩余信道、均衡器、截断与噪声，不能直接称作 TX PA/IQ 参数。L-STF 满足 s*[n]=-j s[n-10]，共轭与时移/复增益有歧义，不能识别 I/Q 约定。频偏仅是模1.25MHz的TX-RX相对量；包内拟合没有跨包或身份学习状态。周期性与tone支撑也可能来自任意周期信号，匹配仅检验更具体的激励对应关系。

合成8项聚焦检查PASS，覆盖实际coeff支撑/归一化、相位增益/时移/CFO/共轭响应、共轭歧义、任意周期tone负例、LTI残差混淆、输入有限性和target/checkpoint拒绝。初始共轭“判定”断言失败，推导发现固有歧义，已移除该不可辨识输出并加入显式等价检查。独立P0/P1审查PASS，无阻断。CPU-only，保护所有其他任务，run/release/log不可覆盖，唯一launch owner/root。

参考：[作者实现固定版本](https://raw.githubusercontent.com/bastibl/gr-ieee802-11/ad0598e4a874f4b8e1f391a1e0323e80df2b34ff/examples/wifi_phy_hier.grc)，[作者关于L-STF周期的论文](https://link.springer.com/article/10.1186/s13638-019-1621-z)。[参考独立读回](evidence/author_reference.json)，[远端资源检查](evidence/remote_preflight.json)。

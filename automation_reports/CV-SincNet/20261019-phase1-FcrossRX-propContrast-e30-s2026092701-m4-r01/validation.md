# P76 已执行的聚焦检查

协议：6组、31负例、5源阈值样例，missing prediction truth closed，PASS。channel：5组／29负例、512次fake process（四格×E21/E30），真实NN/IQ/CP/native process=0，PASS。模型：7负例，同scene OFF loss/gradient/RNG逐位对齐；ON活跃As图、系数.05、ordinary.1、6L768、生成64/copy64、无pair CE保留、finally恢复，PASS。分析：14 CPU fixtures、2 AST PASS；root其余13 Python AST PASS（workers7文件已各自检查）。

原输出位于 local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/{protocol_checks,core_channel_checks_actual,core_model_checks_actual,analysis_checks_actual,syntax_checks}.json。Torch2.1/NumPy2 ABI warning在本地已知环境出现，相关检查exit0且不用失效的numpy桥；真实native图仅在发布launcher首步验证3076NN，不把toy检查写成实际CVS验证。

一次N607普通账号只读preflight通过：project存在，GPU容量可用。独立P0/P1审查进行中，尚未启动训练。

唯一一次连续独立P0/P1审查PASS，覆盖18核心＋2分析文件，记录local_artifacts/cvs_FcrossRX_propContrast_E30_20261019_r01/p0p1_review.md。没有真实NN/IQ/CP或远端访问。

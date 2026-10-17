> 已 ANALYZED／VERIFIED：固定共享响应科学失败，18/18 折及全部 5 RX 均劣于原加性控制，候选关闭。0 IQ／NN／更新／target；不是 Ag 或联合目标完成。见 [完整源结果](source_analysis.md)。下文保留运行前预登记。

# 共享输入响应：已存源矩阵单项预登记

状态：IMPLEMENTED_LOCAL_VERIFIED，尚未运行真实源输入。前一周期信号形状行确认V中跨TX/day的RX关联，但共享加性偏移残余为同TX控制2.609倍，固定资格失败。本行检验不同的输入作用模型，不修改旧测量、阈值或FAIL，也不复活旧FIR／phase／RXMIX固定配方。

唯一行 `SHARED_RX_RESPONSE_L_TO_V`，实际配置见 experiment.json 与 experiments/cvs_rx_response18/design.py。输入仅为已完成 source signalCoherence 行的180个20×20源单元矩阵、其completion/resolved_config；不重新读取IQ、U、source_cache、模型权重或target。不修改任何旧产物。对应物理角色仍是完整L6300／U56700 TX隐藏／原V27000、RX[1,3,4,6,8]／day1/2/3／6TX／256点／split392005。数据句柄复用，禁止增加哈希或数据再验证。

对固定周期20的相关矩阵P做单位DFT，得到G。每个RX有同一20维复响应h，对角D=diag(h)，预测映射为

    Phi_h(G) = D G Dᴴ * tr(G) / tr(D G Dᴴ)

全部20频点保留，参考RX固定为索引1，没有按结果选择。参考响应h=1；其余4RX每折由其他5TX和其他2day的10个对应L单元闭式估计。幅度用log((diagG_r+1e−12)/(diagG_ref+1e−12))在频点中心化后跨L单元平均，再取半指数；相位用跨L单元G_r[k,1]conj(G_ref[k,1])的均值相角。h归一到单位RMS，仅消去不可辨识全局尺度；相位基准固定为周期第一正谐波k=1。无频点筛选、扫描、迭代优化器、classifier、teacher或辅助网络。

每个18折仍保留全部5RX。held TX/day仅为统计泛化检验，不是训练域留出。V中参考RX的矩阵是预测输入，h只由L决定；V不参与拟合、持久状态或模型选择。响应预测转回原坐标后在5RX中心化，和原V RX差异比较Frobenius SSE。两个原控制保持：L-only共享加性偏移与同TX/其他2day条件偏移。共享拟合使用50个L单元/3500包，同TX控制使用10个单元/700包；未把二者说成等样本估计。

固定全部资格：加性控制误差非零；响应相对加性SSE降低至少20%；至少15/18折改善；每个RX的SSE不高于加性控制；总SSE不高于同TX控制的1.05倍；所有相位均值乘积非零。只对SSE加1e−20数值容差。任一失败关闭此固定DFT20／logratio／anchor1／参考RX1／数值floor／总体均值预测配方，不改频点、窗口、阶数、参考RX或阈值救跑。通过仍仅是源预测资格，不自动生成IQ或训练模型。

重要边界：逐包归一化与求平均不交换，Phi(E[P])不必等于E[Phi(P)]；当前是单元均值的预测近似，不是完整物理RX识别。跨RX单元也不是同步配对的同一包；传播、均衡和有限噪声仍混在其中。失败只否定固定近似，不能推断所有共享物理响应不存在；通过也不证明对完整256点施加滤波是身份保持增强，不是替换真实RX。没有改变NoExplicitCFO或网络必要内部补偿。

本地17个负例、已知共享线性响应正例、TX-only／L-V响应反转／TX专属响应反例通过；V变化不改变拟合h，h的全局复尺度不改变预测。720个合成L/V单元矩阵、360次闭式响应拟合、450次合成矩阵预测，0真实IQ／CP／NN／更新。唯一直接P0/P1审查PASS，未发现直接缺陷，未访问真实输入/远程/权重/target或扩大测试。独立recount端到端与2个损坏证据反例也PASS；开发合成总648闭式fit/810矩阵预测，真实NN/更新0。不重复旧候选审查或增加审批。

真实预算固定：180个已有源矩阵／72000复条目，18统计折，72次非参考RX闭式拟合，90个V单元预测，保存1800复响应条目（含参考恒等响应）。0rawIQ／model／CP／NN／teacher／epoch／optimizer更新／头拟合／target。CPU两线程、无GPU，后续模型预算仍E30。参数与原始derived矩阵留在私有run；只提交聚合结果。独立recount从相同L矩阵以独立FFT/归约实现复算全部响应、18折及RX SSE，无新NN或原始IQ读取。

CWD `PROJECT/releases/cvs_F_RX_response_source_20261018_r01`；命令 `CVS-RFFI/bin/python -u -m experiments.cvs_rx_response18.run`；唯一输出 `PROJECT/runs/20261018-diagnostic-F-RXresponse-sourceCells-manysig-m1-r01`；唯一owner `codex/root/F-RX-response-source-20261018`。本地发布入口 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_rx_response18.publish`。实际GitHEAD填入不可变release登记；普通账户一次preflight、单archive一次本远SHA比较、remote compile、launcher源句柄/合成控制smoke通过即继续，不引入未消费CP/NN。

只因所属技术异常、错误输入权限/角色、非有限或输出冲突停止并保留产物，不自动重试，不修改健康旧任务。预期checks、resolved_config、私有responses.npz、全部result/completion、独立recount、同row源报告。历史设计SHA、成员哈希、签名链、额外review或审批均为REJECTED_EXTRA_GATE。

本行没有正式身份识别结果，不替代Ag独立训练价值、As实际适应、clean净正、lowurban>60或suburban接近clean。F＋跨RX同TX／原cosine头联合训练／原U／全源域／256点／E30边界保持，完整联合目标ACTIVE。
